#!/usr/bin/env python3

import json
import re
from pathlib import Path
import logging
from collections import defaultdict
import numpy as np
from typing import List, Dict, Any
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
import librosa
import librosa.display
import subprocess
import plotly.graph_objects as go
import plotly.io as pio
import shutil
import argparse

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DiscoursePatternAnalyzer:
    def __init__(self, processed_dir="data/processed/full_podcast", raw_dir="data/raw/full_podcast"):
        self.processed_dir = Path(processed_dir)
        self.raw_dir = Path(raw_dir)
        self.patterns_dir = self.processed_dir / "discourse_patterns"
        self.patterns_dir.mkdir(parents=True, exist_ok=True)
        
    def load_data(self):
        """Load both Whisper analysis and audio segments data."""
        logging.info("Loading data...")
        
        # Load full Whisper output for broader context
        whisper_path = self.processed_dir / "whisper_output_with_speakers.json"
        logging.info(f"Looking for Whisper transcript at {whisper_path}")
        if not whisper_path.exists():
            raise FileNotFoundError(f"Whisper transcript not found at {whisper_path}")
            
        with open(whisper_path, 'r') as f:
            self.whisper_data = json.load(f)
            
        # Generate right analysis if it doesn't exist
        right_path = self.processed_dir / "right_analysis.json"
        if not right_path.exists():
            logging.info("Generating right analysis from Whisper transcript...")
            self.right_instances = self.extract_right_instances()
            with open(right_path, 'w') as f:
                json.dump(self.right_instances, f, indent=2)
        else:
            with open(right_path, 'r') as f:
                self.right_instances = json.load(f)
                
        # Initialize empty audio segments if file doesn't exist
        segments_path = self.processed_dir / "audio_segments.json"
        if not segments_path.exists():
            logging.info("Initializing empty audio segments...")
            self.audio_segments = {
                'rapid_segments': [],
                'context_segments': []
            }
            with open(segments_path, 'w') as f:
                json.dump(self.audio_segments, f, indent=2)
        else:
            with open(segments_path, 'r') as f:
                self.audio_segments = json.load(f)
                
    def extract_right_instances(self):
        """Extract instances of 'right' from the Whisper transcript."""
        instances = []
        
        for segment in self.whisper_data['segments']:
            if segment.get('speaker', '').lower() != 'dylan patel':  # Only analyze Dylan's segments
                continue
                
            text = segment['text'].lower()
            start_time = segment['start']
            end_time = segment['end']
            
            # Find all instances of 'right' in the segment
            for match in re.finditer(r'\bright\b', text):
                # Get surrounding context (up to 100 characters before and after)
                start_idx = max(0, match.start() - 100)
                end_idx = min(len(text), match.end() + 100)
                context = text[start_idx:end_idx]
                
                # Determine if it's at the end of a sentence
                is_sentence_end = bool(re.search(r'right[.!?]\s*$', context))
                
                instances.append({
                    'start_time': start_time,
                    'end_time': end_time,
                    'timestamp': f"{int(start_time//3600):02d}:{int((start_time%3600)//60):02d}:{int(start_time%60):02d}",
                    'context': context,
                    'is_sentence_end': is_sentence_end,
                    'speaker': 'dylan patel'
                })
                
        return instances
    
    def analyze_surrounding_words(self, window_size=5):
        """Analyze words that commonly appear before and after 'right'."""
        before_words = defaultdict(int)
        after_words = defaultdict(int)
        
        # Common English stop words to filter out
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                     'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                     'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'shall', 'should',
                     'can', 'could', 'may', 'might', 'must', 'that', 'which', 'who', 'whom', 'whose',
                     'what', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few',
                     'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
                     'same', 'so', 'than', 'too', 'very', 'just', 'i', 'you', 'he', 'she', 'it',
                     'we', 'they', 'this', 'these', 'those'}
        
        for instance in self.right_instances:
            if instance['speaker'] != 'Dylan Patel':
                continue
                
            context = instance['context'].lower()
            # Split on word boundaries and clean up
            words = [w.strip('.,!?()[]{}:;"\'') for w in re.findall(r'\b\w+\b', context)]
            
            # Find all instances of 'right' in the text
            for i, word in enumerate(words):
                if word == 'right':
                    # Get surrounding words
                    start_idx = max(0, i - window_size)
                    end_idx = min(len(words), i + window_size + 1)
                    
                    # Words before 'right'
                    for w in words[start_idx:i]:
                        if w.isalnum() and w not in stop_words and len(w) > 1:
                            before_words[w] += 1
                    
                    # Words after 'right'
                    for w in words[i+1:end_idx]:
                        if w.isalnum() and w not in stop_words and len(w) > 1:
                            after_words[w] += 1
        
        return dict(before_words), dict(after_words)
    
    def analyze_sentence_position(self):
        """Analyze where 'right' appears in sentences and its punctuation context."""
        positions = defaultdict(int)
        punctuation_patterns = defaultdict(int)
        
        for instance in self.right_instances:
            if instance['speaker'] != 'Dylan Patel':
                continue
                
            context = instance['context']
            
            # Determine sentence position
            if instance.get('is_sentence_end', False):
                positions['end'] += 1
            elif re.search(r'^\W*right\b', context.lower()):
                positions['start'] += 1
            else:
                positions['middle'] += 1
            
            # Analyze punctuation patterns
            match = re.search(r'right[.,!?;]', context.lower())
            if match:
                punctuation_patterns[match.group()[-1]] += 1
            
        return dict(positions), dict(punctuation_patterns)
    
    def analyze_timing_patterns(self):
        """Analyze timing patterns between instances of 'right'."""
        times = []
        gaps = []
        last_time = None
        
        for instance in sorted(self.right_instances, key=lambda x: float(x['start_time'])):
            if instance['speaker'] != 'Dylan Patel':
                continue
                
            current_time = float(instance['start_time'])
            times.append(current_time)
            
            if last_time is not None:
                gap = current_time - last_time
                gaps.append(gap)
            
            last_time = current_time
            
        return times, gaps
    
    def analyze_sentiment_context(self):
        """Analyze the sentiment of context around 'right' instances."""
        sentiments = []
        
        for instance in self.right_instances:
            if instance['speaker'] != 'Dylan Patel':
                continue
                
            context = instance['context']
            blob = TextBlob(context)
            sentiments.append({
                'timestamp': instance['timestamp'],
                'polarity': blob.sentiment.polarity,
                'subjectivity': blob.sentiment.subjectivity,
                'context': context
            })
            
        return sentiments
    
    def find_topic_transitions(self):
        """Identify when 'right' is used around topic transitions."""
        transitions = []
        
        for i, instance in enumerate(self.right_instances):
            if instance['speaker'] != 'Dylan Patel':
                continue
                
            # Get broader context from Whisper data
            current_time = float(instance['start_time'])
            
            # Find segments before and after
            before_text = ""
            after_text = ""
            
            for segment in self.whisper_data['segments']:
                if segment['start'] < current_time and segment['end'] > current_time - 30:
                    before_text += segment['text'] + " "
                elif segment['start'] > current_time and segment['end'] < current_time + 30:
                    after_text += segment['text'] + " "
            
            # Calculate similarity between before and after context
            if before_text and after_text:
                before_blob = TextBlob(before_text)
                after_blob = TextBlob(after_text)
                
                # If contexts are significantly different, might indicate topic transition
                word_overlap = len(set(before_blob.words) & set(after_blob.words)) / len(set(before_blob.words))
                
                transitions.append({
                    'timestamp': instance['timestamp'],
                    'word_overlap': word_overlap,
                    'before_context': before_text.strip(),
                    'after_context': after_text.strip()
                })
        
        return transitions
    
    def analyze_intonation(self, window_seconds=0.5):
        """Analyze pitch patterns around instances of 'right'."""
        logging.info("Starting intonation analysis...")
        
        # Load audio file
        audio_path = self.raw_dir / "sample_segment.m4a"
        if not audio_path.exists():
            logging.error(f"Audio file not found at {audio_path}")
            return []
            
        logging.info(f"Loading audio file: {audio_path}")
        try:
            # First convert to wav using ffmpeg with higher quality settings
            wav_path = self.processed_dir / "temp.wav"
            subprocess.run([
                'ffmpeg', '-y',
                '-i', str(audio_path),
                '-acodec', 'pcm_s16le',
                '-ar', '44100',  # Higher sample rate
                '-ac', '1',      # Convert to mono
                str(wav_path)
            ], check=True, capture_output=True)
            
            logging.info("Loading converted wav file...")
            y, sr = librosa.load(str(wav_path), sr=44100)  # Match ffmpeg sample rate
            wav_path.unlink()  # Clean up temporary file
            
            logging.info("Extracting pitch features...")
            # Use more sensitive pitch detection
            hop_length = 512  # Smaller hop length for better time resolution
            fmin = librosa.note_to_hz('C2')  # Typical male voice lower bound
            fmax = librosa.note_to_hz('C5')  # Typical male voice upper bound
            
            # Extract pitch using more precise settings
            pitches, magnitudes = librosa.piptrack(
                y=y, sr=sr,
                hop_length=hop_length,
                fmin=fmin,
                fmax=fmax,
                threshold=0.1  # Lower threshold for better detection
            )
            
            intonation_patterns = []
            total_instances = len([i for i in self.right_instances if i['speaker'] == 'Dylan Patel'])
            processed = 0
            
            for instance in self.right_instances:
                if instance['speaker'] != 'Dylan Patel':
                    continue
                    
                processed += 1
                logging.info(f"Processing instance {processed}/{total_instances} at {instance['timestamp']}")
                
                # Convert time to frame index
                center_time = float(instance['start_time'])
                center_frame = int((center_time * sr) / hop_length)
                
                # Define window around the word (wider window)
                window_frames = int((window_seconds * sr) / hop_length)
                start_frame = max(0, center_frame - window_frames)
                end_frame = min(pitches.shape[1], center_frame + window_frames)
                
                # Get pitch contour for this window
                window_pitches = pitches[:, start_frame:end_frame]
                window_magnitudes = magnitudes[:, start_frame:end_frame]
                
                # Extract the most prominent pitch at each time
                pitch_contour = []
                for i in range(window_pitches.shape[1]):
                    pitches_t = window_pitches[:, i]
                    mags_t = window_magnitudes[:, i]
                    strong_pitches = pitches_t[mags_t > np.max(mags_t) * 0.1]
                    if len(strong_pitches) > 0:
                        pitch_contour.append(np.median(strong_pitches))
                    else:
                        pitch_contour.append(0)
                
                # Analyze the pitch trend
                if len(pitch_contour) > 0:
                    # Remove zeros and smooth
                    valid_pitches = np.array([p for p in pitch_contour if p > 0])
                    if len(valid_pitches) > 0:
                        # Use median filtering to remove outliers
                        smoothed = np.convolve(valid_pitches, np.ones(5)/5, mode='valid')
                        
                        if len(smoothed) > 1:
                            # Analyze the last 200ms for trend
                            final_window = smoothed[-int(0.2*sr/hop_length):]
                            if len(final_window) > 1:
                                # Use robust statistics
                                start_pitch = np.median(final_window[:len(final_window)//2])
                                end_pitch = np.median(final_window[len(final_window)//2:])
                                pitch_change = end_pitch - start_pitch
                                
                                # Only include significant changes
                                if abs(pitch_change) > 5:  # Hz threshold
                                    pattern = {
                                        'timestamp': instance['timestamp'],
                                        'pitch_change': float(pitch_change),
                                        'is_rising': pitch_change > 0,
                                        'magnitude': abs(pitch_change),
                                        'context': instance['context'],
                                        'start_pitch': float(start_pitch),
                                        'end_pitch': float(end_pitch),
                                        'confidence': float(np.mean(window_magnitudes[:, -len(final_window):]))
                                    }
                                    intonation_patterns.append(pattern)
                                    logging.info(f"Found {'rising' if pitch_change > 0 else 'falling'} "
                                               f"intonation (change: {pitch_change:.1f} Hz)")
            
            logging.info(f"Completed intonation analysis. Found patterns for "
                        f"{len(intonation_patterns)}/{total_instances} instances")
            return intonation_patterns
            
        except Exception as e:
            logging.error(f"Error during intonation analysis: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())
            return []
    
    def get_transcript_duration(self, stats):
        """Calculate total duration of the transcript in hours and minutes."""
        total_duration = sum(s['total_speaking_time'] for s in stats.values())
        hours = int(total_duration // 3600)
        minutes = int((total_duration % 3600) // 60)
        return f"{hours}h {minutes}m"

    def create_speaker_time_series_visualization(self, stats, bin_size_minutes=5, transcript_type="Whisper"):
        """Create a time series visualization of 'right' usage frequency for each speaker independently."""
        plt.figure(figsize=(20, 8))
        
        # Process each speaker independently
        colors = {'dylan patel': 'blue', 'lex fridman': 'green', 'nathan lambert': 'red'}
        
        for speaker, speaker_stats in stats.items():
            if speaker == 'unknown':
                continue
            
            # Get all timestamps where this speaker said "right"
            timestamps = []
            current_time = 0
            cumulative_speaking_time = 0
            
            # Sort turns by timestamp
            sorted_turns = sorted(
                [(int(turn_idx), turn_time) 
                 for turn_idx, turn_time in speaker_stats.get('turn_timestamps', {}).items()],
                key=lambda x: x[1]
            )
            
            # Process each turn in order
            for turn_idx, turn_time in sorted_turns:
                if turn_idx in speaker_stats.get('turn_rights', {}):
                    rights_in_turn = speaker_stats['turn_rights'][turn_idx]
                    # Add timestamp for each "right" instance, using cumulative speaking time
                    for _ in range(rights_in_turn):
                        timestamps.append(cumulative_speaking_time)
                
                # Add the duration of this turn to cumulative time
                if turn_idx < len(sorted_turns) - 1:
                    next_turn_time = sorted_turns[turn_idx + 1][1]
                    turn_duration = next_turn_time - turn_time
                else:
                    # For the last turn, use average turn duration
                    turn_duration = speaker_stats['total_speaking_time'] / speaker_stats['turns']
                
                cumulative_speaking_time += turn_duration
            
            # Create bins based on cumulative speaking time
            total_speaking_minutes = cumulative_speaking_time / 60
            num_bins = int(np.ceil(total_speaking_minutes / bin_size_minutes))
            bins = np.zeros(num_bins)
            
            # Fill the bins
            for timestamp in timestamps:
                bin_idx = int((timestamp / 60) / bin_size_minutes)
                if bin_idx < num_bins:
                    bins[bin_idx] += 1
            
            # Plot time series for this speaker
            x = np.arange(num_bins) * bin_size_minutes
            plt.plot(x, bins, label=f"{speaker.title()} ({len(timestamps)} total)",
                    color=colors.get(speaker, 'gray'),
                    marker='o', markersize=4, alpha=0.7)
        
        plt.title(f"'Right' Usage Frequency Over Speaking Time\n({transcript_type} Transcript - {bin_size_minutes}-minute bins)",
                 pad=20)
        plt.xlabel("Cumulative Speaking Time (minutes)")
        plt.ylabel(f"Frequency (instances per {bin_size_minutes}-minute bin)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        return plt.gcf()

    def create_temporal_scatter_visualization(self, stats, transcript_type="Whisper"):
        """Create a scatter plot showing temporal distribution of 'right' usage."""
        plt.figure(figsize=(15, 5))
        
        # Process each speaker
        colors = {'dylan patel': '#1f77b4', 'lex fridman': '#2ca02c', 'nathan lambert': '#ff7f0e'}
        y_positions = {'dylan patel': 1, 'lex fridman': 3, 'nathan lambert': 2}
        
        # Store instance data for legend
        instances_data = {}
        
        for speaker, speaker_stats in stats.items():
            if speaker == 'unknown':
                continue
            
            # Get all timestamps where this speaker said "right"
            instances = []
            total_words = speaker_stats['total_words']
            total_minutes = speaker_stats['total_speaking_time'] / 60
            
            # Sort turns by timestamp
            sorted_turns = sorted(
                [(int(turn_idx), turn_time) 
                 for turn_idx, turn_time in speaker_stats.get('turn_timestamps', {}).items()],
                key=lambda x: x[1]
            )
            
            # Process each turn
            for turn_idx, turn_time in sorted_turns:
                if turn_idx in speaker_stats.get('turn_rights', {}):
                    rights_in_turn = speaker_stats['turn_rights'][turn_idx]
                    # Add each instance
                    for _ in range(rights_in_turn):
                        instances.append(turn_time / 60)  # Convert to minutes
            
            if instances:
                # Plot instances
                plt.scatter(instances, [y_positions[speaker]] * len(instances),
                           color=colors[speaker], alpha=0.6, s=50)
                instances_data[speaker] = (len(instances), total_minutes, total_words)
        
        # Customize plot
        plt.title(f"Temporal Distribution of 'right' Usage\n({transcript_type} transcript - with speaking time and word counts)")
        plt.xlabel("Time (minutes)")
        plt.yticks(list(y_positions.values()), 
                   [f"{name.title()}" for name in y_positions.keys()])
        plt.grid(True, alpha=0.3, axis='x')
        
        # Add legend with instance counts and speaking time
        legend_labels = [
            f"{speaker.title()} ({count} instances, {minutes:.1f} min, {words:,} words)"
            for speaker, (count, minutes, words) in instances_data.items()
        ]
        legend_handles = [plt.scatter([], [], color=colors[speaker.lower()], alpha=0.6, s=50)
                         for speaker in instances_data.keys()]
        plt.legend(legend_handles, legend_labels, loc='upper right', bbox_to_anchor=(1.0, -0.15))
        
        plt.tight_layout()
        return plt.gcf()

    def generate_visualizations(self, speaker_stats):
        """Generate visualizations for the patterns found."""
        # Set style parameters
        plt.rcParams['figure.figsize'] = [20, 6]
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.alpha'] = 0.3
        
        # Get whisper transcript duration
        whisper_duration = self.get_transcript_duration(speaker_stats['speaker_stats'])
        
        # Create bar plot visualizations
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        fig.suptitle(f"'Right' Usage Analysis (Whisper Transcript - {whisper_duration} total)", fontsize=14, y=1.05)
        
        # Prepare data
        speakers = []
        per_minute = []
        per_1k = []
        per_turn = []
        
        # Get data from whisper transcript stats
        for speaker, stats in speaker_stats['speaker_stats'].items():
            if speaker != 'unknown' and stats['right_instances'] > 0:
                speakers.append(speaker.title())
                per_minute.append(stats['rights_per_minute'])
                per_1k.append(stats['rights_per_1k_words'])
                per_turn.append(stats['rights_per_turn'])
        
        # 1. Rights per minute by speaker
        bars1 = ax1.bar(speakers, per_minute, color='skyblue', alpha=0.7)
        ax1.set_title("Usage per Minute", pad=20)
        ax1.set_ylabel("Instances per Minute")
        ax1.tick_params(axis='x', rotation=45)
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom')
        
        # 2. Rights per 1000 words by speaker
        bars2 = ax2.bar(speakers, per_1k, color='lightgreen', alpha=0.7)
        ax2.set_title("Usage per 1000 Words", pad=20)
        ax2.set_ylabel("Instances per 1000 Words")
        ax2.tick_params(axis='x', rotation=45)
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom')
        
        # 3. Rights per turn by speaker
        bars3 = ax3.bar(speakers, per_turn, color='coral', alpha=0.7)
        ax3.set_title("Usage per Turn", pad=20)
        ax3.set_ylabel("Instances per Turn")
        ax3.tick_params(axis='x', rotation=45)
        for bar in bars3:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.processed_dir / 'discourse_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Add temporal scatter visualization for Whisper transcript
        scatter_fig = self.create_temporal_scatter_visualization(
            speaker_stats['speaker_stats'],
            transcript_type="Whisper"
        )
        scatter_fig.savefig(self.processed_dir / 'discourse_analysis_scatter.png',
                           dpi=300, bbox_inches='tight')
        plt.close(scatter_fig)
        
        # Add time series visualization for Whisper transcript
        time_series_fig = self.create_speaker_time_series_visualization(
            speaker_stats['speaker_stats'],
            bin_size_minutes=5,
            transcript_type="Whisper"
        )
        time_series_fig.savefig(self.processed_dir / 'discourse_analysis_timeseries.png', 
                               dpi=300, bbox_inches='tight')
        plt.close(time_series_fig)
        
        # Now create the same visualizations for pasted transcript data
        if hasattr(self, 'pasted_transcript_stats'):
            # Get pasted transcript duration
            pasted_duration = self.get_transcript_duration(self.pasted_transcript_stats)
            
            # Create bar plots for pasted transcript
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
            fig.suptitle(f"'Right' Usage Analysis (Pasted Transcript - {pasted_duration} total)", fontsize=14, y=1.05)
            
            # Prepare data from pasted transcript
            speakers = []
            per_minute = []
            per_1k = []
            per_turn = []
            
            for speaker, stats in self.pasted_transcript_stats.items():
                if speaker != 'unknown' and stats['right_instances'] > 0:
                    speakers.append(speaker.title())
                    per_minute.append(stats['rights_per_minute'])
                    per_1k.append(stats['rights_per_1k_words'])
                    per_turn.append(stats['rights_per_turn'])
            
            # 1. Rights per minute by speaker
            bars1 = ax1.bar(speakers, per_minute, color='skyblue', alpha=0.7)
            ax1.set_title("Usage per Minute", pad=20)
            ax1.set_ylabel("Instances per Minute")
            ax1.tick_params(axis='x', rotation=45)
            for bar in bars1:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}', ha='center', va='bottom')
            
            # 2. Rights per 1000 words by speaker
            bars2 = ax2.bar(speakers, per_1k, color='lightgreen', alpha=0.7)
            ax2.set_title("Usage per 1000 Words", pad=20)
            ax2.set_ylabel("Instances per 1000 Words")
            ax2.tick_params(axis='x', rotation=45)
            for bar in bars2:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}', ha='center', va='bottom')
            
            # 3. Rights per turn by speaker
            bars3 = ax3.bar(speakers, per_turn, color='coral', alpha=0.7)
            ax3.set_title("Usage per Turn", pad=20)
            ax3.set_ylabel("Instances per Turn")
            ax3.tick_params(axis='x', rotation=45)
            for bar in bars3:
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(self.processed_dir / 'pasted_transcript_discourse_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # Add temporal scatter visualization for pasted transcript
            pasted_scatter_fig = self.create_temporal_scatter_visualization(
                self.pasted_transcript_stats,
                transcript_type="Pasted"
            )
            pasted_scatter_fig.savefig(
                self.processed_dir / 'pasted_transcript_discourse_analysis_scatter.png',
                dpi=300, bbox_inches='tight'
            )
            plt.close(pasted_scatter_fig)
            
            # Add time series visualization for pasted transcript
            pasted_time_series_fig = self.create_speaker_time_series_visualization(
                self.pasted_transcript_stats,
                bin_size_minutes=5,
                transcript_type="Pasted"
            )
            pasted_time_series_fig.savefig(
                self.processed_dir / 'pasted_transcript_discourse_analysis_timeseries.png',
                dpi=300, bbox_inches='tight'
            )
            plt.close(pasted_time_series_fig)
    
    def generate_interactive_visualization(self, intonation_patterns):
        """Generate an interactive HTML visualization with audio playback."""
        logging.info("Generating interactive visualization...")
        
        # Create directory for audio clips
        audio_clips_dir = self.patterns_dir / "audio_clips"
        audio_clips_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract audio clips for each instance
        for i, pattern in enumerate(intonation_patterns):
            timestamp = pattern['timestamp']
            start_time = float(timestamp.split(':')[0])*3600 + float(timestamp.split(':')[1])*60 + float(timestamp.split(':')[2])
            
            # Extract a 2-second clip centered on the instance
            clip_start = max(0, start_time - 1)  # 1 second before
            clip_duration = 2  # 2 seconds total
            
            output_file = audio_clips_dir / f"right_{i+1}.m4a"
            cmd = [
                'ffmpeg', '-y',
                '-i', str(self.raw_dir / "sample_segment.m4a"),
                '-ss', str(clip_start),
                '-t', str(clip_duration),
                str(output_file)
            ]
            
            try:
                subprocess.run(cmd, check=True, capture_output=True)
            except subprocess.CalledProcessError as e:
                logging.error(f"Error extracting audio clip {i+1}: {str(e)}")
                continue
        
        # Create interactive plot
        times = [float(p['timestamp'].split(':')[0])*3600 + 
                float(p['timestamp'].split(':')[1])*60 + 
                float(p['timestamp'].split(':')[2]) 
                for p in intonation_patterns]
        
        changes = [p['pitch_change'] for p in intonation_patterns]
        
        # Create hover text with context and audio player
        hover_texts = []
        for i, pattern in enumerate(intonation_patterns):
            audio_path = f"audio_clips/right_{i+1}.m4a"
            hover_text = f"""
            Time: {pattern['timestamp']}<br>
            Pitch Change: {pattern['pitch_change']:.1f} Hz<br>
            Context: {pattern['context']}<br>
            <audio controls>
                <source src='{audio_path}' type='audio/mp4'>
                Your browser does not support the audio element.
            </audio>
            """
            hover_texts.append(hover_text)
        
        # Create figure
        fig = go.Figure()
        
        # Add rising intonation points
        rising_indices = [i for i, c in enumerate(changes) if c > 0]
        if rising_indices:
            fig.add_trace(go.Scatter(
                x=[times[i] for i in rising_indices],
                y=[changes[i] for i in rising_indices],
                mode='markers',
                name='Rising Intonation',
                marker=dict(color='red', size=12),
                text=[hover_texts[i] for i in rising_indices],
                hovertemplate="%{text}<extra></extra>"
            ))
        
        # Add falling intonation points
        falling_indices = [i for i, c in enumerate(changes) if c <= 0]
        if falling_indices:
            fig.add_trace(go.Scatter(
                x=[times[i] for i in falling_indices],
                y=[changes[i] for i in falling_indices],
                mode='markers',
                name='Falling Intonation',
                marker=dict(color='blue', size=12),
                text=[hover_texts[i] for i in falling_indices],
                hovertemplate="%{text}<extra></extra>"
            ))
        
        # Update layout
        fig.update_layout(
            title="Interactive Intonation Patterns of 'right' Usage",
            xaxis_title="Time in Podcast (seconds)",
            yaxis_title="Pitch Change (Hz)",
            hovermode='closest',
            template='plotly_white'
        )
        
        # Add zero line
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        
        # Save as standalone HTML
        html_path = self.patterns_dir / "interactive_intonation.html"
        
        # Create a complete HTML file with necessary styling
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Interactive Intonation Analysis</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 20px;
                    background-color: #f5f5f5;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    background-color: white;
                    padding: 20px;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                h1 {{
                    color: #333;
                    text-align: center;
                    margin-bottom: 30px;
                }}
                .description {{
                    margin-bottom: 20px;
                    line-height: 1.6;
                }}
                audio {{
                    margin-top: 10px;
                    width: 200px;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Interactive Intonation Analysis</h1>
                <div class="description">
                    <p>This visualization shows the intonation patterns when Dylan uses the word "right".
                    Click on any point to see the context and hear the audio clip.</p>
                    <ul>
                        <li><strong>Red points:</strong> Rising intonation (question-like)</li>
                        <li><strong>Blue points:</strong> Falling intonation (statement-like)</li>
                    </ul>
                </div>
                {pio.to_html(fig, full_html=False)}
            </div>
        </body>
        </html>
        """
        
        with open(html_path, 'w') as f:
            f.write(html_content)
        
        # Copy audio files to a web-accessible location
        web_audio_dir = self.patterns_dir / "audio_clips"
        if web_audio_dir.exists():
            shutil.rmtree(web_audio_dir)
        shutil.copytree(audio_clips_dir, web_audio_dir)
        
        logging.info(f"Interactive visualization saved to {html_path}")
        return html_path
    
    def save_results(self, results):
        """Save analysis results to JSON."""
        output_path = self.patterns_dir / 'discourse_patterns.json'
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
    
    def analyze_speaker_patterns(self):
        """Analyze and visualize how speakers use the word 'right'."""
        logging.info("Analyzing speaking patterns...")
        
        # Track per-speaker statistics
        speaker_stats = defaultdict(lambda: {
            'total_speaking_time': 0,
            'total_words': 0,
            'right_instances': 0,
            'turns': 0
        })
        
        # Analyze each segment
        for segment in self.whisper_data['segments']:
            # Normalize speaker name to lowercase and handle variations
            speaker = segment.get('speaker', 'unknown').lower().strip()
            if 'dylan' in speaker or 'patel' in speaker:
                speaker = 'dylan patel'
            elif 'lex' in speaker or 'fridman' in speaker:
                speaker = 'lex fridman'
            elif 'nathan' in speaker or 'lambert' in speaker:
                speaker = 'nathan lambert'
            
            duration = segment['end'] - segment['start']
            text = segment['text'].lower()
            words = len(text.split())
            right_count = len(re.findall(r'\bright\b', text))
            
            speaker_stats[speaker]['total_speaking_time'] += duration
            speaker_stats[speaker]['total_words'] += words
            speaker_stats[speaker]['right_instances'] += right_count
            speaker_stats[speaker]['turns'] += 1
        
        # Calculate normalized rates
        for speaker, stats in speaker_stats.items():
            # Per minute rate
            minutes = stats['total_speaking_time'] / 60
            stats['rights_per_minute'] = stats['right_instances'] / minutes if minutes > 0 else 0
            
            # Per 1000 words rate
            per_1k = stats['total_words'] / 1000
            stats['rights_per_1k_words'] = stats['right_instances'] / per_1k if per_1k > 0 else 0
            
            # Per turn rate
            stats['rights_per_turn'] = stats['right_instances'] / stats['turns'] if stats['turns'] > 0 else 0
        
        # Analyze turn context
        turn_context = defaultdict(int)
        previous_speaker = None
        
        for segment in self.whisper_data['segments']:
            current_speaker = segment.get('speaker', 'unknown').lower().strip()
            if 'dylan' in current_speaker or 'patel' in current_speaker:
                current_speaker = 'dylan patel'
                if previous_speaker:
                    turn_context[f"After {previous_speaker}"] += 1
            previous_speaker = current_speaker
        
        return {
            'speaker_stats': dict(speaker_stats),
            'turn_context': dict(turn_context)
        }
    
    def analyze_pasted_transcript(self):
        """Analyze the pasted transcript for 'right' usage patterns."""
        pasted_transcript_path = Path("data/raw/pasted_transcript.txt")
        if not pasted_transcript_path.exists():
            logging.warning("Pasted transcript not found, skipping pasted transcript analysis")
            return
            
        # Initialize stats dictionary with additional tracking for timestamps
        speaker_stats = defaultdict(lambda: {
            'total_speaking_time': 0,
            'total_words': 0,
            'right_instances': 0,
            'turns': 0,
            'turn_timestamps': {},  # Store timestamp for each turn
            'turn_rights': {}      # Store number of "rights" in each turn
        })
        
        current_speaker = None
        current_start_time = None
        turn_counter = defaultdict(int)
        
        with open(pasted_transcript_path, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
            
        for i, line in enumerate(lines):
            # Check if line is a speaker name
            if line in ['Lex Fridman', 'Nathan Lambert', 'Dylan Patel']:
                current_speaker = line.lower()
                continue
                
            # Check if line contains timestamp and text
            timestamp_match = re.match(r'\((\d{2}:\d{2}:\d{2})\)\s*(.*)', line)
            if timestamp_match and current_speaker:
                timestamp, text = timestamp_match.groups()
                h, m, s = map(int, timestamp.split(':'))
                start_time = h * 3600 + m * 60 + s
                
                # Calculate duration
                if current_start_time is not None:
                    duration = start_time - current_start_time
                    if duration > 0:  # Only add if duration is positive
                        speaker_stats[current_speaker]['total_speaking_time'] += duration
                
                current_start_time = start_time
                
                # Count words and 'right' instances
                words = len(text.split())
                right_count = len(re.findall(r'\bright\b', text.lower()))
                
                # Store timestamp and rights count for this turn
                turn_idx = turn_counter[current_speaker]
                speaker_stats[current_speaker]['turn_timestamps'][turn_idx] = start_time
                speaker_stats[current_speaker]['turn_rights'][turn_idx] = right_count
                
                speaker_stats[current_speaker]['total_words'] += words
                speaker_stats[current_speaker]['right_instances'] += right_count
                speaker_stats[current_speaker]['turns'] += 1
                turn_counter[current_speaker] += 1
        
        # Calculate normalized rates
        for speaker, stats in speaker_stats.items():
            # Per minute rate
            minutes = stats['total_speaking_time'] / 60
            stats['rights_per_minute'] = stats['right_instances'] / minutes if minutes > 0 else 0
            
            # Per 1000 words rate
            per_1k = stats['total_words'] / 1000
            stats['rights_per_1k_words'] = stats['right_instances'] / per_1k if per_1k > 0 else 0
            
            # Per turn rate
            stats['rights_per_turn'] = stats['right_instances'] / stats['turns'] if stats['turns'] > 0 else 0
        
        self.pasted_transcript_stats = dict(speaker_stats)
        return self.pasted_transcript_stats
    
    def analyze_patterns(self):
        """Analyze and visualize patterns in the transcript."""
        logging.info("Analyzing patterns...")
        
        # First analyze the whisper transcript
        speaker_stats = self.analyze_speaker_patterns()
        
        # Then analyze the pasted transcript
        self.analyze_pasted_transcript()
        
        # Generate visualizations for both
        self.generate_visualizations(speaker_stats)
        
        logging.info("Pattern analysis complete!")
        return speaker_stats

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Analyze discourse patterns in podcast transcripts.')
    parser.add_argument('--processed_dir', type=str, default="data/processed/full_podcast",
                      help='Directory containing processed data')
    parser.add_argument('--raw_dir', type=str, default="data/raw/full_podcast",
                      help='Directory containing raw data')
    args = parser.parse_args()
    
    # Initialize analyzer with provided directories
    analyzer = DiscoursePatternAnalyzer(
        processed_dir=args.processed_dir,
        raw_dir=args.raw_dir
    )
    
    try:
        # Load data first
        analyzer.load_data()
        
        # Then analyze patterns
        results = analyzer.analyze_patterns()
        
        # Print some key findings
        print("\nKey Findings:")
        
        # Handle sentence position stats
        if 'sentence_position' in results and results['sentence_position'] and len(results['sentence_position']) > 0:
            most_common_pos = max(results['sentence_position'].items(), key=lambda x: x[1])[0]
            print(f"Most common position: {most_common_pos}")
        else:
            print("No sentence position data available")
            
        # Handle timing gaps
        if 'timing' in results and results['timing']['gaps'] and len(results['timing']['gaps']) > 0:
            avg_gap = np.mean(results['timing']['gaps'])
            print(f"Average gap between instances: {avg_gap:.2f} seconds")
        else:
            print("No timing gap data available")
            
        # Handle surrounding words
        if 'surrounding_words' in results and results['surrounding_words']['before']:
            print("\nMost common words before 'right':")
            for word, count in sorted(results['surrounding_words']['before'].items(), 
                                    key=lambda x: x[1], reverse=True)[:5]:
                print(f"  {word}: {count}")
        else:
            print("\nNo data available for words before 'right'")
            
        # Handle speaker patterns
        if 'speaker_stats' in results:
            print("\nSpeaker patterns:")
            for speaker, stats in results['speaker_stats'].items():
                if speaker != 'unknown':
                    print(f"\n{speaker.title()}:")
                    print(f"  Total speaking time: {stats['total_speaking_time']/60:.2f} minutes")
                    print(f"  Total words: {stats['total_words']}")
                    print(f"  'Right' instances: {stats['right_instances']}")
                    print(f"  Rights per minute: {stats['rights_per_minute']:.2f}")
                    print(f"  Rights per 1000 words: {stats['rights_per_1k_words']:.2f}")
                    print(f"  Rights per turn: {stats['rights_per_turn']:.2f}")
        else:
            print("\nNo speaker pattern data available")
            
        logging.info("Pattern analysis completed successfully!")
        
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main() 