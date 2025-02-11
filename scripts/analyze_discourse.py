import re
from collections import defaultdict
import matplotlib.pyplot as plt
import json
import numpy as np
from pathlib import Path
import seaborn as sns
from datetime import datetime, timedelta

class VTTParser:
    """Parser for VTT format transcripts from MacWhisper."""
    
    def __init__(self, vtt_path):
        self.vtt_path = Path(vtt_path)
        
    def parse(self):
        """Parse VTT file into segments with timestamps and content."""
        with open(self.vtt_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            
        segments = []
        current_time = None
        current_content = []
        
        for line in lines:
            line = line.strip()
            if not line or line == 'WEBVTT':
                continue
                
            # Check for timestamp line (e.g., "00:00:00.000 --> 00:00:05.000")
            if '-->' in line:
                # Save previous segment if exists
                if current_time is not None and current_content:
                    segments.append({
                        'timestamp': current_time,
                        'content': ' '.join(current_content)
                    })
                    current_content = []
                
                # Parse new timestamp
                start_time = line.split(' --> ')[0]
                current_time = self._parse_timestamp(start_time)
            else:
                current_content.append(line)
        
        # Add the last segment
        if current_time is not None and current_content:
            segments.append({
                'timestamp': current_time,
                'content': ' '.join(current_content)
            })
            
        return segments
    
    def _parse_timestamp(self, timestamp):
        """Convert VTT timestamp to seconds."""
        # Handle both HH:MM:SS.mmm and MM:SS.mmm formats
        parts = timestamp.replace(',', '.').split(':')
        if len(parts) == 3:  # HH:MM:SS.mmm
            hours, minutes, seconds = parts
            return float(hours) * 3600 + float(minutes) * 60 + float(seconds)
        else:  # MM:SS.mmm
            minutes, seconds = parts
            return float(minutes) * 60 + float(seconds)

class TranscriptParser:
    """Handles the ingestion and parsing of podcast transcripts."""
    
    def __init__(self, pasted_transcript_path, vtt_path=None):
        self.pasted_transcript_path = Path(pasted_transcript_path)
        self.vtt_path = Path(vtt_path) if vtt_path else None
        self.speakers = {'Dylan Patel', 'Nathan Lambert', 'Lex Fridman'}
        
    def parse(self):
        """Parse transcripts into a structured format."""
        # First parse the pasted transcript to get speaker segments
        speaker_segments = self._parse_pasted_transcript()
        
        if self.vtt_path and self.vtt_path.exists():
            # If we have a VTT file, use it for verbatim content
            vtt_parser = VTTParser(self.vtt_path)
            vtt_segments = vtt_parser.parse()
            return self._combine_transcripts(speaker_segments, vtt_segments)
        else:
            # If no VTT file, combine segments by speaker
            return self._combine_speaker_segments(speaker_segments)
    
    def _parse_pasted_transcript(self):
        """Parse the pasted transcript to get speaker segments with timestamps."""
        with open(self.pasted_transcript_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            
        segments = []
        current_speaker = None
        current_timestamp = None
        current_content = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # If line is a speaker name, process previous speaker's content and start new speaker
            if line in self.speakers:
                if current_speaker and current_timestamp is not None and current_content:
                    segments.append({
                        'speaker': current_speaker,
                        'timestamp': current_timestamp,
                        'content': ' '.join(current_content)
                    })
                current_speaker = line
                current_content = []
                current_timestamp = None
                continue
                
            # If line starts with timestamp, extract it and content
            if line.startswith('(') and current_speaker:
                timestamp_end = line.find(')')
                if timestamp_end > 0:
                    timestamp_str = line[1:timestamp_end]
                    current_timestamp = self._parse_timestamp(timestamp_str)
                    content = line[timestamp_end + 1:].strip()
                    if content:
                        current_content = [content]  # Start new content for this timestamp
                        segments.append({
                            'speaker': current_speaker,
                            'timestamp': current_timestamp,
                            'content': content
                        })
            elif current_speaker and current_timestamp is not None:
                current_content.append(line)
        
        return segments
    
    def _combine_speaker_segments(self, segments):
        """Combine segments by speaker when using only pasted transcript."""
        combined = {}
        
        # Sort segments by timestamp
        sorted_segments = sorted(segments, key=lambda x: x['timestamp'] if x['timestamp'] is not None else float('inf'))
        
        # Combine content for each speaker
        for segment in sorted_segments:
            speaker = segment['speaker']
            if speaker not in combined:
                combined[speaker] = {
                    'speaker': speaker,
                    'timestamp': segment['timestamp'],  # Keep first timestamp
                    'content': []
                }
            combined[speaker]['content'].append(segment['content'])
        
        # Join content and convert to list
        result = []
        for speaker_data in combined.values():
            speaker_data['content'] = ' '.join(speaker_data['content'])
            result.append(speaker_data)
        
        return sorted(result, key=lambda x: x['timestamp'] if x['timestamp'] is not None else float('inf'))
    
    def _parse_timestamp(self, timestamp):
        """Convert timestamp string (HH:MM:SS) to seconds."""
        try:
            hours, minutes, seconds = map(int, timestamp.split(':'))
            return hours * 3600 + minutes * 60 + seconds
        except ValueError:
            return None
    
    def _combine_transcripts(self, speaker_segments, vtt_segments):
        """Combine speaker information with verbatim VTT content."""
        combined_segments = []
        current_speaker = None
        unattributed_segments = []
        
        # Sort both segment lists by timestamp
        speaker_segments.sort(key=lambda x: x['timestamp'] if x['timestamp'] is not None else float('inf'))
        vtt_segments.sort(key=lambda x: x['timestamp'])
        
        # Create debug log
        debug_info = {
            'total_vtt_segments': len(vtt_segments),
            'total_speaker_segments': len(speaker_segments),
            'speaker_changes': [],
            'unattributed_segments': [],
            'alignment_issues': []
        }
        
        # Find the speaker for each VTT segment based on timestamps
        for i, vtt_seg in enumerate(vtt_segments):
            # Find the closest speaker segment before this VTT segment
            speaker_seg = None
            for seg in speaker_segments:
                if seg['timestamp'] is not None and seg['timestamp'] <= vtt_seg['timestamp']:
                    speaker_seg = seg
                else:
                    break
            
            if speaker_seg:
                new_speaker = speaker_seg['speaker']
                if new_speaker != current_speaker:
                    debug_info['speaker_changes'].append({
                        'timestamp': vtt_seg['timestamp'],
                        'previous_speaker': current_speaker,
                        'new_speaker': new_speaker,
                        'content_preview': vtt_seg['content'][:100]
                    })
                    current_speaker = new_speaker
            
            if current_speaker:
                combined_segments.append({
                    'speaker': current_speaker,
                    'timestamp': vtt_seg['timestamp'],
                    'content': vtt_seg['content']
                })
            else:
                unattributed_segments.append({
                    'timestamp': vtt_seg['timestamp'],
                    'content': vtt_seg['content']
                })
                debug_info['unattributed_segments'].append({
                    'timestamp': vtt_seg['timestamp'],
                    'content_preview': vtt_seg['content'][:100]
                })
            
            # Check for potential alignment issues
            if i > 0 and current_speaker:
                time_diff = vtt_seg['timestamp'] - vtt_segments[i-1]['timestamp']
                if time_diff > 30:  # Flag gaps longer than 30 seconds
                    debug_info['alignment_issues'].append({
                        'timestamp': vtt_seg['timestamp'],
                        'gap_duration': time_diff,
                        'speaker': current_speaker,
                        'content_preview': vtt_seg['content'][:100]
                    })
        
        # Save debug information
        debug_path = Path('data/processed/transcript_combination_debug.json')
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        with open(debug_path, 'w', encoding='utf-8') as f:
            json.dump(debug_info, f, indent=2)
        
        # Print summary statistics
        print("\nTranscript combination summary:")
        print(f"Total VTT segments: {len(vtt_segments)}")
        print(f"Total speaker segments: {len(speaker_segments)}")
        print(f"Combined segments: {len(combined_segments)}")
        print(f"Unattributed segments: {len(unattributed_segments)}")
        print(f"Speaker changes: {len(debug_info['speaker_changes'])}")
        print(f"Alignment issues: {len(debug_info['alignment_issues'])}")
        
        if unattributed_segments:
            print("\nWARNING: Some segments could not be attributed to a speaker!")
            print("Check data/processed/transcript_combination_debug.json for details")
        
        if debug_info['alignment_issues']:
            print("\nWARNING: Found potential alignment issues!")
            print("Check data/processed/transcript_combination_debug.json for details")
        
        return combined_segments
    
    def get_word_counts(self):
        """Get word counts for each speaker."""
        segments = self.parse()
        word_counts = defaultdict(int)
        
        for segment in segments:
            speaker = segment['speaker']
            content = segment['content']
            words = len(content.split())
            word_counts[speaker] += words
            
        return dict(word_counts)

class DiscourseAnalyzer:
    """Analyzes discourse patterns in podcast transcripts."""
    
    def __init__(self, segments):
        self.segments = segments
        
    def analyze_right_usage(self):
        """Analyze usage of 'right' in the transcript."""
        instances = []
        speaker_counts = defaultdict(int)
        filtered_instances = defaultdict(list)  # Track filtered instances by reason
        
        # Initialize counts for all speakers to 0
        for segment in self.segments:
            speaker_counts[segment['speaker']] = 0
        
        # Common phrases to filter
        skip_phrases = {
            'all_right': r'\ball\s+right\b',
            'right_now': r'\bright\s+now\b',
            'right_after': r'\bright\s+after\b',
            'right_before': r'\bright\s+before\b',
            'right_away': r'\bright\s+away\b',
            'right_hand': r'\bright\s+hand\b',
            'right_side': r'\bright\s+side\b',
            'right_there': r'\bright\s+there\b'
        }
        
        total_instances = 0
        filtered_count = 0
        
        for segment in self.segments:
            speaker = segment['speaker']
            text = segment['content'].lower()
            
            # Count total raw instances
            raw_matches = list(re.finditer(r'\bright\b', text))
            total_instances += len(raw_matches)
            
            for match in raw_matches:
                # Get surrounding context
                pre_context = text[max(0, match.start()-30):match.start()].strip()
                post_context = text[match.end():min(len(text), match.end()+30)].strip()
                context = text[max(0, match.start()-30):min(len(text), match.end()+30)]
                
                # Get a window of text around the match for checking phrases
                window_start = max(0, match.start() - 10)
                window_end = min(len(text), match.end() + 10)
                check_text = text[window_start:window_end]
                
                # Check each phrase
                should_skip = False
                skip_reason = None
                for phrase_name, pattern in skip_phrases.items():
                    if re.search(pattern, check_text):
                        should_skip = True
                        skip_reason = phrase_name
                        break
                
                if should_skip:
                    filtered_count += 1
                    filtered_instances[skip_reason].append({
                        'speaker': speaker,
                        'context': context,
                        'timestamp': segment.get('timestamp')
                    })
                    continue
                
                # Determine if it's at the end of a sentence
                next_char_idx = match.end()
                is_sentence_end = bool(re.match(r'[.!?,]', text[next_char_idx:next_char_idx+1] if next_char_idx < len(text) else ''))
                
                instances.append({
                    'speaker': speaker,
                    'context': context,
                    'is_sentence_end': is_sentence_end,
                    'full_text': text,
                    'timestamp': segment.get('timestamp')
                })
                speaker_counts[speaker] += 1
        
        # Add filtering statistics to debug info
        debug_info = {
            'total_instances': total_instances,
            'kept_instances': len(instances),
            'filtered_instances': filtered_count,
            'filtered_breakdown': {
                phrase: len(instances) 
                for phrase, instances in filtered_instances.items()
            },
            'filtered_examples': {
                phrase: [
                    {'context': inst['context'], 'speaker': inst['speaker']}
                    for inst in instances[:3]  # Show first 3 examples
                ]
                for phrase, instances in filtered_instances.items()
            }
        }
        
        # Save debug information
        debug_path = Path('data/processed/right_analysis_debug.json')
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        with open(debug_path, 'w', encoding='utf-8') as f:
            json.dump(debug_info, f, indent=2)
        
        # Print summary
        print("\nDetailed 'right' usage analysis:")
        print(f"Total instances found: {total_instances}")
        print(f"Instances after filtering: {len(instances)}")
        print(f"Filtered instances: {filtered_count}")
        print("\nFiltered by phrase:")
        for phrase, count in debug_info['filtered_breakdown'].items():
            print(f"- {phrase}: {count} instances")
        
        return instances, dict(speaker_counts)
    
    def plot_results(self, instances, word_counts):
        """Create visualizations of the analysis results."""
        # Set style for better visibility
        sns.set_style("whitegrid")
        sns.set_palette("husl")
        
        # Create figure with subplots
        fig = plt.figure(figsize=(15, 35), facecolor='white')  # Made taller to accommodate new panel
        fig.suptitle("Analysis of 'right' Usage in Lex Fridman Podcast #459\nBased on MacWhisper VTT Transcript with Speaker Attribution\n" + 
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    fontsize=16, y=0.95)
        
        # 1. Raw frequency by speaker
        plt.subplot(5, 1, 1)  # Changed to 5,1,1 for five panels
        speaker_counts = defaultdict(int)
        for instance in instances:
            speaker_counts[instance['speaker']] += 1
            
        speakers = list(speaker_counts.keys())
        counts = [speaker_counts[s] for s in speakers]
        
        sns.barplot(x=speakers, y=counts)
        plt.title("Distribution of 'right' Usage by Speaker\n(Excluding phrases like 'all right', 'right now')")
        plt.ylabel("Number of Instances")
        
        # Add value labels
        for i, v in enumerate(counts):
            plt.text(i, v, str(v), ha='center', va='bottom',
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1))
        
        # 2. Context analysis (sentence position)
        plt.subplot(5, 1, 2)  # Changed to 5,1,2
        end_count = sum(1 for i in instances if i['is_sentence_end'])
        mid_count = len(instances) - end_count
        
        positions = ['End of Sentence', 'Mid-Sentence']
        counts = [end_count, mid_count]
        
        sns.barplot(x=positions, y=counts)
        plt.title("Position of 'right' in Sentences\nAnalysis of Discourse Marker Position")
        plt.ylabel("Count")
        
        # Add value labels and percentages
        total = end_count + mid_count
        for i, v in enumerate(counts):
            percentage = (v / total * 100) if total > 0 else 0
            plt.text(i, v, f'{v}\n({percentage:.1f}%)', ha='center', va='bottom',
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1))
        
        # 3. Normalized usage rates (per 1000 words)
        plt.subplot(5, 1, 3)  # Changed to 5,1,3
        per_word_rates = []
        
        for speaker in speakers:
            if word_counts[speaker] > 0:
                rate = (speaker_counts[speaker] / word_counts[speaker]) * 1000
                per_word_rates.append(rate)
            else:
                per_word_rates.append(0)
        
        sns.barplot(x=speakers, y=per_word_rates)
        plt.title("Normalized 'right' Usage Rates\nInstances per 1000 Words by Speaker")
        plt.ylabel("Instances per 1000 words")
        
        # Add value labels
        for i, v in enumerate(per_word_rates):
            plt.text(i, v, f'{v:.2f}', ha='center', va='bottom',
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1))
        
        # 4. Time-based analysis (time series)
        plt.subplot(5, 1, 4)  # Changed to 5,1,4
        
        # Convert timestamps to minutes and group by speaker
        speaker_times = defaultdict(list)
        for instance in instances:
            if instance['timestamp'] is not None:
                minutes = instance['timestamp'] / 60  # Convert to minutes
                speaker_times[instance['speaker']].append(minutes)
        
        # Calculate rate per minute in 5-minute windows
        window_size = 5  # minutes
        max_time = max(max(times) for times in speaker_times.values())
        windows = np.arange(0, max_time + window_size, window_size)
        
        for speaker in speakers:
            times = speaker_times[speaker]
            counts_per_window = []
            for start in windows[:-1]:
                end = start + window_size
                count = sum(1 for t in times if start <= t < end)
                counts_per_window.append(count / window_size)  # Convert to rate per minute
            
            plt.plot(windows[:-1], counts_per_window, label=speaker, alpha=0.7)
        
        plt.title("Usage Rate Over Time\nInstances per Minute (5-minute windows)")
        plt.xlabel("Time (minutes)")
        plt.ylabel("Instances per minute")
        plt.legend()
        
        # 5. NEW: Average per-minute usage by speaker
        plt.subplot(5, 1, 5)
        
        # Calculate average per-minute rates
        per_minute_rates = []
        for speaker in speakers:
            times = speaker_times[speaker]
            total_instances = len(times)
            rate = total_instances / max_time * 60  # Convert to per hour for readability
            per_minute_rates.append(rate)
        
        sns.barplot(x=speakers, y=per_minute_rates)
        plt.title("Average 'right' Usage Rate\nInstances per Hour by Speaker")
        plt.ylabel("Instances per hour")
        
        # Add value labels
        for i, v in enumerate(per_minute_rates):
            plt.text(i, v, f'{v:.2f}', ha='center', va='bottom',
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1))
        
        # Add a footer with data source and timestamp
        plt.figtext(0.02, 0.02, 
                   f"Data source: MacWhisper VTT transcript with speaker attribution from pasted transcript\n" +
                   f"Total words analyzed: {sum(word_counts.values()):,}\n" +
                   f"Total duration: {int(max_time)} minutes\n" +
                   f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                   fontsize=10, ha='left',
                   bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=5))
        
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        
        # Save with timestamp in filename and ensure white background
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f'data/processed/discourse_analysis_{timestamp}.png', 
                   dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()

def generate_transcript_comparison(pasted_segments, combined_segments):
    """Generate a side-by-side comparison of pasted and combined transcripts, focusing on segments with 'right'."""
    # Sort both segment lists by timestamp
    pasted_segments = sorted(pasted_segments, key=lambda x: x['timestamp'] if x['timestamp'] is not None else float('inf'))
    combined_segments = sorted(combined_segments, key=lambda x: x['timestamp'] if x['timestamp'] is not None else float('inf'))
    
    # Find segments containing "right"
    pasted_right_segments = [
        seg for seg in pasted_segments 
        if 'right' in seg['content'].lower()
    ][:3]  # Take first 3 instances
    
    # Write comparison to file
    with open('data/processed/transcript_comparison.txt', 'w', encoding='utf-8') as f:
        f.write("TRANSCRIPT COMPARISON\n")
        f.write("====================\n\n")
        
        for i, pasted_seg in enumerate(pasted_right_segments):
            # Find matching combined segment by timestamp
            combined_seg = next(
                (seg for seg in combined_segments 
                 if seg['timestamp'] and pasted_seg['timestamp'] and 
                 abs(seg['timestamp'] - pasted_seg['timestamp']) < 5),
                None
            )
            if combined_seg:
                time_str = str(timedelta(seconds=int(pasted_seg['timestamp'])))
                f.write(f"Example {i+1} ({time_str}) - {pasted_seg['speaker']}\n")
                f.write("-" * 50 + "\n")
                f.write("PASTED: " + pasted_seg['content'] + "\n")
                f.write("VTT   : " + combined_seg['content'] + "\n\n")
        
        # Add statistics
        f.write("\nSTATISTICS\n")
        f.write("==========\n")
        
        # Word counts
        pasted_words = sum(len(seg['content'].split()) for seg in pasted_segments)
        combined_words = sum(len(seg['content'].split()) for seg in combined_segments)
        word_diff = combined_words - pasted_words
        word_percent = (word_diff / pasted_words * 100)
        
        f.write("Word Counts:\n")
        f.write(f"- Pasted: {pasted_words:,} words\n")
        f.write(f"- VTT   : {combined_words:,} words\n")
        f.write(f"- Diff  : {word_diff:+,} words ({word_percent:+.1f}%)\n\n")
        
        # Right usage
        pasted_right = sum(1 for seg in pasted_segments if 'right' in seg['content'].lower())
        combined_right = sum(1 for seg in combined_segments if 'right' in seg['content'].lower())
        right_diff = combined_right - pasted_right
        right_percent = (right_diff / pasted_right * 100)
        
        f.write("'Right' Usage:\n")
        f.write(f"- Pasted: {pasted_right} instances\n")
        f.write(f"- VTT   : {combined_right} instances\n")
        f.write(f"- Diff  : {right_diff:+} instances ({right_percent:+.1f}%)\n\n")
        
        # Add notes
        f.write("\nNOTES\n")
        f.write("=====\n")
        f.write("1. Pasted transcript edited for clarity, removing discourse markers\n")
        f.write("2. VTT transcript is verbatim, preserving speech patterns\n")
        f.write("3. Large word count difference suggests significant editing\n")
        f.write("4. VTT shows many more instances of 'right' as discourse marker\n")
    
    return None

def simple_word_count_comparison():
    """Do a simple word count of both files, counting only content lines."""
    # Count words in VTT file
    vtt_words = 0
    with open('data/raw/macwhisper_full_video.vtt', 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            # Skip empty lines, WEBVTT header, and timestamp lines
            if not line or line == 'WEBVTT' or '-->' in line:
                continue
            # Count words in content lines
            vtt_words += len(line.split())
    
    # Count words in pasted transcript
    pasted_words = 0
    with open('data/raw/pasted_transcript.txt', 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            # Skip empty lines and speaker names
            if not line or line in {'Lex Fridman', 'Dylan Patel', 'Nathan Lambert'}:
                continue
            # If it's a timestamp line, get the content after the timestamp
            if line.startswith('('):
                timestamp_end = line.find(')')
                if timestamp_end > 0:
                    content = line[timestamp_end + 1:].strip()
                    if content:  # If there's content on the same line as timestamp
                        pasted_words += len(content.split())
            else:  # Regular content line
                pasted_words += len(line.split())
    
    print("\nSimple word count comparison:")
    print(f"VTT file: {vtt_words:,} words")
    print(f"Pasted transcript: {pasted_words:,} words")
    diff = vtt_words - pasted_words
    percent = (diff / pasted_words * 100) if pasted_words > 0 else 0
    print(f"Difference: {diff:,} words ({percent:+.1f}%)")

def simple_right_and_speaker_count():
    """Do a simple count of 'right' and words by speaker in both files."""
    # Count in VTT file
    vtt_right_count = 0
    vtt_words = 0
    vtt_speaker_words = defaultdict(int)
    current_speaker = None
    
    # First pass through pasted transcript to get speaker timestamps
    speaker_segments = []
    with open('data/raw/pasted_transcript.txt', 'r', encoding='utf-8') as f:
        current_speaker = None
        current_timestamp = None
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line in {'Lex Fridman', 'Dylan Patel', 'Nathan Lambert'}:
                current_speaker = line
            elif line.startswith('(') and current_speaker:
                timestamp_end = line.find(')')
                if timestamp_end > 0:
                    timestamp_str = line[1:timestamp_end]
                    try:
                        hours, minutes, seconds = map(int, timestamp_str.split(':'))
                        timestamp = hours * 3600 + minutes * 60 + seconds
                        speaker_segments.append({
                            'speaker': current_speaker,
                            'timestamp': timestamp
                        })
                    except ValueError:
                        continue
    
    # Sort speaker segments by timestamp
    speaker_segments.sort(key=lambda x: x['timestamp'])
    
    # Count in VTT file
    with open('data/raw/macwhisper_full_video.vtt', 'r', encoding='utf-8') as f:
        current_time = None
        for line in f:
            line = line.strip()
            if not line or line == 'WEBVTT':
                continue
            if '-->' in line:
                # Parse timestamp
                start_time = line.split(' --> ')[0]
                parts = start_time.split(':')
                if len(parts) == 3:  # HH:MM:SS.mmm
                    hours, minutes, seconds = parts
                    current_time = float(hours) * 3600 + float(minutes) * 60 + float(seconds)
                # Find current speaker based on timestamp
                current_speaker = None
                for seg in speaker_segments:
                    if seg['timestamp'] <= current_time:
                        current_speaker = seg['speaker']
                    else:
                        break
            elif current_time is not None:  # Content line
                vtt_right_count += len(re.findall(r'\bright\b', line.lower()))
                words = len(line.split())
                vtt_words += words
                if current_speaker:
                    vtt_speaker_words[current_speaker] += words
    
    # Count in pasted transcript
    pasted_right_count = 0
    pasted_words = 0
    pasted_speaker_words = defaultdict(int)
    current_speaker = None
    
    with open('data/raw/pasted_transcript.txt', 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line in {'Lex Fridman', 'Dylan Patel', 'Nathan Lambert'}:
                current_speaker = line
            else:
                # If it's a timestamp line, get content after timestamp
                if line.startswith('('):
                    timestamp_end = line.find(')')
                    if timestamp_end > 0:
                        content = line[timestamp_end + 1:].strip()
                else:
                    content = line
                
                if content and current_speaker:
                    pasted_right_count += len(re.findall(r'\bright\b', content.lower()))
                    words = len(content.split())
                    pasted_words += words
                    pasted_speaker_words[current_speaker] += words
    
    # Print results
    print("\nSimple word count comparison:")
    print(f"VTT total words: {vtt_words:,}")
    print(f"Pasted total words: {pasted_words:,}")
    print(f"Difference: {vtt_words - pasted_words:,} words ({((vtt_words - pasted_words) / pasted_words * 100):+.1f}%)")
    
    print("\nSimple 'right' count comparison:")
    print(f"VTT 'right' instances: {vtt_right_count}")
    print(f"Pasted 'right' instances: {pasted_right_count}")
    print(f"Difference: {vtt_right_count - pasted_right_count:+} instances ({((vtt_right_count - pasted_right_count) / pasted_right_count * 100):+.1f}%)")
    
    print("\nWords by speaker in VTT file:")
    for speaker, count in sorted(vtt_speaker_words.items()):
        print(f"{speaker}: {count:,} words ({(count/vtt_words*100):.1f}%)")
    
    print("\nWords by speaker in pasted transcript:")
    for speaker, count in sorted(pasted_speaker_words.items()):
        print(f"{speaker}: {count:,} words ({(count/pasted_words*100):.1f}%)")
    
    print("\nDifferences by speaker:")
    for speaker in vtt_speaker_words:
        vtt_count = vtt_speaker_words[speaker]
        pasted_count = pasted_speaker_words[speaker]
        diff = vtt_count - pasted_count
        percent = (diff / pasted_count * 100) if pasted_count > 0 else float('inf')
        print(f"{speaker}: {diff:+,} words ({percent:+.1f}%)")

def main():
    print("\n=== Analysis using combined transcripts (VTT content with speaker attribution) ===")
    # Parse both transcripts
    combined_parser = TranscriptParser(
        pasted_transcript_path='data/raw/pasted_transcript.txt',
        vtt_path='data/raw/macwhisper_full_video.vtt'
    )
    combined_segments = combined_parser.parse()
    combined_word_counts = combined_parser.get_word_counts()
    
    # Print word count statistics
    total_words = sum(combined_word_counts.values())
    print("\nWord counts by speaker:")
    for speaker, count in combined_word_counts.items():
        percentage = (count / total_words * 100) if total_words > 0 else 0
        print(f"{speaker}: {count:,} words ({percentage:.1f}%)")
    
    # Analyze discourse patterns
    analyzer = DiscourseAnalyzer(combined_segments)
    instances, speaker_counts = analyzer.analyze_right_usage()
    
    # Print right usage statistics
    print(f"\nAnalysis of 'right' usage:")
    print(f"Total instances: {len(instances)}")
    
    print("\nUsage by speaker:")
    for speaker, count in speaker_counts.items():
        words = combined_word_counts[speaker]
        rate = (count / words * 1000) if words > 0 else 0
        print(f"{speaker}: {rate:.2f} instances per 1000 words ({count} instances in {words:,} words)")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    analyzer.plot_results(instances, combined_word_counts)
    plt.savefig('data/processed/discourse_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    print("\nPerforming simple right and speaker word count comparison...")
    simple_right_and_speaker_count()
    print("\nPerforming simple word count comparison...")
    simple_word_count_comparison()
    main() 