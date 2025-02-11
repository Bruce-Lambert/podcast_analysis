#!/usr/bin/env python3

import whisper
import torch
import json
import logging
from pathlib import Path
import requests
from bs4 import BeautifulSoup
import re
from datetime import datetime, timedelta
import subprocess
import numpy as np
from tqdm import tqdm
from playwright.sync_api import sync_playwright
import time

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class PodcastAnalyzer:
    def __init__(self):
        self.transcript_url = "https://lexfridman.com/deepseek-dylan-patel-nathan-lambert-transcript"
        self.youtube_url = "https://youtu.be/_1f-o0nqpEI"
        self.raw_dir = Path("data/raw")
        self.processed_dir = Path("data/processed")
        
        # Create directories if they don't exist
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def download_official_transcript(self):
        """Download and save the official transcript"""
        logging.info("Downloading official transcript...")
        
        # First try to load from local file if it exists
        transcript_path = self.raw_dir / "pasted_transcript.txt"
        if transcript_path.exists():
            logging.info("Loading transcript from existing file...")
            transcript_text = transcript_path.read_text()
            return self.parse_official_transcript(transcript_text)
        
        # If no local file exists, download from website
        logging.info("Downloading transcript from website...")
        with sync_playwright() as p:
            # Launch browser
            browser = p.chromium.launch()
            page = browser.new_page()
            
            # Go to transcript page and wait for content to load
            page.goto(self.transcript_url)
            time.sleep(5)  # Give time for JavaScript to execute
            
            # Get all transcript segments
            segments = page.query_selector_all('.ts-segment')
            transcript_text = []
            
            # Extract text from each segment
            for segment in segments:
                # Get speaker name
                name_elem = segment.query_selector('.ts-name')
                name = name_elem.inner_text() if name_elem else ""
                
                # Get timestamp
                time_elem = segment.query_selector('.ts-timestamp')
                timestamp = time_elem.inner_text() if time_elem else ""
                
                # Get text
                text_elem = segment.query_selector('.ts-text')
                text = text_elem.inner_text() if text_elem else ""
                
                # Combine into format matching pasted transcript
                if name:
                    transcript_text.append(name)
                transcript_text.append(f"{timestamp} {text}")
            
            browser.close()
            
            # Join all lines and save to file
            full_transcript = "\n".join(transcript_text)
            transcript_path.write_text(full_transcript)
            
            return self.parse_official_transcript(full_transcript)

    def parse_official_transcript(self, transcript_text):
        """Parse the official transcript to get speaker segments with timestamps"""
        logging.info("Parsing official transcript...")
        
        segments = []
        current_speaker = None
        current_text = []
        
        # Split into lines and process each line
        for line in transcript_text.split('\n'):
            line = line.strip()
            if not line:
                continue
            
            # Check if this is a speaker line (no timestamp)
            if '(' not in line:
                current_speaker = line
                continue
            
            # Look for timestamp pattern (HH:MM:SS)
            timestamp_match = re.search(r'\((\d{2}:\d{2}:\d{2})\)', line)
            if timestamp_match:
                timestamp = timestamp_match.group(1)
                # Get the text after the timestamp
                text = line.split(')', 1)[1].strip() if ')' in line else line
                
                # Convert timestamp to seconds
                time_obj = datetime.strptime(timestamp, '%H:%M:%S')
                seconds = time_obj.hour * 3600 + time_obj.minute * 60 + time_obj.second
                
                segments.append({
                    'speaker': current_speaker,
                    'timestamp': timestamp,
                    'seconds': seconds,
                    'text': text
                })
                
                # Debug logging
                logging.debug(f"Found segment - Speaker: {current_speaker}, Timestamp: {timestamp}, Text: {text[:50]}...")
        
        # Log unique speakers found
        speakers = sorted(set(seg['speaker'] for seg in segments if seg['speaker']))
        logging.info(f"Found speakers in transcript: {speakers}")
        
        # Log a few sample segments
        if segments:
            logging.info("Sample segments:")
            for seg in segments[:3]:
                logging.info(f"  {seg['speaker']} ({seg['timestamp']}): {seg['text'][:50]}...")
        
        return segments

    def download_audio_segment(self, start_time='00:00:00', duration='00:30:00'):
        """Download a segment of the podcast audio"""
        logging.info(f"Downloading {duration} audio segment starting at {start_time}...")
        output_path = self.raw_dir / "sample_segment.m4a"
        
        # Use yt-dlp to download the audio segment
        cmd = [
            'yt-dlp',
            '-x',  # Extract audio
            '--audio-format', 'm4a',
            '--audio-quality', '0',
            '--download-sections', f"*{start_time}-{duration}",
            '-o', str(output_path),
            self.youtube_url
        ]
        
        subprocess.run(cmd, check=True)
        return output_path

    def transcribe_audio_segment(self, audio_path):
        """Transcribe the audio segment using Whisper with word-level timestamps"""
        whisper_output_path = self.processed_dir / "whisper_output.json"
        whisper_transcript_path = self.processed_dir / "whisper_transcript.txt"
        
        # Check if we have existing transcription
        if whisper_output_path.exists():
            logging.info("Loading existing Whisper transcription...")
            with open(whisper_output_path, 'r') as f:
                return json.load(f)
        
        logging.info("Transcribing audio segment with Whisper...")
        model = whisper.load_model("medium")
        result = model.transcribe(
            str(audio_path),
            language="en",
            word_timestamps=True,
            verbose=False
        )
        
        # Immediately save the raw Whisper output
        logging.info("Saving raw Whisper output...")
        with open(whisper_output_path, 'w') as f:
            json.dump(result, f, indent=2)
            
        # Also save a human-readable transcript
        logging.info("Saving human-readable transcript...")
        transcript_text = []
        for segment in result['segments']:
            timestamp = str(timedelta(seconds=int(segment['start'])))
            text = segment['text'].strip()
            transcript_text.append(f"({timestamp}) {text}")
        
        with open(whisper_transcript_path, 'w') as f:
            f.write('\n'.join(transcript_text))
        
        return result

    def analyze_rights(self, whisper_result, official_segments):
        """Analyze occurrences of 'right' in the transcription"""
        logging.info("Analyzing occurrences of 'right'...")
        
        # Create a mapping of time ranges to speakers
        speaker_ranges = []
        for i, seg in enumerate(official_segments):
            end_seconds = official_segments[i+1]['seconds'] if i+1 < len(official_segments) else float('inf')
            speaker_ranges.append({
                'start': seg['seconds'],
                'end': end_seconds,
                'speaker': seg['speaker']
            })
        
        # Process Whisper segments
        right_instances = []
        segments_analyzed = 0
        total_rights = 0
        
        # Process each segment from Whisper
        for segment in whisper_result['segments']:
            start_time = segment['start']
            end_time = segment['end']
            text = segment['text'].lower()
            segments_analyzed += 1
            
            # Find all instances of "right" in this segment
            for match in re.finditer(r'\bright\b', text):
                total_rights += 1
                
                # Calculate approximate start and end times for the word
                word_length = match.end() - match.start()
                segment_length = len(text)
                word_duration = (end_time - start_time) * (word_length / segment_length)
                word_start = start_time + (match.start() / segment_length) * (end_time - start_time)
                word_end = word_start + word_duration
                
                # Get more context (full sentence if possible)
                full_text = text
                context_start = max(0, match.start() - 200)  # Get more context
                context_end = min(len(text), match.end() + 200)
                context = text[context_start:context_end]
                
                # Try to get complete sentences
                if context_start > 0:
                    # Find the start of the sentence
                    sentence_start = context.find('.')
                    if sentence_start != -1:
                        context = context[sentence_start + 1:]
                
                if context_end < len(text):
                    # Find the end of the sentence
                    sentence_end = context.rfind('.')
                    if sentence_end != -1:
                        context = context[:sentence_end + 1]
                
                # Determine if it's at the end of a sentence or phrase
                is_sentence_end = bool(re.search(r'right[?.!,]\s*', context))
                
                # Find the speaker at this timestamp
                current_speaker = None
                for range in speaker_ranges:
                    if range['start'] <= word_start <= range['end']:
                        current_speaker = range['speaker']
                        break
                
                right_instances.append({
                    'timestamp': str(timedelta(seconds=int(word_start))),
                    'start_time': word_start,
                    'end_time': word_end,
                    'speaker': current_speaker or 'Unknown',
                    'context': context.strip(),
                    'is_sentence_end': is_sentence_end,
                    'full_segment': text  # Include full segment for context
                })
        
        logging.info(f"Analyzed {segments_analyzed} Whisper segments")
        logging.info(f"Found {total_rights} total instances of 'right'")
        
        # Group instances by speaker
        by_speaker = {}
        for instance in right_instances:
            speaker = instance['speaker']
            if speaker not in by_speaker:
                by_speaker[speaker] = []
            by_speaker[speaker].append(instance)
        
        # Log instances by speaker
        for speaker, instances in by_speaker.items():
            logging.info(f"Found {len(instances)} instances for {speaker}")
            if instances:
                logging.info(f"Sample from {speaker}:")
                for instance in instances[:2]:
                    logging.info(f"  {instance['timestamp']}: ...{instance['context']}...")
        
        return right_instances

    def save_results(self, right_instances):
        """Save the analysis results"""
        logging.info("Saving analysis results...")
        
        # Save detailed results with full context
        results_path = self.processed_dir / "right_analysis.json"
        with open(results_path, 'w') as f:
            json.dump(right_instances, f, indent=2)
        
        # Generate and save summary
        total_rights = len(right_instances)
        sentence_end_rights = sum(1 for r in right_instances if r['is_sentence_end'])
        
        summary = {
            'total_right_count': total_rights,
            'sentence_end_count': sentence_end_rights,
            'percentage_sentence_end': (sentence_end_rights / total_rights * 100) if total_rights > 0 else 0,
            'instances': [{
                'timestamp': r['timestamp'],
                'context': r['context'],
                'is_sentence_end': r['is_sentence_end']
            } for r in right_instances]
        }
        
        summary_path = self.processed_dir / "analysis_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logging.info(f"Found {total_rights} instances of 'right', {sentence_end_rights} at sentence endings")
        return summary

def main():
    analyzer = PodcastAnalyzer()
    
    try:
        # Step 1: Get official transcript for speaker identification
        official_segments = analyzer.download_official_transcript()
        
        # Step 2: Use existing audio file
        audio_path = analyzer.raw_dir / "sample_segment.m4a"
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found at {audio_path}. Please ensure the file exists.")
        logging.info(f"Using existing audio file: {audio_path}")
        
        # Step 3: Transcribe with Whisper for word-level analysis
        whisper_result = analyzer.transcribe_audio_segment(audio_path)
        
        # Step 4: Analyze "right" occurrences
        right_instances = analyzer.analyze_rights(whisper_result, official_segments)
        
        # Step 5: Save results
        summary = analyzer.save_results(right_instances)
        
        logging.info("Analysis completed successfully!")
        logging.info(f"Summary: {json.dumps(summary, indent=2)}")
        
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main() 