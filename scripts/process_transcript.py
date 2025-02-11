#!/usr/bin/env python3

import requests
from bs4 import BeautifulSoup
import re
import json
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TranscriptProcessor:
    def __init__(self):
        self.transcript_url = "https://lexfridman.com/deepseek-dylan-patel-nathan-lambert-transcript"
        self.raw_dir = Path("data/raw")
        self.processed_dir = Path("data/processed")
        
        # Create directories if they don't exist
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def download_transcript(self):
        """Download the transcript from Lex Fridman's website"""
        logging.info("Downloading transcript...")
        response = requests.get(self.transcript_url)
        response.raise_for_status()
        
        # Save raw HTML
        raw_transcript_path = self.raw_dir / "raw_transcript.html"
        raw_transcript_path.write_text(response.text)
        logging.info(f"Raw transcript saved to {raw_transcript_path}")
        
        return response.text

    def parse_transcript(self, html_content):
        """Parse the HTML content to extract speaker segments"""
        logging.info("Parsing transcript...")
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Find all paragraphs that might contain dialogue
        transcript_segments = []
        current_speaker = None
        current_text = []
        
        for p in soup.find_all('p'):
            text = p.get_text().strip()
            if not text:
                continue
                
            # Look for timestamp pattern (HH:MM:SS)
            timestamp_match = re.search(r'\((\d{2}:\d{2}:\d{2})\)', text)
            if timestamp_match:
                # Extract speaker and text
                parts = text.split(')', 1)
                if len(parts) > 1:
                    speaker_time = parts[0] + ')'
                    content = parts[1].strip()
                    
                    # Extract speaker name and timestamp
                    speaker_match = re.match(r'([^(]+)\s*\(', speaker_time)
                    if speaker_match:
                        speaker = speaker_match.group(1).strip()
                        timestamp = timestamp_match.group(1)
                        
                        transcript_segments.append({
                            'speaker': speaker,
                            'timestamp': timestamp,
                            'text': content
                        })

        return transcript_segments

    def analyze_dylan_rights(self, segments):
        """Analyze Dylan's usage of the word 'right'"""
        logging.info("Analyzing Dylan's usage of 'right'...")
        dylan_segments = [s for s in segments if s['speaker'] == 'Dylan Patel']
        
        right_instances = []
        for segment in dylan_segments:
            # Find all instances of 'right' in the text
            text = segment['text']
            for match in re.finditer(r'\bright\b', text, re.IGNORECASE):
                # Get some context around the word
                start = max(0, match.start() - 50)
                end = min(len(text), match.end() + 50)
                context = text[start:end]
                
                # Determine if it's at the end of a sentence
                is_sentence_end = bool(re.search(r'right[?.!,]\s*', match.group()))
                
                right_instances.append({
                    'timestamp': segment['timestamp'],
                    'context': context,
                    'is_sentence_end': is_sentence_end
                })
        
        return right_instances

    def save_results(self, segments, right_instances):
        """Save the processed results"""
        logging.info("Saving results...")
        
        # Save all transcript segments
        segments_path = self.processed_dir / "transcript_segments.json"
        with open(segments_path, 'w') as f:
            json.dump(segments, f, indent=2)
        
        # Save Dylan's 'right' instances
        rights_path = self.processed_dir / "dylan_rights.json"
        with open(rights_path, 'w') as f:
            json.dump(right_instances, f, indent=2)
        
        # Generate a summary
        total_rights = len(right_instances)
        sentence_end_rights = sum(1 for r in right_instances if r['is_sentence_end'])
        
        summary = {
            'total_right_count': total_rights,
            'sentence_end_count': sentence_end_rights,
            'percentage_sentence_end': (sentence_end_rights / total_rights * 100) if total_rights > 0 else 0
        }
        
        summary_path = self.processed_dir / "analysis_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logging.info(f"Found {total_rights} instances of 'right', {sentence_end_rights} at sentence endings")

def main():
    processor = TranscriptProcessor()
    
    try:
        # Download and process the transcript
        html_content = processor.download_transcript()
        transcript_segments = processor.parse_transcript(html_content)
        right_instances = processor.analyze_dylan_rights(transcript_segments)
        processor.save_results(transcript_segments, right_instances)
        
        logging.info("Processing completed successfully!")
        
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main() 