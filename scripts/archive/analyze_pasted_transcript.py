#!/usr/bin/env python3

import json
import re
from pathlib import Path
import logging
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PastedTranscriptAnalyzer:
    def __init__(self, raw_dir="data/raw", output_dir="data/processed/pasted_transcript_analysis"):
        self.raw_dir = Path(raw_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.transcript_path = self.raw_dir / "pasted_transcript.txt"
        
    def parse_timestamp(self, timestamp):
        """Convert timestamp to seconds."""
        h, m, s = map(int, timestamp.strip('()').split(':'))
        return h * 3600 + m * 60 + s
        
    def load_transcript(self):
        """Load and parse the pasted transcript with timestamps."""
        if not self.transcript_path.exists():
            raise FileNotFoundError(f"Pasted transcript not found at {self.transcript_path}")
            
        segments = []
        current_speaker = None
        
        with open(self.transcript_path, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
            
        for line in lines:
            # Check if line is a speaker name
            if line in ['Lex Fridman', 'Nathan Lambert', 'Dylan Patel']:
                current_speaker = line.lower()
                continue
                
            # Check if line contains timestamp and text
            timestamp_match = re.match(r'\((\d{2}:\d{2}:\d{2})\)\s*(.*)', line)
            if timestamp_match and current_speaker:
                timestamp, text = timestamp_match.groups()
                start_time = self.parse_timestamp(timestamp)
                
                segments.append({
                    'speaker': current_speaker,
                    'text': text,
                    'timestamp': timestamp,
                    'start_time': start_time
                })
        
        return segments
    
    def analyze_right_usage(self, segments):
        """Analyze usage of 'right' across speakers."""
        speaker_stats = defaultdict(lambda: {
            'total_words': 0,
            'right_instances': 0,
            'turns': 0,
            'speaking_time': 0
        })
        
        # Calculate speaking time by assuming each turn ends at the start of the next turn
        for i, segment in enumerate(segments):
            speaker = segment['speaker']
            text = segment['text'].lower()
            words = len(text.split())
            right_count = len(re.findall(r'\bright\b', text))
            
            # Calculate duration (use 10 seconds as default duration for last segment)
            if i < len(segments) - 1:
                duration = segments[i + 1]['start_time'] - segment['start_time']
            else:
                duration = 10
            
            speaker_stats[speaker]['total_words'] += words
            speaker_stats[speaker]['right_instances'] += right_count
            speaker_stats[speaker]['turns'] += 1
            speaker_stats[speaker]['speaking_time'] += duration
        
        # Calculate normalized rates
        for stats in speaker_stats.values():
            minutes = stats['speaking_time'] / 60
            words_per_1k = stats['total_words'] / 1000
            
            stats['rights_per_minute'] = stats['right_instances'] / minutes if minutes > 0 else 0
            stats['rights_per_1k_words'] = stats['right_instances'] / words_per_1k if words_per_1k > 0 else 0
            stats['rights_per_turn'] = stats['right_instances'] / stats['turns'] if stats['turns'] > 0 else 0
        
        return dict(speaker_stats)
    
    def generate_visualizations(self, speaker_stats):
        """Generate visualizations for the 'right' usage patterns."""
        # Set style parameters
        plt.rcParams['figure.figsize'] = [20, 6]
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.alpha'] = 0.3
        
        # Create a figure with three subplots
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        
        # Prepare data
        speakers = list(speaker_stats.keys())
        rights_per_minute = [stats['rights_per_minute'] for stats in speaker_stats.values()]
        rights_per_1k = [stats['rights_per_1k_words'] for stats in speaker_stats.values()]
        rights_per_turn = [stats['rights_per_turn'] for stats in speaker_stats.values()]
        
        # 1. Rights per minute
        bars1 = ax1.bar(speakers, rights_per_minute, color='skyblue', alpha=0.7)
        ax1.set_title("'Right' Usage per Minute\n(From Pasted Transcript)", pad=20)
        ax1.set_ylabel("Instances per Minute")
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom')
        
        # 2. Rights per 1000 words
        bars2 = ax2.bar(speakers, rights_per_1k, color='lightgreen', alpha=0.7)
        ax2.set_title("'Right' Usage per 1000 Words\n(From Pasted Transcript)", pad=20)
        ax2.set_ylabel("Instances per 1000 Words")
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Add value labels
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom')
        
        # 3. Rights per turn
        bars3 = ax3.bar(speakers, rights_per_turn, color='coral', alpha=0.7)
        ax3.set_title("'Right' Usage per Turn\n(From Pasted Transcript)", pad=20)
        ax3.set_ylabel("Instances per Turn")
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # Add value labels
        for bar in bars3:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'pasted_transcript_right_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save statistics to JSON
        with open(self.output_dir / 'pasted_transcript_stats.json', 'w') as f:
            json.dump(speaker_stats, f, indent=2)
        
        # Print summary statistics
        print("\nAnalysis from Pasted Transcript:")
        print("-" * 40)
        for speaker, stats in speaker_stats.items():
            print(f"\n{speaker.title()}:")
            print(f"Total 'right' instances: {stats['right_instances']}")
            print(f"Per minute: {stats['rights_per_minute']:.2f}")
            print(f"Per 1000 words: {stats['rights_per_1k_words']:.2f}")
            print(f"Per turn: {stats['rights_per_turn']:.2f}")
            print(f"Total turns: {stats['turns']}")
            print(f"Total words: {stats['total_words']}")
            print(f"Speaking time: {stats['speaking_time']/60:.1f} minutes")

def main():
    analyzer = PastedTranscriptAnalyzer()
    
    try:
        # Load and analyze transcript
        segments = analyzer.load_transcript()
        speaker_stats = analyzer.analyze_right_usage(segments)
        
        # Generate visualizations
        analyzer.generate_visualizations(speaker_stats)
        
        print("\nAnalysis complete! Check the output directory for visualizations.")
        
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main() 