#!/usr/bin/env python3

import json
import logging
from pathlib import Path
import re
from difflib import SequenceMatcher
from typing import Dict, List, Tuple, Optional
import numpy as np
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class AlignmentConfig:
    """Configuration for speaker alignment parameters."""
    text_similarity_threshold: float = 0.7
    time_window_seconds: float = 5.0
    min_segment_length: float = 1.0
    speaker_consistency_window: int = 3
    confidence_threshold: float = 0.6

class SpeakerAligner:
    def __init__(self, 
                 processed_dir: str = "data/processed/full_podcast", 
                 raw_dir: str = "data/raw",
                 config: Optional[AlignmentConfig] = None):
        self.processed_dir = Path(processed_dir)
        self.raw_dir = Path(raw_dir)
        self.whisper_file = self.processed_dir / "whisper_output.json"
        self.pasted_file = self.raw_dir / "pasted_transcript.txt"
        self.output_file = self.processed_dir / "whisper_output_with_speakers.json"
        self.config = config or AlignmentConfig()
        
        # Speaker name mapping for normalization
        self.speaker_mapping = {
            'lex': 'lex fridman',
            'lex fridman': 'lex fridman',
            'nathan': 'nathan lambert',
            'nathan lambert': 'nathan lambert',
            'dylan': 'dylan patel',
            'dylan patel': 'dylan patel'
        }
        
        # Statistics for reporting
        self.stats = defaultdict(int)
        
    def load_whisper_transcript(self) -> Dict:
        """Load the full Whisper transcript with error handling."""
        if not self.whisper_file.exists():
            raise FileNotFoundError(f"Whisper transcript not found at {self.whisper_file}")
        
        try:
            with open(self.whisper_file, 'r') as f:
                data = json.load(f)
                logger.info(f"Loaded Whisper transcript with {len(data['segments'])} segments")
                return data
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing Whisper JSON: {e}")
            raise
            
    def load_pasted_transcript(self) -> List[Dict]:
        """Load and parse the pasted transcript with robust error handling."""
        if not self.pasted_file.exists():
            raise FileNotFoundError(f"Pasted transcript not found at {self.pasted_file}")
            
        turns = []
        try:
            with open(self.pasted_file, 'r') as f:
                content = f.read()
                
            lines = [line.strip() for line in content.split('\n') if line.strip()]
            i = 0
            current_speaker = None
            
            while i < len(lines):
                line = lines[i]
                
                # Handle speaker names (with or without titles/affiliations)
                if any(name in line for name in ['Lex Fridman', 'Nathan Lambert', 'Dylan Patel']):
                    current_speaker = next(name for name in ['Lex Fridman', 'Nathan Lambert', 'Dylan Patel'] 
                                        if name in line)
                    i += 1
                    continue
                
                # Parse timestamp and text
                timestamp_match = re.match(r'\((\d{2}:\d{2}:\d{2})\)\s*(.*)', line)
                if timestamp_match and current_speaker:
                    timestamp = timestamp_match.group(1)
                    text = timestamp_match.group(2)
                    
                    try:
                        h, m, s = map(int, timestamp.split(':'))
                        time_seconds = h * 3600 + m * 60 + s
                        
                        turns.append({
                            'speaker': self.normalize_speaker_name(current_speaker),
                            'text': text,
                            'timestamp': timestamp,
                            'time_seconds': time_seconds,
                            'original_text': text  # Keep original for reference
                        })
                        self.stats['pasted_segments'] += 1
                        
                    except ValueError as e:
                        logger.warning(f"Error parsing timestamp {timestamp}: {e}")
                        self.stats['timestamp_errors'] += 1
                
                i += 1
            
            # Sort turns by timestamp
            turns.sort(key=lambda x: x['time_seconds'])
            
            logger.info(f"Loaded {len(turns)} speaker turns from pasted transcript")
            return turns
            
        except Exception as e:
            logger.error(f"Error processing pasted transcript: {e}")
            raise

    def clean_text(self, text: str) -> str:
        """Clean text for better matching with configurable options."""
        # Convert to lowercase
        text = text.lower()
        
        # Remove speaker annotations if present
        text = re.sub(r'\[.*?\]', '', text)
        
        # Remove punctuation except apostrophes
        text = re.sub(r'[^\w\s\']', '', text)
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        # Remove common filler words for better matching
        filler_words = {'um', 'uh', 'like', 'you know', 'i mean'}
        for word in filler_words:
            text = text.replace(f' {word} ', ' ')
        
        return text.strip()

    def normalize_speaker_name(self, name: str) -> str:
        """Normalize speaker names with fuzzy matching."""
        name = name.lower().strip()
        return self.speaker_mapping.get(name, name)

    def calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text segments."""
        # Clean both texts
        clean1 = self.clean_text(text1)
        clean2 = self.clean_text(text2)
        
        # Use SequenceMatcher for fuzzy matching
        matcher = SequenceMatcher(None, clean1, clean2)
        return matcher.ratio()

    def find_best_alignment(self, 
                          whisper_segment: dict, 
                          pasted_turns: List[Dict],
                          previous_alignments: List[Tuple[str, float]] = None) -> Tuple[str, float]:
        """Find best matching speaker with confidence score."""
        best_speaker = None
        best_confidence = 0.0
        
        # Extract Whisper segment details
        whisper_start = whisper_segment['start']
        whisper_end = whisper_segment['end']
        whisper_text = whisper_segment['text']
        
        # Filter pasted turns within time window
        time_window = self.config.time_window_seconds
        relevant_turns = [
            turn for turn in pasted_turns
            if abs(turn['time_seconds'] - whisper_start) <= time_window
        ]
        
        if not relevant_turns:
            logger.debug(f"No relevant turns found for segment at {whisper_start:.2f}s")
            return None, 0.0
        
        # Calculate similarities and weights
        for turn in relevant_turns:
            # Text similarity
            text_sim = self.calculate_text_similarity(whisper_text, turn['text'])
            
            # Time proximity (inverse of time difference)
            time_diff = abs(turn['time_seconds'] - whisper_start)
            time_weight = 1 - (time_diff / time_window)
            
            # Speaker consistency with previous alignments
            consistency_weight = 1.0
            if previous_alignments:
                recent_speakers = [s for s, _ in previous_alignments[-self.config.speaker_consistency_window:]]
                if turn['speaker'] in recent_speakers:
                    consistency_weight = 1.2
            
            # Combined confidence score
            confidence = (text_sim * 0.6 + time_weight * 0.4) * consistency_weight
            
            if confidence > best_confidence:
                best_confidence = confidence
                best_speaker = turn['speaker']
        
        return best_speaker, best_confidence

    def align_speakers(self) -> Dict:
        """Main alignment function with improved error handling and reporting."""
        try:
            # Load transcripts
            whisper_data = self.load_whisper_transcript()
            pasted_turns = self.load_pasted_transcript()
            
            # Track alignments for speaker consistency
            previous_alignments = []
            
            # Process each Whisper segment
            for segment in whisper_data['segments']:
                # Skip segments that are too short
                if segment['end'] - segment['start'] < self.config.min_segment_length:
                    self.stats['skipped_short_segments'] += 1
                    continue
                
                # Find best speaker match
                speaker, confidence = self.find_best_alignment(
                    segment, pasted_turns, previous_alignments
                )
                
                # Update segment with speaker information
                segment['speaker'] = speaker if confidence >= self.config.confidence_threshold else 'unknown'
                segment['speaker_confidence'] = float(confidence)
                
                # Update statistics
                self.stats['total_segments'] += 1
                if speaker:
                    self.stats['assigned_speakers'] += 1
                    previous_alignments.append((speaker, confidence))
                else:
                    self.stats['unknown_speakers'] += 1
            
            # Save aligned transcript
            self.save_aligned_transcript(whisper_data)
            
            # Log alignment statistics
            self.log_alignment_stats()
            
            return whisper_data
            
        except Exception as e:
            logger.error(f"Error during speaker alignment: {e}")
            raise

    def save_aligned_transcript(self, aligned_data: Dict):
        """Save aligned transcript with metadata."""
        # Add metadata
        aligned_data['metadata'] = {
            'alignment_timestamp': datetime.now().isoformat(),
            'config': vars(self.config),
            'statistics': dict(self.stats)
        }
        
        # Save with pretty printing
        with open(self.output_file, 'w') as f:
            json.dump(aligned_data, f, indent=2)
        
        logger.info(f"Saved aligned transcript to {self.output_file}")

    def log_alignment_stats(self):
        """Log detailed alignment statistics."""
        logger.info("=== Alignment Statistics ===")
        logger.info(f"Total segments processed: {self.stats['total_segments']}")
        logger.info(f"Segments with assigned speakers: {self.stats['assigned_speakers']}")
        logger.info(f"Segments with unknown speakers: {self.stats['unknown_speakers']}")
        logger.info(f"Short segments skipped: {self.stats['skipped_short_segments']}")
        
        if self.stats['total_segments'] > 0:
            success_rate = (self.stats['assigned_speakers'] / self.stats['total_segments']) * 100
            logger.info(f"Speaker assignment success rate: {success_rate:.1f}%")

def main():
    """Main function with configuration options."""
    # Create configuration
    config = AlignmentConfig(
        text_similarity_threshold=0.7,
        time_window_seconds=5.0,
        min_segment_length=1.0,
        speaker_consistency_window=3,
        confidence_threshold=0.6
    )
    
    try:
        # Initialize and run aligner
        aligner = SpeakerAligner(config=config)
        aligned_data = aligner.align_speakers()
        
        logger.info("Speaker alignment completed successfully")
        
    except Exception as e:
        logger.error(f"Speaker alignment failed: {e}")
        raise

if __name__ == "__main__":
    main() 