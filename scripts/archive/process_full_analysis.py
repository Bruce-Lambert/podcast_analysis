#!/usr/bin/env python3

import json
import logging
from pathlib import Path
import subprocess
from analyze_discourse_patterns import DiscoursePatternAnalyzer
from create_audio_montages import AudioMontageCreator
from create_video_montages import VideoMontageCreator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('analysis.log'),
        logging.StreamHandler()
    ]
)

class FullAnalysisPipeline:
    def __init__(self):
        self.processed_dir = Path("data/processed")
        self.raw_dir = Path("data/raw")
        self.full_podcast_dir = self.processed_dir / "full_podcast"
        self.whisper_output = self.full_podcast_dir / "whisper_output.json"
        
        # Create necessary directories
        self.full_podcast_dir.mkdir(parents=True, exist_ok=True)
        (self.full_podcast_dir / "discourse_patterns").mkdir(parents=True, exist_ok=True)
        (self.full_podcast_dir / "audio_montages").mkdir(parents=True, exist_ok=True)
        (self.full_podcast_dir / "video_montages").mkdir(parents=True, exist_ok=True)
        
    def validate_transcript(self):
        """Validate the Whisper transcript and extract basic statistics."""
        logging.info("Validating transcript...")
        
        if not self.whisper_output.exists():
            raise FileNotFoundError(f"Whisper transcript not found at {self.whisper_output}")
            
        with open(self.whisper_output) as f:
            transcript_data = json.load(f)
            
        # Basic validation and statistics
        segments = transcript_data.get('segments', [])
        total_segments = len(segments)
        total_duration = segments[-1]['end'] if segments else 0
        total_words = sum(len(seg.get('text', '').split()) for seg in segments)
        
        stats = {
            'total_segments': total_segments,
            'duration_hours': total_duration / 3600,
            'total_words': total_words,
            'words_per_minute': (total_words / (total_duration / 60)) if total_duration > 0 else 0
        }
        
        logging.info(f"Transcript statistics:")
        logging.info(f"- Total segments: {stats['total_segments']}")
        logging.info(f"- Duration: {stats['duration_hours']:.2f} hours")
        logging.info(f"- Total words: {stats['total_words']}")
        logging.info(f"- Words per minute: {stats['words_per_minute']:.1f}")
        
        return stats
        
    def run_discourse_analysis(self):
        """Run the full discourse pattern analysis."""
        logging.info("Starting discourse pattern analysis...")
        
        analyzer = DiscoursePatternAnalyzer(
            processed_dir=self.full_podcast_dir,
            raw_dir=self.raw_dir / "full_podcast"
        )
        analyzer.analyze_patterns()
        
        logging.info("Discourse analysis complete!")
        
    def create_montages(self):
        """Create both audio and video montages."""
        logging.info("Creating audio montages...")
        audio_creator = AudioMontageCreator(
            raw_dir=self.raw_dir / "full_podcast",
            processed_dir=self.full_podcast_dir
        )
        audio_creator.create_rapid_montage()
        audio_creator.create_context_montage()
        
        logging.info("Creating video montages...")
        # For now, skip video montages since we don't have a video URL
        logging.info("Skipping video montages - no video URL provided")
        
    def run_full_analysis(self):
        """Run the complete analysis pipeline."""
        try:
            # Step 1: Validate transcript
            stats = self.validate_transcript()
            
            # Step 2: Run discourse analysis
            self.run_discourse_analysis()
            
            # Step 3: Create montages
            self.create_montages()
            
            logging.info("Full analysis pipeline completed successfully!")
            return stats
            
        except Exception as e:
            logging.error(f"Error during analysis: {str(e)}")
            raise

def main():
    pipeline = FullAnalysisPipeline()
    stats = pipeline.run_full_analysis()
    
    # Print final summary
    print("\nAnalysis Complete!")
    print(f"Processed {stats['total_segments']} segments")
    print(f"Total duration: {stats['duration_hours']:.2f} hours")
    print(f"Total words: {stats['total_words']}")
    print(f"Average speaking rate: {stats['words_per_minute']:.1f} words per minute")
    print("\nCheck the following locations for results:")
    print("- Discourse patterns: data/processed/full_podcast/discourse_patterns/")
    print("- Audio montages: data/processed/full_podcast/audio_montages/")
    print("- Video montages: data/processed/full_podcast/video_montages/")

if __name__ == "__main__":
    main() 