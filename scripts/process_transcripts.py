#!/usr/bin/env python3
"""
Transcript Processing Example Script

This script demonstrates how to use the transcript processing package to combine
VTT and text transcripts into a unified format with speaker attribution.
"""

from pathlib import Path
import logging
from transcript_processing import VTTParser, TextParser, TranscriptCombiner

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Process and combine transcripts."""
    try:
        # Define paths
        vtt_path = Path('data/raw/macwhisper_full_video.vtt')
        text_path = Path('data/raw/pasted_transcript.txt')
        output_path = Path('data/processed/combined_transcript.json')
        
        # Create combiner
        logger.info("Initializing transcript combiner...")
        combiner = TranscriptCombiner(
            vtt_path=vtt_path,
            text_path=text_path,
            output_path=output_path
        )
        
        # Combine transcripts
        logger.info("Starting transcript combination process...")
        debug_info = combiner.combine()
        
        # Log results
        logger.info("Transcript combination complete!")
        logger.info(f"VTT segments: {debug_info['vtt_segments']}")
        logger.info(f"Text segments: {debug_info['text_segments']}")
        logger.info(f"Combined segments: {debug_info['combined_segments']}")
        logger.info(f"Speaker changes: {len(debug_info['speaker_changes'])}")
        logger.info(f"Unattributed segments: {debug_info['unattributed_segments']}")
        
        logger.info(f"Combined transcript saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"Error processing transcripts: {e}")
        raise

if __name__ == '__main__':
    main() 