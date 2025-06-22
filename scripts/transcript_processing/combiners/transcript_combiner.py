"""
Transcript Combiner Module

This module handles the combination of VTT transcripts with speaker-attributed text transcripts.
It produces a unified transcript with both precise timestamps and speaker attribution.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import json
import logging
from ..parsers.vtt_parser import VTTParser
from ..parsers.text_parser import TextParser

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TranscriptCombiner:
    """Combines VTT and text transcripts with speaker attribution."""
    
    def __init__(self, vtt_path: Path, text_path: Path, output_path: Path) -> None:
        """
        Initialize the combiner.
        
        Args:
            vtt_path (Path): Path to the VTT file
            text_path (Path): Path to the text transcript
            output_path (Path): Path to save combined output
        """
        self.vtt_path = Path(vtt_path)
        self.text_path = Path(text_path)
        self.output_path = Path(output_path)
        
        # Initialize parsers
        self.vtt_parser = VTTParser(vtt_path)
        self.text_parser = TextParser(text_path)
        
        # Initialize state
        self.previous_speaker = None
        self.last_known_speaker = None
        
    def combine(self) -> Dict[str, Any]:
        """
        Combine VTT and text transcripts.
        
        Returns:
            Dict: Debug information about the combination process
            
        Raises:
            ValueError: If there are issues combining the transcripts
            IOError: If there are issues with file operations
        """
        try:
            # Parse both transcripts
            logger.info("Parsing VTT transcript...")
            vtt_segments = self.vtt_parser.parse()
            logger.info(f"Found {len(vtt_segments)} VTT segments")
            
            logger.info("Parsing text transcript...")
            text_segments = self.text_parser.parse()
            logger.info(f"Found {len(text_segments)} text segments")
            
            # Initialize combined segments
            combined_segments = []
            speaker_changes = []
            self.previous_speaker = None
            self.last_known_speaker = None
            
            # Sort segments by timestamp
            vtt_segments.sort(key=lambda x: x['timestamp'])
            text_segments.sort(key=lambda x: x['timestamp'])
            
            # Process VTT segments and find corresponding text segments
            logger.info("Combining transcripts...")
            current_text_idx = 0
            
            for vtt_segment in vtt_segments:
                # Find the closest text segment before this VTT segment
                while (current_text_idx < len(text_segments) and 
                       text_segments[current_text_idx]['timestamp'] <= vtt_segment['timestamp']):
                    self.last_known_speaker = text_segments[current_text_idx]['speaker']
                    current_text_idx += 1
                
                # Create merged segment
                merged = {
                    'timestamp': vtt_segment['timestamp'],
                    'content': vtt_segment['content'],
                    'speaker': self.last_known_speaker if self.last_known_speaker else 'Unknown'
                }
                
                # Track speaker changes
                if (self.previous_speaker is not None and 
                    merged['speaker'] != self.previous_speaker):
                    speaker_changes.append({
                        'timestamp': merged['timestamp'],
                        'from': self.previous_speaker,
                        'to': merged['speaker']
                    })
                
                self.previous_speaker = merged['speaker']
                combined_segments.append(merged)
            
            # Save combined transcript
            logger.info("Saving combined transcript...")
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.output_path, 'w', encoding='utf-8') as f:
                json.dump(combined_segments, f, indent=2)
            
            # Generate debug information
            debug_info = {
                'vtt_segments': len(vtt_segments),
                'text_segments': len(text_segments),
                'combined_segments': len(combined_segments),
                'speaker_changes': speaker_changes,
                'unattributed_segments': sum(1 for s in combined_segments if s.get('speaker') == 'Unknown')
            }
            
            logger.info("Transcript combination complete")
            logger.info(f"Combined {len(combined_segments)} segments with {len(speaker_changes)} speaker changes")
            return debug_info
            
        except Exception as e:
            logger.error(f"Error combining transcripts: {e}")
            raise 