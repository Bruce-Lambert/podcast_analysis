"""
Text Parser Module

This module handles the parsing of text-based transcripts with speaker attribution.
It's specifically designed for manually prepared transcripts with timestamps and speaker labels.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional, Set
import re
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextParser:
    """Parser for text format transcripts with speaker attribution."""
    
    def __init__(self, transcript_path: Path, speakers: Optional[Set[str]] = None) -> None:
        """
        Initialize the text parser.
        
        Args:
            transcript_path (Path): Path to the transcript file
            speakers (Set[str], optional): Set of valid speaker names
        """
        self.transcript_path = Path(transcript_path)
        if not self.transcript_path.exists():
            raise FileNotFoundError(f"Transcript file not found: {transcript_path}")
            
        self.speakers = speakers or {'Dylan Patel', 'Nathan Lambert', 'Lex Fridman'}
        
    def parse(self) -> List[Dict[str, Any]]:
        """
        Parse the transcript file.
        
        Returns:
            List[Dict]: List of segments, each containing:
                - timestamp (float): Time in seconds
                - speaker (str): Speaker name
                - content (str): Text content
        """
        try:
            with open(self.transcript_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except IOError as e:
            logger.error(f"Error reading transcript file: {e}")
            raise
            
        segments = []
        current_speaker = None
        current_timestamp = None
        current_content = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check if line is a speaker name
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
                
            # Check for timestamp line (e.g., "(01:14:35) Text content")
            if line.startswith('(') and current_speaker:
                timestamp_end = line.find(')')
                if timestamp_end > 0:
                    timestamp_str = line[1:timestamp_end]
                    current_timestamp = self._parse_timestamp(timestamp_str)
                    content = line[timestamp_end + 1:].strip()
                    if content:
                        current_content = [content]
                        segments.append({
                            'speaker': current_speaker,
                            'timestamp': current_timestamp,
                            'content': content
                        })
            elif current_speaker and current_timestamp is not None:
                current_content.append(line)
                
        # Add final segment if exists
        if current_speaker and current_timestamp is not None and current_content:
            segments.append({
                'speaker': current_speaker,
                'timestamp': current_timestamp,
                'content': ' '.join(current_content)
            })
            
        if not segments:
            logger.warning("No segments found in transcript file")
            
        logger.info(f"Successfully parsed {len(segments)} segments from transcript file")
        return segments
            
    def _parse_timestamp(self, timestamp: str) -> float:
        """
        Convert timestamp string (HH:MM:SS) to seconds.
        
        Args:
            timestamp (str): Timestamp in HH:MM:SS format
            
        Returns:
            float: Time in seconds
        """
        try:
            hours, minutes, seconds = map(int, timestamp.split(':'))
            return hours * 3600 + minutes * 60 + seconds
        except ValueError:
            return None 