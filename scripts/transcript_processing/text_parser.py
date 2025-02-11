"""
Text Parser Module

This module handles the parsing of text-based transcripts with speaker attribution and timestamps.
It converts formatted text transcripts into a standardized internal format.
"""

from pathlib import Path
from typing import List, Dict, Any, Set

class TextParser:
    """Parser for text format transcripts with speaker attribution."""
    
    def __init__(self, transcript_path: Path, speakers: Set[str] = None):
        """
        Initialize text parser.
        
        Args:
            transcript_path (Path): Path to the transcript file
            speakers (Set[str], optional): Set of speaker names to recognize
        """
        self.transcript_path = Path(transcript_path)
        self.speakers = speakers or {'Lex Fridman', 'Dylan Patel', 'Nathan Lambert'}
        
    def parse(self) -> List[Dict[str, Any]]:
        """
        Parse text transcript into segments with timestamps and speaker attribution.
        
        Returns:
            List[Dict]: List of segments with speaker, timestamp, and content
        """
        with open(self.transcript_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            
        segments = []
        current_speaker = None
        current_timestamp = None
        current_content = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # If line is a speaker name, process previous speaker's content
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
        
        # Add the last segment if needed
        if current_speaker and current_timestamp is not None and current_content:
            segments.append({
                'speaker': current_speaker,
                'timestamp': current_timestamp,
                'content': ' '.join(current_content)
            })
        
        return segments
    
    def _parse_timestamp(self, timestamp: str) -> float:
        """
        Convert timestamp string (HH:MM:SS) to seconds.
        
        Args:
            timestamp (str): Timestamp string in HH:MM:SS format
            
        Returns:
            float: Time in seconds, or None if parsing fails
        """
        try:
            hours, minutes, seconds = map(int, timestamp.split(':'))
            return hours * 3600 + minutes * 60 + seconds
        except ValueError:
            return None 