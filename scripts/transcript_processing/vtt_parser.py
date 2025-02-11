"""
VTT Parser Module

This module handles the parsing of VTT format transcripts, typically from MacWhisper or other sources.
It converts VTT files into a standardized internal format for further processing.
"""

from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

class VTTParser:
    """Parser for VTT format transcripts."""
    
    def __init__(self, vtt_path: Path):
        """
        Initialize VTT parser.
        
        Args:
            vtt_path (Path): Path to the VTT file
        """
        self.vtt_path = Path(vtt_path)
        
    def parse(self) -> List[Dict[str, Any]]:
        """
        Parse VTT file into segments with timestamps and content.
        
        Returns:
            List[Dict]: List of segments, each containing timestamp and content
        """
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
    
    def _parse_timestamp(self, timestamp: str) -> float:
        """
        Convert VTT timestamp to seconds.
        
        Args:
            timestamp (str): Timestamp string in HH:MM:SS.mmm format
            
        Returns:
            float: Time in seconds
        """
        # Handle both HH:MM:SS.mmm and MM:SS.mmm formats
        parts = timestamp.replace(',', '.').split(':')
        if len(parts) == 3:  # HH:MM:SS.mmm
            hours, minutes, seconds = parts
            return float(hours) * 3600 + float(minutes) * 60 + float(seconds)
        else:  # MM:SS.mmm
            minutes, seconds = parts
            return float(minutes) * 60 + float(seconds) 