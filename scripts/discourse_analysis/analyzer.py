"""
Discourse Analysis Module

This module provides the base functionality for analyzing discourse patterns in transcripts.
It can be extended with specific analyzers for different discourse markers or patterns.
"""

from pathlib import Path
import json
from typing import List, Dict, Any, Optional
from collections import defaultdict

class DiscourseAnalyzer:
    """Base class for discourse analysis."""
    
    def __init__(self, transcript_path: Path):
        """
        Initialize discourse analyzer.
        
        Args:
            transcript_path (Path): Path to the combined transcript JSON file
        """
        self.transcript_path = Path(transcript_path)
        self.segments = self._load_transcript()
        
    def _load_transcript(self) -> List[Dict[str, Any]]:
        """
        Load the combined transcript from JSON.
        
        Returns:
            List[Dict]: List of transcript segments
        """
        with open(self.transcript_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data['segments']
    
    def get_word_counts(self) -> Dict[str, int]:
        """
        Get word counts for each speaker.
        
        Returns:
            Dict[str, int]: Dictionary mapping speakers to their word counts
        """
        word_counts = defaultdict(int)
        
        for segment in self.segments:
            speaker = segment['speaker']
            content = segment['content']
            words = len(content.split())
            word_counts[speaker] += words
            
        return dict(word_counts)
    
    def get_speaker_segments(self, speaker: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get segments for a specific speaker or all segments if no speaker specified.
        
        Args:
            speaker (str, optional): Speaker name to filter by
            
        Returns:
            List[Dict]: List of transcript segments
        """
        if speaker is None:
            return self.segments
        return [seg for seg in self.segments if seg['speaker'] == speaker]
    
    def get_time_window_segments(self, start_time: float, end_time: float) -> List[Dict[str, Any]]:
        """
        Get segments within a specific time window.
        
        Args:
            start_time (float): Start time in seconds
            end_time (float): End time in seconds
            
        Returns:
            List[Dict]: List of segments within the time window
        """
        return [
            seg for seg in self.segments
            if start_time <= seg['timestamp'] <= end_time
        ]
    
    def get_context_window(self, segment: Dict[str, Any], 
                          before_seconds: float = 30, 
                          after_seconds: float = 30) -> List[Dict[str, Any]]:
        """
        Get segments within a time window around a given segment.
        
        Args:
            segment (Dict): The center segment
            before_seconds (float): Seconds to include before the segment
            after_seconds (float): Seconds to include after the segment
            
        Returns:
            List[Dict]: List of segments within the context window
        """
        center_time = segment['timestamp']
        return self.get_time_window_segments(
            center_time - before_seconds,
            center_time + after_seconds
        ) 