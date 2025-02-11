"""
Transcript Combiner Module

This module handles the combination of VTT transcripts with speaker-attributed text transcripts.
It produces a unified transcript with both precise timestamps and speaker attribution.
"""

from pathlib import Path
import json
from typing import List, Dict, Any
from .vtt_parser import VTTParser
from .text_parser import TextParser

class TranscriptCombiner:
    """Combines VTT and text transcripts with speaker attribution."""
    
    def __init__(self, vtt_path: Path, text_path: Path, output_path: Path = None):
        """
        Initialize transcript combiner.
        
        Args:
            vtt_path (Path): Path to the VTT transcript
            text_path (Path): Path to the text transcript with speaker attribution
            output_path (Path, optional): Path to save the combined transcript
        """
        self.vtt_path = Path(vtt_path)
        self.text_path = Path(text_path)
        self.output_path = output_path or Path('data/processed/combined_transcript.json')
        self.debug_info = {
            'total_vtt_segments': 0,
            'total_speaker_segments': 0,
            'speaker_changes': [],
            'unattributed_segments': [],
            'alignment_issues': []
        }
        
    def combine(self) -> List[Dict[str, Any]]:
        """
        Combine VTT and text transcripts.
        
        Returns:
            List[Dict]: Combined transcript with speaker attribution and timestamps
        """
        # Parse both transcripts
        vtt_parser = VTTParser(self.vtt_path)
        text_parser = TextParser(self.text_path)
        
        vtt_segments = vtt_parser.parse()
        speaker_segments = text_parser.parse()
        
        # Update debug info
        self.debug_info['total_vtt_segments'] = len(vtt_segments)
        self.debug_info['total_speaker_segments'] = len(speaker_segments)
        
        # Sort segments by timestamp
        speaker_segments.sort(key=lambda x: x['timestamp'] if x['timestamp'] is not None else float('inf'))
        vtt_segments.sort(key=lambda x: x['timestamp'])
        
        # Combine transcripts
        combined_segments = []
        current_speaker = None
        unattributed_segments = []
        
        for i, vtt_seg in enumerate(vtt_segments):
            # Find the closest speaker segment before this VTT segment
            speaker_seg = None
            for seg in speaker_segments:
                if seg['timestamp'] is not None and seg['timestamp'] <= vtt_seg['timestamp']:
                    speaker_seg = seg
                else:
                    break
            
            if speaker_seg:
                new_speaker = speaker_seg['speaker']
                if new_speaker != current_speaker:
                    self.debug_info['speaker_changes'].append({
                        'timestamp': vtt_seg['timestamp'],
                        'previous_speaker': current_speaker,
                        'new_speaker': new_speaker,
                        'content_preview': vtt_seg['content'][:100]
                    })
                    current_speaker = new_speaker
            
            if current_speaker:
                combined_segments.append({
                    'speaker': current_speaker,
                    'timestamp': vtt_seg['timestamp'],
                    'content': vtt_seg['content']
                })
            else:
                unattributed_segments.append({
                    'timestamp': vtt_seg['timestamp'],
                    'content': vtt_seg['content']
                })
                self.debug_info['unattributed_segments'].append({
                    'timestamp': vtt_seg['timestamp'],
                    'content_preview': vtt_seg['content'][:100]
                })
            
            # Check for potential alignment issues
            if i > 0 and current_speaker:
                time_diff = vtt_seg['timestamp'] - vtt_segments[i-1]['timestamp']
                if time_diff > 30:  # Flag gaps longer than 30 seconds
                    self.debug_info['alignment_issues'].append({
                        'timestamp': vtt_seg['timestamp'],
                        'gap_duration': time_diff,
                        'speaker': current_speaker,
                        'content_preview': vtt_seg['content'][:100]
                    })
        
        # Save combined transcript and debug info
        self.save_output(combined_segments)
        
        return combined_segments
    
    def save_output(self, combined_segments: List[Dict[str, Any]]) -> None:
        """
        Save combined transcript and debug information.
        
        Args:
            combined_segments (List[Dict]): Combined transcript segments
        """
        # Create output directory if it doesn't exist
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save combined transcript
        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump({
                'segments': combined_segments,
                'metadata': {
                    'total_segments': len(combined_segments),
                    'speaker_changes': len(self.debug_info['speaker_changes']),
                    'unattributed_segments': len(self.debug_info['unattributed_segments']),
                    'alignment_issues': len(self.debug_info['alignment_issues'])
                }
            }, f, indent=2)
        
        # Save debug information
        debug_path = self.output_path.parent / 'transcript_combination_debug.json'
        with open(debug_path, 'w', encoding='utf-8') as f:
            json.dump(self.debug_info, f, indent=2)
    
    def get_debug_summary(self) -> str:
        """
        Get a summary of the combination process.
        
        Returns:
            str: Summary of the transcript combination process
        """
        return f"""
Transcript combination summary:
Total VTT segments: {self.debug_info['total_vtt_segments']}
Total speaker segments: {self.debug_info['total_speaker_segments']}
Combined segments: {len(self.debug_info['speaker_changes'])}
Unattributed segments: {len(self.debug_info['unattributed_segments'])}
Speaker changes: {len(self.debug_info['speaker_changes'])}
Alignment issues: {len(self.debug_info['alignment_issues'])}
""" 