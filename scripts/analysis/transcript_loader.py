import re
from pathlib import Path
import json

class VTTParser:
    """Parser for VTT format transcripts from MacWhisper."""
    
    def __init__(self, vtt_path):
        self.vtt_path = Path(vtt_path)
        
    def parse(self):
        """Parse VTT file into segments with timestamps and content."""
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
    
    def _parse_timestamp(self, timestamp):
        """Convert VTT timestamp to seconds."""
        # Handle both HH:MM:SS.mmm and MM:SS.mmm formats
        parts = timestamp.replace(',', '.').split(':')
        if len(parts) == 3:  # HH:MM:SS.mmm
            hours, minutes, seconds = parts
            return float(hours) * 3600 + float(minutes) * 60 + float(seconds)
        else:  # MM:SS.mmm
            minutes, seconds = parts
            return float(minutes) * 60 + float(seconds)

class TranscriptParser:
    """Handles the ingestion and parsing of podcast transcripts."""
    
    def __init__(self, pasted_transcript_path, vtt_path=None):
        self.pasted_transcript_path = Path(pasted_transcript_path)
        self.vtt_path = Path(vtt_path) if vtt_path else None
        self.speakers = {'Dylan Patel', 'Nathan Lambert', 'Lex Fridman'}
        
    def parse(self):
        """Parse transcripts into a structured format."""
        # First parse the pasted transcript to get speaker segments
        speaker_segments = self._parse_pasted_transcript()
        
        if self.vtt_path and self.vtt_path.exists():
            # If we have a VTT file, use it for verbatim content
            vtt_parser = VTTParser(self.vtt_path)
            vtt_segments = vtt_parser.parse()
            return self._combine_transcripts(speaker_segments, vtt_segments)
        else:
            # If no VTT file, combine segments by speaker
            return self._combine_speaker_segments(speaker_segments)
    
    def _parse_pasted_transcript(self):
        """Parse the pasted transcript to get speaker segments with timestamps."""
        with open(self.pasted_transcript_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            
        segments = []
        current_speaker = None
        current_timestamp = None
        current_content = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # If line is a speaker name, process previous speaker's content and start new speaker
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
        
        return segments
    
    def _combine_speaker_segments(self, segments):
        """Combine segments by speaker when using only pasted transcript."""
        combined = {}
        
        # Sort segments by timestamp
        sorted_segments = sorted(segments, key=lambda x: x['timestamp'] if x['timestamp'] is not None else float('inf'))
        
        # Combine content for each speaker
        for segment in sorted_segments:
            speaker = segment['speaker']
            if speaker not in combined:
                combined[speaker] = {
                    'speaker': speaker,
                    'timestamp': segment['timestamp'],  # Keep first timestamp
                    'content': []
                }
            combined[speaker]['content'].append(segment['content'])
        
        # Join content and convert to list
        result = []
        for speaker_data in combined.values():
            speaker_data['content'] = ' '.join(speaker_data['content'])
            result.append(speaker_data)
        
        return sorted(result, key=lambda x: x['timestamp'] if x['timestamp'] is not None else float('inf'))
    
    def _parse_timestamp(self, timestamp):
        """Convert timestamp string (HH:MM:SS) to seconds."""
        try:
            hours, minutes, seconds = map(int, timestamp.split(':'))
            return hours * 3600 + minutes * 60 + seconds
        except ValueError:
            return None
    
    def _combine_transcripts(self, speaker_segments, vtt_segments):
        """Combine speaker information with verbatim VTT content."""
        combined_segments = []
        current_speaker = None
        unattributed_segments = []
        
        # Sort both segment lists by timestamp
        speaker_segments.sort(key=lambda x: x['timestamp'] if x['timestamp'] is not None else float('inf'))
        vtt_segments.sort(key=lambda x: x['timestamp'])
        
        # Create debug log
        debug_info = {
            'total_vtt_segments': len(vtt_segments),
            'total_speaker_segments': len(speaker_segments),
            'speaker_changes': [],
            'unattributed_segments': [],
            'alignment_issues': []
        }
        
        # Find the speaker for each VTT segment based on timestamps
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
                    debug_info['speaker_changes'].append({
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
                debug_info['unattributed_segments'].append({
                    'timestamp': vtt_seg['timestamp'],
                    'content_preview': vtt_seg['content'][:100]
                })
            
            # Check for potential alignment issues
            if i > 0 and current_speaker:
                time_diff = vtt_seg['timestamp'] - vtt_segments[i-1]['timestamp']
                if time_diff > 30:  # Flag gaps longer than 30 seconds
                    debug_info['alignment_issues'].append({
                        'timestamp': vtt_seg['timestamp'],
                        'gap_duration': time_diff,
                        'speaker': current_speaker,
                        'content_preview': vtt_seg['content'][:100]
                    })
        
        # Save debug information
        debug_path = Path('data/processed/transcript_combination_debug.json')
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        with open(debug_path, 'w', encoding='utf-8') as f:
            json.dump(debug_info, f, indent=2)
        
        # Print summary statistics
        print("\nTranscript combination summary:")
        print(f"Total VTT segments: {len(vtt_segments)}")
        print(f"Total speaker segments: {len(speaker_segments)}")
        print(f"Combined segments: {len(combined_segments)}")
        print(f"Unattributed segments: {len(unattributed_segments)}")
        print(f"Speaker changes: {len(debug_info['speaker_changes'])}")
        print(f"Alignment issues: {len(debug_info['alignment_issues'])}")
        
        return combined_segments, unattributed_segments

    def get_word_counts(self):
        """Get word counts for each speaker from the pasted transcript."""
        segments = self._parse_pasted_transcript()
        word_counts = defaultdict(int)
        for segment in segments:
            word_counts[segment['speaker']] += len(segment['content'].split())
        return word_counts
