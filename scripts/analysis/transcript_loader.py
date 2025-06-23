import re
from pathlib import Path
import json
from collections import defaultdict

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
    
    def __init__(self, pasted_transcript_path=None, vtt_path=None, whisper_json_path=None, elevenlabs_json_path=None):
        self.pasted_transcript_path = Path(pasted_transcript_path) if pasted_transcript_path else None
        self.vtt_path = Path(vtt_path) if vtt_path else None
        self.whisper_json_path = Path(whisper_json_path) if whisper_json_path else None
        self.elevenlabs_json_path = Path(elevenlabs_json_path) if elevenlabs_json_path else None
        self.speakers = {'Dylan Patel', 'Nathan Lambert', 'Lex Fridman'}
        
    def parse(self):
        """Parse transcripts into a structured format."""
        if self.elevenlabs_json_path and self.elevenlabs_json_path.exists():
            print(f"Using Eleven Labs JSON for high-accuracy transcript: {self.elevenlabs_json_path}")
            return self._parse_elevenlabs_json(), []

        # Fallback to older methods if elevenlabs not present
        if not self.pasted_transcript_path or not self.pasted_transcript_path.exists():
             raise FileNotFoundError("pasted_transcript_path is required for VTT or Whisper sources.")
        
        speaker_segments = self._parse_pasted_transcript()

        if self.whisper_json_path and self.whisper_json_path.exists():
            # If we have a Whisper JSON file, use it for word-level timestamps
            print("Using Whisper JSON for high-accuracy timestamps.")
            whisper_segments = self._parse_whisper_json()
            return self._combine_with_speaker_data(whisper_segments, speaker_segments)
        
        elif self.vtt_path and self.vtt_path.exists():
            # If we have a VTT file, use it for verbatim content
            print(f"Using VTT file for transcript content: {self.vtt_path}")
            vtt_parser = VTTParser(self.vtt_path)
            vtt_segments = vtt_parser.parse()
            # This method will need to be robust enough to handle the new VTT data
            return self._combine_with_speaker_data_vtt(vtt_segments, speaker_segments)
        
        else:
            # If no VTT/JSON file, combine segments by speaker from pasted transcript
            print("Using pasted transcript only.")
            return self._combine_speaker_segments(speaker_segments)

    def _parse_elevenlabs_json(self):
        """Load and parse the Eleven Labs JSON, mapping speaker_id to names."""
        speaker_mapping = {
            "speaker_0": "Lex Fridman",
            "speaker_1": "Nathan Lambert",
            "speaker_2": "Dylan Patel" # Correct mapping
        }

        with open(self.elevenlabs_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        word_list = data.get('words', [])
        enriched_segments = []

        main_speaker_ids = ["speaker_0", "speaker_1", "speaker_2"]

        for word_info in word_list:
            speaker_id = word_info.get('speaker_id')

            # Only process words from the main speakers
            if speaker_id not in main_speaker_ids:
                continue

            # Defensively check for required keys
            if 'text' in word_info and 'start' in word_info and 'end' in word_info:
                # Use the mapping but keep original ID if not found, to include all speakers
                speaker_name = speaker_mapping.get(speaker_id, speaker_id) 

                enriched_segments.append({
                    'speaker': speaker_name,
                    'word': word_info['text'],
                    'start': word_info['start'],
                    'end': word_info['end'],
                    'speaker_id': speaker_id # Keep original ID for reference
                })

        print(f"Successfully parsed Eleven Labs JSON, producing {len(enriched_segments)} word segments.")
        return enriched_segments

    def _parse_whisper_json(self):
        """Load and parse the Whisper JSON output file."""
        with open(self.whisper_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data.get('segments', [])

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
    
    def get_word_counts(self):
        """Get word counts for each speaker from the pasted transcript."""
        segments = self._parse_pasted_transcript()
        word_counts = defaultdict(int)
        for segment in segments:
            word_counts[segment['speaker']] += len(segment['content'].split())
        return word_counts

    def _combine_with_speaker_data(self, whisper_segments, speaker_segments):
        """Combines whisper segments with speaker attribution."""
        enriched_segments = []
        speaker_segments.sort(key=lambda x: x['timestamp'] if x['timestamp'] is not None else float('inf'))
        
        for whisper_seg in whisper_segments:
            segment_start_time = whisper_seg['start']
            speaker = self._find_speaker_for_time(segment_start_time, speaker_segments)
            
            # Add speaker info to each word in the segment
            if 'words' in whisper_seg:
                for word_info in whisper_seg['words']:
                    enriched_segments.append({
                        'speaker': speaker,
                        'word': word_info['word'],
                        'start': word_info['start'],
                        'end': word_info['end']
                    })
        
        print(f"Successfully combined Whisper JSON with speaker data, producing {len(enriched_segments)} word segments.")
        return enriched_segments, [] # Return empty list for unattributed for now

    def _combine_with_speaker_data_vtt(self, vtt_segments, speaker_segments):
        """Combines VTT segments with speaker attribution."""
        enriched_segments = []
        speaker_segments.sort(key=lambda x: x['timestamp'] if x['timestamp'] is not None else float('inf'))

        for vtt_seg in vtt_segments:
            segment_start_time = vtt_seg['timestamp']
            speaker = self._find_speaker_for_time(segment_start_time, speaker_segments)
            
            # Split content into words and create word-level segments
            words = vtt_seg['content'].split()
            # Estimate duration of each word
            num_words = len(words)
            segment_duration = 5.0  # Default duration if no next segment
            
            # Find time of next segment to estimate duration
            current_index = vtt_segments.index(vtt_seg)
            if current_index + 1 < len(vtt_segments):
                segment_duration = vtt_segments[current_index + 1]['timestamp'] - segment_start_time
            
            if num_words > 0:
                word_duration = segment_duration / num_words
                for i, word in enumerate(words):
                    word_start = segment_start_time + (i * word_duration)
                    word_end = word_start + word_duration
                    enriched_segments.append({
                        'speaker': speaker,
                        'word': word,
                        'start': word_start,
                        'end': word_end
                    })

        print(f"Successfully combined VTT with speaker data, producing {len(enriched_segments)} word segments.")
        return enriched_segments, []

    def _find_speaker_for_time(self, timestamp, speaker_segments):
        """Finds the speaker for a given timestamp."""
        current_speaker = "Unknown"
        for segment in speaker_segments:
            if segment['timestamp'] is not None and segment['timestamp'] <= timestamp:
                current_speaker = segment['speaker']
            else:
                break
        return current_speaker
