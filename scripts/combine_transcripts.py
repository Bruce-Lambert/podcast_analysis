import json
import re
from datetime import datetime, timedelta

def parse_timestamp(timestamp):
    """Convert timestamp string to seconds"""
    # Extract just the timestamp part from the line
    match = re.match(r'\((\d{2}):(\d{2}):(\d{2})\)', timestamp)
    if not match:
        return 0
    hours, minutes, seconds = map(int, match.groups())
    return hours * 3600 + minutes * 60 + seconds

def is_valid_speaker(name):
    """Check if a line represents a valid speaker name"""
    valid_speakers = {'Lex Fridman', 'Dylan Patel', 'Nathan Lambert'}
    return name in valid_speakers

def load_pasted_transcript(file_path):
    """Load and parse the pasted transcript with speaker attribution"""
    speaker_segments = []
    current_speaker = None
    current_timestamp = None
    current_text = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
            
        # Check for speaker name (line without timestamp that's not a continuation)
        if not line.startswith('(') and (i == 0 or not lines[i-1].strip() or lines[i-1].strip().endswith('.')):
            if is_valid_speaker(line):
                if current_speaker and current_timestamp is not None:
                    speaker_segments.append({
                        'speaker': current_speaker,
                        'timestamp': current_timestamp,
                        'text': ' '.join(current_text)
                    })
                current_speaker = line
                current_text = []
        # Check for timestamp
        elif line.startswith('('):
            if current_timestamp is not None and current_speaker and current_text:
                speaker_segments.append({
                    'speaker': current_speaker,
                    'timestamp': current_timestamp,
                    'text': ' '.join(current_text)
                })
            current_timestamp = parse_timestamp(line)
            current_text = []
        else:
            current_text.append(line)
    
    # Add the last segment
    if current_speaker and current_timestamp is not None and current_text:
        speaker_segments.append({
            'speaker': current_speaker,
            'timestamp': current_timestamp,
            'text': ' '.join(current_text)
        })
    
    return speaker_segments

def parse_vtt(file_path):
    """Parse the MacWhisper VTT file"""
    segments = []
    current_segment = None
    current_text = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    for line in lines:
        line = line.strip()
        if not line or line == 'WEBVTT':
            continue
            
        # Check for timestamp line
        if '-->' in line:
            if current_segment and current_text:
                current_segment['text'] = ' '.join(current_text)
                segments.append(current_segment)
            
            times = line.split(' --> ')
            start_time = times[0].split(':')
            start_seconds = float(start_time[0]) * 3600 + float(start_time[1]) * 60 + float(start_time[2])
            end_time = times[1].split(':')
            end_seconds = float(end_time[0]) * 3600 + float(end_time[1]) * 60 + float(end_time[2])
            current_segment = {
                'start_time': start_seconds,
                'end_time': end_seconds,
                'text': ''
            }
            current_text = []
        # Text content
        elif current_segment is not None:
            current_text.append(line)
            
    # Add the last segment
    if current_segment and current_text:
        current_segment['text'] = ' '.join(current_text)
        segments.append(current_segment)
            
    return segments

def find_speaker_for_timestamp(timestamp, speaker_segments):
    """Find the speaker for a given timestamp"""
    # Sort segments by timestamp to ensure proper ordering
    sorted_segments = sorted(speaker_segments, key=lambda x: x['timestamp'])
    
    # Find the last speaker before this timestamp
    current_speaker = None
    for segment in sorted_segments:
        if segment['timestamp'] <= timestamp:
            current_speaker = segment['speaker']
        else:
            break
            
    # If we couldn't find a speaker, check if this is within 5 seconds of the next speaker
    if current_speaker is None and sorted_segments:
        next_segment = sorted_segments[0]
        if abs(timestamp - next_segment['timestamp']) <= 5:
            current_speaker = next_segment['speaker']
    
    return current_speaker if current_speaker else "Unknown"

def analyze_right_usage(vtt_segments, speaker_segments):
    """Analyze usage of 'right' in the transcript with proper speaker attribution"""
    right_instances = []
    
    # Sort speaker segments by timestamp
    speaker_segments = sorted(speaker_segments, key=lambda x: x['timestamp'])
    
    for segment in vtt_segments:
        # Find all instances of "right" in the segment
        text = segment['text'].lower()
        for match in re.finditer(r'\bright\b', text):
            # Skip instances that are part of phrases like "all right" or "right now"
            pre_context = text[max(0, match.start()-10):match.start()].strip()
            post_context = text[match.end():min(len(text), match.end()+10)].strip()
            if any(phrase in f"{pre_context} right {post_context}" for phrase in ['all right', 'right now']):
                continue
                
            # Get context around "right"
            start_idx = match.start()
            context_start = max(0, start_idx - 30)
            context_end = min(len(text), start_idx + 35)
            context = text[context_start:context_end]
            
            # Determine if it's at the end of a sentence
            next_char_idx = match.end()
            is_sentence_end = bool(re.match(r'[.!?]', text[next_char_idx:next_char_idx+1] if next_char_idx < len(text) else ''))
            
            # Find the speaker
            speaker = find_speaker_for_timestamp(segment['start_time'], speaker_segments)
            
            # Only include if we have a valid speaker
            if speaker != "Unknown":
                right_instances.append({
                    'start_time': segment['start_time'],
                    'end_time': segment['end_time'],
                    'timestamp': str(timedelta(seconds=int(segment['start_time']))),
                    'context': context,
                    'is_sentence_end': is_sentence_end,
                    'speaker': speaker
                })
    
    return right_instances

def main():
    # Load transcripts
    print("Loading pasted transcript...")
    pasted_transcript = load_pasted_transcript('data/raw/pasted_transcript.txt')
    print(f"Found {len(pasted_transcript)} segments in pasted transcript")
    
    print("\nLoading VTT file...")
    vtt_segments = parse_vtt('data/raw/macwhisper_full_video.vtt')
    print(f"Found {len(vtt_segments)} segments in VTT file")
    
    # Analyze right usage
    print("\nAnalyzing 'right' usage...")
    right_instances = analyze_right_usage(vtt_segments, pasted_transcript)
    
    # Save results
    with open('data/processed/full_podcast/right_analysis.json', 'w') as f:
        json.dump(right_instances, f, indent=2)
    
    # Print summary
    print(f"\nFound {len(right_instances)} instances of 'right'")
    speaker_counts = {}
    for instance in right_instances:
        speaker = instance['speaker']
        speaker_counts[speaker] = speaker_counts.get(speaker, 0) + 1
    
    print("\nUsage by speaker:")
    for speaker, count in sorted(speaker_counts.items()):
        print(f"{speaker}: {count} instances")

if __name__ == '__main__':
    main() 