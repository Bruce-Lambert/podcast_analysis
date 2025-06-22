#!/usr/bin/env python3
"""
Match VTT transcript segments to speakers using text similarity matching.
Collects turn-length segments from both transcripts and uses TF-IDF weighted comparison
within time windows to find the best matches.
"""

import json
from pathlib import Path
import logging
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_pasted_turns(file_path: Path) -> list:
    """
    Extract turn-length segments from pasted transcript with timestamps.
    """
    turns = []
    current_speaker = None
    current_content = []
    current_start = None
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    for i, line in enumerate(lines):
        # Check for speaker name
        if line in {'Dylan Patel', 'Nathan Lambert', 'Lex Fridman'}:
            # Save previous turn if exists
            if current_speaker and current_content and current_start is not None:
                # Look ahead for next timestamp to get end point
                end_time = None
                for next_line in lines[i:]:
                    if next_line.startswith('('):
                        timestamp_end = next_line.find(')')
                        if timestamp_end > 0:
                            timestamp_str = next_line[1:timestamp_end]
                            h, m, s = map(int, timestamp_str.split(':'))
                            end_time = h * 3600 + m * 60 + s - 1
                            break
                
                turns.append({
                    'speaker': current_speaker,
                    'content': ' '.join(current_content),
                    'start_time': current_start,
                    'end_time': end_time
                })
            
            # Start new turn
            current_speaker = line
            current_content = []
            current_start = None
            
        # Check for timestamped line
        elif line.startswith('(') and current_speaker:
            timestamp_end = line.find(')')
            if timestamp_end > 0:
                timestamp_str = line[1:timestamp_end]
                h, m, s = map(int, timestamp_str.split(':'))
                if current_start is None:
                    current_start = h * 3600 + m * 60 + s
                content = line[timestamp_end + 1:].strip()
                if content:
                    current_content.append(content)
        elif current_speaker:
            current_content.append(line)
    
    # Add final turn if needed
    if current_speaker and current_content and current_start is not None:
        turns.append({
            'speaker': current_speaker,
            'content': ' '.join(current_content),
            'start_time': current_start,
            'end_time': None
        })
    
    return turns

def get_vtt_turns(segments: list) -> list:
    """
    Collect turn-length segments from VTT transcript using ">>" as turn boundaries.
    """
    turns = []
    current_content = []
    current_start = None
    
    for segment in segments:
        content = segment['content'].strip()
        timestamp = segment['timestamp']
        
        # Check for turn boundary
        if content.startswith('>>'):
            # Save previous turn if exists
            if current_content:
                turns.append({
                    'content': ' '.join(current_content),
                    'start_time': current_start,
                    'end_time': timestamp - 0.001  # Just before current timestamp
                })
            # Start new turn
            current_content = [content[2:].strip()]  # Remove ">>" and start new turn
            current_start = timestamp
        else:
            if not current_content:  # Start first turn
                current_start = timestamp
            current_content.append(content)
    
    # Add final turn if needed
    if current_content:
        turns.append({
            'content': ' '.join(current_content),
            'start_time': current_start,
            'end_time': None
        })
    
    return turns

def group_vtt_segments(segments: list) -> list:
    """
    Group consecutive VTT segments that belong to the same speaker.
    Uses ">>" markers and natural pauses to detect turn boundaries.
    """
    grouped_segments = []
    current_group = []
    current_start = None
    last_timestamp = None
    
    def save_current_group():
        if current_group:
            grouped_segments.append({
                'content': ' '.join(current_group),
                'start_time': current_start,
                'end_time': last_timestamp - 0.001 if last_timestamp else None
            })
    
    for segment in segments:
        content = segment['content'].strip()
        timestamp = segment['timestamp']
        
        # Start new group on ">>" marker or long pause (>2 seconds)
        start_new_group = (
            content.startswith('>>') or
            (last_timestamp and timestamp - last_timestamp > 2.0) or
            not current_group  # First group
        )
        
        if start_new_group:
            save_current_group()
            current_group = [content[2:].strip() if content.startswith('>>') else content]
            current_start = timestamp
        else:
            current_group.append(content)
        
        last_timestamp = timestamp
    
    # Add final group
    save_current_group()
    
    return grouped_segments

def find_candidate_turns(timestamp: float, pasted_turns: list, margin: int = 30, max_candidates: int = 8) -> list:
    """
    Find the closest turns to a given timestamp, within a margin.
    Returns up to max_candidates turns, sorted by time proximity.
    """
    candidates = []
    
    for turn in pasted_turns:
        if turn['start_time'] is None:
            continue
            
        # Check if turn overlaps with or is close to timestamp
        time_diff = abs(turn['start_time'] - timestamp)
        if time_diff <= margin:
            candidates.append((turn, time_diff))
            
        # Also check end time if available
        if turn['end_time'] is not None:
            time_diff_end = abs(turn['end_time'] - timestamp)
            if time_diff_end <= margin:
                candidates.append((turn, time_diff_end))
    
    # Sort by time difference and take top candidates
    candidates.sort(key=lambda x: x[1])
    return [turn for turn, _ in candidates[:max_candidates]]

def get_words_from_text(text: str) -> list:
    """Split text into words, handling punctuation and whitespace."""
    return [word.strip().lower() for word in text.split() if word.strip()]

def compute_window_similarity(window_words: list, target_words: list) -> float:
    """
    Compute similarity between two word sequences using character bigrams.
    """
    # Join words back to text for bigram comparison
    window_text = ' '.join(window_words)
    target_text = ' '.join(target_words)
    
    # Create character bigrams
    def get_bigrams(text):
        return set(text[i:i+2] for i in range(len(text)-1))
    
    bigrams1 = get_bigrams(window_text)
    bigrams2 = get_bigrams(target_text)
    
    # Compute Jaccard similarity
    if not bigrams1 or not bigrams2:
        return 0.0
        
    intersection = len(bigrams1 & bigrams2)
    union = len(bigrams1 | bigrams2)
    
    # Also boost score for exact substring matches
    similarity = intersection / union
    if window_text in target_text or target_text in window_text:
        similarity += 0.3
        
    return min(similarity, 1.0)

def find_best_window_match(query_words: list, target_text: str) -> float:
    """
    Slide query window across target text and find highest similarity.
    Uses the natural length of the query (VTT line) as the window size.
    Allows for flexible window sizes (±20% of query length).
    """
    base_window_size = len(query_words)
    target_words = get_words_from_text(target_text)
    
    if len(target_words) < base_window_size * 0.8:  # Target too short
        return compute_window_similarity(query_words, target_words)
    
    max_similarity = 0.0
    
    # Try different window sizes around the base size
    min_size = max(int(base_window_size * 0.8), 1)
    max_size = min(int(base_window_size * 1.2), len(target_words))
    
    for window_size in range(min_size, max_size + 1):
        for i in range(len(target_words) - window_size + 1):
            target_window = target_words[i:i + window_size]
            window_similarity = compute_window_similarity(query_words, target_window)
            max_similarity = max(max_similarity, window_similarity)
    
    return max_similarity

def match_vtt_segments(vtt_segments: list, pasted_turns: list) -> list:
    """
    Match grouped VTT segments to pasted turns using sliding window similarity.
    """
    matched_segments = []
    
    for segment in vtt_segments:
        content = segment['content']
        start_time = segment['start_time']
        
        # Skip empty segments
        query_words = get_words_from_text(content)
        if not query_words:
            continue
        
        # Adjust time window based on segment length (1s per 3 words, min 10s, max 30s)
        time_margin = min(max(10, len(query_words) // 3), 30)
        
        # Find candidate turns based on time proximity
        candidates = find_candidate_turns(start_time, pasted_turns, margin=time_margin)
        if not candidates:
            continue
        
        # Find best matching turn
        best_score = 0
        best_match = None
        
        for turn in candidates:
            score = find_best_window_match(query_words, turn['content'])
            if score > best_score:
                best_score = score
                best_match = turn
        
        if best_match and best_score > 0.05:
            matched_segments.append({
                'vtt_content': content,
                'start_time': start_time,
                'end_time': segment['end_time'],
                'speaker': best_match['speaker'],
                'match_score': best_score,
                'pasted_turn': best_match
            })
    
    return matched_segments

def main():
    """Match turns between VTT and pasted transcripts."""
    try:
        # Load transcripts
        pasted_path = Path('data/raw/pasted_transcript.txt')
        vtt_path = Path('data/raw/full_video_sentences.vtt')  # Use new sentence-based VTT
        
        logger.info("Loading transcripts...")
        pasted_turns = get_pasted_turns(pasted_path)
        
        # Count turns and words by speaker in pasted transcript
        pasted_stats = {'Dylan Patel': {'turns': 0, 'words': 0},
                       'Nathan Lambert': {'turns': 0, 'words': 0},
                       'Lex Fridman': {'turns': 0, 'words': 0}}
        
        for turn in pasted_turns:
            speaker = turn['speaker']
            pasted_stats[speaker]['turns'] += 1
            pasted_stats[speaker]['words'] += len(get_words_from_text(turn['content']))
            
        print("\nPasted Transcript Statistics:")
        print("-" * 80)
        print(f"{'Speaker':<15} {'Turns':>8} {'Words':>12}")
        print("-" * 80)
        total_pasted_turns = 0
        total_pasted_words = 0
        for speaker, stats in pasted_stats.items():
            print(f"{speaker:<15} {stats['turns']:>8,d} {stats['words']:>12,d}")
            total_pasted_turns += stats['turns']
            total_pasted_words += stats['words']
        print("-" * 80)
        print(f"{'TOTAL':<15} {total_pasted_turns:>8,d} {total_pasted_words:>12,d}")
        print("-" * 80 + "\n")
        
        # Parse VTT file
        vtt_segments = []
        current_segment = None
        
        with open(vtt_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
            
        for line in lines:
            if line == 'WEBVTT':
                continue
                
            if '-->' in line:
                if current_segment:
                    vtt_segments.append(current_segment)
                
                start_time = line.split(' --> ')[0]
                h, m, s = map(float, start_time.split(':'))
                timestamp = h * 3600 + m * 60 + s
                
                current_segment = {
                    'timestamp': timestamp,
                    'content': ''
                }
            elif current_segment is not None:
                if current_segment['content']:
                    current_segment['content'] += ' '
                current_segment['content'] += line
        
        # Add final segment
        if current_segment:
            vtt_segments.append(current_segment)
        
        logger.info(f"Found {len(pasted_turns)} turns in pasted transcript")
        logger.info(f"Found {len(vtt_segments)} lines in VTT transcript")
        
        # Match segments to turns
        matched_segments = []
        
        for segment in vtt_segments:
            content = segment['content']
            timestamp = segment['timestamp']
            
            # Skip empty segments
            query_words = get_words_from_text(content)
            if not query_words:
                continue
            
            # Use larger time window for longer segments
            time_margin = min(max(15, len(query_words) // 5), 45)  # 1s per 5 words, min 15s, max 45s
            
            # Find candidate turns based on time proximity
            candidates = find_candidate_turns(timestamp, pasted_turns, margin=time_margin)
            if not candidates:
                continue
            
            # Find best matching turn
            best_score = 0
            best_match = None
            
            for turn in candidates:
                score = find_best_window_match(query_words, turn['content'])
                if score > best_score:
                    best_score = score
                    best_match = turn
            
            if best_match and best_score > 0.1:  # Higher threshold for longer segments
                matched_segments.append({
                    'vtt_content': content,
                    'start_time': timestamp,
                    'speaker': best_match['speaker'],
                    'match_score': best_score,
                    'pasted_turn': best_match
                })
        
        logger.info(f"Successfully matched {len(matched_segments)} segments")
        
        # Group matched segments by speaker and count words
        matched_stats = {'Dylan Patel': {'turns': 0, 'words': 0},
                        'Nathan Lambert': {'turns': 0, 'words': 0},
                        'Lex Fridman': {'turns': 0, 'words': 0}}
        
        for segment in matched_segments:
            speaker = segment['speaker']
            matched_stats[speaker]['turns'] += 1
            matched_stats[speaker]['words'] += len(get_words_from_text(segment['vtt_content']))
        
        print("\nMatched Transcript Statistics:")
        print("-" * 80)
        print(f"{'Speaker':<15} {'Turns':>8} {'Words':>12} {'Turn %':>8} {'Word %':>8}")
        print("-" * 80)
        total_matched_turns = 0
        total_matched_words = 0
        for speaker in matched_stats:
            pasted = pasted_stats[speaker]
            matched = matched_stats[speaker]
            turn_pct = (matched['turns'] / pasted['turns'] * 100) if pasted['turns'] > 0 else 0
            word_pct = (matched['words'] / pasted['words'] * 100) if pasted['words'] > 0 else 0
            print(f"{speaker:<15} {matched['turns']:>8,d} {matched['words']:>12,d} {turn_pct:>7.1f}% {word_pct:>7.1f}%")
            total_matched_turns += matched['turns']
            total_matched_words += matched['words']
        print("-" * 80)
        total_turn_pct = (total_matched_turns / total_pasted_turns * 100) if total_pasted_turns > 0 else 0
        total_word_pct = (total_matched_words / total_pasted_words * 100) if total_pasted_words > 0 else 0
        print(f"{'TOTAL':<15} {total_matched_turns:>8,d} {total_matched_words:>12,d} {total_turn_pct:>7.1f}% {total_word_pct:>7.1f}%")
        print("-" * 80)
        
        # Sort segments by start time
        speaker_segments = {'Dylan Patel': [], 'Nathan Lambert': [], 'Lex Fridman': []}
        for segment in matched_segments:
            speaker_segments[segment['speaker']].append(segment)
        for speaker in speaker_segments:
            speaker_segments[speaker].sort(key=lambda x: x['start_time'] or 0)
        
        # Print sample segments for each speaker
        for speaker in ['Dylan Patel', 'Nathan Lambert', 'Lex Fridman']:
            print(f"\n{speaker} Segments")
            print("=" * 120)
            
            segments = speaker_segments[speaker]
            if not segments:
                print("No segments found for this speaker")
                continue
            
            # Print beginning, middle, and end segments
            for i, position in [(0, "BEGINNING"), 
                              (len(segments)//2, "MIDDLE"), 
                              (-1, "END")]:
                segment = segments[i]
                print(f"\n{position} SEGMENT (Time: {segment['start_time']}, Match Score: {segment['match_score']:.3f}):")
                print("-" * 120)
                print("Pasted Transcript:")
                print(segment['pasted_turn']['content'][:500] + "..." if len(segment['pasted_turn']['content']) > 500 else segment['pasted_turn']['content'])
                print("\nVTT Transcript:")
                print(segment['vtt_content'][:500] + "..." if len(segment['vtt_content']) > 500 else segment['vtt_content'])
                print("-" * 120)
        
    except Exception as e:
        logger.error(f"Error matching transcripts: {e}")
        raise

if __name__ == '__main__':
    main() 