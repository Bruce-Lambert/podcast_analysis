#!/usr/bin/env python3
"""
Show multiple turns from each speaker across the transcript to verify speaker attribution consistency.
Shows beginning, middle, and end turns for each speaker.
"""

import json
from pathlib import Path
import logging
import webvtt
from difflib import SequenceMatcher
import re
from typing import List, Dict, Tuple
from dataclasses import dataclass
from collections import defaultdict

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Turn:
    speaker: str
    text: str
    start_time: float = None
    end_time: float = None

def get_all_turns_from_pasted(file_path: Path) -> list:
    """
    Create a sequential database of all turns from the pasted transcript.
    Each turn has a sequential number, speaker, and time interval.
    """
    turns = []  # List of all turns in sequence
    current_speaker = None
    current_content = []
    current_start = None
    turn_number = 0  # Sequential turn number across all speakers
    
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
                            end_time = h * 3600 + m * 60 + s - 1  # Subtract 1 second
                            break
                
                turns.append({
                    'turn_number': turn_number,
                    'speaker': current_speaker,
                    'start_time': current_start,
                    'end_time': end_time,
                    'content': ' '.join(current_content)
                })
                turn_number += 1
            
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
            'turn_number': turn_number,
            'speaker': current_speaker,
            'start_time': current_start,
            'end_time': None,  # Last turn has no end time
            'content': ' '.join(current_content)
        })
    
    return turns

def create_time_to_speaker_mapping(turns: list) -> list:
    """
    Create a sorted list of time boundaries with speaker assignments.
    Each entry contains the start time and the speaker active from that time until the next entry.
    """
    boundaries = []
    
    for turn in turns:
        # Add turn start
        boundaries.append({
            'time': turn['start_time'],
            'speaker': turn['speaker']
        })
        # Add turn end if it exists
        if turn['end_time'] is not None:
            boundaries.append({
                'time': turn['end_time'] + 1,  # Start of next speaker's time
                'speaker': None  # Will be filled by next turn
            })
    
    # Sort boundaries by time
    boundaries.sort(key=lambda x: x['time'])
    
    # Fill in None speakers with the next known speaker
    last_known_speaker = None
    for i in range(len(boundaries)):
        if boundaries[i]['speaker'] is not None:
            last_known_speaker = boundaries[i]['speaker']
        else:
            boundaries[i]['speaker'] = last_known_speaker
    
    return boundaries

def find_speaker_for_time(timestamp: float, time_mapping: list) -> str:
    """
    Binary search to find the speaker for a given timestamp.
    """
    left = 0
    right = len(time_mapping) - 1
    
    while left <= right:
        mid = (left + right) // 2
        
        # Check if this is the right boundary
        if mid < len(time_mapping) - 1 and (
            time_mapping[mid]['time'] <= timestamp < time_mapping[mid + 1]['time']):
            return time_mapping[mid]['speaker']
        elif timestamp < time_mapping[mid]['time']:
            right = mid - 1
        else:
            left = mid + 1
    
    # If we're past the last boundary, use the last known speaker
    if timestamp >= time_mapping[-1]['time']:
        return time_mapping[-1]['speaker']
    
    return None

def get_speaker_turns_combined(segments: list) -> dict:
    """Map VTT segments to turns from pasted transcript."""
    # First get all turns in sequence from pasted transcript
    pasted_path = Path('data/raw/pasted_transcript.txt')
    all_turns = get_all_turns_from_pasted(pasted_path)
    
    # Create efficient time-to-speaker mapping
    time_mapping = create_time_to_speaker_mapping(all_turns)
    
    # Initialize combined turns by speaker
    speaker_turns = {'Dylan Patel': [], 'Nathan Lambert': [], 'Lex Fridman': []}
    current_turn = None
    current_content = []
    
    for segment in segments:
        timestamp = segment['timestamp']
        content = segment['content'].strip()
        
        # Find the speaker for this timestamp
        speaker = find_speaker_for_time(timestamp, time_mapping)
        
        # Find the corresponding turn from all_turns
        matching_turn = None
        for turn in all_turns:
            if (turn['speaker'] == speaker and 
                turn['start_time'] <= timestamp and 
                (turn['end_time'] is None or timestamp <= turn['end_time'])):
                matching_turn = turn
                break
        
        if matching_turn:
            if current_turn != matching_turn:
                # Save previous turn if exists
                if current_turn and current_content:
                    speaker_turns[current_turn['speaker']].append({
                        'turn_number': current_turn['turn_number'],
                        'content': ' '.join(current_content),
                        'timestamp': current_turn['start_time']
                    })
                # Start new turn
                current_turn = matching_turn
                current_content = []
            
            # Handle content
            if content.startswith('>>'):
                # Remove ">>" from start of line
                content = content[2:].strip()
            current_content.append(content)
    
    # Add final turn if needed
    if current_turn and current_content:
        speaker_turns[current_turn['speaker']].append({
            'turn_number': current_turn['turn_number'],
            'content': ' '.join(current_content),
            'timestamp': current_turn['start_time']
        })
    
    # Sort each speaker's turns by turn number to maintain sequence
    for speaker in speaker_turns:
        speaker_turns[speaker].sort(key=lambda x: x['turn_number'])
    
    return speaker_turns

def print_turn_comparison(pasted_turn, combined_turn, turn_position):
    """Print a side-by-side comparison of turns."""
    print(f"\n{turn_position} Turn (Turn #{pasted_turn['turn_number']}, Time: {pasted_turn['start_time']}):")
    print("-" * 120)
    print("Pasted Transcript:")
    print(pasted_turn['content'][:500] + "..." if len(pasted_turn['content']) > 500 else pasted_turn['content'])
    print("\nCombined Transcript:")
    print(combined_turn['content'][:500] + "..." if len(combined_turn['content']) > 500 else combined_turn['content'])
    print("-" * 120)

def load_pasted_transcript(file_path: str) -> List[Turn]:
    """Load and parse the pasted transcript with timing info if available."""
    turns = []
    current_speaker = None
    current_text = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                if current_speaker and current_text:
                    turns.append(Turn(
                        speaker=current_speaker,
                        text=' '.join(current_text)
                    ))
                    current_text = []
                continue
            
            if ':' in line:
                if current_speaker and current_text:
                    turns.append(Turn(
                        speaker=current_speaker,
                        text=' '.join(current_text)
                    ))
                    current_text = []
                
                speaker, text = line.split(':', 1)
                current_speaker = speaker.strip()
                if text.strip():
                    current_text.append(text.strip())
            else:
                current_text.append(line)
    
    if current_speaker and current_text:
        turns.append(Turn(
            speaker=current_speaker,
            text=' '.join(current_text)
        ))
    
    return turns

def load_vtt_sentences(file_path: str) -> List[Turn]:
    """Load sentence-based VTT transcript."""
    vtt_turns = []
    
    for caption in webvtt.read(file_path):
        text = caption.text.strip()
        if ':' in text:
            speaker, content = text.split(':', 1)
            vtt_turns.append(Turn(
                speaker=speaker.strip(),
                text=content.strip(),
                start_time=caption.start_in_seconds,
                end_time=caption.end_in_seconds
            ))
        else:
            vtt_turns.append(Turn(
                speaker="UNKNOWN",
                text=text,
                start_time=caption.start_in_seconds,
                end_time=caption.end_in_seconds
            ))
    
    return vtt_turns

def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    text = re.sub(r'\s+', ' ', text.lower())
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()

def calculate_text_similarity(text1: str, text2: str) -> float:
    """Calculate similarity between two text strings."""
    text1 = normalize_text(text1)
    text2 = normalize_text(text2)
    return SequenceMatcher(None, text1, text2).ratio()

def find_best_matching_turn(vtt_turn: Turn, pasted_turns: List[Turn], 
                          similarity_threshold: float = 0.3) -> Tuple[Turn, float]:
    """Find the best matching turn from pasted transcript."""
    best_match = None
    best_similarity = 0
    
    for turn in pasted_turns:
        similarity = calculate_text_similarity(vtt_turn.text, turn.text)
        if similarity > best_similarity:
            best_similarity = similarity
            best_match = turn
    
    if best_similarity >= similarity_threshold:
        return best_match, best_similarity
    return None, 0

def align_transcripts(vtt_turns: List[Turn], pasted_turns: List[Turn]) -> List[Turn]:
    """
    Create optimal alignment between VTT sentences and pasted transcript turns.
    Uses global statistics and temporal proximity for matching.
    """
    # Get target statistics from pasted transcript
    pasted_stats = get_speaker_stats(pasted_turns)
    current_stats = defaultdict(lambda: {"words": 0, "turns": 0})
    aligned_turns = []
    
    # Group VTT turns into temporal windows
    window_size = 10  # seconds
    current_window = []
    current_window_start = 0
    
    for turn in vtt_turns:
        if turn.start_time - current_window_start > window_size and current_window:
            # Process current window
            window_assignments = optimize_window_assignments(
                current_window, 
                pasted_stats, 
                current_stats
            )
            
            # Update statistics and add aligned turns
            for assigned_turn in window_assignments:
                current_stats[assigned_turn.speaker]["words"] += get_word_count(assigned_turn.text)
                current_stats[assigned_turn.speaker]["turns"] += 1
                aligned_turns.append(assigned_turn)
            
            # Start new window
            current_window = [turn]
            current_window_start = turn.start_time
        else:
            current_window.append(turn)
    
    # Process final window
    if current_window:
        window_assignments = optimize_window_assignments(
            current_window,
            pasted_stats,
            current_stats
        )
        for assigned_turn in window_assignments:
            current_stats[assigned_turn.speaker]["words"] += get_word_count(assigned_turn.text)
            current_stats[assigned_turn.speaker]["turns"] += 1
            aligned_turns.append(assigned_turn)
    
    return aligned_turns

def optimize_window_assignments(window_turns: List[Turn], 
                              target_stats: Dict, 
                              current_stats: Dict) -> List[Turn]:
    """Optimize speaker assignments for a window of turns."""
    speakers = list(target_stats.keys())
    assignments = []
    
    # Calculate target ratios
    total_target_words = sum(stats["words"] for stats in target_stats.values())
    speaker_ratios = {
        speaker: stats["words"] / total_target_words 
        for speaker, stats in target_stats.items()
    }
    
    # Sort turns by confidence (if they already have speaker assignments)
    sorted_turns = sorted(
        window_turns,
        key=lambda t: 1 if t.speaker in speakers else 0
    )
    
    for turn in sorted_turns:
        best_speaker = None
        min_cost = float('inf')
        
        for speaker in speakers:
            # Calculate cost based on:
            # 1. Word count deviation from target ratio
            # 2. Turn count deviation
            # 3. Temporal continuity (prefer same speaker for adjacent turns)
            
            words = get_word_count(turn.text)
            current_ratio = (current_stats[speaker]["words"] + words) / (sum(s["words"] for s in current_stats.values()) + words)
            ratio_cost = abs(current_ratio - speaker_ratios[speaker])
            
            turn_ratio = current_stats[speaker]["turns"] / max(1, sum(s["turns"] for s in current_stats.values()))
            turn_cost = abs(turn_ratio - speaker_ratios[speaker])
            
            # Add temporal continuity cost
            continuity_cost = 0
            if assignments:
                last_turn = assignments[-1]
                if last_turn.speaker != speaker:
                    continuity_cost = 5
            
            total_cost = ratio_cost + turn_cost * 2 + continuity_cost
            
            if total_cost < min_cost:
                min_cost = total_cost
                best_speaker = speaker
        
        assignments.append(Turn(
            speaker=best_speaker,
            text=turn.text,
            start_time=turn.start_time,
            end_time=turn.end_time
        ))
    
    return assignments

def calculate_metrics(turns: List[Turn]) -> Dict:
    """Calculate word and turn counts per speaker."""
    metrics = {}
    for turn in turns:
        if turn.speaker not in metrics:
            metrics[turn.speaker] = {"words": 0, "turns": 0}
        words = len(normalize_text(turn.text).split())
        metrics[turn.speaker]["words"] += words
        metrics[turn.speaker]["turns"] += 1
    return metrics

def main():
    """Show turns from different parts of the transcript for each speaker."""
    try:
        # Load transcripts
        pasted_path = Path('data/raw/pasted_transcript.txt')
        combined_path = Path('data/processed/combined_transcript.json')
        
        logger.info("Loading transcripts...")
        all_pasted_turns = get_all_turns_from_pasted(pasted_path)
        
        # Group pasted turns by speaker
        pasted_turns = {'Dylan Patel': [], 'Nathan Lambert': [], 'Lex Fridman': []}
        for turn in all_pasted_turns:
            pasted_turns[turn['speaker']].append(turn)
        
        with open(combined_path, 'r') as f:
            combined_segments = json.load(f)
        combined_turns = get_speaker_turns_combined(combined_segments)
        
        # For each speaker, show turns from beginning, middle, and end
        for speaker in ['Dylan Patel', 'Nathan Lambert', 'Lex Fridman']:
            print(f"\n{speaker} Turns")
            print("=" * 120)
            
            # Get all turns for this speaker
            speaker_pasted_turns = pasted_turns[speaker]
            speaker_combined_turns = combined_turns[speaker]
            
            print(f"\nTotal turns in pasted transcript: {len(speaker_pasted_turns)}")
            print(f"Total turns in combined transcript: {len(speaker_combined_turns)}")
            
            if not speaker_pasted_turns or not speaker_combined_turns:
                print("No turns found for this speaker")
                continue
            
            # Get beginning turn
            begin_idx = 0
            print("\nBEGINNING TURN:")
            if begin_idx < len(speaker_pasted_turns) and begin_idx < len(speaker_combined_turns):
                print_turn_comparison(
                    speaker_pasted_turns[begin_idx],
                    speaker_combined_turns[begin_idx],
                    "Beginning"
                )
            
            # Get middle turn
            mid_idx = len(speaker_pasted_turns) // 2
            mid_combined_idx = len(speaker_combined_turns) // 2
            print("\nMIDDLE TURN:")
            if mid_idx < len(speaker_pasted_turns) and mid_combined_idx < len(speaker_combined_turns):
                print_turn_comparison(
                    speaker_pasted_turns[mid_idx],
                    speaker_combined_turns[mid_combined_idx],
                    "Middle"
                )
            
            # Get end turn
            end_idx = len(speaker_pasted_turns) - 1
            end_combined_idx = len(speaker_combined_turns) - 1
            print("\nEND TURN:")
            if end_idx >= 0 and end_combined_idx >= 0:
                print_turn_comparison(
                    speaker_pasted_turns[end_idx],
                    speaker_combined_turns[end_combined_idx],
                    "End"
                )
        
        # Load VTT sentences
        vtt_turns = load_vtt_sentences("data/raw/macwhisper_full_video.vtt")
        
        # Print original metrics
        print("\nPasted Transcript Metrics:")
        pasted_metrics = calculate_metrics(pasted_turns)
        for speaker, stats in pasted_metrics.items():
            print(f"{speaker}: {stats['words']} words, {stats['turns']} turns")
        
        print("\nVTT Transcript Metrics:")
        vtt_metrics = calculate_metrics(vtt_turns)
        for speaker, stats in vtt_metrics.items():
            print(f"{speaker}: {stats['words']} words, {stats['turns']} turns")
        
        # Align transcripts
        aligned_turns = align_transcripts(vtt_turns, pasted_turns)
        
        # Print aligned metrics
        print("\nAligned Transcript Metrics:")
        aligned_metrics = calculate_metrics(aligned_turns)
        for speaker, stats in aligned_metrics.items():
            print(f"{speaker}: {stats['words']} words, {stats['turns']} turns")
        
        # Save aligned transcript
        output_path = Path("data/processed/full_podcast/aligned_transcript.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        aligned_data = [
            {
                "speaker": turn.speaker,
                "text": turn.text,
                "start_time": turn.start_time,
                "end_time": turn.end_time
            }
            for turn in aligned_turns
        ]
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(aligned_data, f, indent=2)
        
        print(f"\nAligned transcript saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Error comparing transcripts: {e}")
        raise

if __name__ == '__main__':
    main() 