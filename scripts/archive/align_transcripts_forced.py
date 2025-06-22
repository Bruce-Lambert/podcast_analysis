#!/usr/bin/env python3
"""
Align VTT transcript with pasted transcript using forced alignment.
Uses aeneas for forced alignment to get word-level timestamps,
then maps those to speaker segments from the pasted transcript.
"""

import json
from pathlib import Path
import logging
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import re
from collections import defaultdict
import webvtt
import tempfile
import os
from aeneas.tools.execute_task import ExecuteTaskCLI
from aeneas.executetask import ExecuteTask
from aeneas.task import Task
from aeneas.textfile import TextFile
from aeneas.language import Language

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class Turn:
    """Represents a turn from either transcript."""
    speaker: str
    content: str
    start_time: float
    end_time: Optional[float] = None

@dataclass
class Word:
    """Represents a word with timing information."""
    word: str
    start_time: float
    end_time: float
    speaker: Optional[str] = None

def parse_timestamp(timestamp: str) -> float:
    """Convert timestamp string to seconds."""
    h, m, s = map(float, timestamp.split(':'))
    return h * 3600 + m * 60 + s

def extract_pasted_turns(file_path: Path) -> List[Turn]:
    """Extract turns from pasted transcript with speaker attribution."""
    turns = []
    current_speaker = None
    current_content = []
    current_start = None
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    for i, line in enumerate(lines):
        if line in {'Dylan Patel', 'Nathan Lambert', 'Lex Fridman'}:
            # Save previous turn if exists
            if current_speaker and current_content and current_start is not None:
                # Look ahead for next timestamp
                end_time = None
                for next_line in lines[i:]:
                    if next_line.startswith('('):
                        timestamp_end = next_line.find(')')
                        if timestamp_end > 0:
                            timestamp_str = next_line[1:timestamp_end]
                            end_time = parse_timestamp(timestamp_str) - 0.001
                            break
                
                turns.append(Turn(
                    speaker=current_speaker,
                    content=' '.join(current_content),
                    start_time=current_start,
                    end_time=end_time
                ))
            
            current_speaker = line
            current_content = []
            current_start = None
            
        elif line.startswith('(') and current_speaker:
            timestamp_end = line.find(')')
            if timestamp_end > 0:
                timestamp_str = line[1:timestamp_end]
                current_start = parse_timestamp(timestamp_str)
                content = line[timestamp_end + 1:].strip()
                if content:
                    current_content.append(content)
        elif current_speaker:
            current_content.append(line)
    
    # Add final turn if needed
    if current_speaker and current_content and current_start is not None:
        turns.append(Turn(
            speaker=current_speaker,
            content=' '.join(current_content),
            start_time=current_start,
            end_time=None
        ))
    
    return turns

def extract_vtt_content(file_path: Path) -> str:
    """Extract full text content from VTT transcript."""
    content = []
    
    for caption in webvtt.read(str(file_path)):
        text = caption.text.strip()
        # Remove '>>' speaker indicators
        text = re.sub(r'^>>\s*', '', text)
        if text:
            content.append(text)
    
    return ' '.join(content)

def run_forced_alignment(audio_path: Path, text: str) -> List[Word]:
    """Run aeneas forced alignment to get word-level timestamps."""
    # Create temporary files
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt') as text_file, \
         tempfile.NamedTemporaryFile(mode='w', suffix='.json') as sync_map_file:
        
        # Write text to file
        text_file.write(text)
        text_file.flush()
        
        # Create and execute task
        config_string = "task_language=eng|is_text_type=plain|os_task_file_format=json|task_adjust_boundary_nonspeech_min=0.1|task_adjust_boundary_nonspeech_string=REMOVE|task_adjust_boundary_algorithm=percent|task_adjust_boundary_percent_value=50|is_audio_file_detect_head_max=0.0|is_audio_file_detect_head_min=0.0|is_audio_file_detect_tail_max=0.0|is_audio_file_detect_tail_min=0.0"
        
        task = Task(config_string=config_string)
        task.audio_file_path_absolute = str(audio_path)
        task.text_file_path_absolute = text_file.name
        task.sync_map_file_path_absolute = sync_map_file.name
        
        # Execute task
        ExecuteTask(task).execute()
        task.output_sync_map_file()
        
        # Read results
        with open(sync_map_file.name, 'r') as f:
            alignment = json.load(f)
        
        # Convert to Word objects
        words = []
        for fragment in alignment['fragments']:
            words.append(Word(
                word=fragment['lines'][0],
                start_time=float(fragment['begin']),
                end_time=float(fragment['end'])
            ))
        
        return words

def assign_speakers_to_words(words: List[Word], turns: List[Turn]) -> List[Word]:
    """Assign speakers to words based on turn timestamps."""
    for word in words:
        # Find the turn that contains this word
        for turn in turns:
            if (turn.start_time <= word.start_time and 
                (turn.end_time is None or word.end_time <= turn.end_time)):
                word.speaker = turn.speaker
                break
    
    return words

def group_words_into_sentences(words: List[Word]) -> List[Dict]:
    """Group words into sentences based on punctuation and pauses."""
    sentences = []
    current_sentence = []
    current_speaker = None
    sentence_start = None
    
    for word in words:
        # Skip words without speaker assignment
        if not word.speaker:
            continue
        
        # Start new sentence if:
        # 1. First word
        # 2. Speaker changes
        # 3. Long pause (> 1 second)
        # 4. Previous word ends with sentence-ending punctuation
        if (not current_sentence or
            word.speaker != current_speaker or
            (word.start_time - current_sentence[-1].end_time > 1.0) or
            current_sentence[-1].word.rstrip(',.!?').lower() != current_sentence[-1].word.lower()):
            
            # Save previous sentence
            if current_sentence:
                sentences.append({
                    'speaker': current_speaker,
                    'content': ' '.join(w.word for w in current_sentence),
                    'start_time': sentence_start,
                    'end_time': current_sentence[-1].end_time
                })
            
            # Start new sentence
            current_sentence = [word]
            current_speaker = word.speaker
            sentence_start = word.start_time
        else:
            current_sentence.append(word)
    
    # Add final sentence
    if current_sentence:
        sentences.append({
            'speaker': current_speaker,
            'content': ' '.join(w.word for w in current_sentence),
            'start_time': sentence_start,
            'end_time': current_sentence[-1].end_time
        })
    
    return sentences

def print_alignment_stats(pasted_turns: List[Turn], aligned_sentences: List[Dict]):
    """Print detailed alignment statistics."""
    # Count original turns and words by speaker
    pasted_stats = defaultdict(lambda: {'turns': 0, 'words': 0})
    for turn in pasted_turns:
        pasted_stats[turn.speaker]['turns'] += 1
        pasted_stats[turn.speaker]['words'] += len(turn.content.split())
    
    # Count aligned turns and words
    aligned_stats = defaultdict(lambda: {'turns': 0, 'words': 0})
    for sentence in aligned_sentences:
        speaker = sentence['speaker']
        aligned_stats[speaker]['turns'] += 1
        aligned_stats[speaker]['words'] += len(sentence['content'].split())
    
    # Print statistics
    print("\nAlignment Statistics:")
    print("=" * 120)
    print(f"{'Speaker':<15} {'Pasted':^24} {'Aligned':^24} {'Coverage':^35}")
    print(f"{'':15} {'Turns':>11} {'Words':>12} {'Turns':>11} {'Words':>12} {'Turns %':>11} {'Words %':>11}")
    print("-" * 120)
    
    total_pasted = {'turns': 0, 'words': 0}
    total_aligned = {'turns': 0, 'words': 0}
    
    for speaker in sorted(pasted_stats.keys()):
        pasted = pasted_stats[speaker]
        aligned = aligned_stats[speaker]
        
        # Calculate coverage
        turn_pct = (aligned['turns'] / pasted['turns'] * 100) if pasted['turns'] > 0 else 0
        word_pct = (aligned['words'] / pasted['words'] * 100) if pasted['words'] > 0 else 0
        
        print(f"{speaker:<15} {pasted['turns']:>11,d} {pasted['words']:>12,d} "
              f"{aligned['turns']:>11,d} {aligned['words']:>12,d} "
              f"{turn_pct:>10.1f}% {word_pct:>10.1f}%")
        
        # Print example sentences
        speaker_sentences = [s for s in aligned_sentences if s['speaker'] == speaker]
        if speaker_sentences:
            print(f"\nExample sentences for {speaker}:")
            for i, sentence in enumerate(speaker_sentences[:3], 1):
                print(f"{i}. [{sentence['start_time']:.1f}-{sentence['end_time']:.1f}] {sentence['content']}")
        
        total_pasted['turns'] += pasted['turns']
        total_pasted['words'] += pasted['words']
        total_aligned['turns'] += aligned['turns']
        total_aligned['words'] += aligned['words']
    
    print("-" * 120)
    total_turn_pct = (total_aligned['turns'] / total_pasted['turns'] * 100) if total_pasted['turns'] > 0 else 0
    total_word_pct = (total_aligned['words'] / total_pasted['words'] * 100) if total_pasted['words'] > 0 else 0
    
    print(f"{'TOTAL':<15} {total_pasted['turns']:>11,d} {total_pasted['words']:>12,d} "
          f"{total_aligned['turns']:>11,d} {total_aligned['words']:>12,d} "
          f"{total_turn_pct:>10.1f}% {total_word_pct:>10.1f}%")
    print("=" * 120)

def main():
    """Align VTT transcript with pasted transcript using forced alignment."""
    try:
        # Setup paths
        pasted_path = Path('data/raw/pasted_transcript.txt')
        vtt_path = Path('data/raw/full_video_sentences.vtt')
        audio_path = Path('data/raw/full_podcast/full_video.mp4')
        output_path = Path('data/processed/full_podcast/aligned_transcript.json')
        
        logger.info("Loading transcripts...")
        pasted_turns = extract_pasted_turns(pasted_path)
        vtt_content = extract_vtt_content(vtt_path)
        
        logger.info("Running forced alignment...")
        words = run_forced_alignment(audio_path, vtt_content)
        
        logger.info("Assigning speakers...")
        words = assign_speakers_to_words(words, pasted_turns)
        
        logger.info("Grouping into sentences...")
        sentences = group_words_into_sentences(words)
        
        logger.info(f"Successfully aligned {len(sentences)} segments")
        
        # Print statistics
        print_alignment_stats(pasted_turns, sentences)
        
        # Create output directory if needed
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save aligned transcript
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(sentences, f, indent=2)
        
        logger.info(f"Saved aligned transcript to {output_path}")
        
    except Exception as e:
        logger.error(f"Error aligning transcripts: {e}")
        raise

if __name__ == '__main__':
    main() 