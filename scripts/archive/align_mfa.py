#!/usr/bin/env python3
"""
Align transcript with audio using Montreal Forced Aligner (MFA).
This script takes a sample segment and performs forced alignment to get word-level timestamps.
"""

import json
from pathlib import Path
import logging
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import re
from collections import defaultdict
import tempfile
import os
import subprocess
import shutil

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

def prepare_mfa_input(turns: List[Turn], work_dir: Path) -> Tuple[Path, Path, Path]:
    """
    Prepare input files for MFA.
    Returns paths to the corpus directory, lab file, and speaker mapping file.
    """
    # Create corpus directory structure
    corpus_dir = work_dir / "corpus"
    corpus_dir.mkdir(exist_ok=True)
    
    # Create speaker directories and lab files
    speaker_files = {}
    for speaker in {'Dylan Patel', 'Nathan Lambert', 'Lex Fridman'}:
        speaker_dir = corpus_dir / speaker.replace(' ', '_')
        speaker_dir.mkdir(exist_ok=True)
        speaker_files[speaker] = speaker_dir
    
    # Create lab files for each speaker's turns
    for turn in turns:
        speaker_dir = speaker_files[turn.speaker]
        # Create unique filename based on start time
        lab_file = speaker_dir / f"{int(turn.start_time):06d}.lab"
        with open(lab_file, "w", encoding="utf-8") as f:
            # Clean text for alignment
            text = re.sub(r'[^\w\s]', '', turn.content)
            text = text.lower()
            f.write(f"{text}\n")
    
    return corpus_dir

def parse_textgrid(textgrid_file: Path) -> List[Word]:
    """Parse TextGrid file to extract word alignments."""
    words = []
    current_word = None
    in_words = False
    in_interval = False
    
    with open(textgrid_file, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f]
    
    # Extract speaker from filename
    speaker = textgrid_file.parent.name.replace('_', ' ')
    
    for i, line in enumerate(lines):
        if 'name = "words"' in line:
            in_words = True
            continue
        
        if in_words:
            if 'intervals [' in line:
                current_word = None
                in_interval = True
            elif in_interval and 'xmin = ' in line:
                if not current_word:
                    current_word = {}
                current_word['start'] = float(line.split('=')[1].strip())
            elif in_interval and 'xmax = ' in line:
                if current_word:
                    current_word['end'] = float(line.split('=')[1].strip())
            elif in_interval and 'text = ' in line:
                text = line.split('=')[1].strip().strip('"')
                if text and text != "sp" and text != "<unk>" and text != "":
                    if current_word:
                        current_word['text'] = text
                        words.append(Word(
                            word=text,
                            start_time=current_word['start'],
                            end_time=current_word['end'],
                            speaker=speaker
                        ))
                current_word = None
                in_interval = False
            elif 'item [' in line and 'item [1]' not in line:
                in_words = False
    
    return words

def run_mfa_align(audio_path: Path, corpus_dir: Path, work_dir: Path) -> List[Dict]:
    """
    Run MFA alignment on the audio and transcript.
    Returns list of word alignments.
    """
    try:
        # Create audio files for each speaker's turns
        for speaker_dir in corpus_dir.iterdir():
            if speaker_dir.is_dir():
                for lab_file in speaker_dir.glob("*.lab"):
                    wav_file = lab_file.with_suffix(".wav")
                    if not wav_file.exists():
                        subprocess.run([
                            "ffmpeg", "-i", str(audio_path),
                            "-acodec", "pcm_s16le",
                            "-ar", "16000",
                            "-ac", "1",
                            str(wav_file)
                        ], check=True)
        
        # Run alignment
        output_dir = work_dir / "aligned"
        output_dir.mkdir(exist_ok=True)
        
        logger.info("Running alignment...")
        subprocess.run([
            "mfa", "model", "download", "acoustic", "english_mfa"
        ], check=True)
        subprocess.run([
            "mfa", "model", "download", "dictionary", "english_mfa"
        ], check=True)
        subprocess.run([
            "mfa", "align",
            str(corpus_dir),
            "english_mfa",
            "english_mfa",
            str(output_dir),
            "--clean",
            "--overwrite"
        ], check=True)
        
        # Parse MFA output - combine all TextGrid files
        words = []
        for textgrid_file in output_dir.glob("**/*.TextGrid"):
            words.extend(parse_textgrid(textgrid_file))
        
        return sorted(words, key=lambda w: w.start_time)
        
    except Exception as e:
        logger.error(f"Error in MFA alignment: {e}")
        raise

def assign_speakers_to_words(words: List[Word], turns: List[Turn]) -> List[Word]:
    """Assign speakers to words based on turn timestamps."""
    # First, sort words by start time
    words = sorted(words, key=lambda w: w.start_time)
    
    # Create a list of non-overlapping turns
    sorted_turns = sorted(turns, key=lambda t: t.start_time)
    final_turns = []
    current_turn = None
    
    for turn in sorted_turns:
        if not current_turn:
            current_turn = turn
        elif turn.start_time > current_turn.end_time:
            final_turns.append(current_turn)
            current_turn = turn
        else:
            # Handle overlapping turns - take the one that ends first
            if turn.end_time < current_turn.end_time:
                current_turn = turn
    
    if current_turn:
        final_turns.append(current_turn)
    
    # Assign speakers to words
    for word in words:
        # Skip if word already has speaker assigned from TextGrid
        if word.speaker:
            continue
            
        # Find the turn that contains this word
        for turn in final_turns:
            if (turn.start_time <= word.start_time and 
                (turn.end_time is None or word.end_time <= turn.end_time)):
                word.speaker = turn.speaker
                break
    
    return words

def clean_text(text):
    """Clean text by removing filler words and normalizing whitespace, but preserve discourse markers."""
    # Split into words and filter out empty strings
    words = [w.strip() for w in text.split() if w.strip()]
    
    # List of filler words to remove - explicitly exclude "right"
    fillers = {'um', 'uh', 'like', 'you know', 'i mean', 'sort of', 'kind of'}
    
    # Remove consecutive duplicates and filler words while preserving order
    # and keeping the word "right"
    cleaned_words = []
    for i, word in enumerate(words):
        word_lower = word.lower()
        if (word_lower == 'right' or  # Always keep "right"
            ((i == 0 or word != words[i-1]) and word_lower not in fillers)):
            cleaned_words.append(word)
    
    # Join words back together
    return ' '.join(cleaned_words)

def group_words_into_turns(words_by_speaker):
    """Group words into turns based on timing and speaker."""
    MIN_TURN_DURATION = 1.0  # Minimum turn duration in seconds
    MAX_TURN_DURATION = 20.0  # Maximum turn duration in seconds
    MAX_PAUSE = 0.75  # Maximum pause between words in the same turn
    MIN_OVERLAP = 0.2  # Minimum overlap to consider as overlapping speech
    
    turns = []
    
    # First pass: merge overlapping or closely timed words by same speaker
    for speaker, words in words_by_speaker.items():
        merged_words = []
        current_group = []
        
        for word in sorted(words, key=lambda x: x['start']):
            if not current_group:
                current_group.append(word)
            else:
                # Check if words overlap or are close in time
                last_word = current_group[-1]
                gap = word['start'] - last_word['end']
                duration = word['end'] - current_group[0]['start']
                
                if gap < MAX_PAUSE and duration < MAX_TURN_DURATION:
                    current_group.append(word)
                else:
                    # Process current group if it's long enough
                    if len(current_group) > 0:
                        start_time = min(w['start'] for w in current_group)
                        end_time = max(w['end'] for w in current_group)
                        if end_time - start_time >= MIN_TURN_DURATION:
                            text = ' '.join(w['word'] for w in current_group)
                            merged_words.append({
                                'word': clean_text(text),
                                'start': start_time,
                                'end': end_time
                            })
                    current_group = [word]
        
        # Handle last group
        if current_group:
            start_time = min(w['start'] for w in current_group)
            end_time = max(w['end'] for w in current_group)
            if end_time - start_time >= MIN_TURN_DURATION:
                text = ' '.join(w['word'] for w in current_group)
                merged_words.append({
                    'word': clean_text(text),
                    'start': start_time,
                    'end': end_time
                })
        
        # Update words for this speaker
        words_by_speaker[speaker] = merged_words
    
    # Second pass: create turns and handle overlaps between speakers
    all_words = []
    for speaker, words in words_by_speaker.items():
        for word in words:
            word['speaker'] = speaker
            all_words.append(word)
    
    all_words.sort(key=lambda x: x['start'])
    
    current_turn = None
    for word in all_words:
        if not current_turn:
            current_turn = {
                'speaker': word['speaker'],
                'words': [word],
                'start_time': word['start'],
                'end_time': word['end']
            }
        else:
            # Check if this word should start a new turn
            gap = word['start'] - current_turn['end_time']
            same_speaker = word['speaker'] == current_turn['speaker']
            duration = word['end'] - current_turn['start_time']
            
            if same_speaker and gap < MAX_PAUSE and duration < MAX_TURN_DURATION:
                # Extend current turn
                current_turn['words'].append(word)
                current_turn['end_time'] = max(current_turn['end_time'], word['end'])
            else:
                # Finalize current turn if it's long enough
                if current_turn['end_time'] - current_turn['start_time'] >= MIN_TURN_DURATION:
                    turn_text = ' '.join(w['word'] for w in current_turn['words'])
                    turns.append({
                        'speaker': current_turn['speaker'],
                        'words': current_turn['words'],
                        'start_time': current_turn['start_time'],
                        'end_time': current_turn['end_time'],
                        'text': clean_text(turn_text)
                    })
                # Start new turn
                current_turn = {
                    'speaker': word['speaker'],
                    'words': [word],
                    'start_time': word['start'],
                    'end_time': word['end']
                }
    
    # Handle last turn
    if current_turn and current_turn['end_time'] - current_turn['start_time'] >= MIN_TURN_DURATION:
        turn_text = ' '.join(w['word'] for w in current_turn['words'])
        turns.append({
            'speaker': current_turn['speaker'],
            'words': current_turn['words'],
            'start_time': current_turn['start_time'],
            'end_time': current_turn['end_time'],
            'text': clean_text(turn_text)
        })
    
    # Final pass: fix any obviously incorrect timestamps
    for turn in turns:
        if turn['end_time'] > 3600:  # More than 1 hour
            # Find the next turn's start time or use a reasonable maximum
            next_turn_start = None
            for next_turn in turns:
                if next_turn['start_time'] > turn['start_time']:
                    next_turn_start = next_turn['start_time']
                    break
            
            # Set a reasonable end time
            if next_turn_start:
                turn['end_time'] = min(turn['start_time'] + 30.0, next_turn_start)
            else:
                turn['end_time'] = turn['start_time'] + 30.0
    
    return turns

def print_alignment_stats(words: List[Word], turns: List[Turn]):
    """Print detailed alignment statistics."""
    # Count words by speaker
    word_counts = defaultdict(int)
    for word in words:
        if word.speaker:
            word_counts[word.speaker] += 1
    
    # Print statistics
    print("\nAlignment Statistics:")
    print("=" * 80)
    print(f"{'Speaker':<20} {'Words':<10} {'Example Words'}")
    print("-" * 80)
    
    for speaker in sorted(word_counts.keys()):
        # Get example words for this speaker
        examples = [w.word for w in words if w.speaker == speaker][:5]
        print(f"{speaker:<20} {word_counts[speaker]:<10} {' '.join(examples)}")
    
    print("=" * 80)
    print(f"Total words aligned: {sum(word_counts.values())}")

def main():
    """Align sample segment using MFA."""
    try:
        # Setup paths
        sample_dir = Path("data/raw/sample_segment")
        audio_path = sample_dir / "sample_segment_16k.wav"
        transcript_path = sample_dir / "segment_transcript.txt"
        output_path = Path("data/processed/sample_segment/mfa_aligned.json")
        
        # Create work directory
        work_dir = Path("data/processed/sample_segment/mfa_work")
        work_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Loading transcript...")
        turns = extract_pasted_turns(transcript_path)
        
        logger.info("Preparing MFA input...")
        corpus_dir = prepare_mfa_input(turns, work_dir)
        
        logger.info("Running MFA alignment...")
        word_alignments = run_mfa_align(audio_path, corpus_dir, work_dir)
        
        logger.info("Assigning speakers...")
        words = assign_speakers_to_words(word_alignments, turns)
        
        # Group words into turns
        logger.info("Grouping words into turns...")
        words_by_speaker = defaultdict(list)
        for word in words:
            words_by_speaker[word.speaker].append({
                'word': word.word,
                'start': word.start_time,
                'end': word.end_time
            })
        aligned_turns = group_words_into_turns(words_by_speaker)
        
        # Print statistics
        print_alignment_stats(words, turns)
        
        # Save aligned transcript
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(aligned_turns, f, indent=2)
        
        logger.info(f"Saved aligned transcript to {output_path}")
        
    except Exception as e:
        logger.error(f"Error in MFA alignment: {e}")
        raise

if __name__ == "__main__":
    main() 