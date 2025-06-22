#!/usr/bin/env python3
"""
Match VTT transcript sentences to pasted transcript turns using a sliding window approach.
Optimizes for maximizing turn and word coverage while maintaining temporal alignment.
"""

import json
from pathlib import Path
import logging
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Turn:
    """Represents a turn from either transcript."""
    speaker: str
    content: str
    start_time: float
    end_time: Optional[float]
    turn_index: int = -1  # Position in original sequence
    word_count: int = 0
    
    def __post_init__(self):
        self.word_count = len(self.get_words())
    
    def get_words(self) -> List[str]:
        """Get cleaned word list from content."""
        return [w.strip().lower() for w in re.findall(r'\b\w+\b', self.content)]
    
    def get_time_span(self) -> Tuple[float, float]:
        """Get start and end time, using reasonable default for missing end."""
        return (self.start_time, self.end_time if self.end_time is not None 
                else self.start_time + (self.word_count * 0.3))  # ~300ms per word

def extract_pasted_turns(file_path: Path) -> List[Turn]:
    """Extract turns from pasted transcript with proper speaker attribution."""
    turns = []
    current_speaker = None
    current_content = []
    current_start = None
    turn_index = 0
    
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
                            h, m, s = map(int, timestamp_str.split(':'))
                            end_time = h * 3600 + m * 60 + s - 1
                            break
                
                turns.append(Turn(
                    speaker=current_speaker,
                    content=' '.join(current_content),
                    start_time=current_start,
                    end_time=end_time,
                    turn_index=turn_index
                ))
                turn_index += 1
            
            current_speaker = line
            current_content = []
            current_start = None
            
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
        turns.append(Turn(
            speaker=current_speaker,
            content=' '.join(current_content),
            start_time=current_start,
            end_time=None,
            turn_index=turn_index
        ))
    
    return turns

def extract_vtt_sentences(file_path: Path) -> List[Turn]:
    """Extract sentence-level segments from VTT transcript."""
    sentences = []
    current_sentence = None
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    for line in lines:
        if line == 'WEBVTT':
            continue
            
        if '-->' in line:
            if current_sentence:
                sentences.append(current_sentence)
            
            start_time = line.split(' --> ')[0]
            h, m, s = map(float, start_time.split(':'))
            timestamp = h * 3600 + m * 60 + s
            
            current_sentence = Turn(
                speaker='',  # Will be filled in during matching
                content='',
                start_time=timestamp,
                end_time=None,
                turn_index=len(sentences)
            )
        elif current_sentence is not None:
            if current_sentence.content:
                current_sentence.content += ' '
            current_sentence.content += line
    
    # Add final sentence
    if current_sentence:
        sentences.append(current_sentence)
    
    return sentences

class TurnMatcher:
    """Matches VTT sentences to pasted transcript turns using sliding windows."""
    
    def __init__(self, pasted_turns: List[Turn], vtt_sentences: List[Turn]):
        self.pasted_turns = pasted_turns
        self.vtt_sentences = vtt_sentences
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            token_pattern=r'\b\w+\b',
            ngram_range=(1, 2)  # Use both unigrams and bigrams
        )
        
    def find_candidate_turns(self, sentence: Turn, margin: float) -> List[Turn]:
        """Find pasted turns within temporal vicinity of VTT sentence."""
        candidates = []
        sentence_span = sentence.get_time_span()
        
        for turn in self.pasted_turns:
            turn_span = turn.get_time_span()
            
            # Check for temporal overlap or proximity
            if (abs(turn_span[0] - sentence_span[0]) <= margin or
                abs(turn_span[1] - sentence_span[1]) <= margin or
                (turn_span[0] <= sentence_span[0] <= turn_span[1]) or
                (sentence_span[0] <= turn_span[0] <= sentence_span[1])):
                candidates.append(turn)
        
        return candidates
    
    def compute_similarity(self, text1: str, text2: str) -> float:
        """Compute TF-IDF weighted cosine similarity between texts."""
        try:
            tfidf_matrix = self.vectorizer.fit_transform([text1, text2])
            return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        except:
            return 0.0
    
    def find_best_window_match(self, sentence: Turn, turn: Turn) -> Tuple[float, str]:
        """Find best matching window within turn using sliding approach."""
        sentence_words = sentence.get_words()
        turn_words = turn.get_words()
        
        if not sentence_words or not turn_words:
            return 0.0, ''
        
        # Try different window sizes around sentence length
        base_size = len(sentence_words)
        min_size = max(int(base_size * 0.8), 1)
        max_size = min(int(base_size * 1.5), len(turn_words))
        
        best_score = 0.0
        best_text = ''
        
        for size in range(min_size, max_size + 1):
            for i in range(len(turn_words) - size + 1):
                window = turn_words[i:i + size]
                window_text = ' '.join(window)
                score = self.compute_similarity(' '.join(sentence_words), window_text)
                
                if score > best_score:
                    best_score = score
                    best_text = window_text
        
        return best_score, best_text
    
    def match_turns(self) -> List[Dict]:
        """Match VTT sentences to pasted turns optimizing for coverage."""
        matches = []
        used_turns = set()
        
        # Sort sentences by length (longer ones first as they're more distinctive)
        sorted_sentences = sorted(
            self.vtt_sentences,
            key=lambda s: len(s.get_words()),
            reverse=True
        )
        
        for sentence in sorted_sentences:
            # Adjust time margin based on sentence length
            time_margin = min(max(10, sentence.word_count * 0.5), 30)
            candidates = self.find_candidate_turns(sentence, time_margin)
            
            if not candidates:
                continue
            
            # Find best matching turn
            best_score = 0.0
            best_match = None
            best_text = ''
            
            for turn in candidates:
                if turn.turn_index in used_turns:
                    continue
                    
                score, matched_text = self.find_best_window_match(sentence, turn)
                if score > best_score:
                    best_score = score
                    best_match = turn
                    best_text = matched_text
            
            # Accept match if similarity is high enough
            if best_match and best_score > 0.3:  # Adjusted threshold
                matches.append({
                    'vtt_sentence': sentence,
                    'pasted_turn': best_match,
                    'matched_text': best_text,
                    'similarity_score': best_score
                })
                used_turns.add(best_match.turn_index)
        
        return matches

def print_statistics(pasted_turns: List[Turn], matches: List[Dict]):
    """Print detailed matching statistics."""
    # Count original turns and words by speaker
    pasted_stats = defaultdict(lambda: {'turns': 0, 'words': 0})
    for turn in pasted_turns:
        pasted_stats[turn.speaker]['turns'] += 1
        pasted_stats[turn.speaker]['words'] += turn.word_count
    
    # Count matched turns and words
    matched_stats = defaultdict(lambda: {'turns': 0, 'words': 0})
    for match in matches:
        speaker = match['pasted_turn'].speaker
        matched_stats[speaker]['turns'] += 1
        matched_stats[speaker]['words'] += len(match['matched_text'].split())
    
    # Print statistics
    print("\nPasted Transcript Statistics:")
    print("-" * 80)
    print(f"{'Speaker':<15} {'Turns':>8} {'Words':>12}")
    print("-" * 80)
    
    total_pasted = {'turns': 0, 'words': 0}
    for speaker in sorted(pasted_stats.keys()):
        stats = pasted_stats[speaker]
        print(f"{speaker:<15} {stats['turns']:>8,d} {stats['words']:>12,d}")
        total_pasted['turns'] += stats['turns']
        total_pasted['words'] += stats['words']
    
    print("-" * 80)
    print(f"{'TOTAL':<15} {total_pasted['turns']:>8,d} {total_pasted['words']:>12,d}")
    print("-" * 80)
    
    print("\nMatched Transcript Statistics:")
    print("-" * 80)
    print(f"{'Speaker':<15} {'Turns':>8} {'Words':>12} {'Turn %':>8} {'Word %':>8}")
    print("-" * 80)
    
    total_matched = {'turns': 0, 'words': 0}
    for speaker in sorted(pasted_stats.keys()):
        pasted = pasted_stats[speaker]
        matched = matched_stats[speaker]
        turn_pct = (matched['turns'] / pasted['turns'] * 100) if pasted['turns'] > 0 else 0
        word_pct = (matched['words'] / pasted['words'] * 100) if pasted['words'] > 0 else 0
        
        print(f"{speaker:<15} {matched['turns']:>8,d} {matched['words']:>12,d} "
              f"{turn_pct:>7.1f}% {word_pct:>7.1f}%")
        
        total_matched['turns'] += matched['turns']
        total_matched['words'] += matched['words']
    
    print("-" * 80)
    turn_pct = (total_matched['turns'] / total_pasted['turns'] * 100) if total_pasted['turns'] > 0 else 0
    word_pct = (total_matched['words'] / total_pasted['words'] * 100) if total_pasted['words'] > 0 else 0
    print(f"{'TOTAL':<15} {total_matched['turns']:>8,d} {total_matched['words']:>12,d} "
          f"{turn_pct:>7.1f}% {word_pct:>7.1f}%")
    print("-" * 80)

def print_sample_matches(matches: List[Dict]):
    """Print sample matches from beginning, middle, and end."""
    # Group matches by speaker
    speaker_matches = defaultdict(list)
    for match in matches:
        speaker = match['pasted_turn'].speaker
        speaker_matches[speaker].append(match)
    
    # Sort matches by timestamp within each speaker
    for speaker in speaker_matches:
        speaker_matches[speaker].sort(key=lambda m: m['vtt_sentence'].start_time)
    
    # Print samples for each speaker
    for speaker in sorted(speaker_matches.keys()):
        matches = speaker_matches[speaker]
        if not matches:
            continue
        
        print(f"\n{speaker} Matches")
        print("=" * 120)
        
        # Print beginning, middle, and end matches
        for i, position in [(0, "BEGINNING"), 
                          (len(matches)//2, "MIDDLE"), 
                          (-1, "END")]:
            match = matches[i]
            sentence = match['vtt_sentence']
            turn = match['pasted_turn']
            
            print(f"\n{position} MATCH (Time: {sentence.start_time:.2f}, "
                  f"Score: {match['similarity_score']:.3f}):")
            print("-" * 120)
            print("VTT Sentence:")
            print(sentence.content[:500] + "..." if len(sentence.content) > 500 
                  else sentence.content)
            print("\nMatched Turn Excerpt:")
            print(match['matched_text'])
            print("\nFull Pasted Turn:")
            print(turn.content[:500] + "..." if len(turn.content) > 500 
                  else turn.content)
            print("-" * 120)

def main():
    """Match VTT sentences to pasted transcript turns."""
    try:
        # Load transcripts
        pasted_path = Path('data/raw/pasted_transcript.txt')
        vtt_path = Path('data/raw/full_video_sentences.vtt')
        
        logger.info("Loading transcripts...")
        pasted_turns = extract_pasted_turns(pasted_path)
        vtt_sentences = extract_vtt_sentences(vtt_path)
        
        logger.info(f"Found {len(pasted_turns)} turns in pasted transcript")
        logger.info(f"Found {len(vtt_sentences)} sentences in VTT transcript")
        
        # Match turns
        matcher = TurnMatcher(pasted_turns, vtt_sentences)
        matches = matcher.match_turns()
        
        logger.info(f"Successfully matched {len(matches)} segments")
        
        # Print statistics and samples
        print_statistics(pasted_turns, matches)
        print_sample_matches(matches)
        
    except Exception as e:
        logger.error(f"Error matching transcripts: {e}")
        raise

if __name__ == '__main__':
    main() 