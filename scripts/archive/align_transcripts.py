#!/usr/bin/env python3
"""
Align VTT transcript with pasted transcript using simple text similarity matching.
"""

import json
from pathlib import Path
import logging
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import re
from collections import defaultdict
import webvtt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from difflib import SequenceMatcher

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

def extract_vtt_sentences(file_path: Path) -> List[Turn]:
    """Extract segments from VTT transcript."""
    sentences = []
    
    for caption in webvtt.read(str(file_path)):
        start_time = parse_timestamp(caption.start)
        end_time = parse_timestamp(caption.end)
        content = caption.text.strip()
        
        # Skip empty captions
        if not content:
            continue
        
        sentences.append(Turn(
            speaker='',  # Will be filled during matching
            content=content,
            start_time=start_time,
            end_time=end_time
        ))
    
    return sentences

def clean_text(text: str) -> str:
    """Clean text for comparison."""
    # Remove punctuation and convert to lowercase
    text = re.sub(r'[^\w\s]', '', text.lower())
    # Normalize whitespace
    return ' '.join(text.split())

def get_similarity_metrics(text1: str, text2: str) -> Dict[str, float]:
    """Calculate multiple similarity metrics between two texts."""
    # Clean texts
    text1 = clean_text(text1)
    text2 = clean_text(text2)
    
    metrics = {}
    
    # 1. Sequence Matcher (similar to edit distance)
    metrics['sequence'] = SequenceMatcher(None, text1, text2).ratio()
    
    # 2. Character trigram similarity
    def get_trigrams(s):
        return set(s[i:i+3] for i in range(len(s)-2))
    
    if len(text1) >= 3 and len(text2) >= 3:
        trigrams1 = get_trigrams(text1)
        trigrams2 = get_trigrams(text2)
        if trigrams1 and trigrams2:
            metrics['trigram'] = len(trigrams1 & trigrams2) / len(trigrams1 | trigrams2)
        else:
            metrics['trigram'] = 0.0
    else:
        metrics['trigram'] = 0.0
    
    # 3. TF-IDF Cosine Similarity
    try:
        vectorizer = TfidfVectorizer(lowercase=True, token_pattern=r'\b\w+\b')
        tfidf_matrix = vectorizer.fit_transform([text1, text2])
        metrics['tfidf'] = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    except:
        metrics['tfidf'] = 0.0
    
    # Combined score (weighted average)
    metrics['combined'] = (
        metrics['sequence'] * 0.4 +
        metrics['trigram'] * 0.3 +
        metrics['tfidf'] * 0.3
    )
    
    return metrics

def find_best_match(sentence: Turn, pasted_turns: List[Turn], time_window: int = 30) -> Tuple[Optional[Turn], float]:
    """Find the best matching turn within the time window."""
    candidates = []
    
    # Get temporally close turns
    for turn in pasted_turns:
        time_diff = abs(turn.start_time - sentence.start_time)
        if time_diff <= time_window:
            candidates.append(turn)
    
    if not candidates:
        return None, 0.0
    
    # Find best matching turn
    best_score = 0.0
    best_match = None
    
    for turn in candidates:
        metrics = get_similarity_metrics(sentence.content, turn.content)
        score = metrics['combined']
        if score > best_score:
            best_score = score
            best_match = turn
    
    return best_match, best_score

def align_transcripts(vtt_sentences: List[Turn], pasted_turns: List[Turn]) -> List[Dict]:
    """Align VTT sentences with pasted turns."""
    matches = []
    
    for sentence in vtt_sentences:
        best_match, score = find_best_match(sentence, pasted_turns)
        
        # Accept match if similarity is high enough
        if best_match and score > 0.5:  # Adjust threshold if needed
            matches.append({
                'speaker': best_match.speaker,
                'content': sentence.content,
                'start_time': sentence.start_time,
                'end_time': sentence.end_time,
                'similarity_score': score,
                'pasted_text': best_match.content  # For verification
            })
    
    return matches

def print_alignment_stats(pasted_turns: List[Turn], aligned_matches: List[Dict]):
    """Print detailed alignment statistics."""
    # Count original turns and words by speaker
    pasted_stats = defaultdict(lambda: {'turns': 0, 'words': 0})
    for turn in pasted_turns:
        pasted_stats[turn.speaker]['turns'] += 1
        pasted_stats[turn.speaker]['words'] += len(turn.content.split())
    
    # Count aligned turns and words
    aligned_stats = defaultdict(lambda: {'turns': 0, 'words': 0})
    for match in aligned_matches:
        speaker = match['speaker']
        aligned_stats[speaker]['turns'] += 1
        aligned_stats[speaker]['words'] += len(match['content'].split())
    
    # Print statistics
    print("\nAlignment Statistics:")
    print("=" * 120)
    print(f"{'Speaker':<15} {'Pasted':^24} {'Aligned':^24} {'Coverage':^35}")
    print(f"{'':15} {'Turns':>11} {'Words':>12} {'Turns':>11} {'Words':>12} {'Turns %':>11} {'Words %':>11} {'Avg Sim':>11}")
    print("-" * 120)
    
    total_pasted = {'turns': 0, 'words': 0}
    total_aligned = {'turns': 0, 'words': 0}
    total_similarity = 0
    match_count = 0
    
    for speaker in sorted(pasted_stats.keys()):
        pasted = pasted_stats[speaker]
        aligned = aligned_stats[speaker]
        
        # Calculate coverage
        turn_pct = (aligned['turns'] / pasted['turns'] * 100) if pasted['turns'] > 0 else 0
        word_pct = (aligned['words'] / pasted['words'] * 100) if pasted['words'] > 0 else 0
        
        # Calculate average similarity for this speaker
        speaker_matches = [m for m in aligned_matches if m['speaker'] == speaker]
        avg_sim = sum(m['similarity_score'] for m in speaker_matches) / len(speaker_matches) if speaker_matches else 0
        
        print(f"{speaker:<15} {pasted['turns']:>11,d} {pasted['words']:>12,d} "
              f"{aligned['turns']:>11,d} {aligned['words']:>12,d} "
              f"{turn_pct:>10.1f}% {word_pct:>10.1f}% {avg_sim:>10.3f}")
        
        # Print example matches
        if speaker_matches:
            # Print best match
            best_match = max(speaker_matches, key=lambda m: m['similarity_score'])
            print(f"\nBest match (score: {best_match['similarity_score']:.3f}):")
            print(f"VTT   : {best_match['content']}")
            print(f"Pasted: {best_match['pasted_text']}")
            
            # Print worst match
            worst_match = min(speaker_matches, key=lambda m: m['similarity_score'])
            print(f"\nWorst match (score: {worst_match['similarity_score']:.3f}):")
            print(f"VTT   : {worst_match['content']}")
            print(f"Pasted: {worst_match['pasted_text']}")
        
        total_pasted['turns'] += pasted['turns']
        total_pasted['words'] += pasted['words']
        total_aligned['turns'] += aligned['turns']
        total_aligned['words'] += aligned['words']
        total_similarity += sum(m['similarity_score'] for m in speaker_matches)
        match_count += len(speaker_matches)
    
    print("-" * 120)
    total_turn_pct = (total_aligned['turns'] / total_pasted['turns'] * 100) if total_pasted['turns'] > 0 else 0
    total_word_pct = (total_aligned['words'] / total_pasted['words'] * 100) if total_pasted['words'] > 0 else 0
    avg_similarity = total_similarity / match_count if match_count > 0 else 0
    
    print(f"{'TOTAL':<15} {total_pasted['turns']:>11,d} {total_pasted['words']:>12,d} "
          f"{total_aligned['turns']:>11,d} {total_aligned['words']:>12,d} "
          f"{total_turn_pct:>10.1f}% {total_word_pct:>10.1f}% {avg_similarity:>10.3f}")
    print("=" * 120)

def count_discourse_markers(text: str, marker: str = "right") -> int:
    """Count occurrences of a discourse marker in text."""
    # Convert to lowercase and split into words
    words = text.lower().split()
    # Count standalone "right" and "right?" (including punctuation)
    count = 0
    for i, word in enumerate(words):
        # Remove punctuation from the end of the word
        clean_word = word.rstrip('.,!?')
        if clean_word == marker:
            # Check if it's standalone (not part of another word)
            if (i == 0 or not words[i-1].endswith(marker)) and \
               (i == len(words)-1 or not words[i+1].startswith(marker)):
                count += 1
    return count

def analyze_discourse_markers(vtt_path: Path, pasted_path: Path):
    """Analyze discourse markers in both transcripts and calculate correction factors."""
    # Read VTT transcript
    vtt_text = ""
    for caption in webvtt.read(str(vtt_path)):
        # Remove speaker indicators and clean up text
        text = re.sub(r'^>>\s*', '', caption.text)
        vtt_text += " " + text
    
    # Read pasted transcript
    with open(pasted_path, 'r', encoding='utf-8') as f:
        pasted_lines = [line.strip() for line in f if line.strip()]
    
    # Process pasted transcript to remove timestamps and speaker labels
    pasted_text = ""
    for line in pasted_lines:
        # Skip speaker names and timestamp lines
        if line in {'Dylan Patel', 'Nathan Lambert', 'Lex Fridman'} or \
           line.startswith('(') and line.find(')') > 0:
            continue
        pasted_text += " " + line
    
    # Count markers in each transcript
    vtt_rights = count_discourse_markers(vtt_text)
    pasted_rights = count_discourse_markers(pasted_text)
    
    # Calculate correction factor
    correction_factor = vtt_rights / pasted_rights if pasted_rights > 0 else 1.0
    
    # Count by speaker in pasted transcript
    pasted_turns = extract_pasted_turns(pasted_path)
    speaker_counts = defaultdict(int)
    speaker_words = defaultdict(int)
    
    for turn in pasted_turns:
        rights = count_discourse_markers(turn.content)
        # Count actual words (excluding punctuation and numbers)
        words = len([w for w in turn.content.split() if any(c.isalpha() for c in w)])
        speaker_counts[turn.speaker] += rights
        speaker_words[turn.speaker] += words
    
    # Print analysis
    print("\nDiscourse Marker Analysis:")
    print("=" * 80)
    print(f"Total 'right' instances in VTT: {vtt_rights}")
    print(f"Total 'right' instances in pasted: {pasted_rights}")
    print(f"Correction factor: {correction_factor:.2f}")
    print("\nBy speaker (from pasted transcript):")
    print("-" * 80)
    print(f"{'Speaker':<20} {'Raw Count':<12} {'Estimated':<12} {'Per 1000 words':<15}")
    
    for speaker in sorted(speaker_counts.keys()):
        raw_count = speaker_counts[speaker]
        estimated = raw_count * correction_factor
        per_thousand = (estimated / speaker_words[speaker]) * 1000 if speaker_words[speaker] > 0 else 0
        print(f"{speaker:<20} {raw_count:<12d} {estimated:< 12.1f} {per_thousand:< 15.1f}")
    
    print("=" * 80)
    
    # Add some context analysis
    print("\nSample contexts (from pasted transcript):")
    print("-" * 80)
    
    # Find some example contexts
    contexts = []
    for turn in pasted_turns:
        words = turn.content.split()
        for i, word in enumerate(words):
            if word.lower().rstrip('.,!?') == 'right':
                # Get context (up to 5 words before and after)
                start = max(0, i - 5)
                end = min(len(words), i + 6)
                context = ' '.join(words[start:end])
                contexts.append((turn.speaker, context))
    
    # Print up to 5 examples per speaker
    speaker_examples = defaultdict(list)
    for speaker, context in contexts:
        if len(speaker_examples[speaker]) < 5:
            speaker_examples[speaker].append(context)
    
    for speaker in sorted(speaker_examples.keys()):
        print(f"\n{speaker}:")
        for context in speaker_examples[speaker]:
            print(f"  - {context}")
    
    return {
        'correction_factor': correction_factor,
        'speaker_counts': dict(speaker_counts),
        'speaker_words': dict(speaker_words),
        'vtt_total': vtt_rights,
        'pasted_total': pasted_rights,
        'contexts': {speaker: contexts for speaker, contexts in speaker_examples.items()}
    }

def simple_word_count(text: str, word: str = "right") -> int:
    """Simply count occurrences of a word in text."""
    return text.lower().count(word.lower())

def analyze_transcripts(vtt_path: Path, pasted_path: Path):
    """Simple analysis of 'right' usage in both transcripts."""
    # Count in VTT transcript
    vtt_text = ""
    for caption in webvtt.read(str(vtt_path)):
        vtt_text += " " + caption.text
    vtt_count = simple_word_count(vtt_text)
    
    # Count in pasted transcript
    with open(pasted_path, 'r', encoding='utf-8') as f:
        pasted_text = f.read()
    pasted_count = simple_word_count(pasted_text)
    
    # Calculate inflation factor
    inflation_factor = vtt_count / pasted_count if pasted_count > 0 else 1.0
    
    # Analyze by speaker
    turns = extract_pasted_turns(pasted_path)
    speaker_stats = defaultdict(lambda: {"count": 0, "words": 0})
    
    for turn in turns:
        rights = simple_word_count(turn.content)
        words = len(turn.content.split())
        speaker_stats[turn.speaker]["count"] += rights
        speaker_stats[turn.speaker]["words"] += words
    
    # Print results
    print("\nSimple 'right' count analysis:")
    print("=" * 80)
    print(f"VTT transcript total: {vtt_count}")
    print(f"Pasted transcript total: {pasted_count}")
    print(f"Inflation factor: {inflation_factor:.2f}")
    
    print("\nBy speaker (from pasted transcript):")
    print("-" * 80)
    print(f"{'Speaker':<20} {'Raw Count':<12} {'Inflated':<12} {'Per 1K words':<15} {'Infl. per 1K':<15}")
    
    for speaker, stats in sorted(speaker_stats.items()):
        raw_count = stats["count"]
        inflated = raw_count * inflation_factor
        per_1k = (raw_count / stats["words"]) * 1000 if stats["words"] > 0 else 0
        infl_per_1k = per_1k * inflation_factor
        print(f"{speaker:<20} {raw_count:<12d} {inflated:< 12.1f} {per_1k:< 15.1f} {infl_per_1k:< 15.1f}")
    
    print("=" * 80)
    
    return {
        "vtt_count": vtt_count,
        "pasted_count": pasted_count,
        "inflation_factor": inflation_factor,
        "speaker_stats": {
            speaker: {
                "raw_count": stats["count"],
                "inflated_count": stats["count"] * inflation_factor,
                "words": stats["words"],
                "per_1k": (stats["count"] / stats["words"]) * 1000 if stats["words"] > 0 else 0,
                "inflated_per_1k": ((stats["count"] / stats["words"]) * 1000 * inflation_factor) if stats["words"] > 0 else 0
            }
            for speaker, stats in speaker_stats.items()
        }
    }

def main():
    """Simple analysis of discourse markers."""
    try:
        pasted_path = Path('data/raw/pasted_transcript.txt')
        vtt_path = Path('data/raw/macwhisper_full_video.vtt')
        output_path = Path('data/processed/full_podcast/right_analysis.json')
        
        logger.info("Analyzing 'right' usage...")
        results = analyze_transcripts(vtt_path, pasted_path)
        
        # Save results
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Saved analysis results to {output_path}")
        
    except Exception as e:
        logger.error(f"Error analyzing transcripts: {e}")
        raise

if __name__ == '__main__':
    main() 