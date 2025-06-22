#!/usr/bin/env python3
"""
Verify transcript combination by comparing word counts and discourse marker usage
between the pasted transcript and the combined transcript.
"""

import json
from pathlib import Path
import re
from collections import defaultdict
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def count_words_and_markers(text):
    """Count total words and instances of 'right' in text."""
    words = len(text.split())
    # Count standalone "right" (not part of other words)
    rights = len(re.findall(r'\bright\b', text.lower()))
    return words, rights

def analyze_pasted_transcript(file_path):
    """Analyze the pasted transcript for word counts and markers."""
    stats = defaultdict(lambda: {'words': 0, 'rights': 0})
    current_speaker = None
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    for line in lines:
        # Check for speaker name
        if line in {'Dylan Patel', 'Nathan Lambert', 'Lex Fridman'}:
            current_speaker = line
            continue
        
        # Skip timestamp-only lines
        if line.startswith('(') and line.endswith(')'):
            continue
            
        # Process content
        if current_speaker and (not line.startswith('(') or len(line) > 12):
            # If line starts with timestamp, remove it
            content = re.sub(r'^\(\d{2}:\d{2}:\d{2}\)\s*', '', line)
            words, rights = count_words_and_markers(content)
            stats[current_speaker]['words'] += words
            stats[current_speaker]['rights'] += rights
    
    return stats

def analyze_combined_transcript(file_path):
    """Analyze the combined transcript for word counts and markers."""
    stats = defaultdict(lambda: {'words': 0, 'rights': 0})
    
    with open(file_path, 'r', encoding='utf-8') as f:
        segments = json.load(f)
    
    for segment in segments:
        speaker = segment.get('speaker')
        if speaker and 'content' in segment:
            words, rights = count_words_and_markers(segment['content'])
            stats[speaker]['words'] += words
            stats[speaker]['rights'] += rights
    
    return stats

def main():
    """Compare word counts and discourse marker usage between transcripts."""
    try:
        # Analyze pasted transcript
        pasted_path = Path('data/raw/pasted_transcript.txt')
        logger.info("Analyzing pasted transcript...")
        pasted_stats = analyze_pasted_transcript(pasted_path)
        
        # Analyze combined transcript
        combined_path = Path('data/processed/combined_transcript.json')
        logger.info("Analyzing combined transcript...")
        combined_stats = analyze_combined_transcript(combined_path)
        
        # Print comparison
        print("\nTranscript Comparison:")
        print("=" * 80)
        print(f"{'Speaker':<15} {'Pasted Words':>12} {'Combined Words':>15} {'Diff %':>8} {'Pasted Rights':>12} {'Combined Rights':>15} {'Diff %':>8}")
        print("-" * 80)
        
        for speaker in sorted(set(pasted_stats.keys()) | set(combined_stats.keys())):
            pasted = pasted_stats.get(speaker, {'words': 0, 'rights': 0})
            combined = combined_stats.get(speaker, {'words': 0, 'rights': 0})
            
            # Calculate differences
            word_diff_pct = ((combined['words'] - pasted['words']) / pasted['words'] * 100 
                           if pasted['words'] > 0 else float('inf'))
            right_diff_pct = ((combined['rights'] - pasted['rights']) / pasted['rights'] * 100 
                            if pasted['rights'] > 0 else float('inf'))
            
            print(f"{speaker:<15} {pasted['words']:>12,d} {combined['words']:>15,d} {word_diff_pct:>7.1f}% "
                  f"{pasted['rights']:>12,d} {combined['rights']:>15,d} {right_diff_pct:>7.1f}%")
        
        # Print totals
        print("-" * 80)
        total_pasted_words = sum(s['words'] for s in pasted_stats.values())
        total_combined_words = sum(s['words'] for s in combined_stats.values())
        total_pasted_rights = sum(s['rights'] for s in pasted_stats.values())
        total_combined_rights = sum(s['rights'] for s in combined_stats.values())
        
        total_word_diff_pct = ((total_combined_words - total_pasted_words) / total_pasted_words * 100 
                              if total_pasted_words > 0 else float('inf'))
        total_right_diff_pct = ((total_combined_rights - total_pasted_rights) / total_pasted_rights * 100 
                               if total_pasted_rights > 0 else float('inf'))
        
        print(f"{'TOTAL':<15} {total_pasted_words:>12,d} {total_combined_words:>15,d} {total_word_diff_pct:>7.1f}% "
              f"{total_pasted_rights:>12,d} {total_combined_rights:>15,d} {total_right_diff_pct:>7.1f}%")
        
    except Exception as e:
        logger.error(f"Error comparing transcripts: {e}")
        raise

if __name__ == '__main__':
    main() 