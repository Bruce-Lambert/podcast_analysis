import re
import json
from pathlib import Path
from collections import Counter

def get_vtt_word_counts(vtt_path: Path) -> Counter:
    """Parses a VTT file and returns a Counter object with word frequencies."""
    word_counts = Counter()
    with open(vtt_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line == 'WEBVTT' or '-->' in line or line.startswith('NOTE'):
                continue
            words_in_line = line.split()
            for word in words_in_line:
                normalized_word = word.lower().strip(".,?!();:")
                if normalized_word:
                    word_counts[normalized_word] += 1
    return word_counts

def get_elevenlabs_word_counts(json_path: Path) -> Counter:
    """Parses an ElevenLabs JSON and returns a Counter object with word frequencies."""
    word_counts = Counter()
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    word_list = data.get('words', [])
    for item in word_list:
        if item.get('type') == 'word':
            word = item.get('text', '')
            normalized_word = word.lower().strip(".,?!();:")
            if normalized_word:
                word_counts[normalized_word] += 1
    return word_counts

def main():
    """
    Generates and displays a side-by-side comparison of word frequencies
    between the Whisper VTT and ElevenLabs JSON transcripts.
    """
    vtt_file_path = Path('data/raw/macwhisper_full_video.vtt')
    json_file_path = Path('data/processed/full_podcast/elevenlabs_transcript.json')

    print("--- Generating side-by-side word frequency comparison ---")
    vtt_counts = get_vtt_word_counts(vtt_file_path)
    elevenlabs_counts = get_elevenlabs_word_counts(json_file_path)

    all_words = set(vtt_counts.keys()) | set(elevenlabs_counts.keys())
    
    comparison_data = []
    for word in all_words:
        vtt_count = vtt_counts.get(word, 0)
        el_count = elevenlabs_counts.get(word, 0)
        difference = el_count - vtt_count
        comparison_data.append({
            'word': word,
            'vtt': vtt_count,
            'el': el_count,
            'diff': difference
        })

    # Sort by the words that are most over-represented in the ElevenLabs transcript
    sorted_data = sorted(comparison_data, key=lambda x: x['diff'], reverse=True)

    print("\\n" + "="*60)
    print(" Top 30 Words More Frequent in ElevenLabs vs. Whisper VTT")
    print("="*60)
    print(f"{'Word':<15} | {'ElevenLabs #':>12} | {'Whisper VTT #':>14} | {'Difference':>12}")
    print("-" * 60)

    for item in sorted_data[:30]:
        print(f"{item['word']:<15} | {item['el']:>12,} | {item['vtt']:>14,} | {item['diff']:>+12,}")

    print("="*60)
    print("\\nNote: A positive difference means the word appeared more in ElevenLabs.")
    print("\\n")

if __name__ == "__main__":
    main() 