import json
from pathlib import Path
from collections import Counter

def deep_analyze_json(json_path: Path):
    """
    Performs a deep analysis of the ElevenLabs JSON transcript to understand
    its composition and verify the word count.
    """
    print(f"--- Deep Analysis of: {json_path} ---")
    if not json_path.exists():
        print(f"Error: JSON file not found at {json_path}")
        return

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    word_list = [item for item in data.get('words', []) if item.get('type') == 'word']
    word_count = len(word_list)
    print(f"\\n1. Basic Count Verification (type=='word'): {word_count:,} words")
    print("-" * 40)

    # --- Analysis 2: Word Frequency ---
    print("\\n2. Most Common Words Analysis")
    all_words_text = [item.get('text', '').lower().strip(".,?!") for item in word_list]
    word_freq = Counter(all_words_text)
    print("   Top 20 most frequent words:")
    for i, (word, count) in enumerate(word_freq.most_common(20)):
        print(f"   {i+1: >2}. {word:<10} ({count:,} times)")
    print("-" * 40)

    # --- Analysis 3: Timestamp Overlaps ---
    print("\\n3. Timestamp Overlap Analysis")
    overlap_count = 0
    for i in range(len(word_list) - 1):
        current_word = word_list[i]
        next_word = word_list[i+1]
        if current_word['end'] > next_word['start']:
            overlap_count += 1
    print(f"   Found {overlap_count:,} instances of overlapping timestamps.")
    print("-" * 40)

    # --- Analysis 4: Consecutive Duplicates ---
    print("\\n4. Consecutive Duplicate Word Analysis (Stutter Check)")
    duplicate_count = 0
    for i in range(len(word_list) - 1):
        current_word_text = word_list[i].get('text', '').lower()
        next_word_text = word_list[i+1].get('text', '').lower()
        if current_word_text == next_word_text:
            duplicate_count += 1
    print(f"   Found {duplicate_count:,} instances of consecutive identical words.")
    print("-" * 40)

def main():
    """Main function to run the deep analysis."""
    json_file_path = Path('data/processed/full_podcast/elevenlabs_transcript.json')
    deep_analyze_json(json_file_path)

if __name__ == "__main__":
    main() 