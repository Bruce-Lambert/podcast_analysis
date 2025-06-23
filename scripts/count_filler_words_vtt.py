import re
from pathlib import Path
from collections import Counter

def count_specific_words_in_vtt(vtt_path: Path, words_to_count: list):
    """
    Parses a VTT file and counts occurrences of specific words, ignoring metadata.
    This function is intentionally simple for transparency.
    """
    if not vtt_path.exists():
        print(f"Error: VTT file not found at {vtt_path}")
        return None

    word_counts = Counter()
    # Normalize the target words to lowercase for case-insensitive comparison
    words_to_count_lower = [word.lower() for word in words_to_count]

    with open(vtt_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            
            # Skip VTT metadata
            if not line or line == 'WEBVTT' or '-->' in line or line.startswith('NOTE'):
                continue
            
            # This is a content line, process the words
            words_in_line = line.split()
            for word in words_in_line:
                # Normalize the word from the file
                normalized_word = word.lower().strip(".,?!")
                if normalized_word in words_to_count_lower:
                    word_counts[normalized_word] += 1
            
    return word_counts

def main():
    """
    Main function to execute the word count on the specified VTT file.
    """
    vtt_file_path = Path('data/raw/macwhisper_full_video.vtt')
    target_words = ['uh', 'like']
    
    print(f"--- Counting specific words ('uh', 'like') in: {vtt_file_path} ---")
    
    counts = count_specific_words_in_vtt(vtt_file_path, target_words)
    
    print("\\n" + "="*40)
    if counts is not None:
        for word in target_words:
            count = counts[word.lower()]
            print(f"Total count for '{word}': {count:,}")
    print("="*40 + "\\n")

if __name__ == "__main__":
    main() 