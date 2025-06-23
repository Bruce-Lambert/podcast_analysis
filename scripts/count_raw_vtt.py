import re
from pathlib import Path

def count_words_in_vtt(vtt_path: Path):
    """
    Parses a VTT file and returns the total word count, ignoring all metadata.
    This function is intentionally simple to be transparent and avoid complex logic.
    """
    if not vtt_path.exists():
        print(f"Error: VTT file not found at {vtt_path}")
        return 0

    total_words = 0
    with open(vtt_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            
            # Skip VTT metadata: empty lines, header, comments, timestamps.
            if not line or line == 'WEBVTT' or '-->' in line or line.startswith('NOTE'):
                continue
            
            # This is a content line, count the words.
            words = line.split()
            total_words += len(words)
            
    return total_words

def main():
    """
    Main function to execute the word count on the specified VTT file.
    """
    vtt_file_path = Path('data/raw/full_video_sentences.vtt')
    print(f"--- Counting total words in raw VTT file: {vtt_file_path} ---")
    
    word_count = count_words_in_vtt(vtt_file_path)
    
    print("\\n" + "="*40)
    print(f"Total Word Count: {word_count:,}")
    print("="*40 + "\\n")

if __name__ == "__main__":
    main() 