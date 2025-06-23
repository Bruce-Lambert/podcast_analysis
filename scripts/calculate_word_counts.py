import sys
from pathlib import Path
from collections import defaultdict

# Add the script's parent directory to the Python path
# to allow for relative imports
script_dir = Path(__file__).resolve().parent
sys.path.append(str(script_dir.parent))

from analysis.transcript_loader import TranscriptParser

def calculate_and_print_word_counts_correctly():
    """
    Loads the ElevenLabs transcript, which is a list of words, and correctly
    counts the number of words for each speaker.
    """
    print("--- Calculating word counts from ElevenLabs transcript... ---")
    
    elevenlabs_json_path = Path('data/processed/full_podcast/elevenlabs_transcript.json')
    
    # This will correctly call the _parse_elevenlabs_json method
    loader = TranscriptParser(elevenlabs_json_path=elevenlabs_json_path)
    # The 'parsed_data' from this method is a list of word dictionaries
    word_list, _ = loader.parse()
    
    if not word_list:
        print("Error: Could not parse transcript data.")
        return
        
    speaker_word_counts = defaultdict(int)
    
    # Each item in word_list is a dictionary for a single word
    for word_data in word_list:
        speaker = word_data.get('speaker', 'Unknown')
        speaker_word_counts[speaker] += 1 # Just count the word entry
        
    total_words = len(word_list)
        
    print("\\n--- Speaker Word Counts (from ElevenLabs) ---")
    # Print in a consistent order
    speakers = ['Dylan Patel', 'Nathan Lambert', 'Lex Fridman']
    for speaker in speakers:
        count = speaker_word_counts.get(speaker, 0)
        print(f"{speaker}: {count} words")
        
    print("---------------------------")
    print(f"Total Words in Transcript: {total_words}")
    print("---------------------------\\n")

if __name__ == '__main__':
    calculate_and_print_word_counts_correctly() 