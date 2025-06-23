import sys
from pathlib import Path
from collections import defaultdict

# Add the script's parent directory to the Python path
script_dir = Path(__file__).resolve().parent
sys.path.append(str(script_dir.parent))

from analysis.transcript_loader import TranscriptParser

def calculate_pasted_counts(loader):
    """Calculates word counts from the pasted transcript source."""
    # The get_word_counts method is specifically designed for the pasted transcript.
    word_counts = loader.get_word_counts()
    if not word_counts:
        print("Error: Could not parse pasted transcript data.")
        return None, 0
    total_words = sum(word_counts.values())
    return word_counts, total_words

def calculate_vtt_counts(loader):
    """Calculates word counts from the VTT source (which returns word segments)."""
    # The 'parse' method for VTT+pasted returns a flat list of word dictionaries
    word_list, _ = loader.parse()
    
    if not word_list:
        print("Error: Could not parse VTT transcript data.")
        return None, 0
        
    speaker_word_counts = defaultdict(int)
    
    # The data is a list of words, so we just count them
    for word_data in word_list:
        speaker = word_data.get('speaker', 'Unknown')
        speaker_word_counts[speaker] += 1
        
    total_words = len(word_list)
        
    return speaker_word_counts, total_words

def calculate_elevenlabs_counts(loader):
    """Calculates word counts from the ElevenLabs source."""
    # The 'parse' method for ElevenLabs returns a flat list of words
    word_list, _ = loader.parse()
    
    if not word_list:
        print("Error: Could not parse ElevenLabs transcript data.")
        return None, 0
        
    speaker_word_counts = defaultdict(int)
    
    for word_data in word_list:
        speaker = word_data.get('speaker', 'Unknown')
        speaker_word_counts[speaker] += 1
        
    total_words = len(word_list)
        
    return speaker_word_counts, total_words

def compare_counts():
    """
    Compares word counts from the VTT transcript and the ElevenLabs transcript.
    """
    pasted_transcript_path = Path('data/raw/pasted_transcript.txt')
    vtt_path = Path('data/raw/macwhisper_full_video.vtt')
    elevenlabs_json_path = Path('data/processed/full_podcast/elevenlabs_transcript.json')
    
    speakers = ['Dylan Patel', 'Nathan Lambert', 'Lex Fridman']

    # --- 1. Calculate from Pasted Transcript ---
    print("--- 1. Calculating Word Counts from Pasted Transcript (Lex's Website) ---")
    pasted_loader = TranscriptParser(pasted_transcript_path=pasted_transcript_path)
    pasted_counts, pasted_total = calculate_pasted_counts(pasted_loader)

    # --- 2. Calculate from VTT ---
    print("\\n--- 2. Calculating Word Counts from VTT Transcript ---")
    vtt_loader = TranscriptParser(pasted_transcript_path=pasted_transcript_path, vtt_path=vtt_path)
    vtt_counts, vtt_total = calculate_vtt_counts(vtt_loader)

    # --- 3. Calculate from ElevenLabs ---
    print("\\n--- 3. Calculating Word Counts from ElevenLabs Transcript ---")
    elevenlabs_loader = TranscriptParser(elevenlabs_json_path=elevenlabs_json_path)
    elevenlabs_counts, elevenlabs_total = calculate_elevenlabs_counts(elevenlabs_loader)
    
    # --- 4. Print Comparison ---
    print("\\n\\n--- Comparison of Speaker Word Counts ---")
    print("-" * 70)
    print(f"{'Speaker':<15} | {'Pasted Count':>12} | {'VTT Count':>12} | {'ElevenLabs Count':>18}")
    print("-" * 70)

    if pasted_counts and vtt_counts and elevenlabs_counts:
        for speaker in speakers:
            pasted_val = pasted_counts.get(speaker, 0)
            vtt_val = vtt_counts.get(speaker, 0)
            el_val = elevenlabs_counts.get(speaker, 0)
            print(f"{speaker:<15} | {pasted_val:>12,} | {vtt_val:>12,} | {el_val:>18,}")
    
    print("-" * 70)
    print(f"{'TOTAL':<15} | {pasted_total:>12,} | {vtt_total:>12,} | {elevenlabs_total:>18,}")
    print("-" * 70)
    print("\\n")


if __name__ == '__main__':
    compare_counts() 