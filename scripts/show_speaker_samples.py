import json
from collections import defaultdict

def show_speaker_samples():
    """
    Loads the Eleven Labs transcript and prints the first few words spoken by each speaker
    to help with speaker identification, using the 'words' data.
    """
    transcript_path = 'data/processed/full_podcast/elevenlabs_transcript.json'
    print(f"--- Loading transcript from: {transcript_path} ---")

    try:
        with open(transcript_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Transcript file not found at {transcript_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON from {transcript_path}")
        return

    words = data.get('words')

    if not words:
        print("The JSON file does not contain a 'words' key or it is empty.")
        return

    samples_by_speaker = defaultdict(list)
    word_limit = 25  # Get the first 25 words for each speaker

    for word_info in words:
        speaker = word_info.get('speaker_id')
        word = word_info.get('text')

        if speaker is not None and word and len(samples_by_speaker[speaker]) < word_limit:
            samples_by_speaker[speaker].append(word)

    print("\\n--- Sample Text by Speaker ID ---")
    if not samples_by_speaker:
        print("Could not extract any samples.")
        return

    sorted_speaker_ids = sorted(samples_by_speaker.keys())

    for speaker_id in sorted_speaker_ids:
        print(f"\\n--- Speaker {speaker_id} ---")
        sample_text = " ".join(samples_by_speaker[speaker_id])
        print(f"  Sample: \"{sample_text}...\"")

if __name__ == '__main__':
    show_speaker_samples() 