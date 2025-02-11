def count_words_by_speaker(file_path):
    """Count words spoken by each speaker in the pasted transcript"""
    speaker_words = {}
    current_speaker = None
    current_content = []
    
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
                
            # Check if line is a speaker name
            if line in {'Lex Fridman', 'Dylan Patel', 'Nathan Lambert'}:
                # Process previous speaker's content if exists
                if current_speaker and current_content:
                    content_text = ' '.join(current_content)
                    word_count = len(content_text.split())
                    speaker_words[current_speaker] = speaker_words.get(current_speaker, 0) + word_count
                
                current_speaker = line
                current_content = []
                continue
            
            # Skip timestamp lines
            if line.startswith('[') and line.endswith(']'):
                continue
                
            # Skip section headers (usually in all caps)
            if line.isupper() and len(line) > 3:
                continue
                
            # Add content to current speaker
            if current_speaker:
                current_content.append(line)
    
    # Process the last speaker's content
    if current_speaker and current_content:
        content_text = ' '.join(current_content)
        word_count = len(content_text.split())
        speaker_words[current_speaker] = speaker_words.get(current_speaker, 0) + word_count
    
    # Calculate total words and percentages
    total_words = sum(speaker_words.values())
    speaker_percentages = {
        speaker: (count / total_words * 100) if total_words > 0 else 0
        for speaker, count in speaker_words.items()
    }
    
    # Calculate speaking ratios relative to the speaker with least words
    min_words = min(speaker_words.values()) if speaker_words else 1
    speaking_ratios = {
        speaker: count / min_words
        for speaker, count in speaker_words.items()
    }
    
    # Print results
    print("\nWord counts by speaker:")
    print("-" * 50)
    for speaker, count in speaker_words.items():
        percentage = speaker_percentages[speaker]
        print(f"{speaker}: {count:,} words ({percentage:.1f}%)")
    
    print("\nSpeaking ratios (relative to speaker with least words):")
    print("-" * 50)
    for speaker, ratio in speaking_ratios.items():
        print(f"{speaker}: {ratio:.1f}x")

if __name__ == "__main__":
    count_words_by_speaker("data/raw/pasted_transcript.txt") 