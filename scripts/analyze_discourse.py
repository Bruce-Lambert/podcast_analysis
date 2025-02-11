import re
from collections import defaultdict
import matplotlib.pyplot as plt
import json
import numpy as np
from pathlib import Path
import seaborn as sns
from datetime import datetime, timedelta

class DiscourseAnalyzer:
    def __init__(self, raw_dir="data/raw", processed_dir="data/processed"):
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        
    def load_whisper_results(self):
        """Load the Whisper analysis results."""
        results_path = self.processed_dir / "right_analysis.json"
        with open(results_path, 'r') as f:
            return json.load(f)
            
    def load_transcript(self):
        """Load the transcript from the raw directory (from original version)."""
        transcript_path = self.raw_dir / "pasted_transcript.txt"
        with open(transcript_path, 'r', encoding='utf-8') as f:
            return f.read()
            
    def parse_segments(self, text):
        """Parse the transcript into segments with speaker and text (from original version)."""
        segments = []
        current_speaker = None
        current_text = []
        
        for line in text.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            # Check if line starts with timestamp
            if line.startswith('('):
                if current_speaker and current_text:
                    segments.append({
                        'speaker': current_speaker,
                        'text': ' '.join(current_text)
                    })
                current_text = [line]
            elif not line.startswith('('):
                if any(char.isdigit() for char in line):
                    continue
                if ':' in line:
                    if current_speaker and current_text:
                        segments.append({
                            'speaker': current_speaker,
                            'text': ' '.join(current_text)
                        })
                    current_speaker = line.split(':')[0].strip()
                    current_text = []
                else:
                    current_text.append(line)
                    
        if current_speaker and current_text:
            segments.append({
                'speaker': current_speaker,
                'text': ' '.join(current_text)
            })
            
        return segments

    def get_context(self, text, start, end, context_chars=50):
        """Get context around a specific portion of text (from original version)."""
        context_start = max(0, start - context_chars)
        context_end = min(len(text), end + context_chars)
        return text[context_start:context_end]

    def calculate_speaking_time(self):
        """Calculate total speaking time for each speaker from Whisper output."""
        whisper_path = self.processed_dir / "whisper_output.json"
        with open(whisper_path, 'r') as f:
            whisper_data = json.load(f)
            
        speaking_time = defaultdict(float)
        total_segments = defaultdict(int)
        
        # Process each segment
        for segment in whisper_data['segments']:
            start_time = segment['start']
            end_time = segment['end']
            duration = end_time - start_time
            
            # For now, treat all speech as from a single speaker since we don't have diarization
            speaker = "Speaker"
            speaking_time[speaker] += duration
            total_segments[speaker] += 1
        
        return speaking_time, total_segments
    
    def calculate_word_counts(self):
        """Calculate total words spoken from Whisper output."""
        whisper_path = self.processed_dir / "whisper_output.json"
        with open(whisper_path, 'r') as f:
            whisper_data = json.load(f)
            
        word_counts = defaultdict(int)
        right_counts = defaultdict(int)
        
        # Process each segment
        for segment in whisper_data['segments']:
            # For now, treat all speech as from a single speaker
            speaker = "Speaker"
            # Count words in the text
            text = segment['text']
            words = len(text.split())
            word_counts[speaker] += words
            
            # Count instances of "right"
            right_count = len(re.findall(r'\bright\b', text.lower()))
            right_counts[speaker] += right_count
        
        return word_counts, right_counts
            
    def plot_results(self, instances):
        """Create visualizations of the analysis results using pasted transcript data."""
        # Get word counts from pasted transcript
        word_counts = defaultdict(int)
        speaking_time = defaultdict(float)
        right_counts = defaultdict(int)
        
        # Count total words and instances per speaker
        for instance in instances:
            speaker = instance['speaker']
            right_counts[speaker] += 1
            # Approximate words in the context
            word_counts[speaker] += len(instance['text'].split())
        
        # Create figure with subplots
        fig = plt.figure(figsize=(15, 25))
        
        # 1. Raw frequency by speaker
        plt.subplot(3, 1, 1)
        speakers = list(right_counts.keys())
        counts = [right_counts[s] for s in speakers]
        
        bars = plt.bar(speakers, counts)
        plt.title("Distribution of 'right' Usage by Speaker")
        plt.ylabel("Number of Instances")
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom')
        
        # 2. Context analysis (sentence position)
        plt.subplot(3, 1, 2)
        end_count = sum(1 for i in instances if i.get('is_sentence_end', False))
        mid_count = len(instances) - end_count
        
        positions = ['End of Sentence', 'Mid-Sentence']
        counts = [end_count, mid_count]
        
        bars = plt.bar(positions, counts)
        plt.title("Position of 'right' in Sentences")
        plt.ylabel("Count")
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom')
        
        # Add percentage labels
        total = end_count + mid_count
        if total > 0:
            plt.text(bars[0].get_x() + bars[0].get_width()/2., end_count/2,
                    f'{end_count/total*100:.1f}%',
                    ha='center', va='center')
            plt.text(bars[1].get_x() + bars[1].get_width()/2., mid_count/2,
                    f'{mid_count/total*100:.1f}%',
                    ha='center', va='center')
        
        # 3. Normalized usage rates (per 1000 words)
        plt.subplot(3, 1, 3)
        per_word_rates = []
        
        for speaker in speakers:
            if word_counts[speaker] > 0:
                rate = (right_counts[speaker] / word_counts[speaker]) * 1000
                per_word_rates.append(rate)
            else:
                per_word_rates.append(0)
        
        bars = plt.bar(speakers, per_word_rates)
        plt.title("Normalized 'right' Usage Rates (per 1000 words)")
        plt.ylabel("Instances per 1000 words")
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.processed_dir / 'discourse_analysis_pasted.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Print summary statistics
        print("\nAnalysis Summary:")
        print(f"Total instances of 'right': {len(instances)}")
        
        print("\nUsage by speaker:")
        for speaker in speakers:
            count = right_counts[speaker]
            words = word_counts[speaker]
            rate = (count / words * 1000) if words > 0 else 0
            print(f"\n{speaker}:")
            print(f"  Total instances: {count}")
            print(f"  Words in context: {words}")
            print(f"  Rate per 1000 words: {rate:.2f}")
        
        print(f"\nSentence position:")
        if total > 0:
            print(f"  End of sentence: {end_count} ({end_count/total*100:.1f}%)")
            print(f"  Mid-sentence: {mid_count} ({mid_count/total*100:.1f}%)")

    def analyze_pasted_transcript_right_usage(self):
        """Analyze usage of 'right' directly from the pasted transcript."""
        transcript_path = self.raw_dir / "pasted_transcript.txt"
        right_instances = []
        current_speaker = None
        current_content = []
        
        print("\nDebug: Starting transcript analysis...")
        
        with open(transcript_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        print(f"Debug: Read {len(lines)} lines from transcript")
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if not line:
                i += 1
                continue
            
            # Check if line is a speaker name
            if line in {'Lex Fridman', 'Dylan Patel', 'Nathan Lambert'}:
                # Process any accumulated text before changing speakers
                if current_speaker and current_content:
                    full_text = ' '.join(current_content)
                    print(f"\nDebug: Processing text for {current_speaker}:")
                    print(f"Text: {full_text[:100]}...")
                    
                    for match in re.finditer(r'\bright\b', full_text.lower()):
                        # Get surrounding context
                        pre_context = full_text[max(0, match.start()-30):match.start()].strip()
                        post_context = full_text[match.end():min(len(full_text), match.end()+30)].strip()
                        full_context = full_text[max(0, match.start()-30):min(len(full_text), match.end()+30)]
                        
                        # Skip if part of certain phrases
                        skip_phrases = [
                            'all right', 'right now', 'right after', 'right before', 'right away',
                            'right hand', 'right side', 'right there'
                        ]
                        context_phrase = f"{pre_context} right {post_context}".lower()
                        should_skip = any(phrase in context_phrase for phrase in skip_phrases)
                        
                        if should_skip:
                            print(f"Debug: Skipping instance due to phrase match: {context_phrase}")
                            continue
                        
                        # Determine if it's at the end of a sentence
                        next_char_idx = match.end()
                        is_sentence_end = bool(re.match(r'[.!?,]', full_text[next_char_idx:next_char_idx+1] if next_char_idx < len(full_text) else ''))
                        
                        print(f"Debug: Found 'right' instance for {current_speaker}")
                        print(f"Debug: Context: {full_context}")
                        
                        right_instances.append({
                            'speaker': current_speaker,
                            'context': full_context,
                            'is_sentence_end': is_sentence_end,
                            'text': full_text
                        })
                
                current_speaker = line
                current_content = []
                print(f"\nDebug: Found speaker: {current_speaker}")
                i += 1
                continue
            
            # If we have a timestamp, collect all content until next timestamp or speaker
            if line.startswith('('):
                content_lines = []
                i += 1  # Move past timestamp line
                
                # Collect content until we hit another timestamp, speaker, or empty line
                while i < len(lines):
                    next_line = lines[i].strip()
                    if not next_line or next_line.startswith('(') or next_line in {'Lex Fridman', 'Dylan Patel', 'Nathan Lambert'}:
                        break
                    content_lines.append(next_line)
                    i += 1
                
                if content_lines:
                    content_text = ' '.join(content_lines)
                    current_content.append(content_text)
                    print(f"Debug: Added content: {content_text[:100]}...")
                continue
            
            # Skip section headers (usually in all caps)
            if line.isupper() and len(line) > 3:
                i += 1
                continue
            
            # Add any other content lines
            if current_speaker and not line.startswith('('):
                current_content.append(line)
                print(f"Debug: Added line: {line[:100]}...")
            
            i += 1
        
        # Process any remaining text
        if current_speaker and current_content:
            full_text = ' '.join(current_content)
            print(f"\nDebug: Processing final text for {current_speaker}:")
            print(f"Text: {full_text[:100]}...")
            
            for match in re.finditer(r'\bright\b', full_text.lower()):
                # Get surrounding context
                pre_context = full_text[max(0, match.start()-30):match.start()].strip()
                post_context = full_text[match.end():min(len(full_text), match.end()+30)].strip()
                full_context = full_text[max(0, match.start()-30):min(len(full_text), match.end()+30)]
                
                # Skip if part of certain phrases
                skip_phrases = [
                    'all right', 'right now', 'right after', 'right before', 'right away',
                    'right hand', 'right side', 'right there'
                ]
                context_phrase = f"{pre_context} right {post_context}".lower()
                should_skip = any(phrase in context_phrase for phrase in skip_phrases)
                
                if should_skip:
                    print(f"Debug: Skipping instance due to phrase match: {context_phrase}")
                    continue
                
                # Determine if it's at the end of a sentence
                next_char_idx = match.end()
                is_sentence_end = bool(re.match(r'[.!?,]', full_text[next_char_idx:next_char_idx+1] if next_char_idx < len(full_text) else ''))
                
                print(f"Debug: Found 'right' instance for {current_speaker}")
                print(f"Debug: Context: {full_context}")
                
                right_instances.append({
                    'speaker': current_speaker,
                    'context': full_context,
                    'is_sentence_end': is_sentence_end,
                    'text': full_text
                })
        
        # Print summary statistics
        print(f"\nAnalysis of 'right' usage from pasted transcript:")
        print(f"Total instances: {len(right_instances)}")
        
        speaker_counts = {}
        for instance in right_instances:
            speaker = instance['speaker']
            speaker_counts[speaker] = speaker_counts.get(speaker, 0) + 1
        
        print("\nUsage by speaker:")
        total_instances = len(right_instances)
        if total_instances > 0:
            for speaker, count in sorted(speaker_counts.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / total_instances) * 100
                print(f"{speaker}: {count} instances ({percentage:.1f}%)")
                
                # Print some example contexts
                print(f"\nExample contexts for {speaker}:")
                speaker_instances = [i for i in right_instances if i['speaker'] == speaker]
                for instance in speaker_instances[:3]:  # Show first 3 examples for each speaker
                    print(f"  - {instance['context']}")
        
        # Save results
        output_path = self.processed_dir / "full_podcast" / "right_analysis_pasted.json"
        with open(output_path, 'w') as f:
            json.dump(right_instances, f, indent=2)
        
        return right_instances

def analyze_transcript(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    speakers = {'Dylan Patel', 'Nathan Lambert', 'Lex Fridman'}
    current_speaker = None
    current_content = []
    instances = []
    speaker_counts = {}
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # If line is a speaker name, process previous speaker's content and start new speaker
        if line in speakers:
            if current_speaker and current_content:
                text = ' '.join(current_content)
                process_text_for_right(text, current_speaker, instances, speaker_counts)
                current_content = []
            current_speaker = line
            continue
            
        # If line starts with timestamp (e.g. "(00:00:00)"), add its content to current speaker
        if line.startswith('(') and len(line) > 10:  # Basic length check for timestamp
            content = line[line.find(')')+1:].strip()  # Get everything after the timestamp
            if content:
                current_content.append(content)
    
    # Process the last speaker's content
    if current_speaker and current_content:
        text = ' '.join(current_content)
        process_text_for_right(text, current_speaker, instances, speaker_counts)

    # Print analysis results
    print("\nAnalysis of 'right' usage from pasted transcript:")
    print(f"Total instances: {len(instances)}")
    
    print("\nUsage by speaker:")
    for speaker, count in speaker_counts.items():
        if count > 0:
            print(f"{speaker}: {count} instances")
            
    return instances

def process_text_for_right(text, speaker, instances, speaker_counts):
    # Split into sentences
    sentences = text.split('.')
    
    for sentence in sentences:
        sentence = sentence.strip().lower()
        if not sentence:
            continue
            
        # Look for standalone "right" with word boundaries
        words = sentence.split()
        for i, word in enumerate(words):
            clean_word = word.strip('.,!?')
            if clean_word == 'right':
                # Skip common phrases
                if i > 0 and words[i-1].strip('.,!?') == 'all':  # Skip "all right"
                    continue
                if i < len(words)-1 and words[i+1].strip('.,!?') in ['now', 'after', 'before']:  # Skip "right now/after/before"
                    continue
                    
                # Found a valid instance
                print(f"Found 'right' instance in: {sentence}")
                instances.append({
                    'speaker': speaker,
                    'sentence': sentence,
                    'position': 'start' if i == 0 else 'middle' if i < len(words)-1 else 'end'
                })
                speaker_counts[speaker] = speaker_counts.get(speaker, 0) + 1

def count_words_by_speaker(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    speakers = {'Dylan Patel', 'Nathan Lambert', 'Lex Fridman'}
    current_speaker = None
    word_counts = defaultdict(int)
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # If line is a speaker name, switch current speaker
        if line in speakers:
            current_speaker = line
            continue
            
        # If line starts with timestamp, count its words
        if line.startswith('(') and current_speaker:
            content = line[line.find(')')+1:].strip()
            if content:
                words = len(content.split())
                word_counts[current_speaker] += words
    
    # Print results
    total_words = sum(word_counts.values())
    print("\nWord counts by speaker:")
    for speaker, count in word_counts.items():
        percentage = (count / total_words * 100) if total_words > 0 else 0
        print(f"{speaker}: {count:,} words ({percentage:.1f}%)")
        
    return word_counts

def main():
    # First analyze "right" usage
    instances = analyze_transcript('data/raw/pasted_transcript.txt')
    
    print("\nComparing with total word counts:")
    word_counts = count_words_by_speaker('data/raw/pasted_transcript.txt')
    
    # Calculate and print "right" usage rates per 1000 words
    print("\nUsage rates per 1000 words:")
    speaker_right_counts = defaultdict(int)
    for instance in instances:
        speaker_right_counts[instance['speaker']] += 1
    
    for speaker, words in word_counts.items():
        rights = speaker_right_counts[speaker]
        rate = (rights / words * 1000) if words > 0 else 0
        print(f"{speaker}: {rate:.2f} instances per 1000 words ({rights} instances in {words:,} words)")

if __name__ == "__main__":
    main() 