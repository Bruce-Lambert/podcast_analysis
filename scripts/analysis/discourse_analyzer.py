import re
from collections import defaultdict
from datetime import timedelta
import json
import string # Import string module

class DiscourseAnalyzer:
    """Analyzes discourse markers in a transcript."""

    def __init__(self, segments):
        self.segments = segments
        self.total_words = 0
        self.word_counts_by_speaker = defaultdict(int)
        self._calculate_initial_stats()

    def _calculate_initial_stats(self):
        """Calculate total words and word counts per speaker."""
        # Check for a more reliable key first, like 'speaker_id' from ElevenLabs
        is_word_based = 'speaker_id' in self.segments[0] if self.segments else False

        if is_word_based:
            for segment in self.segments:
                # Use the 'speaker' key which should be remapped in main.py
                self.word_counts_by_speaker[segment['speaker']] += 1
                self.total_words += 1
        else: # Fallback for other transcript types
            for segment in self.segments:
                words = segment.get('content', '').split()
                self.word_counts_by_speaker[segment['speaker']] += len(words)
                self.total_words += len(words)

    def analyze_right_usage(self, output_path, phrases_to_exclude=None):
        """
        Analyzes the usage of the word 'right', excluding specified phrases.
        This now only handles word-based (high-accuracy) data.
        """
        if phrases_to_exclude is None:
            phrases_to_exclude = ['all right', 'right now', 'right there', 'right after']

        right_instances = []
        excluded_counts = defaultdict(int)

        # High-accuracy word-based analysis
        words_data = self.segments # Keep original segments for metadata

        for i, segment in enumerate(words_data):
            # More robustly clean the word by removing all punctuation and making it lowercase
            original_word = segment.get('word', '')
            # Create a translation table to remove all punctuation
            translator = str.maketrans('', '', string.punctuation)
            word = original_word.translate(translator).strip().lower()

            if word == 'right':
                is_excluded = False
                # Check for excluded phrases using word-level matching
                for phrase in phrases_to_exclude:
                    phrase_words = phrase.lower().split()
                    try:
                        # Find where "right" is in the phrase, e.g., 1 for "all right"
                        right_index_in_phrase = phrase_words.index('right')

                        # Determine the start and end of the phrase in the main words list
                        start_index = i - right_index_in_phrase
                        end_index = start_index + len(phrase_words)

                        if start_index < 0:
                            continue

                        if end_index <= len(words_data):
                            candidate_words = [
                                w.get('word', '').translate(translator).strip().lower()
                                for w in words_data[start_index:end_index]
                            ]

                            if candidate_words == phrase_words:
                                is_excluded = True
                                excluded_counts[phrase] += 1
                                break # Move to next word once an exclusion is found
                    except ValueError:
                        # "right" not in phrase_words, should not happen with default list
                        continue
                
                if is_excluded:
                    continue
                
                # Get context directly from surrounding words
                context_words = [s['word'] for s in words_data[max(0, i-10):min(len(words_data), i+11)]]
                context = " ".join(context_words)
                
                right_instances.append({
                    'start_time': segment['start'],
                    'end_time': segment['end'],
                    'time_formatted': str(timedelta(seconds=int(segment['start']))),
                    'context': context,
                    'speaker': segment['speaker'] # Use the remapped speaker name
                })

        # Save analysis results
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({
                'right_instances': right_instances,
                'excluded_phrases_counts': dict(excluded_counts)
            }, f, indent=2)

        print(f"Saved analysis to {output_path}")
        return right_instances, dict(excluded_counts)
        
    def get_speaking_time(self):
        """Calculate total speaking time for each speaker."""
        speaking_time = defaultdict(float)
        last_timestamp = 0
        
        # Assuming segments are sorted by timestamp
        for i, segment in enumerate(self.segments):
            # Calculate duration of the segment
            if i + 1 < len(self.segments):
                duration = self.segments[i+1]['timestamp'] - segment['timestamp']
            else:
                # Estimate duration of last segment as 5 seconds (can be refined)
                duration = 5
            
            speaking_time[segment['speaker']] += duration
            last_timestamp = segment['timestamp']
            
        return dict(speaking_time)
