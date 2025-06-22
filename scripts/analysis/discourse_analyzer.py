import re
from collections import defaultdict
from datetime import timedelta
import json

class DiscourseAnalyzer:
    """Analyzes discourse markers in a transcript."""

    def __init__(self, segments):
        self.segments = segments
        self.total_words = 0
        self.word_counts_by_speaker = defaultdict(int)
        self._calculate_initial_stats()

    def _calculate_initial_stats(self):
        """Calculate total words and word counts per speaker."""
        for segment in self.segments:
            words = segment['content'].split()
            self.word_counts_by_speaker[segment['speaker']] += len(words)
            self.total_words += len(words)

    def analyze_right_usage(self, phrases_to_exclude=None):
        """
        Analyzes the usage of the word 'right', excluding specified phrases.
        """
        if phrases_to_exclude is None:
            phrases_to_exclude = ['all right', 'right now', 'right there', 'right after']

        right_instances = []
        excluded_counts = defaultdict(int)

        for segment in self.segments:
            text = segment['content'].lower()
            
            # Find all instances of "right"
            for match in re.finditer(r'\bright\b', text):
                
                # Check if the instance should be excluded
                is_excluded = False
                for phrase in phrases_to_exclude:
                    # Check for phrase at the match position
                    phrase_match_start = text.find(phrase, match.start() - len(phrase) + 1, match.end() + len(phrase) -1)
                    if phrase_match_start != -1:
                        # Quick check to see if 'right' in the found phrase is our 'right'
                        if phrase.find("right") + phrase_match_start == match.start():
                           is_excluded = True
                           excluded_counts[phrase] += 1
                           break
                
                if is_excluded:
                    continue

                # Get context around "right"
                start_idx = match.start()
                context_start = max(0, start_idx - 30)
                context_end = min(len(text), start_idx + 35)
                context = segment['content'][context_start:context_end]

                # Determine if it's at the end of a sentence
                next_char_idx = match.end()
                is_sentence_end = bool(re.match(r'[.!?]', text[next_char_idx:next_char_idx+1] if next_char_idx < len(text) else ''))
                
                right_instances.append({
                    'timestamp': segment['timestamp'],
                    'time_formatted': str(timedelta(seconds=int(segment['timestamp']))),
                    'context': context,
                    'is_sentence_end': is_sentence_end,
                    'speaker': segment['speaker']
                })
        
        # Save analysis results
        analysis_path = 'data/processed/full_podcast/right_analysis_results.json'
        with open(analysis_path, 'w', encoding='utf-8') as f:
            json.dump({
                'right_instances': right_instances,
                'excluded_phrases_counts': dict(excluded_counts)
            }, f, indent=2)

        print(f"Saved 'right' usage analysis to {analysis_path}")
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
