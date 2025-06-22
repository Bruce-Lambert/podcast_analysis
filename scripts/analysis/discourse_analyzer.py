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
        is_word_based = 'word' in self.segments[0] if self.segments else False

        if is_word_based:
            for segment in self.segments:
                self.word_counts_by_speaker[segment['speaker']] += 1
                self.total_words += 1
        else:
            for segment in self.segments:
                words = segment['content'].split()
                self.word_counts_by_speaker[segment['speaker']] += len(words)
                self.total_words += len(words)

    def analyze_right_usage(self, phrases_to_exclude=None):
        """
        Analyzes the usage of the word 'right', excluding specified phrases.
        This method can handle both segment-based and word-based data.
        """
        if phrases_to_exclude is None:
            phrases_to_exclude = ['all right', 'right now', 'right there', 'right after']

        right_instances = []
        excluded_counts = defaultdict(int)

        # Check if segments are word-based (high-accuracy) or content-based
        is_word_based = 'word' in self.segments[0] if self.segments else False

        if is_word_based:
            # High-accuracy word-based analysis
            for i, segment in enumerate(self.segments):
                if segment['word'].strip().lower() == 'right':
                    # Check for excluded phrases by looking at surrounding words
                    full_phrase_text = " ".join(
                        self.segments[j]['word'] for j in range(max(0, i-2), min(len(self.segments), i+3))
                    ).lower()
                    
                    is_excluded = False
                    for phrase in phrases_to_exclude:
                        if phrase in full_phrase_text:
                            is_excluded = True
                            excluded_counts[phrase] += 1
                            break
                    if is_excluded:
                        continue
                    
                    # Get context directly from surrounding words
                    context_words = [self.segments[j]['word'] for j in range(max(0, i-10), min(len(self.segments), i+11))]
                    context = " ".join(context_words)
                    
                    right_instances.append({
                        'start_time': segment['start'],
                        'end_time': segment['end'],
                        'time_formatted': str(timedelta(seconds=int(segment['start']))),
                        'context': context,
                        'speaker': segment['speaker']
                    })
            analysis_path = 'data/processed/full_podcast/accurate_analysis_results.json'

        else:
            # Original content-based analysis
            for segment in self.segments:
                text = segment['content'].lower()
                for match in re.finditer(r'\bright\b', text):
                    is_excluded = False
                    for phrase in phrases_to_exclude:
                        phrase_match_start = text.find(phrase, match.start() - len(phrase) + 1, match.end() + len(phrase) -1)
                        if phrase_match_start != -1 and phrase.find("right") + phrase_match_start == match.start():
                           is_excluded = True
                           excluded_counts[phrase] += 1
                           break
                    if is_excluded:
                        continue

                    start_idx = match.start()
                    context_start = max(0, start_idx - 30)
                    context_end = min(len(text), start_idx + 35)
                    context = segment['content'][context_start:context_end]
                    is_sentence_end = bool(re.match(r'[.!?]', text[next_char_idx:next_char_idx+1] if next_char_idx < len(text) else ''))
                    
                    right_instances.append({
                        'timestamp': segment['timestamp'],
                        'time_formatted': str(timedelta(seconds=int(segment['timestamp']))),
                        'context': context,
                        'is_sentence_end': is_sentence_end,
                        'speaker': segment['speaker']
                    })
            analysis_path = 'data/processed/full_podcast/right_analysis_results.json'

        # Save analysis results
        with open(analysis_path, 'w', encoding='utf-8') as f:
            json.dump({
                'right_instances': right_instances,
                'excluded_phrases_counts': dict(excluded_counts)
            }, f, indent=2)

        print(f"Saved analysis to {analysis_path}")
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
