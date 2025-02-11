"""
Right Usage Analyzer Module

This module provides specific analysis for the discourse marker "right".
It extends the base DiscourseAnalyzer with functionality for analyzing "right" usage patterns.
"""

import re
from pathlib import Path
from typing import List, Dict, Any, Tuple
from collections import defaultdict
from ..analyzer import DiscourseAnalyzer

class RightAnalyzer(DiscourseAnalyzer):
    """Analyzer for 'right' usage patterns."""
    
    def __init__(self, transcript_path: Path):
        """
        Initialize right usage analyzer.
        
        Args:
            transcript_path (Path): Path to the combined transcript JSON file
        """
        super().__init__(transcript_path)
        self.skip_phrases = {
            'all_right': r'\ball\s+right\b',
            'right_now': r'\bright\s+now\b',
            'right_after': r'\bright\s+after\b',
            'right_before': r'\bright\s+before\b',
            'right_away': r'\bright\s+away\b',
            'right_hand': r'\bright\s+hand\b',
            'right_side': r'\bright\s+side\b',
            'right_there': r'\bright\s+there\b'
        }
    
    def analyze(self) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
        """
        Analyze usage of 'right' in the transcript.
        
        Returns:
            Tuple[List[Dict], Dict]: List of instances and count by speaker
        """
        instances = []
        speaker_counts = defaultdict(int)
        filtered_instances = defaultdict(list)
        
        # Initialize counts for all speakers
        for segment in self.segments:
            speaker_counts[segment['speaker']] = 0
        
        total_instances = 0
        filtered_count = 0
        
        for segment in self.segments:
            speaker = segment['speaker']
            text = segment['content'].lower()
            
            # Count total raw instances
            raw_matches = list(re.finditer(r'\bright\b', text))
            total_instances += len(raw_matches)
            
            for match in raw_matches:
                # Get surrounding context
                pre_context = text[max(0, match.start()-30):match.start()].strip()
                post_context = text[match.end():min(len(text), match.end()+30)].strip()
                context = text[max(0, match.start()-30):min(len(text), match.end()+30)]
                
                # Check for phrases to skip
                window_start = max(0, match.start() - 10)
                window_end = min(len(text), match.end() + 10)
                check_text = text[window_start:window_end]
                
                should_skip = False
                skip_reason = None
                for phrase_name, pattern in self.skip_phrases.items():
                    if re.search(pattern, check_text):
                        should_skip = True
                        skip_reason = phrase_name
                        break
                
                if should_skip:
                    filtered_count += 1
                    filtered_instances[skip_reason].append({
                        'speaker': speaker,
                        'context': context,
                        'timestamp': segment.get('timestamp')
                    })
                    continue
                
                # Determine if it's at the end of a sentence
                next_char_idx = match.end()
                is_sentence_end = bool(re.match(r'[.!?,]', text[next_char_idx:next_char_idx+1] if next_char_idx < len(text) else ''))
                
                instances.append({
                    'speaker': speaker,
                    'context': context,
                    'is_sentence_end': is_sentence_end,
                    'full_text': text,
                    'timestamp': segment.get('timestamp')
                })
                speaker_counts[speaker] += 1
        
        # Save analysis results
        self._save_analysis_results(instances, speaker_counts, filtered_instances, total_instances)
        
        return instances, dict(speaker_counts)
    
    def _save_analysis_results(self, instances: List[Dict[str, Any]], 
                             speaker_counts: Dict[str, int],
                             filtered_instances: Dict[str, List[Dict[str, Any]]],
                             total_instances: int) -> None:
        """
        Save analysis results to a JSON file.
        
        Args:
            instances (List[Dict]): List of right instances
            speaker_counts (Dict[str, int]): Count of instances by speaker
            filtered_instances (Dict[str, List]): Filtered instances by reason
            total_instances (int): Total number of instances found
        """
        debug_info = {
            'total_instances': total_instances,
            'kept_instances': len(instances),
            'filtered_instances': sum(len(insts) for insts in filtered_instances.values()),
            'filtered_breakdown': {
                phrase: len(instances) 
                for phrase, instances in filtered_instances.items()
            },
            'filtered_examples': {
                phrase: [
                    {'context': inst['context'], 'speaker': inst['speaker']}
                    for inst in instances[:3]  # Show first 3 examples
                ]
                for phrase, instances in filtered_instances.items()
            },
            'speaker_counts': speaker_counts
        }
        
        # Save debug information
        output_dir = self.transcript_path.parent
        debug_path = output_dir / 'right_analysis_debug.json'
        with open(debug_path, 'w', encoding='utf-8') as f:
            json.dump(debug_info, f, indent=2)
    
    def get_usage_rates(self) -> Dict[str, float]:
        """
        Calculate usage rates per 1000 words for each speaker.
        
        Returns:
            Dict[str, float]: Usage rates by speaker
        """
        instances, speaker_counts = self.analyze()
        word_counts = self.get_word_counts()
        
        rates = {}
        for speaker, count in speaker_counts.items():
            if word_counts[speaker] > 0:
                rates[speaker] = (count / word_counts[speaker]) * 1000
            else:
                rates[speaker] = 0
        
        return rates
    
    def get_temporal_distribution(self, window_size: float = 300) -> Dict[str, List[float]]:
        """
        Get temporal distribution of usage in time windows.
        
        Args:
            window_size (float): Size of time windows in seconds
            
        Returns:
            Dict[str, List[float]]: Rates per minute by speaker for each window
        """
        instances, _ = self.analyze()
        
        # Group instances by speaker
        speaker_times = defaultdict(list)
        for instance in instances:
            if instance['timestamp'] is not None:
                speaker_times[instance['speaker']].append(instance['timestamp'])
        
        # Calculate rates in windows
        max_time = max(max(times) for times in speaker_times.values())
        windows = list(range(0, int(max_time) + int(window_size), int(window_size)))
        
        rates = defaultdict(list)
        for speaker, times in speaker_times.items():
            for start in windows[:-1]:
                end = start + window_size
                count = sum(1 for t in times if start <= t < end)
                rate = count / (window_size / 60)  # Convert to per minute
                rates[speaker].append(rate)
        
        return dict(rates) 