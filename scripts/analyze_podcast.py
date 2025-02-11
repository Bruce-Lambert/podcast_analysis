#!/usr/bin/env python3
"""
Podcast Analysis Script

This script demonstrates how to use the transcript processing and discourse analysis modules
to analyze discourse patterns in podcast transcripts.
"""

from pathlib import Path
from transcript_processing import TranscriptCombiner
from discourse_analysis import RightAnalyzer, DiscourseVisualizer

def main():
    # Step 1: Combine transcripts
    print("Combining transcripts...")
    combiner = TranscriptCombiner(
        vtt_path='data/raw/macwhisper_full_video.vtt',
        text_path='data/raw/pasted_transcript.txt'
    )
    combined_segments = combiner.combine()
    print(combiner.get_debug_summary())
    
    # Step 2: Analyze discourse patterns
    print("\nAnalyzing discourse patterns...")
    analyzer = RightAnalyzer('data/processed/combined_transcript.json')
    instances, speaker_counts = analyzer.analyze()
    word_counts = analyzer.get_word_counts()
    temporal_rates = analyzer.get_temporal_distribution(window_size=300)  # 5-minute windows
    
    # Print analysis results
    total_words = sum(word_counts.values())
    print("\nWord counts by speaker:")
    for speaker, count in word_counts.items():
        percentage = (count / total_words * 100) if total_words > 0 else 0
        print(f"{speaker}: {count:,} words ({percentage:.1f}%)")
    
    print(f"\nAnalysis of 'right' usage:")
    print(f"Total instances: {len(instances)}")
    
    print("\nUsage by speaker:")
    for speaker, count in speaker_counts.items():
        words = word_counts[speaker]
        rate = (count / words * 1000) if words > 0 else 0
        print(f"{speaker}: {rate:.2f} instances per 1000 words ({count} instances in {words:,} words)")
    
    # Step 3: Generate visualizations
    print("\nGenerating visualizations...")
    visualizer = DiscourseVisualizer()
    visualizer.plot_results(
        instances=instances,
        word_counts=word_counts,
        title="Analysis of 'right' Usage in Lex Fridman Podcast #459",
        temporal_rates=temporal_rates
    )
    print("Analysis complete! Check the data/processed directory for results.")

if __name__ == "__main__":
    main() 