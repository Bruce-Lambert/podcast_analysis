import argparse
from pathlib import Path
from analysis.transcript_loader import TranscriptParser
from analysis.discourse_analyzer import DiscourseAnalyzer
from analysis.visualizer import Visualizer

def main(pasted_transcript_path, vtt_path=None):
    """
    Main function to run the full analysis pipeline.
    """
    print("Starting analysis pipeline...")

    # 1. Load and parse transcripts
    print("\nStep 1: Loading transcripts...")
    loader = TranscriptParser(pasted_transcript_path, vtt_path)
    combined_segments, _ = loader.parse()
    
    if not combined_segments:
        print("Error: No segments were produced after combining transcripts. Exiting.")
        return

    print(f"Successfully loaded and combined {len(combined_segments)} segments.")

    # 2. Analyze discourse
    print("\nStep 2: Analyzing discourse for 'right' usage...")
    analyzer = DiscourseAnalyzer(combined_segments)
    right_instances, excluded_counts = analyzer.analyze_right_usage()
    
    print(f"Found {len(right_instances)} instances of 'right'.")
    print("Excluded phrases counts:", excluded_counts)

    # 3. Get auxiliary data for visualization
    word_counts = loader.get_word_counts()
    speaking_time = analyzer.get_speaking_time()

    # 4. Generate visualizations
    print("\nStep 3: Generating visualizations...")
    visualizer = Visualizer()
    visualizer.plot_right_usage_by_speaker(right_instances)
    visualizer.plot_sentence_position_analysis(right_instances)
    visualizer.plot_normalized_usage_rate(right_instances, word_counts)
    visualizer.plot_usage_over_time(right_instances, speaking_time)

    print("\nAnalysis pipeline finished successfully!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the podcast discourse analysis pipeline.")
    parser.add_argument(
        "pasted_transcript",
        type=str,
        help="Path to the pasted transcript with speaker names."
    )
    parser.add_argument(
        "--vtt",
        type=str,
        help="Optional path to the VTT transcript for verbatim content."
    )
    args = parser.parse_args()

    pasted_path = Path(args.pasted_transcript)
    vtt_file_path = Path(args.vtt) if args.vtt else None

    main(pasted_path, vtt_file_path)
