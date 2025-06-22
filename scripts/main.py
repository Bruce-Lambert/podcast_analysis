import argparse
from pathlib import Path
from analysis.transcript_loader import TranscriptParser
from analysis.discourse_analyzer import DiscourseAnalyzer
from analysis.visualizer import Visualizer
from create_video_montages_accurate import create_montage_from_results
from create_fcpxml_accurate import generate_fcpxml_from_results

def main(pasted_transcript_path, vtt_path=None, whisper_json_path=None, accurate=False):
    """
    Main function to run the full analysis pipeline.
    """
    print("Starting analysis pipeline...")

    # Define paths based on the accurate flag
    if accurate:
        if not whisper_json_path:
            print("Error: --accurate flag requires --whisper_json_path to be set.")
            return
        print("\n*** Running in High-Accuracy Mode ***")
        output_suffix = "_accurate"
        loader = TranscriptParser(pasted_transcript_path, whisper_json_path=whisper_json_path)
    else:
        print("\n*** Running in Standard Mode ***")
        output_suffix = ""
        loader = TranscriptParser(pasted_transcript_path, vtt_path=vtt_path)

    # 1. Load and parse transcripts
    print("\nStep 1: Loading transcripts...")
    # The new loader returns a flat list of words in accurate mode
    parsed_data, _ = loader.parse()
    
    if not parsed_data:
        print("Error: No segments were produced. Exiting.")
        return

    print(f"Successfully loaded and processed {len(parsed_data)} data segments.")

    # 2. Analyze discourse
    print("\nStep 2: Analyzing discourse for 'right' usage...")
    analyzer = DiscourseAnalyzer(parsed_data)
    right_instances, excluded_phrases = analyzer.analyze_right_usage()
    
    print(f"Found {len(right_instances)} instances of 'right'.")
    print(f"Excluded phrases counts: {excluded_phrases}")

    # 3. Create visualizations
    print("\nStep 3: Creating visualizations...")
    visualizer = Visualizer()
    visualizer.plot_right_usage_by_speaker(right_instances)

    if accurate:
        # Get total podcast duration for x-axis in time plot
        last_segment = parsed_data[-1] if parsed_data else {}
        total_duration = last_segment.get('end', 0)
        visualizer.plot_usage_over_time_accurate(right_instances, total_duration)
        visualizer.plot_usage_dot_plot(right_instances, total_duration)
    
    print("Visualizations created.")

    # 4. Generate Video Assets (only in accurate mode)
    if accurate:
        print("\nStep 4: Generating high-accuracy video assets...")
        video_source_path = "data/raw/full_podcast/full_video.mp4"
        analysis_results_path = "data/processed/full_podcast/accurate_analysis_results.json"
        
        # Create video montage
        create_montage_from_results(
            results_path=analysis_results_path,
            video_path=video_source_path,
            output_dir=f"data/processed/full_podcast/video_montages{output_suffix}",
            clips_dir=f"data/processed/full_podcast/video_clips{output_suffix}",
            montage_name="dylan_patel_right_montage_accurate.mp4"
        )
        
        # Create FCPXML
        generate_fcpxml_from_results(
            results_path=analysis_results_path,
            output_fcpxml_path=f"data/processed/full_podcast/video_montages{output_suffix}/montage_accurate.fcpxml",
            video_path=video_source_path
        )
        print("High-accuracy video assets generated.")

    print("\nAnalysis pipeline complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the podcast analysis pipeline.")
    parser.add_argument("pasted_transcript_path", help="Path to the pasted transcript file with speaker names.")
    parser.add_argument("--vtt_path", help="Path to the VTT transcript file (for standard mode).")
    parser.add_argument("--whisper_json_path", help="Path to the Whisper JSON output file (for accurate mode).")
    parser.add_argument("--accurate", action="store_true", help="Run the high-accuracy video asset generation pipeline.")
    
    args = parser.parse_args()
    
    main(args.pasted_transcript_path, args.vtt_path, args.whisper_json_path, args.accurate)
