import argparse
from pathlib import Path
from analysis.transcript_loader import TranscriptParser
from analysis.discourse_analyzer import DiscourseAnalyzer
from analysis.visualizer import Visualizer
from create_video_montages_accurate import create_montage_from_results
from create_fcpxml_accurate import generate_fcpxml_from_results
from add_counter_overlay import create_counter_video

def main():
    """Main function to run the analysis."""
    parser = argparse.ArgumentParser(description="Analyze discourse markers in a podcast transcript.")
    parser.add_argument('--pasted_transcript_path', type=str, default=None,
                        help='Path to the pasted transcript file with speaker names. Required for whisper or vtt sources.')
    parser.add_argument('--source', type=str, default='elevenlabs',
                        choices=['elevenlabs', 'whisper', 'vtt'],
                        help='The primary transcript source to use.')
    
    args = parser.parse_args()

    # Define file paths based on the source
    output_dir = Path('data/processed/full_podcast')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    vtt_path = None
    whisper_json_path = None
    elevenlabs_json_path = None
    is_accurate_mode = False

    if args.source == 'elevenlabs':
        elevenlabs_json_path = output_dir / 'elevenlabs_transcript.json'
        analysis_results_path = output_dir / 'elevenlabs_analysis_results.json'
        is_accurate_mode = True
    elif args.source == 'whisper':
        whisper_json_path = output_dir / 'whisper_output.json'
        analysis_results_path = output_dir / 'whisper_analysis_results.json'
        is_accurate_mode = True
    else: # vtt
        vtt_path = 'data/raw/full_video_sentences.vtt'
        analysis_results_path = output_dir / 'vtt_analysis_results.json'
        if not args.pasted_transcript_path:
            parser.error("--pasted_transcript_path is required when --source is 'vtt'")

    # 1. Parse the transcript
    print(f"Step 1: Parsing transcript from source: {args.source.upper()}...")
    loader = TranscriptParser(
        pasted_transcript_path=args.pasted_transcript_path,
        vtt_path=vtt_path,
        whisper_json_path=whisper_json_path,
        elevenlabs_json_path=elevenlabs_json_path
    )
    parsed_data, unattributed = loader.parse()
    
    if not parsed_data:
        print("Exiting: No parsed data from transcript loader.")
        return

    print(f"Successfully loaded and processed {len(parsed_data)} data segments.")

    # 2. Analyze discourse
    print("\nStep 2: Analyzing discourse for 'right' usage...")
    analyzer = DiscourseAnalyzer(parsed_data)
    right_instances, excluded_phrases = analyzer.analyze_right_usage(analysis_results_path)
    
    print(f"Found {len(right_instances)} instances of 'right'.")
    print(f"Excluded phrases counts: {excluded_phrases}")

    # 3. Create visualizations
    print("\nStep 3: Creating visualizations...")
    visualizer = Visualizer()
    visualizer.plot_right_usage_by_speaker(right_instances)

    if is_accurate_mode:
        # Get total podcast duration for x-axis in time plot
        last_segment = parsed_data[-1] if parsed_data else {}
        total_duration = last_segment.get('end', 0)
        visualizer.plot_usage_over_time_accurate(right_instances, total_duration)
        visualizer.plot_usage_dot_plot(right_instances, total_duration)
    
    print("Visualizations created.")

    # 4. Generate Video Assets (only in accurate mode)
    if is_accurate_mode:
        print("\nStep 4: Generating high-accuracy video assets...")
        video_source_path = "data/raw/full_podcast/full_video.mp4"
        montage_dir = Path(f"data/processed/full_podcast/video_montages_{args.source}")
        clips_dir = Path(f"data/processed/full_podcast/video_clips_{args.source}")
        
        # Define config for video creation
        video_config = {
            "SPEAKER_NAME": "Dylan Patel",
            "MONTAGE_NAME": f"dylan_patel_right_montage_{args.source}.mp4",
            "FINAL_MONTAGE_NAME": f"dylan_patel_right_montage_{args.source}_with_counter.mp4",
            "CONTEXT_BEFORE": 0.5,
            "CONTEXT_AFTER": 0.5
        }

        # Create the initial, fast montage WITHOUT overlays
        base_montage_path = create_montage_from_results(
            results_path=str(analysis_results_path),
            video_path=video_source_path,
            output_dir=str(montage_dir),
            clips_dir=str(clips_dir),
            config=video_config
        )
        
        if base_montage_path:
            # Add the counter overlay using the faster subtitle method
            print("\nStep 4b: Adding counter and timestamp overlays...")
            create_counter_video(
                results_path=str(analysis_results_path),
                montage_path=str(base_montage_path),
                output_path=str(montage_dir / video_config['FINAL_MONTAGE_NAME']),
                speaker_name=video_config['SPEAKER_NAME']
            )

        # Create FCPXML (referencing the base montage)
        generate_fcpxml_from_results(
            results_path=str(analysis_results_path),
            output_fcpxml_path=f"data/processed/full_podcast/video_montages_{args.source}/montage_{args.source}.fcpxml",
            video_path=video_source_path
        )
        print("High-accuracy video assets generated.")

    print("\nAnalysis pipeline complete.")

if __name__ == '__main__':
    main()
