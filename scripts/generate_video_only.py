import argparse
from pathlib import Path
from create_video_montages_accurate import create_montage_from_results
from add_counter_overlay import create_counter_video

def generate_video_only():
    """
    Generates the video montage and overlays from an existing analysis file.
    This allows for rapid testing of the video generation steps without re-running analysis.
    """
    parser = argparse.ArgumentParser(description="Generate video assets from existing analysis.")
    parser.add_argument('--source', type=str, default='elevenlabs', help='The primary transcript source.')
    parser.add_argument('--speaker', type=str, default='Dylan Patel', help='Speaker to generate montage for.')
    args = parser.parse_args()

    # Define paths based on source
    analysis_results_path = Path(f'data/processed/full_podcast/elevenlabs_analysis_results.json')
    video_source_path = "data/raw/full_podcast/full_video.mp4"
    montage_dir = Path(f"data/processed/full_podcast/video_montages_{args.source}")
    clips_dir = Path(f"data/processed/full_podcast/video_clips_{args.source}")

    # Define filenames
    base_montage_name = f"{args.speaker.replace(' ', '_').lower()}_right_montage_{args.source}.mp4"
    final_montage_name = f"{args.speaker.replace(' ', '_').lower()}_right_montage_{args.source}_with_counter.mp4"

    print(f"--- Generating fast montage for {args.speaker} ---")
    base_montage_path = create_montage_from_results(
        results_path=str(analysis_results_path),
        video_path=video_source_path,
        output_dir=str(montage_dir),
        clips_dir=str(clips_dir),
        montage_name=base_montage_name,
        speaker_name=args.speaker
    )
    
    if base_montage_path:
        print(f"\\n--- Adding overlays to montage for {args.speaker} ---")
        create_counter_video(
            results_path=str(analysis_results_path),
            montage_path=str(base_montage_path),
            output_path=str(montage_dir / final_montage_name)
        )
    
    print("\\n--- Video generation complete. ---")

if __name__ == '__main__':
    generate_video_only() 