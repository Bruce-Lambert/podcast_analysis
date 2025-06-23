import argparse
import json
import tempfile
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from pydub import AudioSegment
import parselmouth
import numpy as np
import shutil

def analyze_intonation(results_path, full_audio_path, output_dir):
    """
    Analyzes the intonation of 'right' instances using parselmouth's core functions.
    """
    print("Starting intonation analysis with direct parselmouth methods...")

    results_path = Path(results_path)
    full_audio_path = Path(full_audio_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(results_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    dylan_instances = [inst for inst in data.get('right_instances', []) if inst.get('speaker') == 'Dylan Patel']
    print(f"Found {len(dylan_instances)} instances of 'right' by Dylan Patel.")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        audio_clips_dir = temp_path / "audio_clips"
        audio_clips_dir.mkdir()

        print("Step 1: Extracting precise audio clips...")
        full_audio = AudioSegment.from_file(full_audio_path)
        for i, instance in enumerate(dylan_instances):
            start_ms = instance['start_time'] * 1000
            end_ms = instance['end_time'] * 1000
            clip = full_audio[start_ms:end_ms]
            clip.export(audio_clips_dir / f"right_clip_{i:04d}.wav", format="wav")
        print(f"Extracted {len(dylan_instances)} audio clips.")

        print("Step 2: Analyzing intonation using direct Parselmouth functions...")
        intonation_results = []
        for audio_file in sorted(audio_clips_dir.glob('*.wav')):
            try:
                snd = parselmouth.Sound(str(audio_file))
                # Extract pitch using Praat's default autocorrelation method
                pitch = snd.to_pitch()
                # Get the standard deviation of the pitch values in Hertz
                # We only consider voiced frames for a more accurate measure
                f0_values = pitch.selected_array['frequency']
                f0_values = f0_values[f0_values > 0] # Filter out unvoiced frames (f0=0)
                
                if len(f0_values) > 1:
                    f0_sd = np.std(f0_values)
                    # Use a threshold to classify intonation
                    if f0_sd > 15:
                        intonation_results.append('Modulated')
                    else:
                        intonation_results.append('Flat')
                else:
                    intonation_results.append('Unclassified (No pitch found)')
            except Exception as e:
                print(f"Could not analyze {audio_file.name}: {e}")
                intonation_results.append('Error')

        print("Step 3: Generating visualization...")
        counts = Counter(intonation_results)
        plt.figure(figsize=(10, 6))
        sns.barplot(x=list(counts.keys()), y=list(counts.values()))
        plt.title("Intonation Analysis of 'Right'")
        plt.ylabel("Number of Instances")
        plt.xlabel("Intonation Type")
        viz_path = output_dir / "intonation_analysis.png"
        plt.savefig(viz_path, dpi=300)
        plt.close()
        print(f"Intonation analysis complete. Visualization saved to {viz_path}")
        print(f"Results: {counts}")

def main():
    """Main function to run the analysis."""
    parser = argparse.ArgumentParser(description="Analyze the intonation of 'right' in a podcast.")
    # Set default paths based on the project structure
    default_results_path = "data/processed/full_podcast/accurate_analysis_results.json"
    default_audio_path = "data/raw/full_podcast/full_video.mp4"
    default_output_dir = "data/processed/full_podcast/visualizations_accurate"
    
    parser.add_argument('results_path', type=str, nargs='?', default=default_results_path,
                        help=f'Path to the accurate_analysis_results.json file. Defaults to {default_results_path}')
    parser.add_argument('--full_audio_path', type=str, default=default_audio_path,
                        help=f'Path to the full podcast audio/video file. Defaults to {default_audio_path}')
    parser.add_argument('--output_dir', type=str, default=default_output_dir,
                        help=f'Directory to save the visualization. Defaults to {default_output_dir}')

    args = parser.parse_args()

    analyze_intonation(args.results_path, args.full_audio_path, args.output_dir)

if __name__ == "__main__":
    main() 