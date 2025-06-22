import json
import subprocess
from pathlib import Path
import tempfile
from typing import List, Dict

def create_montage_from_results(
    results_path: str, 
    video_path: str, 
    output_dir: str, 
    clips_dir: str,
    montage_name: str,
    context_before: float = 0.5,
    context_after: float = 0.5
):
    """
    Creates a video montage from a JSON results file containing precise start and end times.
    """
    results_path = Path(results_path)
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    clips_dir = Path(clips_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    clips_dir.mkdir(parents=True, exist_ok=True)

    with open(results_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    instances = data.get('right_instances', [])
    if not instances:
        print("No 'right' instances found in the results file.")
        return

    # Filter for Dylan Patel
    instances = [inst for inst in instances if inst.get('speaker') == 'Dylan Patel']
    print(f"Found {len(instances)} instances for Dylan Patel.")

    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as concat_file:
        for i, instance in enumerate(instances):
            clip_start = instance['start_time']
            clip_end = instance['end_time']
            
            # Use the provided context window
            start_time = max(0, clip_start - context_before)
            end_time = clip_end + context_after
            duration = end_time - start_time

            clip_filename = clips_dir / f"accurate_clip_{i:04d}.mp4"

            # Use ffmpeg to create the clip
            cmd = [
                'ffmpeg', '-y',
                '-ss', str(start_time),
                '-i', str(video_path),
                '-t', str(duration),
                '-c:v', 'libx264', '-preset', 'medium', '-crf', '23',
                '-c:a', 'aac', '-b:a', '128k',
                str(clip_filename)
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            concat_file.write(f"file '{clip_filename.resolve()}'\n")

    # Now, concatenate the clips into the final montage
    montage_output_path = output_dir / montage_name
    concat_cmd = [
        'ffmpeg', '-y',
        '-f', 'concat',
        '-safe', '0',
        '-i', concat_file.name,
        '-c', 'copy',
        str(montage_output_path)
    ]
    subprocess.run(concat_cmd, check=True, capture_output=True)

    print(f"Montage created at {montage_output_path}") 