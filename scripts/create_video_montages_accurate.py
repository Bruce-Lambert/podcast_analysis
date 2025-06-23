import json
import subprocess
from pathlib import Path
import tempfile
from typing import List, Dict
import os
import shutil

def create_montage_from_results(
    results_path: str, 
    video_path: str, 
    output_dir: str, 
    clips_dir: str,
    config: dict
):
    """
    Creates a video montage from a JSON results file containing precise start and end times.
    This version does NOT add overlays, for speed.
    """
    results_path = Path(results_path)
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    clips_dir = Path(clips_dir)
    speaker_name = config['SPEAKER_NAME']

    # --- Clean and Create Directories ---
    # Ensure a clean slate for the clips
    if clips_dir.exists():
        shutil.rmtree(clips_dir)
    clips_dir.mkdir(parents=True, exist_ok=True)
    
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(results_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    instances = data.get('right_instances', [])
    if speaker_name:
        instances = [inst for inst in instances if inst.get('speaker') == speaker_name]
    
    print(f"Found {len(instances)} instances for {speaker_name}.", flush=True)
    
    if not instances:
        print("No instances found for the specified speaker. Montage creation skipped.", flush=True)
        return

    # --- Create individual clips ONLY if they are missing ---
    print("--- Verifying clips and creating only missing ones... ---", flush=True)
    clips_dir.mkdir(parents=True, exist_ok=True) # Ensure directory exists, DO NOT DELETE.

    for i, instance in enumerate(instances):
        clip_path = clips_dir / f"accurate_clip_{i:04d}.mp4"

        if clip_path.exists():
            # Uncomment the next line for verbose skipping, but it's too noisy for 900+ clips
            # print(f"  - Clip {i+1}/{len(instances)} exists. Skipping.", flush=True)
            continue

        # If we are here, the clip is missing, so we create it.
        print(f"  - Clip {i+1}/{len(instances)} is missing. Creating...", flush=True)
        start_time = max(0, instance['start_time'] - config['CONTEXT_BEFORE'])
        duration = (instance['end_time'] + config['CONTEXT_AFTER']) - start_time
        
        source_video_rel_path = os.path.relpath(video_path, clips_dir)
        
        cmd = [
            'ffmpeg', '-y', '-ss', str(start_time), '-i', str(source_video_rel_path),
            '-t', str(duration), '-c', 'copy', clip_path.name
        ]
        subprocess.run(cmd, check=True, cwd=str(clips_dir), stdin=subprocess.DEVNULL)
    print("--- All necessary clips are present. ---", flush=True)

    # --- Create the concatenation file ---
    print("\\n--- Generating concatenation file... ---", flush=True)
    concat_list_path = clips_dir / 'concat.txt'
    with open(concat_list_path, 'w') as f:
        for i in range(len(instances)):
            line = f"file 'accurate_clip_{i:04d}.mp4'{os.linesep}"
            f.write(line)

    # --- Concatenation Step ---
    print("\\n--- Concatenating clips into final montage... ---", flush=True)
    montage_output_path = output_dir / config['MONTAGE_NAME']
    # Make the output path relative to the clips_dir for the cwd command
    relative_output_path = os.path.relpath(montage_output_path, clips_dir)
    
    concat_cmd = [
        'ffmpeg', '-y', '-f', 'concat', '-safe', '0',
        '-i', 'concat.txt', 
        '-c', 'copy', str(relative_output_path)
    ]
    # Prevent hanging by disabling stdin
    subprocess.run(concat_cmd, check=True, cwd=str(clips_dir), stdin=subprocess.DEVNULL)

    # Clean up the clips and the concat list
    print("\\n--- Cleaning up temporary files... ---", flush=True)
    for f in clips_dir.glob('accurate_clip_*.mp4'):
        f.unlink()
    concat_list_path.unlink()

    print(f"Fast montage created at {montage_output_path}")
    return montage_output_path 