#!/usr/bin/env python3

import json
from pathlib import Path
import datetime

def format_seconds(seconds):
    """Convert seconds to HH:MM:SS format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f'{hours:02d}:{minutes:02d}:{secs:02d}'

def create_summary():
    """Create a summary file from the transcription log."""
    # Read the log file
    with open('transcription.log', 'r') as f:
        log_content = f.read()

    # Extract instances
    instances = []
    for line in log_content.split('\n'):
        if 'Found \'right\' at' in line:
            time_str = line.split('Found \'right\' at ')[1].split('s:')[0]
            time = float(time_str)
            context = line.split('s: ')[1]
            instances.append({
                'time': time,
                'context': context
            })

    # Create summary file
    output_file = Path('data/processed/full_podcast/right_instances_summary.txt')
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        f.write('# Dylan Patel "Right" Usage Summary\n')
        f.write('# Format: [Clip Name] | [Start Time (HH:MM:SS)] | [End Time (HH:MM:SS)] | [Context]\n\n')
        
        for i, instance in enumerate(instances, 1):
            start_time = max(0, instance['time'] - 5)  # 5 seconds before
            end_time = instance['time'] + 1  # 1 second after
            
            f.write(f'full_podcast_right_{i:03d} | {format_seconds(start_time)} | {format_seconds(end_time)} | {instance["context"]}\n')
        
        f.write(f'\n# Total instances: {len(instances)}\n')
        f.write('# Format suitable for FCPXML conversion\n')
        f.write('# Each clip has 5 seconds before and 1 second after the "right" instance')

if __name__ == '__main__':
    create_summary() 