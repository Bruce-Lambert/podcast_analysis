import json
import subprocess
from pathlib import Path

def format_time(seconds: float) -> str:
    """Converts seconds to HH:MM:SS.ms format for subtitles."""
    # The first digit is hours. Then MM:SS. two digits for centiseconds
    # e.g., 0:00:01.23
    hours = int(seconds / 3600)
    minutes = int((seconds % 3600) / 60)
    secs = seconds % 60
    return f"{hours}:{minutes:02d}:{secs:06.2f}"

def create_counter_video(
    results_path: str,
    montage_path: str,
    output_path: str,
    context_before: float = 0.5,
    context_after: float = 0.5
):
    """
    Generates a video with a 'Right' counter overlay.
    """
    results_path = Path(results_path)
    montage_path = Path(montage_path)
    output_path = Path(output_path)
    ass_path = output_path.with_suffix('.ass')

    with open(results_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    instances = [inst for inst in data.get('right_instances', []) if inst.get('speaker') == 'Dylan Patel']
    if not instances:
        print("No 'right' instances found for Dylan Patel.")
        return

    # Sort instances by start time to ensure correct counting
    instances.sort(key=lambda x: x['start_time'])
    
    # --- FIX: Calculate timestamps relative to the montage ---
    montage_events = []
    current_montage_time = 0.0
    
    for instance in instances:
        # The counter for this instance should appear at the beginning of its clip in the montage
        montage_events.append({'start': current_montage_time, 'count': len(montage_events) + 1})
        
        # Calculate the duration of this specific clip and add it to our timeline
        clip_start = instance['start_time'] - context_before
        clip_end = instance['end_time'] + context_after
        clip_duration = clip_end - clip_start
        current_montage_time += clip_duration
    # --- END FIX ---

    # Get total montage duration
    ffprobe_cmd = [
        'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1', str(montage_path)
    ]
    duration_str = subprocess.check_output(ffprobe_cmd).decode('utf-8').strip()
    total_duration = float(duration_str)

    # Generate ASS subtitle content
    ass_content = """
[Script Info]
Title: Right Counter
ScriptType: v4.00+
WrapStyle: 0
ScaledBorderAndShadow: yes
YCbCr Matrix: None

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,36,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,2,3,10,10,10,1
Style: Timestamp,Arial,24,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,1,1,9,10,10,10,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    
    if not montage_events:
        print("No events to create a counter for.")
        return

    # Initial state: Counter at 0
    source_time_text_initial = f"Source: {format_time(instances[0]['start_time'])}"
    ass_content += f"Dialogue: 0,{format_time(0)},{format_time(montage_events[0]['start'])},Default,,0,0,0,,Right Count: 0\\N\n"
    ass_content += f"Dialogue: 0,{format_time(0)},{format_time(montage_events[0]['start'])},Timestamp,,0,0,0,,{source_time_text_initial}\\N\n"

    for i, event in enumerate(montage_events):
        start_time = event['start']
        count = event['count']
        original_start_time = instances[i]['start_time'] # Get original timestamp
        source_time_text = f"Source: {format_time(original_start_time)}"
        
        # Determine the end time for this counter text
        if i + 1 < len(montage_events):
            end_time = montage_events[i+1]['start']
        else:
            # Last event runs to the end of the video
            end_time = total_duration

        # Add the line for the incremented counter and the timestamp
        ass_content += f"Dialogue: 0,{format_time(start_time)},{format_time(end_time)},Default,,0,0,0,,Right Count: {count}\\N\n"
        ass_content += f"Dialogue: 0,{format_time(start_time)},{format_time(end_time)},Timestamp,,0,0,0,,{source_time_text}\\N\n"

    with open(ass_path, 'w', encoding='utf-8') as f:
        f.write(ass_content)

    print(f"Generated corrected subtitle file at {ass_path}")

    # Use ffmpeg to burn subtitles onto the video
    ffmpeg_cmd = [
        'ffmpeg', '-y',
        '-i', str(montage_path),
        '-vf', f"ass='{str(ass_path)}'",
        '-c:v', 'libx264', '-preset', 'medium', '-crf', '23',
        '-c:a', 'aac', '-b:a', '128k',
        str(output_path)
    ]
    
    print("Burning subtitles onto video... This may take a moment.")
    subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
    print(f"Successfully created video with counter at {output_path}")

if __name__ == '__main__':
    create_counter_video(
        results_path="data/processed/full_podcast/accurate_analysis_results.json",
        montage_path="data/processed/full_podcast/video_montages_accurate/dylan_patel_right_montage_accurate.mp4",
        output_path="data/processed/full_podcast/video_montages_accurate/dylan_patel_right_montage_accurate_with_counter_and_timestamp.mp4"
    ) 