import json
from pathlib import Path
from typing import List, Dict

def generate_fcpxml_from_results(
    results_path: str,
    output_fcpxml_path: str,
    video_path: str,
    project_name: str = "Right Montage Accurate",
    context_before: float = 0.5,
    context_after: float = 0.5
):
    """
    Generates an FCPXML file from an analysis results JSON file with precise timestamps.
    """
    results_path = Path(results_path)
    output_fcpxml_path = Path(output_fcpxml_path)
    video_path = Path(video_path).resolve()

    with open(results_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    instances = [inst for inst in data.get('right_instances', []) if inst.get('speaker') == 'Dylan Patel']
    if not instances:
        print("No 'right' instances found for Dylan Patel.")
        return

    # FCPXML Header
    fcpxml_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE fcpxml>
<fcpxml version="1.9">
    <resources>
        <asset id="r1" name="{video_path.name}" src="file://{video_path}" format="r2" hasVideo="1" hasAudio="1" audioSources="1" audioChannels="2" duration="{int(3600*3*60)}s"/>
        <format id="r2" name="FFVideoFormat720p24" width="1280" height="720" frameDuration="1/24s" pasp="1/1"/>
    </resources>
    <library>
        <event name="Generated Clips">
            <project name="{project_name}">
                <sequence format="r2" duration="{len(instances) * 5 * 24}/24s" tcFormat="DF">
                    <spine>
"""

    # Add clips to the spine
    for i, instance in enumerate(instances):
        start_time_s = instance['start_time'] - context_before
        end_time_s = instance['end_time'] + context_after
        duration_s = end_time_s - start_time_s
        
        # Convert to frame-based units for FCPXML (assuming 24fps)
        start_frames = int(start_time_s * 24)
        duration_frames = int(duration_s * 24)

        fcpxml_content += f"""
                        <asset-clip name="{video_path.name}" asset-id="r1" start="{start_frames}/24s" duration="{duration_frames}/24s" format="r2" tcFormat="DF" audioRole="dialogue">
                            <marker start="0s" duration="1/24s" value="Right instance {i+1}: {instance['context']}" completed="0"/>
                        </asset-clip>
"""

    # FCPXML Footer
    fcpxml_content += """
                    </spine>
                </sequence>
            </project>
        </event>
    </library>
</fcpxml>
"""

    with open(output_fcpxml_path, 'w', encoding='utf-8') as f:
        f.write(fcpxml_content)

    print(f"FCPXML file created at {output_fcpxml_path}") 