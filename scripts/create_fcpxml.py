#!/usr/bin/env python3

import datetime
from pathlib import Path

def timecode_to_seconds(timecode):
    """Convert HH:MM:SS timecode to seconds."""
    time_obj = datetime.datetime.strptime(timecode.strip(), '%H:%M:%S')
    return time_obj.hour * 3600 + time_obj.minute * 60 + time_obj.second

def create_fcpxml(summary_file, output_file, video_file):
    """Create FCPXML file from summary file."""
    # FCPXML header
    xml_content = '''<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE fcpxml>
<fcpxml version="1.9">
    <resources>
        <format id="r1" name="FFVideoFormat1080p30" frameDuration="1/30s"/>
        <asset id="r2" name="full_video" src="file://{}"/>
    </resources>
    <library>
        <event name="Right Analysis">
            <project name="Dylan Patel Right Usage">
                <sequence format="r1">
                    <spine>'''.format(video_file)
    
    # Read summary file and create clips
    with open(summary_file, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
                
            # Parse line
            parts = line.strip().split('|')
            if len(parts) < 3:
                continue
                
            clip_name = parts[0].strip()
            start_time = timecode_to_seconds(parts[1])
            end_time = timecode_to_seconds(parts[2])
            
            # Create clip XML
            clip_xml = f'''
                        <asset-clip name="{clip_name}" 
                                   ref="r2" 
                                   offset="{start_time}s" 
                                   duration="{end_time - start_time}s" 
                                   start="{start_time}s"/>'''
            xml_content += clip_xml
    
    # Close XML
    xml_content += '''
                    </spine>
                </sequence>
            </project>
        </event>
    </library>
</fcpxml>'''
    
    # Write output file
    with open(output_file, 'w') as f:
        f.write(xml_content)

def main():
    # File paths
    summary_file = Path('data/processed/full_podcast/right_instances_summary.txt')
    output_file = Path('data/processed/full_podcast/right_instances.fcpxml')
    video_file = Path('data/raw/full_podcast/full_video.mp4').absolute()
    
    create_fcpxml(summary_file, output_file, video_file)
    print(f"Created FCPXML file at {output_file}")

if __name__ == '__main__':
    main() 