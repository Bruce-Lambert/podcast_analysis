#!/usr/bin/env python3

import json
import subprocess
from pathlib import Path
import logging
import tempfile
import re
from typing import List, Dict

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class VideoMontageCreator:
    def __init__(self, video_path: str, transcript_path: str):
        """Initialize the video montage creator."""
        self.video_path = video_path
        self.transcript_path = transcript_path
        self.video_duration = self._get_video_duration()
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        
        # Create output directories if they don't exist
        self.output_dir = Path('data/processed/video_montages')
        self.clips_dir = Path('data/processed/video_clips')
        for dir_path in [self.output_dir, self.clips_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Video duration: {self.video_duration:.2f} seconds")
        
        # Get segment start time from path
        self.segment_start = 300 if 'sample_segment' in str(self.video_path) else 0
        self.logger.info(f"Segment start time: {self.segment_start} seconds")
    
    def parse_timestamp(self, timestamp_str):
        """Parse timestamp string (HH:MM:SS) into seconds."""
        if not timestamp_str:
            return None
        try:
            # Extract timestamp from format (HH:MM:SS)
            match = re.search(r'\((\d{2}):(\d{2}):(\d{2})\)', timestamp_str)
            if not match:
                return None
            hours, minutes, seconds = map(int, match.groups())
            total_seconds = hours * 3600 + minutes * 60 + seconds
            # Convert to relative time from video start
            first_timestamp = None
            with open(self.transcript_path, 'r') as f:
                for line in f:
                    if '(' in line:
                        match = re.search(r'\((\d{2}):(\d{2}):(\d{2})\)', line)
                        if match:
                            h, m, s = map(int, match.groups())
                            first_timestamp = h * 3600 + m * 60 + s
                            break
            if first_timestamp is None:
                return None
            relative_seconds = total_seconds - first_timestamp
            logging.debug(f"Timestamp: {timestamp_str} -> {total_seconds}s -> relative: {relative_seconds}s")
            return relative_seconds
        except Exception as e:
            logging.error(f"Error parsing timestamp {timestamp_str}: {e}")
            return None
    
    def is_discourse_marker(self, text: str) -> bool:
        """Check if 'right' is used as a discourse marker in the given text."""
        # Simple heuristic - check if "right" appears at the start or end of a phrase
        # or is surrounded by punctuation/pauses
        text = text.lower()
        if not 'right' in text:
            return False
            
        # Check for "right?" or "right!" or "right." at the end
        if re.search(r'right[?.!]?\s*$', text):
            return True
            
        # Check for "right" after common pause markers
        if re.search(r'[,.!?]\s+right\b', text):
            return True
            
        # Check for "right" followed by common continuations
        if re.search(r'\bright\b\s*(?:so|and|but|well)', text):
            return True
            
        return False
    
    def find_right_instances(self):
        """Find instances of when Dylan says 'right' in the transcript."""
        instances = []
        current_speaker = None
        
        with open(self.transcript_path, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check if this is a speaker line
            if not line.startswith('('):
                current_speaker = line
                continue
                
            # Only process Dylan's lines
            if current_speaker and current_speaker.lower() != 'dylan patel':
                continue
                
            # Get timestamp
            timestamp_match = re.search(r'\((\d{2}:\d{2}:\d{2})\)', line)
            if not timestamp_match:
                continue
                
            # Get the text after timestamp
            text = re.sub(r'\(\d{2}:\d{2}:\d{2}\)\s*', '', line)
            
            # Find all instances of "right" in this line
            for match in re.finditer(r'\bright\b', text.lower()):
                # Get the timestamp for this instance
                line_timestamp = self.parse_timestamp(timestamp_match.group(0))
                if line_timestamp is None or line_timestamp >= self.video_duration:
                    continue
                
                # Calculate more precise timestamp for this instance of "right"
                text_before = text[:match.start()]
                words_before = len(text_before.split())
                # Assume average speaking rate of 3 words per second
                time_offset = words_before / 3.0
                instance_timestamp = line_timestamp + time_offset
                
                # Get context around "right"
                start_idx = max(0, match.start() - 100)
                end_idx = min(len(text), match.end() + 100)
                context = text[start_idx:end_idx]
                
                # Only include if it's a discourse marker
                if self.is_discourse_marker(text[max(0, match.start()-20):min(len(text), match.end()+20)]):
                    instances.append({
                        'timestamp': instance_timestamp,
                        'context': context,
                        'full_text': text,
                        'word_start': match.start(),
                        'word_end': match.end()
                    })
                    logging.info(f"Found 'right' at {instance_timestamp:.1f}s: {context}")
        
        # Sort instances by timestamp
        instances.sort(key=lambda x: x['timestamp'])
        
        # Log all instances
        logging.info(f"Found {len(instances)} instances of Dylan saying 'right'")
        for i, instance in enumerate(instances):
            logging.info(f"Instance {i+1}: {instance['timestamp']:.1f}s - {instance['context']}")
        
        return instances
    
    def create_montage(self, instances: List[Dict], output_name: str,
                     context_before: float = 1.0,
                     context_after: float = 1.0) -> Path:
        """Create a montage of the selected instances."""
        output_file = self.output_dir / output_name
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            concat_file = temp_path / "concat.txt"
            
            # Create a 1-second black video for transition
            transition = temp_path / "transition.mp4"
            cmd = [
                'ffmpeg', '-y',
                '-f', 'lavfi',
                '-i', 'color=c=black:s=1280x720:d=1.0',
                '-c:v', 'libx264',
                '-preset', 'ultrafast',
                str(transition)
            ]
            subprocess.run(cmd, capture_output=True, check=True)
            
            # Extract each segment
            for i, instance in enumerate(instances):
                start_time = max(0, instance['timestamp'] - context_before)
                duration = context_before + context_after
                output_segment = temp_path / f"segment_{i:03d}.mp4"
                
                # Extract segment using ffmpeg with better seeking
                cmd = [
                    'ffmpeg', '-y',
                    '-ss', str(start_time),
                    '-i', str(self.video_path),
                    '-t', str(duration),
                    '-c:v', 'libx264', '-preset', 'ultrafast',
                    '-c:a', 'aac',
                    str(output_segment)
                ]
                subprocess.run(cmd, capture_output=True, check=True)
                logging.info(f"Extracted clip {i+1} from {start_time:.1f}s to {start_time + duration:.1f}s")
                
                # Write segment path to concat file
                with open(concat_file, 'a') as f:
                    f.write(f"file '{output_segment}'\n")
                    # Add transition between clips (except after the last one)
                    if i < len(instances) - 1:
                        f.write(f"file '{transition}'\n")
            
            # Concatenate all segments with proper encoding
            cmd = [
                'ffmpeg', '-y',
                '-f', 'concat',
                '-safe', '0',
                '-i', str(concat_file),
                '-c:v', 'libx264',
                '-preset', 'medium',
                '-crf', '23',
                '-c:a', 'aac',
                '-b:a', '128k',
                '-ar', '44100',
                '-movflags', '+faststart',
                str(output_file)
            ]
            subprocess.run(cmd, capture_output=True, check=True)
            
            logging.info(f"Created montage with {len(instances)} clips at {output_file}")
            
        return output_file

    def create_rapid_montage(self, num_examples: int = 20):
        """Create a rapid-fire montage of Dylan saying 'right'."""
        instances = self.find_right_instances()[:num_examples]
        logging.info(f"Creating rapid montage with {len(instances)} instances")
        # Use asymmetric timing to capture the build-up and the word clearly
        # More time after to ensure we hear "right" completely
        source_type = "full_podcast" if "full_podcast" in str(self.video_path) else "sample"
        return self.create_montage(instances, f"{source_type}_rapid_montage_{num_examples}.mp4", 
                                 context_before=0.75, context_after=1.0)
    
    def create_context_montage(self, num_examples: int = 10):
        """Create a montage with more context around each instance."""
        instances = self.find_right_instances()[:num_examples]
        # Filter out instances that are too close together (within 3 seconds)
        filtered_instances = []
        for i, instance in enumerate(instances):
            if i == 0 or instance['timestamp'] - filtered_instances[-1]['timestamp'] > 3.0:
                filtered_instances.append(instance)
        
        logging.info(f"Creating context montage with {len(filtered_instances)} instances after filtering")
        source_type = "full_podcast" if "full_podcast" in str(self.video_path) else "sample"
        return self.create_montage(filtered_instances, f"{source_type}_context_montage_{num_examples}.mp4", 
                                 context_before=6.0, context_after=3.0)

    def _get_video_duration(self) -> float:
        """Get the duration of the video in seconds."""
        cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
               '-of', 'default=noprint_wrappers=1:nokey=1', str(self.video_path)]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(result.stdout.strip())

    def create_individual_clips(self, context_before: float = 5.0, context_after: float = 1.0):
        """Create individual clips for each instance of 'right'."""
        instances = self.find_right_instances()
        source_type = "full_podcast" if "full_podcast" in str(self.video_path) else "sample"
        
        self.logger.info(f"Creating {len(instances)} individual clips with {context_before}s before and {context_after}s after")
        
        for i, instance in enumerate(instances):
            timestamp = instance['timestamp']
            # Format timestamp as HH:MM:SS
            timestamp_str = f"{int(timestamp/3600):02d}_{int((timestamp%3600)/60):02d}_{int(timestamp%60):02d}"
            
            output_file = self.clips_dir / f"{source_type}_right_{i+1:03d}_at_{timestamp_str}.mp4"
            
            start_time = max(0, timestamp - context_before)
            duration = context_before + context_after
            
            # Extract clip using ffmpeg with better seeking and higher quality settings
            cmd = [
                'ffmpeg', '-y',
                '-ss', str(start_time),
                '-i', str(self.video_path),
                '-t', str(duration),
                '-c:v', 'libx264',
                '-preset', 'veryslow',  # Higher quality encoding
                '-crf', '18',           # Higher quality (lower value = better quality)
                '-c:a', 'aac',
                '-b:a', '192k',         # Better audio quality
                '-ar', '48000',         # Higher audio sample rate
                '-vsync', 'vfr',        # Variable framerate for better sync
                '-movflags', '+faststart',
                str(output_file)
            ]
            
            try:
                subprocess.run(cmd, capture_output=True, check=True)
                self.logger.info(f"Created clip {i+1} at {timestamp_str} ({timestamp:.1f}s)")
                self.logger.info(f"Context: {instance['context']}")
            except subprocess.CalledProcessError as e:
                self.logger.error(f"Error creating clip {i+1}: {e}")
                continue

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Create video clips of discourse marker 'right' instances")
    parser.add_argument('--video', required=True, help='Path to the video file')
    parser.add_argument('--transcript', required=True, help='Path to the pasted transcript file')
    args = parser.parse_args()
    
    try:
        creator = VideoMontageCreator(args.video, args.transcript)
        creator.create_individual_clips()
    except Exception as e:
        logging.error(f"Error creating clips: {str(e)}")
        raise

if __name__ == "__main__":
    main() 