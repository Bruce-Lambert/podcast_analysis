#!/usr/bin/env python3

import json
import subprocess
from pathlib import Path
import logging
import tempfile
import shutil

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AudioMontageCreator:
    def __init__(self, raw_dir="data/raw", processed_dir="data/processed"):
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.audio_montages_dir = self.processed_dir / "audio_montages"
        self.audio_montages_dir.mkdir(parents=True, exist_ok=True)
        
    def load_segments(self):
        """Load the audio segments from the JSON file."""
        segments_path = self.processed_dir / "audio_segments.json"
        with open(segments_path, 'r') as f:
            return json.load(f)
            
    def create_rapid_montage(self):
        """Create a rapid-fire audio montage of 'right' instances."""
        logging.info("Creating rapid-fire audio montage...")
        
        # Load segments
        segments_data = self.load_segments()
        rapid_segments = segments_data['rapid_segments']
        
        if not rapid_segments:
            logging.warning("No rapid segments found!")
            return
            
        # Create temporary directory for intermediate files
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Extract each segment
            segment_files = []
            for i, segment in enumerate(rapid_segments, 1):
                output_file = temp_path / f"rapid_{i}.m4a"
                segment_files.append(output_file)
                
                # Extract segment using ffmpeg
                cmd = [
                    'ffmpeg', '-y',
                    '-i', str(self.raw_dir / "sample_segment.m4a"),
                    '-ss', str(segment['start']),
                    '-t', str(segment['end'] - segment['start']),
                    str(output_file)
                ]
                
                logging.info(f"Extracting segment {i}/{len(rapid_segments)}")
                subprocess.run(cmd, check=True, capture_output=True)
            
            # Create concat file
            concat_file = temp_path / "concat.txt"
            with open(concat_file, 'w') as f:
                for file in segment_files:
                    f.write(f"file '{file.name}'\n")
            
            # Concatenate all segments
            output_path = self.audio_montages_dir / "rapid_audio_montage.m4a"
            cmd = [
                'ffmpeg', '-y',
                '-f', 'concat',
                '-safe', '0',
                '-i', str(concat_file),
                '-c', 'copy',
                str(output_path)
            ]
            
            logging.info("Concatenating segments into final audio montage...")
            subprocess.run(cmd, check=True, capture_output=True)
            
            logging.info(f"Rapid audio montage created at {output_path}")
            
            # Calculate total duration
            total_duration = sum(seg['segment_duration'] for seg in rapid_segments)
            logging.info(f"Total audio montage duration: {total_duration:.1f} seconds")
            
    def create_context_montage(self):
        """Create a montage with full context for each 'right' instance."""
        logging.info("Creating context audio montage...")
        
        # Load segments
        segments_data = self.load_segments()
        context_segments = segments_data['context_segments']
        
        if not context_segments:
            logging.warning("No context segments found!")
            return
            
        # Create temporary directory for intermediate files
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Extract each segment
            segment_files = []
            for i, segment in enumerate(context_segments, 1):
                output_file = temp_path / f"context_{i}.m4a"
                segment_files.append(output_file)
                
                # Extract segment using ffmpeg
                cmd = [
                    'ffmpeg', '-y',
                    '-i', str(self.raw_dir / "sample_segment.m4a"),
                    '-ss', str(segment['start']),
                    '-t', str(segment['end'] - segment['start']),
                    str(output_file)
                ]
                
                logging.info(f"Extracting segment {i}/{len(context_segments)}")
                subprocess.run(cmd, check=True, capture_output=True)
            
            # Create concat file
            concat_file = temp_path / "concat.txt"
            with open(concat_file, 'w') as f:
                for file in segment_files:
                    f.write(f"file '{file.name}'\n")
            
            # Concatenate all segments
            output_path = self.audio_montages_dir / "context_audio_montage.m4a"
            cmd = [
                'ffmpeg', '-y',
                '-f', 'concat',
                '-safe', '0',
                '-i', str(concat_file),
                '-c', 'copy',
                str(output_path)
            ]
            
            logging.info("Concatenating segments into final audio montage...")
            subprocess.run(cmd, check=True, capture_output=True)
            
            logging.info(f"Context audio montage created at {output_path}")
            
            # Calculate total duration
            total_duration = sum(seg['segment_duration'] for seg in context_segments)
            logging.info(f"Total audio montage duration: {total_duration:.1f} seconds")

def main():
    creator = AudioMontageCreator()
    
    try:
        # Create both types of audio montages
        creator.create_rapid_montage()
        creator.create_context_montage()
        
        logging.info("Audio montage creation completed successfully!")
        
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main() 