#!/usr/bin/env python3

import whisper
import json
from pathlib import Path
import logging
import torch
import signal
import sys
from tqdm import tqdm
import time
import threading

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('transcription.log')
    ]
)

# Global variables
current_result = None
progress_bar = None
transcription_started = False
last_segment_time = 0

def update_progress():
    """Update the progress bar based on the output file size."""
    global progress_bar, transcription_started, last_segment_time
    
    output_dir = Path('data/processed/full_podcast')
    temp_file = output_dir / 'whisper_temp.json'
    
    while not transcription_started:
        time.sleep(0.1)
    
    while progress_bar is not None:
        try:
            if temp_file.exists():
                with open(temp_file, 'r') as f:
                    data = json.load(f)
                    if 'segments' in data and data['segments']:
                        current_time = data['segments'][-1]['end']
                        if current_time > last_segment_time:
                            last_segment_time = current_time
                            hours = int(current_time // 3600)
                            minutes = int((current_time % 3600) // 60)
                            seconds = int(current_time % 60)
                            progress_bar.set_description(
                                f"Transcribing [{hours:02d}:{minutes:02d}:{seconds:02d}]"
                            )
                            # Estimate progress based on video duration
                            if hasattr(progress_bar, 'duration') and progress_bar.duration:
                                progress = min(100, int((current_time / progress_bar.duration) * 100))
                                progress_bar.n = progress
                                progress_bar.refresh()
        except Exception as e:
            pass
        time.sleep(1)

def signal_handler(signum, frame):
    """Handle interrupt signals by saving current progress."""
    if current_result is not None:
        logging.info('Received interrupt signal. Saving partial progress...')
        output_path = Path('data/processed/full_podcast/whisper_output_partial.json')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(current_result, f, indent=2)
        
        logging.info(f'Partial progress saved to {output_path}')
    
    if progress_bar:
        progress_bar.close()
    
    sys.exit(1)

def get_video_duration(video_path):
    """Get the duration of the video using ffmpeg."""
    try:
        import ffmpeg
        probe = ffmpeg.probe(str(video_path))
        duration = float(probe['streams'][0]['duration'])
        return duration
    except Exception as e:
        logging.warning(f"Could not get video duration: {e}")
        return None

def transcribe_video():
    """Transcribe the full video using Whisper."""
    global current_result, progress_bar, transcription_started
    
    try:
        # Register signal handlers
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Log GPU availability
        if torch.cuda.is_available():
            device = torch.cuda.get_device_name(0)
            logging.info(f'Using GPU: {device}')
        else:
            logging.info('No GPU available, using CPU')

        logging.info('Loading Whisper model...')
        model = whisper.load_model('medium')

        video_path = Path('data/raw/full_podcast/full_video.mp4')
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found at {video_path}")

        logging.info(f'Starting transcription of full video: {video_path}')
        video_size_gb = video_path.stat().st_size / (1024*1024*1024)
        logging.info(f'Video size: {video_size_gb:.1f} GB')
        
        # Get video duration for progress tracking
        duration = get_video_duration(video_path)
        if duration:
            hours = int(duration // 3600)
            minutes = int((duration % 3600) // 60)
            seconds = int(duration % 60)
            logging.info(f'Video duration: {hours:02d}:{minutes:02d}:{seconds:02d}')
        
        # Initialize progress bar
        progress_bar = tqdm(total=100, desc="Transcribing", unit="%")
        if duration:
            progress_bar.duration = duration
        
        # Start progress monitoring thread
        progress_thread = threading.Thread(target=update_progress, daemon=True)
        progress_thread.start()
        
        # Load Whisper settings from cursor rules
        settings = {
            'language': 'en',
            'word_timestamps': True,
            'verbose': False  # Set to False to avoid cluttering the progress bar
        }
        
        # Signal that transcription is starting
        transcription_started = True
        
        current_result = model.transcribe(
            str(video_path),
            **settings
        )

        # Close progress bar
        if progress_bar:
            progress_bar.close()
            progress_bar = None

        logging.info('Saving complete transcript...')
        output_path = Path('data/processed/full_podcast/whisper_output.json')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(current_result, f, indent=2)

        logging.info('Transcription complete!')
        
        # Log some statistics
        num_segments = len(current_result['segments'])
        total_duration = current_result['segments'][-1]['end'] if current_result['segments'] else 0
        logging.info(f'Generated {num_segments} segments')
        logging.info(f'Total duration: {total_duration/60:.1f} minutes')
        
        return True

    except KeyboardInterrupt:
        logging.info('Received keyboard interrupt')
        signal_handler(signal.SIGINT, None)
    except Exception as e:
        logging.error(f'Error during transcription: {str(e)}')
        if progress_bar:
            progress_bar.close()
            progress_bar = None
        raise
    finally:
        if progress_bar:
            progress_bar.close()
            progress_bar = None

if __name__ == '__main__':
    transcribe_video() 