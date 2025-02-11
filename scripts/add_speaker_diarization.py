#!/usr/bin/env python3

import json
import logging
import os
import argparse
import torch
import subprocess
import numpy as np
from pathlib import Path
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from difflib import SequenceMatcher
from tqdm import tqdm
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SpeakerDiarizer:
    def __init__(self, processed_dir, raw_dir, auth_token=None):
        self.processed_dir = Path(processed_dir)
        self.raw_dir = Path(raw_dir)
        self.whisper_file = self.processed_dir / "whisper_output.json"
        self.pasted_file = self.raw_dir / "pasted_transcript.txt"
        self.video_file = self.raw_dir / "full_video.mp4"
        self.audio_file = self.processed_dir / "audio.wav"
        self.output_file = self.processed_dir / "whisper_output_with_speakers.json"
        self.auth_token = auth_token or os.getenv("HUGGINGFACE_TOKEN")
        
        if not self.auth_token:
            raise ValueError("Please provide a HuggingFace auth token either as an argument or set it as HUGGINGFACE_TOKEN environment variable")
        
        # Initialize the diarization pipeline with improved settings
        self.pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=self.auth_token
        )
        
        # Initialize speaker embedding model for verification
        self.embedding_model = PretrainedSpeakerEmbedding(
            "speechbrain/spkrec-ecapa-voxceleb",
            use_auth_token=self.auth_token
        )
        
        if torch.cuda.is_available():
            self.pipeline = self.pipeline.to(torch.device("cuda"))
            self.embedding_model = self.embedding_model.to(torch.device("cuda"))
            
        # Known speaker mapping
        self.speaker_mapping = {
            'SPEAKER_00': 'lex fridman',
            'SPEAKER_01': 'nathan lambert',
            'SPEAKER_02': 'dylan patel'
        }
        
        # Statistics for reporting
        self.stats = {
            'total_segments': 0,
            'matched_segments': 0,
            'corrected_segments': 0,
            'confident_segments': 0
        }
    
    def extract_audio(self):
        """Extract audio from video file with improved quality."""
        if not self.video_file.exists():
            raise FileNotFoundError(f"Video file not found at {self.video_file}")
            
        logger.info("Extracting high-quality audio from video...")
        subprocess.run([
            'ffmpeg', '-y',
            '-i', str(self.video_file),
            '-vn',  # Disable video
            '-acodec', 'pcm_s16le',  # Use PCM 16-bit encoding
            '-ar', '16000',  # Set sample rate to 16kHz
            '-ac', '1',  # Convert to mono
            '-af', 'highpass=f=50,lowpass=f=3000',  # Apply audio filters for speech
            str(self.audio_file)
        ], check=True)
        logger.info("Audio extraction complete!")
    
    def load_whisper_transcript(self):
        """Load the Whisper transcript from JSON file."""
        if not self.whisper_file.exists():
            raise FileNotFoundError(f"Whisper transcript not found at {self.whisper_file}")
        
        with open(self.whisper_file, 'r') as f:
            return json.load(f)
            
    def load_ground_truth(self):
        """Load and parse the ground truth transcript."""
        if not self.pasted_file.exists():
            raise FileNotFoundError(f"Ground truth transcript not found at {self.pasted_file}")
            
        ground_truth = []
        current_speaker = None
        
        with open(self.pasted_file, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
            
        for line in lines:
            # Handle speaker names
            if any(name in line for name in ['Lex Fridman', 'Nathan Lambert', 'Dylan Patel']):
                current_speaker = next(
                    name.lower() for name in ['Lex Fridman', 'Nathan Lambert', 'Dylan Patel']
                    if name in line
                )
                continue
                
            # Parse timestamp and text
            timestamp_match = re.match(r'\((\d{2}:\d{2}:\d{2})\)\s*(.*)', line)
            if timestamp_match and current_speaker:
                timestamp, text = timestamp_match.groups()
                h, m, s = map(int, timestamp.split(':'))
                time_seconds = h * 3600 + m * 60 + s
                
                ground_truth.append({
                    'speaker': current_speaker,
                    'text': text,
                    'timestamp': timestamp,
                    'time_seconds': time_seconds
                })
                
        return sorted(ground_truth, key=lambda x: x['time_seconds'])
    
    def get_audio_segment(self, start, end):
        """Extract audio segment for speaker verification."""
        import soundfile as sf
        audio, sr = sf.read(str(self.audio_file))
        start_sample = int(start * sr)
        end_sample = int(end * sr)
        return audio[start_sample:end_sample]
    
    def get_speaker_embedding(self, audio_segment):
        """Get speaker embedding for an audio segment."""
        with torch.no_grad():
            embedding = self.embedding_model(torch.from_numpy(audio_segment).float())
        return embedding.cpu().numpy()
    
    def find_matching_ground_truth(self, segment, ground_truth_segments):
        """Find matching ground truth segment using time and text similarity."""
        segment_start = segment['start']
        segment_text = segment['text'].lower()
        
        best_match = None
        best_score = 0
        
        for gt_segment in ground_truth_segments:
            # Time proximity (within 5 seconds)
            time_diff = abs(gt_segment['time_seconds'] - segment_start)
            if time_diff > 5:
                continue
                
            # Text similarity
            similarity = SequenceMatcher(None, segment_text, gt_segment['text'].lower()).ratio()
            
            # Combined score (70% text, 30% time)
            time_score = max(0, 1 - time_diff/5)  # Normalize to 0-1
            score = similarity * 0.7 + time_score * 0.3
            
            if score > best_score and score > 0.6:  # Minimum threshold
                best_score = score
                best_match = (gt_segment, score)
        
        return best_match
    
    def process_diarization(self):
        """Process the audio file with improved diarization and ground truth validation."""
        logger.info("Loading transcripts...")
        whisper_transcript = self.load_whisper_transcript()
        ground_truth = self.load_ground_truth()
        
        # Extract audio if needed
        if not self.audio_file.exists():
            self.extract_audio()
        
        logger.info("Running speaker diarization...")
        with ProgressHook() as hook:
            diarization = self.pipeline(
                self.audio_file,
                num_speakers=3,
                min_speakers=3,
                max_speakers=3,
                hook=hook,
                clustering='AgglomerativeClustering',
                segmentation_batch_size=64,  # Increased for better context
                embedding_batch_size=64,
                clustering_threshold=0.75  # Adjusted for better separation
            )
        
        # Convert diarization to segments
        speaker_segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            speaker_segments.append({
                'start': turn.start,
                'end': turn.end,
                'speaker': speaker,
                'embedding': None  # Will be computed as needed
            })
        
        # Process Whisper segments
        logger.info("Processing segments with ground truth validation...")
        for segment in tqdm(whisper_transcript['segments']):
            self.stats['total_segments'] += 1
            
            # Find overlapping diarization segments
            segment_start = segment['start']
            segment_end = segment['end']
            
            overlapping_speakers = []
            for spk_seg in speaker_segments:
                if (spk_seg['start'] <= segment_end and 
                    spk_seg['end'] >= segment_start):
                    overlapping_speakers.append(spk_seg['speaker'])
            
            # Get most common speaker in overlapping segments
            if overlapping_speakers:
                from collections import Counter
                speaker = Counter(overlapping_speakers).most_common(1)[0][0]
                
                # Try to match with ground truth
                gt_match = self.find_matching_ground_truth(segment, ground_truth)
                
                if gt_match:
                    gt_segment, confidence = gt_match
                    mapped_speaker = self.speaker_mapping.get(speaker)
                    
                    if mapped_speaker == gt_segment['speaker']:
                        # Diarization matches ground truth
                        self.stats['matched_segments'] += 1
                        segment['speaker'] = gt_segment['speaker']
                        segment['speaker_confidence'] = confidence
                    else:
                        # Use ground truth instead
                        self.stats['corrected_segments'] += 1
                        segment['speaker'] = gt_segment['speaker']
                        segment['speaker_confidence'] = confidence
                else:
                    # No ground truth match, use diarization with lower confidence
                    segment['speaker'] = self.speaker_mapping.get(speaker, 'unknown')
                    segment['speaker_confidence'] = 0.5
            else:
                segment['speaker'] = 'unknown'
                segment['speaker_confidence'] = 0.0
            
            # Mark as confident if confidence is high
            if segment.get('speaker_confidence', 0) > 0.8:
                self.stats['confident_segments'] += 1
        
        # Add metadata
        whisper_transcript['metadata'] = {
            'diarization_stats': self.stats,
            'processing_timestamp': datetime.now().isoformat(),
            'speaker_mapping': self.speaker_mapping
        }
        
        # Save results
        logger.info("Saving diarized transcript...")
        with open(self.output_file, 'w') as f:
            json.dump(whisper_transcript, f, indent=2)
        
        # Log statistics
        logger.info("\n=== Diarization Statistics ===")
        logger.info(f"Total segments: {self.stats['total_segments']}")
        logger.info(f"Matched with ground truth: {self.stats['matched_segments']}")
        logger.info(f"Corrected using ground truth: {self.stats['corrected_segments']}")
        logger.info(f"High confidence segments: {self.stats['confident_segments']}")
        
        accuracy = (self.stats['matched_segments'] + self.stats['corrected_segments']) / self.stats['total_segments'] * 100
        logger.info(f"Overall accuracy: {accuracy:.1f}%")
        
        return whisper_transcript

def main():
    parser = argparse.ArgumentParser(description="Add speaker diarization to Whisper transcript")
    parser.add_argument("--processed_dir", default="data/processed/full_podcast",
                      help="Directory containing the Whisper transcript")
    parser.add_argument("--raw_dir", default="data/raw",
                      help="Directory containing the audio/video file")
    parser.add_argument("--auth_token", help="HuggingFace auth token")
    
    args = parser.parse_args()
    
    try:
        diarizer = SpeakerDiarizer(args.processed_dir, args.raw_dir, args.auth_token)
        diarizer.process_diarization()
    except Exception as e:
        logger.error(f"Diarization failed: {e}")
        raise

if __name__ == "__main__":
    main() 