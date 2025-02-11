#!/usr/bin/env python3

import subprocess
import os
from pathlib import Path
import logging
import sys
import json
import time
import re

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('whisper_cpp_setup.log')
    ]
)

class WhisperCppSetup:
    def __init__(self):
        self.base_dir = Path(__file__).parent.parent
        self.third_party_dir = self.base_dir / "third_party"
        self.whisper_cpp_dir = self.third_party_dir / "whisper.cpp"
        self.models_dir = self.whisper_cpp_dir / "models"
        
    def setup(self):
        """Set up whisper.cpp with Metal support."""
        # Create directories
        self.third_party_dir.mkdir(exist_ok=True)
        
        # Clone or update repository
        if self.whisper_cpp_dir.exists():
            logging.info("Updating whisper.cpp repository...")
            subprocess.run(["git", "-C", str(self.whisper_cpp_dir), "pull"], check=True)
        else:
            logging.info("Cloning whisper.cpp repository...")
            subprocess.run(["git", "clone", "https://github.com/ggerganov/whisper.cpp.git", 
                          str(self.whisper_cpp_dir)], check=True)
        
        # Build with Metal support
        logging.info("Building whisper.cpp with Metal support...")
        env = os.environ.copy()
        env["WHISPER_METAL"] = "1"
        
        subprocess.run(["make"], cwd=self.whisper_cpp_dir, env=env, check=True)
        
    def download_model(self, model_name="medium"):
        """Download the model."""
        model_path = self.models_dir / f"ggml-{model_name}.bin"
        if not model_path.exists():
            logging.info(f"Downloading {model_name} model...")
            download_script = self.whisper_cpp_dir / "models" / "download-ggml-model.sh"
            subprocess.run([str(download_script), model_name], cwd=self.models_dir, check=True)
        return model_path
    
    def test_transcription(self, audio_path, model_path):
        """Test transcription with Metal acceleration."""
        logging.info("Testing transcription...")
        
        main_exe = self.whisper_cpp_dir / "build" / "bin" / "whisper-cli"
        if not main_exe.exists():
            raise FileNotFoundError(f"whisper-cli executable not found at {main_exe}")
        
        cmd = [
            str(main_exe),
            "-m", str(model_path),
            "-f", str(audio_path),
            "--output-txt",
            "-ml", "1",  # Enable Metal
            "-t", "8",   # Use 8 threads
            "-p", "4"    # 4 processors
        ]
        
        try:
            start_time = time.time()
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            duration = time.time() - start_time
            
            # Save output
            output_dir = Path("data/processed/whisper_cpp")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save the raw output
            output_path = output_dir / "transcription.txt"
            with open(output_path, "w") as f:
                f.write(result.stdout)
            
            logging.info(f"Transcription completed in {duration:.2f} seconds")
            
            # Compare with reference transcript
            self.compare_accuracy(output_path)
            
            return output_path
            
        except subprocess.CalledProcessError as e:
            logging.error("Transcription failed")
            logging.error(f"stdout: {e.stdout}")
            logging.error(f"stderr: {e.stderr}")
            raise
            
    def compare_accuracy(self, whisper_output):
        """Compare transcription accuracy with reference transcript."""
        logging.info("Comparing transcription accuracy...")
        
        # Load reference transcript
        ref_path = Path("data/raw/pasted_transcript.txt")
        if not ref_path.exists():
            logging.warning("Reference transcript not found")
            return
            
        # Load whisper output
        with open(whisper_output) as f:
            whisper_text = f.read().lower()
            
        # Load reference
        with open(ref_path) as f:
            ref_text = f.read().lower()
            
        # Count "right" occurrences
        whisper_right_count = len(re.findall(r'\bright\b', whisper_text))
        ref_right_count = len(re.findall(r'\bright\b', ref_text))
        
        # Get some context for each "right"
        whisper_rights = []
        ref_rights = []
        
        for match in re.finditer(r'.{0,50}\bright\b.{0,50}', whisper_text):
            whisper_rights.append(match.group().strip())
            
        for match in re.finditer(r'.{0,50}\bright\b.{0,50}', ref_text):
            ref_rights.append(match.group().strip())
            
        # Calculate basic accuracy metrics
        total_words_whisper = len(whisper_text.split())
        total_words_ref = len(ref_text.split())
        
        # Print comparison
        print("\nAccuracy Comparison:")
        print("-" * 50)
        print(f"Total words in whisper.cpp output: {total_words_whisper:,}")
        print(f"Total words in reference: {total_words_ref:,}")
        print(f"\nOccurrences of 'right':")
        print(f"  whisper.cpp: {whisper_right_count}")
        print(f"  reference:   {ref_right_count}")
        
        if whisper_rights:
            print("\nSample contexts from whisper.cpp:")
            for i, context in enumerate(whisper_rights[:5], 1):
                print(f"{i}. ...{context}...")
                
        # Save detailed comparison
        comparison = {
            "word_counts": {
                "whisper": total_words_whisper,
                "reference": total_words_ref
            },
            "right_counts": {
                "whisper": whisper_right_count,
                "reference": ref_right_count
            },
            "whisper_contexts": whisper_rights,
            "reference_contexts": ref_rights
        }
        
        output_dir = Path("data/processed/whisper_cpp")
        with open(output_dir / "accuracy_comparison.json", "w") as f:
            json.dump(comparison, f, indent=2)

def main():
    setup = WhisperCppSetup()
    
    try:
        # Setup and build
        setup.setup()
        
        # Download model
        model_path = setup.download_model("medium")
        
        # Test with sample audio
        audio_path = Path("data/raw/sample_segment/sample_segment.m4a")
        if audio_path.exists():
            output_path = setup.test_transcription(audio_path, model_path)
            print(f"\nTranscription successful! Output saved to: {output_path}")
        else:
            print(f"\nTest audio file not found at {audio_path}")
            print("Please ensure you have a test audio file available.")
        
    except Exception as e:
        logging.error(f"Setup failed: {str(e)}")
        raise

if __name__ == "__main__":
    main() 