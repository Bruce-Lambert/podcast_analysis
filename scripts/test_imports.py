#!/usr/bin/env python3

import sys
import subprocess
from pathlib import Path

def test_imports():
    """Test importing all required packages."""
    print("Testing imports...")
    try:
        import yt_dlp
        import ffmpeg
        import whisper
        import torch
        import numpy as np
        import matplotlib.pyplot as plt
        
        print("✓ All core packages imported successfully")
        print(f"Python version: {sys.version}")
        print(f"PyTorch version: {torch.__version__}")
        print(f"NumPy version: {np.__version__}")
        return True
    except ImportError as e:
        print(f"✗ Import error: {str(e)}")
        return False

def test_ffmpeg():
    """Test ffmpeg installation."""
    print("\nTesting ffmpeg...")
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, 
                              text=True, 
                              check=True)
        version = result.stdout.split('\n')[0]
        print(f"✓ ffmpeg installed: {version}")
        return True
    except Exception as e:
        print(f"✗ ffmpeg error: {str(e)}")
        return False

def test_file_access():
    """Test access to key project files."""
    print("\nTesting file access...")
    base_dir = Path(__file__).parent.parent
    files_to_check = [
        'environment.yml',
        'data/raw/full_podcast/full_video.mp4',
        'data/processed/full_podcast/whisper_output.json'
    ]
    
    all_good = True
    for file_path in files_to_check:
        path = base_dir / file_path
        if path.exists():
            size = path.stat().st_size / (1024 * 1024)  # Convert to MB
            print(f"✓ Found {file_path} ({size:.1f} MB)")
        else:
            print(f"✗ Missing {file_path}")
            all_good = False
    return all_good

def main():
    print("Running system check...\n")
    
    imports_ok = test_imports()
    ffmpeg_ok = test_ffmpeg()
    files_ok = test_file_access()
    
    print("\nTest Summary:")
    print("-------------")
    print(f"Imports: {'✓' if imports_ok else '✗'}")
    print(f"FFmpeg: {'✓' if ffmpeg_ok else '✗'}")
    print(f"Files: {'✓' if files_ok else '✗'}")
    
    if imports_ok and ffmpeg_ok and files_ok:
        print("\n✓ All systems ready!")
        return 0
    else:
        print("\n✗ Some tests failed. Please check the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 