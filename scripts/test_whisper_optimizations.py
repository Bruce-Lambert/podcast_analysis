#!/usr/bin/env python3

import whisper
import torch
import subprocess
import json
import time
import psutil
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from faster_whisper import WhisperModel
import pandas as pd
import GPUtil
from tqdm import tqdm
import signal
import sys
import threading
import os
from contextlib import contextmanager
from typing import Dict, Any
try:
    import mlx.core as mx
    import mlx_whisper
    HAS_MLX = True
except ImportError:
    HAS_MLX = False

# Set environment variable to avoid Numba threading issues
os.environ["NUMBA_NUM_THREADS"] = "1"

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('optimization_test.log')
    ]
)

class TimeoutException(Exception):
    pass

@contextmanager
def timeout(seconds):
    """Context manager for timing out operations."""
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    
    # Register a function to raise a TimeoutException on the signal
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    
    try:
        yield
    finally:
        # Disable the alarm
        signal.alarm(0)

class WhisperOptimizationTester:
    def __init__(self, sample_duration=600):  # 600 seconds = 10 minutes
        """Initialize the optimization tester."""
        self.base_dir = Path(__file__).parent.parent
        self.data_dir = self.base_dir / "data"
        self.raw_dir = self.data_dir / "raw"
        self.test_dir = self.data_dir / "optimization_test"
        self.test_dir.mkdir(parents=True, exist_ok=True)
        
        # Test parameters
        self.sample_duration = sample_duration
        self.models_to_test = ['small', 'medium']
        self.results = []
        self.timeout_seconds = 900  # 15 minutes timeout
        
        # Initialize test sample path
        self.sample_path = self.test_dir / "test_sample.mp4"
        
    def prepare_test_sample(self):
        """Extract a test sample from the full video."""
        logging.info("Preparing test sample...")
        
        source_video = self.raw_dir / "full_podcast" / "full_video.mp4"
        if not source_video.exists():
            raise FileNotFoundError(f"Source video not found at {source_video}")
        
        # Extract sample using ffmpeg
        cmd = [
            'ffmpeg', '-y',
            '-i', str(source_video),
            '-t', str(self.sample_duration),
            '-c', 'copy',
            str(self.sample_path)
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            logging.info(f"Created {self.sample_duration}s test sample at {self.sample_path}")
        except subprocess.CalledProcessError as e:
            logging.error(f"Error creating test sample: {e.stderr.decode()}")
            raise
    
    def get_system_metrics(self):
        """Get current system metrics."""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        # Get GPU metrics if available
        gpu_util = 0
        gpu_memory = 0
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                gpu_util = gpu.load * 100
                gpu_memory = gpu.memoryUsed
        except:
            pass
        
        return {
            'timestamp': time.time(),
            'cpu_percent': cpu_percent,
            'memory_mb': memory,
            'gpu_util': gpu_util,
            'gpu_memory': gpu_memory
        }
    
    def patch_dtw(self):
        """Patch the DTW function to handle MPS tensors and ensure float32 precision."""
        import whisper.timing
        import torch
        import numpy as np
        
        original_dtw = whisper.timing.dtw

        def dtw_cpu(x):
            """CPU implementation of DTW that works with numpy arrays."""
            if len(x.shape) != 2:
                raise ValueError("expected 2D array")
            
            N, M = x.shape
            D = np.full((N + 1, M + 1), np.inf)
            D[0, 0] = 0
            
            # Forward pass
            for i in range(1, N + 1):
                for j in range(1, M + 1):
                    D[i, j] = x[i - 1, j - 1] + min(
                        D[i - 1, j],     # insertion
                        D[i, j - 1],     # deletion
                        D[i - 1, j - 1]  # match
                    )
            
            # Backtrace
            i, j = N, M
            path = [(i - 1, j - 1)]
            
            while i > 1 or j > 1:
                choices = [
                    (D[i - 1, j], (i - 1, j)),      # insertion
                    (D[i, j - 1], (i, j - 1)),      # deletion
                    (D[i - 1, j - 1], (i - 1, j - 1))  # match
                ]
                _, (i, j) = min(choices, key=lambda x: x[0])
                if i > 0 and j > 0:
                    path.append((i - 1, j - 1))
            
            path = path[::-1]  # Reverse path to get correct order
            text_indices, time_indices = zip(*path)
            return np.array(text_indices), np.array(time_indices)

        def patched_dtw(x):
            """
            Patched version of DTW that handles both numpy arrays and torch tensors.
            Ensures float32 precision and handles MPS tensors.
            """
            if isinstance(x, torch.Tensor):
                # Handle MPS tensors
                if x.device.type == 'mps':
                    x = x.float().cpu()  # Convert to float32 and move to CPU
                elif x.device.type == 'cuda':
                    return original_dtw(x)  # Use original for CUDA tensors
                # For CPU tensors, convert to numpy
                x = x.detach().numpy()
            
            # For numpy arrays, ensure float32
            if isinstance(x, np.ndarray):
                x = x.astype(np.float32)
                return dtw_cpu(x)
            
            return original_dtw(x)
        
        # Replace the original DTW function
        whisper.timing.dtw = patched_dtw

    def test_pytorch_whisper(self, model_name: str, device: str) -> Dict[str, Any]:
        """Test PyTorch Whisper model performance"""
        try:
            print(f"\nTesting PyTorch Whisper - Model: {model_name}, Device: {device}")
            
            print(f"\nLoading {model_name} model...")
            model = whisper.load_model(model_name)
            
            if device == 'mps' and torch.backends.mps.is_available():
                print("Moving model to MPS (Metal) with CPU fallback for sparse operations...")
                self.patch_dtw()  # Patch DTW function for MPS compatibility
                
                # Custom to_device function that handles sparse tensors
                def to_device(t):
                    if not isinstance(t, torch.Tensor):
                        return t
                    if t.is_sparse:
                        logging.info(f"Keeping sparse tensor on CPU: shape={t.shape}, dtype={t.dtype}")
                        return t
                    logging.info(f"Moving dense tensor to MPS: shape={t.shape}, dtype={t.dtype}")
                    return t.to(torch.device('mps'))
                
                # Apply the custom device movement
                model = model._apply(to_device)
                
                # Log device placement summary
                cpu_tensors = sum(1 for p in model.parameters() if p.device.type == 'cpu')
                mps_tensors = sum(1 for p in model.parameters() if p.device.type == 'mps')
                logging.info(f"Device placement summary - CPU: {cpu_tensors}, MPS: {mps_tensors} tensors")
            
            # Get initial metrics
            metrics = []
            metrics.append(self.get_system_metrics())
            
            # Run transcription with progress bar and timeout
            print(f"\nTranscribing with PyTorch Whisper ({model_name}, {device})...")
            pbar = tqdm(total=100, desc="Transcribing")
            
            try:
                with timeout(self.timeout_seconds):
                    result = model.transcribe(
                        str(self.sample_path),
                        language='en',
                        word_timestamps=True,
                        verbose=False
                    )
                pbar.n = 100
                pbar.refresh()
            except TimeoutException:
                logging.error(f"Transcription timed out after {self.timeout_seconds} seconds")
                return None
            finally:
                pbar.close()
            
            # Get final metrics
            metrics.append(self.get_system_metrics())
            
            # Calculate metrics
            end_time = time.time()
            duration = end_time - start_time
            
            # Save output for accuracy comparison
            output_path = self.test_dir / f"pytorch_{model_name}_{device}_output.json"
            with open(output_path, 'w') as f:
                json.dump(result, f, indent=2)
            
            # Compare accuracy if this is the small model (using medium as reference)
            accuracy_metrics = None
            if model_name == 'small' and device == 'cpu':
                reference_path = self.test_dir / "pytorch_medium_cpu_output.json"
                if reference_path.exists():
                    print("\nComparing accuracy with medium model...")
                    accuracy_metrics = self.compare_accuracy(reference_path, output_path)
                    if accuracy_metrics:
                        print(f"Word match ratio: {accuracy_metrics['word_match_ratio']:.2%}")
                        print(f"'Right' detection accuracy: {accuracy_metrics['right_detection_accuracy']:.2%}")
                        print(f"Reference 'right' count: {accuracy_metrics['reference_right_count']}")
                        print(f"Test 'right' count: {accuracy_metrics['test_right_count']}")
            
            print(f"\nCompleted in {duration:.1f} seconds")
            return {
                'implementation': 'pytorch',
                'model': model_name,
                'device': device,
                'duration': duration,
                'metrics': metrics,
                'output_path': str(output_path),
                'accuracy': accuracy_metrics
            }
            
        except Exception as e:
            logging.error(f"Error testing PyTorch Whisper: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())
            return None

    def test_faster_whisper(self, model_name, compute_type='float16'):
        """Test Faster Whisper implementation."""
        logging.info(f"Testing Faster Whisper - Model: {model_name}, Compute: {compute_type}")
        
        metrics = []
        start_time = time.time()
        
        try:
            # Load model with progress bar
            print(f"\nLoading {model_name} model (Faster Whisper)...")
            model = WhisperModel(
                model_name,
                device='cpu',  # Only use CPU for now since metal is not supported
                compute_type=compute_type
            )
            
            # Get initial metrics
            metrics.append(self.get_system_metrics())
            
            # Run transcription with progress bar and timeout
            print(f"\nTranscribing with Faster Whisper ({model_name}, {compute_type})...")
            pbar = tqdm(total=100, desc="Transcribing")
            
            try:
                with timeout(self.timeout_seconds):
                    segments, info = model.transcribe(
                        str(self.sample_path),
                        language='en',
                        word_timestamps=True,
                        verbose=False
                    )
                pbar.n = 100
                pbar.refresh()
            except TimeoutException:
                logging.error(f"Transcription timed out after {self.timeout_seconds} seconds")
                return None
            finally:
                pbar.close()
            
            # Get final metrics
            metrics.append(self.get_system_metrics())
            
            # Calculate metrics
            end_time = time.time()
            duration = end_time - start_time
            
            # Save output for accuracy comparison
            output = {
                'segments': [s._asdict() for s in segments],
                'info': info
            }
            output_path = self.test_dir / f"faster_{model_name}_{compute_type}_output.json"
            with open(output_path, 'w') as f:
                json.dump(output, f, indent=2)
            
            print(f"\nCompleted in {duration:.1f} seconds")
            return {
                'implementation': 'faster',
                'model': model_name,
                'compute_type': compute_type,
                'duration': duration,
                'metrics': metrics,
                'output_path': str(output_path)
            }
            
        except Exception as e:
            logging.error(f"Error testing Faster Whisper: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())
            return None
    
    def compare_accuracy(self, reference_path, test_output_path):
        """Compare transcription accuracy between two outputs."""
        try:
            # Load transcripts
            with open(reference_path) as f:
                reference = json.load(f)
            with open(test_output_path) as f:
                test = json.load(f)
            
            # Extract text segments
            ref_text = ' '.join(s['text'] for s in reference['segments']).lower()
            test_text = ' '.join(s['text'] for s in test['segments']).lower()
            
            # Count "right" instances
            ref_right_count = ref_text.count('right')
            test_right_count = test_text.count('right')
            
            # Calculate basic metrics
            word_match_ratio = len(set(test_text.split()) & set(ref_text.split())) / len(set(ref_text.split()))
            right_detection_accuracy = min(test_right_count, ref_right_count) / max(test_right_count, ref_right_count)
            
            return {
                'word_match_ratio': word_match_ratio,
                'right_detection_accuracy': right_detection_accuracy,
                'reference_right_count': ref_right_count,
                'test_right_count': test_right_count
            }
            
        except Exception as e:
            logging.error(f"Error comparing accuracy: {str(e)}")
            return None
    
    def test_mlx_whisper(self, model_name: str) -> Dict[str, Any]:
        """Test MLX Whisper model performance"""
        if not HAS_MLX:
            logging.warning("MLX Whisper not available. Skipping test.")
            return None
            
        try:
            print(f"\nTesting MLX Whisper - Model: {model_name}")
            
            print(f"\nLoading {model_name} model...")
            model = mlx_whisper.load_model(model_name)
            
            # Get initial metrics
            metrics = []
            metrics.append(self.get_system_metrics())
            
            # Run transcription with progress bar and timeout
            print(f"\nTranscribing with MLX Whisper ({model_name})...")
            pbar = tqdm(total=100, desc="Transcribing")
            start_time = time.time()
            
            try:
                with timeout(self.timeout_seconds):
                    result = model.transcribe(
                        str(self.sample_path),
                        language='en',
                        word_timestamps=True,
                        verbose=False
                    )
                pbar.n = 100
                pbar.refresh()
            except TimeoutException:
                logging.error(f"Transcription timed out after {self.timeout_seconds} seconds")
                return None
            finally:
                pbar.close()
            
            # Get final metrics
            metrics.append(self.get_system_metrics())
            
            # Calculate metrics
            end_time = time.time()
            duration = end_time - start_time
            
            # Save output for accuracy comparison
            output_path = self.test_dir / f"mlx_{model_name}_output.json"
            with open(output_path, 'w') as f:
                json.dump(result, f, indent=2)
            
            # Compare accuracy if this is the small model (using medium as reference)
            accuracy_metrics = None
            if model_name == 'small':
                reference_path = self.test_dir / "pytorch_medium_cpu_output.json"
                if reference_path.exists():
                    print("\nComparing accuracy with medium model...")
                    accuracy_metrics = self.compare_accuracy(reference_path, output_path)
                    if accuracy_metrics:
                        print(f"Word match ratio: {accuracy_metrics['word_match_ratio']:.2%}")
                        print(f"'Right' detection accuracy: {accuracy_metrics['right_detection_accuracy']:.2%}")
                        print(f"Reference 'right' count: {accuracy_metrics['reference_right_count']}")
                        print(f"Test 'right' count: {accuracy_metrics['test_right_count']}")
            
            print(f"\nCompleted in {duration:.1f} seconds")
            return {
                'implementation': 'mlx',
                'model': model_name,
                'device': 'apple_silicon',
                'duration': duration,
                'metrics': metrics,
                'output_path': str(output_path),
                'accuracy': accuracy_metrics
            }
            
        except Exception as e:
            logging.error(f"Error testing MLX Whisper: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())
            return None

    def plot_results(self):
        """Generate visualization of test results."""
        if not self.results:
            logging.error("No results to plot")
            return
        
        # Create results directory
        results_dir = self.test_dir / "results"
        results_dir.mkdir(exist_ok=True)
        
        # Prepare data for plotting
        df = pd.DataFrame(self.results)
        
        # 1. Speed Comparison
        plt.figure(figsize=(12, 6))
        sns.barplot(data=df, x='model', y='duration', hue='implementation')
        plt.title('Transcription Speed Comparison')
        plt.ylabel('Duration (seconds)')
        plt.tight_layout()
        plt.savefig(results_dir / 'speed_comparison.png')
        plt.close()
        
        # 2. Resource Usage
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        for result in self.results:
            metrics_df = pd.DataFrame(result['metrics'])
            label = f"{result['implementation']}_{result['model']}"
            
            # CPU Usage
            ax1.plot(metrics_df['timestamp'] - metrics_df['timestamp'].iloc[0],
                    metrics_df['cpu_percent'],
                    label=label)
            
            # Memory Usage
            ax2.plot(metrics_df['timestamp'] - metrics_df['timestamp'].iloc[0],
                    metrics_df['memory_mb'],
                    label=label)
        
        ax1.set_title('CPU Usage Over Time')
        ax1.set_ylabel('CPU %')
        ax1.legend()
        
        ax2.set_title('Memory Usage Over Time')
        ax2.set_xlabel('Time (seconds)')
        ax2.set_ylabel('Memory (MB)')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(results_dir / 'resource_usage.png')
        plt.close()
        
        # 3. Accuracy Comparison
        plt.figure(figsize=(12, 6))
        metrics = ['word_match_ratio', 'right_detection_accuracy']
        accuracy_data = []
        
        for result in self.results:
            if 'accuracy' in result:
                for metric in metrics:
                    accuracy_data.append({
                        'implementation': result['implementation'],
                        'model': result['model'],
                        'metric': metric,
                        'value': result['accuracy'][metric]
                    })
        
        if accuracy_data:
            acc_df = pd.DataFrame(accuracy_data)
            sns.barplot(data=acc_df, x='implementation', y='value', hue='metric')
            plt.title('Accuracy Comparison')
            plt.ylabel('Score')
            plt.tight_layout()
            plt.savefig(results_dir / 'accuracy_comparison.png')
        plt.close()
        
        # Save numerical results
        results_df = pd.DataFrame(self.results)
        results_df.to_csv(results_dir / 'numerical_results.csv', index=False)
        
        # Generate summary report
        with open(results_dir / 'summary.md', 'w') as f:
            f.write('# Whisper Optimization Test Results\n\n')
            f.write(f'Test conducted on: {datetime.now()}\n')
            f.write(f'Sample duration: {self.sample_duration} seconds\n\n')
            
            f.write('## Speed Comparison\n')
            f.write(results_df[['implementation', 'model', 'duration']].to_markdown())
            f.write('\n\n')
            
            f.write('## Resource Usage\n')
            for result in self.results:
                metrics_df = pd.DataFrame(result['metrics'])
                f.write(f"\n### {result['implementation']} - {result['model']}\n")
                f.write(f"- Peak CPU: {metrics_df['cpu_percent'].max():.1f}%\n")
                f.write(f"- Peak Memory: {metrics_df['memory_mb'].max():.1f} MB\n")
            
            if 'accuracy' in results_df.columns:
                f.write('\n## Accuracy Comparison\n')
                f.write(results_df[['implementation', 'model', 'accuracy']].to_markdown())
    
    def run_tests(self):
        """Run all optimization tests."""
        # Test MLX Whisper if available
        if HAS_MLX:
            for model_name in ['medium', 'small']:
                print(f"\nTesting {model_name} model with MLX...")
                result = self.test_mlx_whisper(model_name)
                if result:
                    self.results.append(result)
        else:
            print("\nMLX Whisper not available. Skipping MLX tests.")

        # Only test PyTorch Whisper on MPS since we already have CPU results
        if torch.backends.mps.is_available():
            for model_name in ['medium', 'small']:
                print(f"\nTesting {model_name} model on MPS (Metal)...")
                result = self.test_pytorch_whisper(model_name, device='mps')
                if result:
                    self.results.append(result)

        # Plot results
        self.plot_results()
        return self.results

def main():
    # Run optimization tests
    print("Starting optimization tests...")
    tester = WhisperOptimizationTester(sample_duration=600)  # 10 minutes
    
    # Prepare test sample first
    print("\nPreparing test sample...")
    tester.prepare_test_sample()
    
    # Run the tests
    print("\nRunning optimization tests...")
    results = tester.run_tests()
    
    # Print summary
    print("\nTest Summary:")
    print("-" * 50)
    for result in results:
        print(f"\nImplementation: {result['implementation']}")
        print(f"Model: {result['model']}")
        if 'device' in result:
            print(f"Device: {result['device']}")
        if 'compute_type' in result:
            print(f"Compute Type: {result['compute_type']}")
        print(f"Duration: {result['duration']:.2f}s")
        if 'accuracy' in result:
            print(f"Word Match Ratio: {result['accuracy']['word_match_ratio']:.2%}")
            print(f"'Right' Detection Accuracy: {result['accuracy']['right_detection_accuracy']:.2%}")
        print("-" * 50)

if __name__ == '__main__':
    main() 