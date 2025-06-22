import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import defaultdict

from analysis.transcript_loader import TranscriptParser

def analyze_technical_context(pasted_transcript_path, whisper_json_path):
    """
    Analyzes the frequency of 'right' in technical vs. non-technical contexts.
    """
    print("Starting technical context analysis...")

    # 1. Load the data using the existing parser
    print("Step 1: Loading transcript data...")
    loader = TranscriptParser(pasted_transcript_path, whisper_json_path=whisper_json_path)
    parsed_data, _ = loader.parse()
    
    if not parsed_data:
        print("Error: No data loaded. Exiting.")
        return

    dylan_words = [seg for seg in parsed_data if seg.get('speaker') == 'Dylan Patel']
    if not dylan_words:
        print("No speech from Dylan Patel found.")
        return
        
    print(f"Loaded {len(dylan_words)} words from Dylan Patel.")

    # 2. Define technical keywords
    technical_keywords = [
        'gpu', 'cpu', 'semiconductor', 'cuda', 'moe', 'mixture of experts', 
        'transformer', 'parameter', 'neuron', 'training', 'inference', 'flops', 
        'nccl', 'pytorch', 'tpu', 'h100', 'chip', 'wafer', 'tsmc', 'nvidia',
        'model', 'architecture', 'scaling', 'attention', 'llm', 'api',
        'data', 'compute', 'algorithm', 'network', 'node', 'latency'
    ]

    # 3. Group words into time windows and classify
    print("Step 2: Classifying speech into technical/non-technical windows...")
    window_size_sec = 60
    
    technical_windows = {'right_count': 0, 'word_count': 0}
    non_technical_windows = {'right_count': 0, 'word_count': 0}
    
    if not dylan_words:
        return

    total_duration = dylan_words[-1]['end']
    num_windows = int(total_duration // window_size_sec) + 1

    for i in range(num_windows):
        start_time = i * window_size_sec
        end_time = start_time + window_size_sec
        
        window_words = [
            word for word in dylan_words 
            if start_time <= word['start'] < end_time
        ]

        if not window_words:
            continue

        window_text = ' '.join(word['word'].lower() for word in window_words)
        
        keyword_count = sum(1 for keyword in technical_keywords if keyword in window_text)
        
        # Classify as technical if at least 2 different technical keywords are present
        is_technical = keyword_count >= 2

        right_count = window_text.count('right')
        word_count = len(window_words)
        
        if is_technical:
            technical_windows['right_count'] += right_count
            technical_windows['word_count'] += word_count
        else:
            non_technical_windows['right_count'] += right_count
            non_technical_windows['word_count'] += word_count

    print("Step 3: Calculating frequencies...")
    # 4. Calculate frequencies
    tech_freq = (technical_windows['right_count'] / technical_windows['word_count']) * 100 if technical_windows['word_count'] > 0 else 0
    non_tech_freq = (non_technical_windows['right_count'] / non_technical_windows['word_count']) * 100 if non_technical_windows['word_count'] > 0 else 0

    print(f"Frequency in technical segments: {tech_freq:.2f} per 100 words.")
    print(f"Frequency in non-technical segments: {non_tech_freq:.2f} per 100 words.")

    # 5. Visualize the results
    print("Step 4: Generating visualization...")
    output_dir = Path('data/processed/full_podcast/visualizations_accurate')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    categories = ['Technical', 'Non-Technical']
    frequencies = [tech_freq, non_tech_freq]

    plt.figure(figsize=(8, 6))
    ax = sns.barplot(x=categories, y=frequencies)
    plt.title("Frequency of 'Right' in Technical vs. Non-Technical Contexts")
    plt.ylabel("Instances per 100 Words")
    plt.ylim(0, max(frequencies) * 1.2) # Add some space at the top

    # Add text labels on bars
    for i, v in enumerate(frequencies):
        ax.text(i, v + 0.01, f"{v:.2f}", color='black', ha='center')

    plt.tight_layout()
    plt.savefig(output_dir / 'technical_context_frequency.png', dpi=300)
    plt.close()
    
    print(f"Saved context frequency plot to {output_dir}")
    print("Analysis complete.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Analyze 'right' usage in technical vs. non-technical contexts.")
    parser.add_argument("pasted_transcript_path", help="Path to the pasted transcript file with speaker names.")
    parser.add_argument("--whisper_json_path", required=True, help="Path to the Whisper JSON output file.")
    
    args = parser.parse_args()
    
    analyze_technical_context(args.pasted_transcript_path, args.whisper_json_path) 