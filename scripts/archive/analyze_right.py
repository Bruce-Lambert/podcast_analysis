#!/usr/bin/env python3

import json
from pathlib import Path
import logging
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import re
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def simple_word_count(text: str, word: str = "right") -> int:
    """Simply count occurrences of a word in text."""
    return len(re.findall(r'\bright\b', text.lower()))

def analyze_transcripts(vtt_path: Path, pasted_path: Path):
    """Analyze 'right' usage in both transcripts and calculate inflation factor."""
    # Count in VTT transcript
    logger.info("Processing VTT transcript...")
    vtt_count = 0
    with open(vtt_path, 'r', encoding='utf-8') as f:
        # Count total lines first for progress bar
        total_lines = sum(1 for _ in f)
        f.seek(0)  # Reset file pointer
        
        for line in tqdm(f, total=total_lines, desc="Reading VTT"):
            if '-->' not in line:  # Skip timestamp lines
                count = simple_word_count(line)
                if count > 0:
                    logger.debug(f"VTT line with 'right': {line.strip()}")
                vtt_count += count
    
    # Count in pasted transcript and gather speaker stats
    logger.info("Processing pasted transcript...")
    pasted_count = 0
    speaker_stats = defaultdict(lambda: {"count": 0, "words": 0})
    current_speaker = None
    
    with open(pasted_path, 'r', encoding='utf-8') as f:
        # Count total lines first for progress bar
        total_lines = sum(1 for _ in f)
        f.seek(0)  # Reset file pointer
        
        prev_line = ""
        for line in tqdm(f, total=total_lines, desc="Reading pasted"):
            line = line.strip()
            if not line:
                continue
            
            # Check if the previous line was a speaker name and current line is a timestamp
            if prev_line in ["Dylan Patel", "Nathan Lambert", "Lex Fridman"] and line.startswith('('):
                current_speaker = prev_line
                logger.debug(f"Found speaker: {current_speaker}")
                prev_line = line
                continue
            
            # If line is just a speaker name, store it for the next iteration
            if line in ["Dylan Patel", "Nathan Lambert", "Lex Fridman"]:
                prev_line = line
                continue
            
            if current_speaker:
                rights = simple_word_count(line)
                if rights > 0:
                    logger.debug(f"Found {rights} instances of 'right' in line from {current_speaker}: {line}")
                words = len(line.split())
                speaker_stats[current_speaker]["count"] += rights
                speaker_stats[current_speaker]["words"] += words
                pasted_count += rights
            
            prev_line = line
    
    # Calculate inflation factor
    inflation_factor = vtt_count / pasted_count if pasted_count > 0 else 1.0
    
    # Calculate per-speaker statistics
    results = {
        "vtt_count": vtt_count,
        "pasted_count": pasted_count,
        "inflation_factor": inflation_factor,
        "speaker_stats": {}
    }
    
    logger.info("Calculating speaker statistics...")
    for speaker, stats in speaker_stats.items():
        raw_count = stats["count"]
        words = stats["words"]
        logger.debug(f"Speaker {speaker}: {raw_count} instances of 'right' in {words} words")
        results["speaker_stats"][speaker] = {
            "raw_count": raw_count,
            "inflated_count": raw_count * inflation_factor,
            "words": words,
            "per_1k": (raw_count / words) * 1000 if words > 0 else 0,
            "inflated_per_1k": ((raw_count / words) * 1000 * inflation_factor) if words > 0 else 0
        }
    
    return results

def plot_results(results: dict, output_dir: Path):
    """Create visualizations for both raw and inflated counts."""
    logger.info("Creating visualizations...")
    plt.style.use('default')  # Use default style instead of seaborn
    
    # Create a figure with two rows of subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("Analysis of 'right' Usage in Podcast\nRaw vs Inflated Counts", fontsize=16)
    
    speakers = list(results["speaker_stats"].keys())
    raw_counts = [stats["raw_count"] for stats in results["speaker_stats"].values()]
    inflated_counts = [stats["inflated_count"] for stats in results["speaker_stats"].values()]
    raw_per_1k = [stats["per_1k"] for stats in results["speaker_stats"].values()]
    inflated_per_1k = [stats["inflated_per_1k"] for stats in results["speaker_stats"].values()]
    
    # Raw counts
    sns.barplot(x=speakers, y=raw_counts, ax=ax1, color='skyblue')
    ax1.set_title("Raw 'right' Counts")
    ax1.set_ylabel("Number of Instances")
    for i, v in enumerate(raw_counts):
        ax1.text(i, v, f'{v:.0f}', ha='center', va='bottom')
    
    # Inflated counts
    sns.barplot(x=speakers, y=inflated_counts, ax=ax2, color='lightgreen')
    ax2.set_title("Inflated 'right' Counts")
    ax2.set_ylabel("Number of Instances")
    for i, v in enumerate(inflated_counts):
        ax2.text(i, v, f'{v:.1f}', ha='center', va='bottom')
    
    # Raw per 1000 words
    sns.barplot(x=speakers, y=raw_per_1k, ax=ax3, color='coral')
    ax3.set_title("Raw Usage per 1000 Words")
    ax3.set_ylabel("Instances per 1000 Words")
    for i, v in enumerate(raw_per_1k):
        ax3.text(i, v, f'{v:.1f}', ha='center', va='bottom')
    
    # Inflated per 1000 words
    sns.barplot(x=speakers, y=inflated_per_1k, ax=ax4, color='purple')
    ax4.set_title("Inflated Usage per 1000 Words")
    ax4.set_ylabel("Instances per 1000 Words")
    for i, v in enumerate(inflated_per_1k):
        ax4.text(i, v, f'{v:.1f}', ha='center', va='bottom')
    
    # Add inflation factor to figure title
    plt.figtext(0.02, 0.02, 
                f"Inflation factor: {results['inflation_factor']:.2f}\n" +
                f"VTT total: {results['vtt_count']}\n" +
                f"Pasted total: {results['pasted_count']}",
                fontsize=10, ha='left')
    
    plt.tight_layout()
    logger.info("Saving visualization...")
    plt.savefig(output_dir / 'right_analysis_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Analyze 'right' usage in both transcripts."""
    try:
        pasted_path = Path('data/raw/pasted_transcript.txt')
        vtt_path = Path('data/raw/macwhisper_full_video.vtt')
        output_dir = Path('data/processed/full_podcast')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Starting analysis...")
        results = analyze_transcripts(vtt_path, pasted_path)
        
        # Print summary
        logger.info(f"VTT count: {results['vtt_count']}")
        logger.info(f"Pasted count: {results['pasted_count']}")
        logger.info(f"Inflation factor: {results['inflation_factor']:.2f}")
        
        # Save results
        logger.info("Saving results...")
        with open(output_dir / 'right_analysis.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        
        # Create visualizations
        plot_results(results, output_dir)
        
        logger.info("Analysis complete!")
        
    except Exception as e:
        logger.error(f"Error analyzing transcripts: {e}")
        raise

if __name__ == '__main__':
    main()