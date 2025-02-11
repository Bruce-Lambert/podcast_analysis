"""
Discourse Analysis Visualizer Module

This module handles the creation of visualizations for discourse analysis results.
It provides flexible plotting functions that can be used with different types of analysis.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import List, Dict, Any, Optional

class DiscourseVisualizer:
    """Creates visualizations for discourse analysis results."""
    
    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize visualizer.
        
        Args:
            output_dir (Path, optional): Directory to save visualizations
        """
        self.output_dir = output_dir or Path('data/processed')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set default style
        sns.set_style("whitegrid")
        sns.set_palette("husl")
    
    def plot_results(self, instances: List[Dict[str, Any]], 
                    word_counts: Dict[str, int],
                    title: str,
                    temporal_rates: Optional[Dict[str, List[float]]] = None) -> None:
        """
        Create comprehensive visualization of analysis results.
        
        Args:
            instances (List[Dict]): List of discourse marker instances
            word_counts (Dict[str, int]): Word counts by speaker
            title (str): Title for the visualization
            temporal_rates (Dict[str, List[float]], optional): Temporal distribution data
        """
        # Create figure
        n_plots = 4 if temporal_rates else 3
        fig = plt.figure(figsize=(15, 8 * n_plots), facecolor='white')
        fig.suptitle(f"{title}\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                    fontsize=16, y=0.95)
        
        # 1. Raw frequency by speaker
        plt.subplot(n_plots, 1, 1)
        speaker_counts = self._get_speaker_counts(instances)
        self._plot_speaker_counts(speaker_counts, "Distribution by Speaker")
        
        # 2. Context analysis (sentence position)
        plt.subplot(n_plots, 1, 2)
        self._plot_sentence_position(instances, "Position in Sentences")
        
        # 3. Normalized usage rates
        plt.subplot(n_plots, 1, 3)
        self._plot_normalized_rates(speaker_counts, word_counts, 
                                  "Normalized Usage Rates\nInstances per 1000 Words")
        
        # 4. Temporal distribution (if provided)
        if temporal_rates:
            plt.subplot(n_plots, 1, 4)
            self._plot_temporal_distribution(temporal_rates, 
                                          "Usage Rate Over Time\nInstances per Minute")
        
        # Add footer with metadata
        plt.figtext(0.02, 0.02,
                   f"Total words analyzed: {sum(word_counts.values()):,}\n" +
                   f"Total instances: {len(instances)}\n" +
                   f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                   fontsize=10, ha='left',
                   bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=5))
        
        # Save plot
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(self.output_dir / f'discourse_analysis_{timestamp}.png',
                   dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close()
    
    def _get_speaker_counts(self, instances: List[Dict[str, Any]]) -> Dict[str, int]:
        """Get count of instances by speaker."""
        counts = {}
        for instance in instances:
            speaker = instance['speaker']
            counts[speaker] = counts.get(speaker, 0) + 1
        return counts
    
    def _plot_speaker_counts(self, counts: Dict[str, int], title: str) -> None:
        """Plot raw frequency by speaker."""
        speakers = list(counts.keys())
        values = [counts[s] for s in speakers]
        
        sns.barplot(x=speakers, y=values)
        plt.title(title)
        plt.ylabel("Number of Instances")
        
        # Add value labels
        for i, v in enumerate(values):
            plt.text(i, v, str(v), ha='center', va='bottom',
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1))
    
    def _plot_sentence_position(self, instances: List[Dict[str, Any]], title: str) -> None:
        """Plot sentence position analysis."""
        end_count = sum(1 for i in instances if i['is_sentence_end'])
        mid_count = len(instances) - end_count
        
        positions = ['End of Sentence', 'Mid-Sentence']
        counts = [end_count, mid_count]
        
        sns.barplot(x=positions, y=counts)
        plt.title(title)
        plt.ylabel("Count")
        
        # Add value labels and percentages
        total = end_count + mid_count
        for i, v in enumerate(counts):
            percentage = (v / total * 100) if total > 0 else 0
            plt.text(i, v, f'{v}\n({percentage:.1f}%)', ha='center', va='bottom',
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1))
    
    def _plot_normalized_rates(self, counts: Dict[str, int], 
                             word_counts: Dict[str, int], 
                             title: str) -> None:
        """Plot normalized usage rates."""
        speakers = list(counts.keys())
        rates = []
        
        for speaker in speakers:
            if word_counts[speaker] > 0:
                rate = (counts[speaker] / word_counts[speaker]) * 1000
                rates.append(rate)
            else:
                rates.append(0)
        
        sns.barplot(x=speakers, y=rates)
        plt.title(title)
        plt.ylabel("Instances per 1000 words")
        
        # Add value labels
        for i, v in enumerate(rates):
            plt.text(i, v, f'{v:.2f}', ha='center', va='bottom',
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1))
    
    def _plot_temporal_distribution(self, rates: Dict[str, List[float]], title: str) -> None:
        """Plot temporal distribution."""
        for speaker, values in rates.items():
            plt.plot(values, label=speaker, alpha=0.7)
        
        plt.title(title)
        plt.xlabel("Time Window")
        plt.ylabel("Instances per Minute")
        plt.legend() 