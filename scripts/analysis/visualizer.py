import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import defaultdict
from pathlib import Path

class Visualizer:
    """Handles the creation of all analysis visualizations."""

    def __init__(self, output_dir='data/processed/full_podcast/visualizations'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        sns.set_style("whitegrid")

    def plot_right_usage_by_speaker(self, instances):
        """Plot the total number of 'right' instances for each speaker."""
        speaker_counts = defaultdict(int)
        for instance in instances:
            speaker_counts[instance['speaker']] += 1
        
        speakers = list(speaker_counts.keys())
        counts = list(speaker_counts.values())

        plt.figure(figsize=(10, 6))
        sns.barplot(x=speakers, y=counts)
        plt.title("Total 'Right' Usage by Speaker")
        plt.ylabel("Number of Instances")
        plt.xlabel("Speaker")
        plt.tight_layout()
        plt.savefig(self.output_dir / 'right_usage_by_speaker.png', dpi=300)
        plt.close()
        print(f"Saved speaker usage plot to {self.output_dir}")

    def plot_sentence_position_analysis(self, instances):
        """Plot the distribution of 'right' at mid vs. end of sentence."""
        position_counts = defaultdict(lambda: defaultdict(int))
        for instance in instances:
            position = 'End of Sentence' if instance['is_sentence_end'] else 'Mid-sentence'
            position_counts[instance['speaker']][position] += 1

        # Data processing for stacked bar chart
        speakers = sorted(position_counts.keys())
        mid_sentence = [position_counts[s]['Mid-sentence'] for s in speakers]
        end_sentence = [position_counts[s]['End of Sentence'] for s in speakers]

        plt.figure(figsize=(12, 7))
        plt.bar(speakers, mid_sentence, label='Mid-sentence')
        plt.bar(speakers, end_sentence, bottom=mid_sentence, label='End of Sentence')
        plt.title("Sentence Position of 'Right' by Speaker")
        plt.ylabel("Number of Instances")
        plt.xlabel("Speaker")
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.output_dir / 'sentence_position_analysis.png', dpi=300)
        plt.close()
        print(f"Saved sentence position plot to {self.output_dir}")

    def plot_normalized_usage_rate(self, instances, word_counts):
        """Plot 'right' usage per 1000 words for each speaker."""
        speaker_counts = defaultdict(int)
        for instance in instances:
            speaker_counts[instance['speaker']] += 1

        rates = {}
        for speaker, count in speaker_counts.items():
            if word_counts.get(speaker, 0) > 0:
                rates[speaker] = (count / word_counts[speaker]) * 1000

        speakers = list(rates.keys())
        rate_values = list(rates.values())

        plt.figure(figsize=(10, 6))
        sns.barplot(x=speakers, y=rate_values)
        plt.title("Normalized 'Right' Usage Rate (per 1000 words)")
        plt.ylabel("Instances per 1000 words")
        plt.xlabel("Speaker")
        plt.tight_layout()
        plt.savefig(self.output_dir / 'normalized_usage_rate.png', dpi=300)
        plt.close()
        print(f"Saved normalized usage plot to {self.output_dir}")

    def plot_usage_over_time(self, instances, speaking_time, window_minutes=5):
        """Plot the usage of 'right' over time for each speaker."""
        window_sec = window_minutes * 60
        
        plt.figure(figsize=(15, 8))
        
        for speaker in sorted(speaking_time.keys()):
            speaker_instances = [inst for inst in instances if inst['speaker'] == speaker]
            if not speaker_instances:
                continue

            timestamps = [inst['timestamp'] for inst in speaker_instances]
            total_duration = max(timestamps) if timestamps else 0
            
            # Create bins
            bins = np.arange(0, total_duration + window_sec, window_sec)
            if len(bins) < 2: continue # Not enough data to plot

            counts, _ = np.histogram(timestamps, bins=bins)
            
            # Normalize by the time window to get a rate
            rates = counts / window_minutes

            plt.plot(bins[:-1] / 3600, rates, label=speaker, marker='o', linestyle='-')

        plt.title(f"Usage Rate of 'Right' Over Time ({window_minutes}-minute windows)")
        plt.xlabel("Time (hours)")
        plt.ylabel(f"Instances per {window_minutes} minutes")
        plt.legend()
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'usage_over_time.png', dpi=300)
        plt.close()
        print(f"Saved usage-over-time plot to {self.output_dir}")
