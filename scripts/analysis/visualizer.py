import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import defaultdict
from pathlib import Path

class Visualizer:
    """Handles the creation of all analysis visualizations."""

    def __init__(self, output_dir='data/processed/full_podcast/visualizations_accurate'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        sns.set_style("whitegrid")

    def plot_right_usage_by_speaker(self, instances):
        """Plot the total number of 'right' instances for each speaker."""
        speaker_counts = defaultdict(int)
        for instance in instances:
            speaker_counts[instance['speaker']] += 1
        
        # Ensure there is data to plot
        if not speaker_counts:
            print("No speaker data to plot for 'right' usage.")
            return

        speakers = list(speaker_counts.keys())
        counts = list(speaker_counts.values())

        plt.figure(figsize=(10, 6))
        sns.barplot(x=speakers, y=counts)
        plt.title("Total 'Right' Usage by Speaker (Accurate)")
        plt.ylabel("Number of Instances")
        plt.xlabel("Speaker")
        # Add text labels on bars
        for index, value in enumerate(counts):
            plt.text(index, value, str(value), ha='center', va='bottom')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'right_usage_by_speaker_accurate.png', dpi=300)
        plt.close()
        print(f"Saved speaker usage plot to {self.output_dir}")

    def plot_usage_over_time_accurate(self, instances, total_duration_seconds, window_minutes=5):
        """Plot the raw count of 'right' usage over time for each speaker."""
        window_sec = window_minutes * 60
        
        plt.figure(figsize=(15, 8))
        
        speakers_with_instances = {inst['speaker'] for inst in instances}
        if not speakers_with_instances:
            print("No instances to plot for usage over time.")
            return

        # Create bins for the entire duration of the podcast
        bins = np.arange(0, total_duration_seconds + window_sec, window_sec)
        if len(bins) < 2: 
            print("Not enough duration to create time bins for plotting.")
            return
            
        # We will plot the center of each time window on the x-axis
        bin_centers = (bins[:-1] + bins[1:]) / 2

        for speaker in sorted(list(speakers_with_instances)):
            speaker_instances = [inst for inst in instances if inst['speaker'] == speaker]
            if not speaker_instances:
                continue

            # Use 'start_time' for accurate timestamps
            timestamps = [inst['start_time'] for inst in speaker_instances]
            
            # This directly gives the raw count in each bin
            counts, _ = np.histogram(timestamps, bins=bins)

            plt.plot(bin_centers / 3600, counts, label=speaker, marker='o', linestyle='-')

        plt.title(f"Raw Count of 'Right' Over Time ({window_minutes}-Minute Windows)")
        plt.xlabel("Time in Podcast (Hours)")
        plt.ylabel(f"Total Occurrences in Window")
        plt.legend()
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'usage_over_time_accurate.png', dpi=300)
        plt.close()
        print(f"Saved usage-over-time plot to {self.output_dir}")

    def plot_usage_dot_plot(self, instances, total_duration_seconds):
        """Creates a dot plot of 'right' instances over time for each speaker."""
        if not instances:
            print("No instances to create a dot plot.")
            return

        times = [inst['start_time'] / 3600 for inst in instances]  # Convert to hours
        speakers = [inst['speaker'] for inst in instances]

        plt.figure(figsize=(15, 8))
        
        # Using seaborn's stripplot is great for this kind of categorical scatter plot
        sns.stripplot(x=times, y=speakers, jitter=0.2, alpha=0.8, orient='h', size=6)
        
        plt.title("Instances of 'Right' Over Time by Speaker (Accurate)")
        plt.xlabel("Time (hours)")
        plt.ylabel("Speaker")
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        
        # Set x-axis limits to represent the full podcast duration
        plt.xlim(0, total_duration_seconds / 3600)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'usage_dot_plot_accurate.png', dpi=300)
        plt.close()
        print(f"Saved usage dot plot to {self.output_dir}")
