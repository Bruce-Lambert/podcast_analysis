import re
from collections import defaultdict
import matplotlib.pyplot as plt
import json
import numpy as np
from pathlib import Path
import seaborn as sns
from datetime import datetime, timedelta

class DiscourseAnalyzer:
    def __init__(self, raw_dir="data/raw", processed_dir="data/processed"):
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        
    def load_whisper_results(self):
        """Load the Whisper analysis results."""
        results_path = self.processed_dir / "right_analysis.json"
        with open(results_path, 'r') as f:
            return json.load(f)
            
    def calculate_speaking_time(self):
        """Calculate total speaking time for each speaker from Whisper output."""
        whisper_path = self.processed_dir / "whisper_output.json"
        with open(whisper_path, 'r') as f:
            whisper_data = json.load(f)
            
        speaking_time = defaultdict(float)
        total_segments = defaultdict(int)
        
        # Process each segment
        for segment in whisper_data['segments']:
            start_time = segment['start']
            end_time = segment['end']
            duration = end_time - start_time
            
            # For now, treat all speech as from a single speaker since we don't have diarization
            speaker = "Speaker"
            speaking_time[speaker] += duration
            total_segments[speaker] += 1
        
        return speaking_time, total_segments
    
    def calculate_word_counts(self):
        """Calculate total words spoken from Whisper output."""
        whisper_path = self.processed_dir / "whisper_output.json"
        with open(whisper_path, 'r') as f:
            whisper_data = json.load(f)
            
        word_counts = defaultdict(int)
        right_counts = defaultdict(int)
        
        # Process each segment
        for segment in whisper_data['segments']:
            # For now, treat all speech as from a single speaker
            speaker = "Speaker"
            # Count words in the text
            text = segment['text']
            words = len(text.split())
            word_counts[speaker] += words
            
            # Count instances of "right"
            right_count = len(re.findall(r'\bright\b', text.lower()))
            right_counts[speaker] += right_count
        
        return word_counts, right_counts
            
    def plot_results(self, instances):
        """Create visualizations of the analysis results."""
        # Calculate speaking time and word counts for normalization
        speaking_time, total_segments = self.calculate_speaking_time()
        word_counts, right_counts = self.calculate_word_counts()
        
        # Create figure with subplots
        fig = plt.figure(figsize=(15, 25))
        
        # 1. Raw frequency of 'right' usage
        plt.subplot(4, 1, 1)
        times = []
        for instance in instances:
            t = instance['timestamp']
            if isinstance(t, str):
                h, m, s = map(int, t.split(':'))
                minutes = h * 60 + m + s/60
            else:
                minutes = float(t)/60
            times.append(minutes)
        
        plt.hist(times, bins=50, color='blue', alpha=0.6)
        plt.title("Distribution of 'right' Usage Over Time")
        plt.xlabel("Time (minutes)")
        plt.ylabel("Frequency")
        plt.grid(True, alpha=0.3)
        
        # 2. Context analysis (sentence position)
        plt.subplot(4, 1, 2)
        end_count = sum(1 for i in instances if i.get('is_sentence_end', False))
        mid_count = len(instances) - end_count
        
        positions = ['End of Sentence', 'Mid-Sentence']
        counts = [end_count, mid_count]
        
        bars = plt.bar(positions, counts)
        plt.title("Position of 'right' in Sentences")
        plt.ylabel("Count")
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom')
        
        # Add percentage labels
        total = end_count + mid_count
        plt.text(bars[0].get_x() + bars[0].get_width()/2., end_count/2,
                f'{end_count/total*100:.1f}%',
                ha='center', va='center')
        plt.text(bars[1].get_x() + bars[1].get_width()/2., mid_count/2,
                f'{mid_count/total*100:.1f}%',
                ha='center', va='center')
        
        # 3. Usage rate over time
        plt.subplot(4, 1, 3)
        
        # Calculate rate per minute in 5-minute windows
        window_size = 5  # minutes
        max_time = max(times)
        windows = np.arange(0, max_time + window_size, window_size)
        rates = []
        
        for start in windows[:-1]:
            end = start + window_size
            count = sum(1 for t in times if start <= t < end)
            rates.append(count / window_size)  # instances per minute
        
        plt.plot(windows[:-1] + window_size/2, rates, color='green', alpha=0.6)
        plt.title("Usage Rate of 'right' Over Time\n(5-minute windows)")
        plt.xlabel("Time (minutes)")
        plt.ylabel("Instances per minute")
        plt.grid(True, alpha=0.3)
        
        # 4. Normalized usage rates
        plt.subplot(4, 1, 4)
        
        speakers = list(word_counts.keys())
        x = np.arange(len(speakers))
        width = 0.35
        
        # Calculate normalized rates
        per_minute_rates = [right_counts[s] / (speaking_time[s] / 60) for s in speakers]
        per_word_rates = [right_counts[s] / word_counts[s] * 1000 for s in speakers]
        
        # Create grouped bar plot
        fig, ax = plt.subplots()
        rects1 = ax.bar(x - width/2, per_minute_rates, width, label='Per Minute')
        rects2 = ax.bar(x + width/2, per_word_rates, width, label='Per 1000 Words')
        
        ax.set_ylabel('Usage Rate')
        ax.set_title("Normalized 'right' Usage Rates")
        ax.set_xticks(x)
        ax.set_xticklabels(speakers)
        ax.legend()
        
        # Add value labels
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height:.2f}',
                          xy=(rect.get_x() + rect.get_width() / 2, height),
                          xytext=(0, 3),  # 3 points vertical offset
                          textcoords="offset points",
                          ha='center', va='bottom')
        
        autolabel(rects1)
        autolabel(rects2)
        
        plt.tight_layout()
        plt.savefig(self.processed_dir / 'discourse_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Print summary statistics
        print("\nAnalysis Summary:")
        print(f"Total instances of 'right': {len(instances)}")
        print(f"Speaking duration: {speaking_time['Speaker']/60:.1f} minutes")
        print(f"Total words: {word_counts['Speaker']}")
        print(f"Overall rate: {len(instances)/(speaking_time['Speaker']/60):.2f} instances per minute")
        print(f"Rate per 1000 words: {len(instances)/word_counts['Speaker']*1000:.2f}")
        print(f"\nSentence position:")
        print(f"  End of sentence: {end_count} ({end_count/total*100:.1f}%)")
        print(f"  Mid-sentence: {mid_count} ({mid_count/total*100:.1f}%)")
        
        # Print normalized rates per speaker
        print("\nNormalized rates per speaker:")
        for speaker in speakers:
            print(f"\n{speaker}:")
            print(f"  Instances per minute: {right_counts[speaker]/(speaking_time[speaker]/60):.2f}")
            print(f"  Instances per 1000 words: {right_counts[speaker]/word_counts[speaker]*1000:.2f}")
            print(f"  Total instances: {right_counts[speaker]}")
            print(f"  Speaking time: {speaking_time[speaker]/60:.1f} minutes")
            print(f"  Total words: {word_counts[speaker]}")

def main():
    analyzer = DiscourseAnalyzer()
    
    # Load Whisper results
    instances = analyzer.load_whisper_results()
    
    # Generate visualizations
    analyzer.plot_results(instances)
    
    print("\nAnalysis complete!")
    print(f"\nFound {len(instances)} instances of 'right'")
    print(f"Speaking duration: {sum(float(i['end_time']) - float(i['start_time']) for i in instances)/60:.1f} minutes")
    print(f"Total words: {sum(len(i['text'].split()) for i in instances)}")
    print(f"Overall rate: {len(instances)/(sum(float(i['end_time']) - float(i['start_time']) for i in instances)/60):.2f} instances per minute")
    print(f"Rate per 1000 words: {len(instances)/sum(len(i['text'].split()) for i in instances)*1000:.2f}")

if __name__ == "__main__":
    main() 