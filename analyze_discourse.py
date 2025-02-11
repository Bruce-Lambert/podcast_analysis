import re
from collections import defaultdict
import matplotlib.pyplot as plt
from pathlib import Path

class DiscourseAnalyzer:
    def __init__(self, raw_dir="data/raw"):
        self.raw_dir = Path(raw_dir)
        
    def load_transcript(self):
        """Load the transcript from the raw directory."""
        transcript_path = self.raw_dir / "pasted_transcript.txt"
        with open(transcript_path, 'r', encoding='utf-8') as f:
            return f.read()
            
    def parse_segments(self, text):
        """Parse the transcript into segments with speaker and text."""
        segments = []
        current_speaker = None
        current_text = []
        
        for line in text.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            # Check if line starts with timestamp
            if line.startswith('('):
                if current_speaker and current_text:
                    segments.append({
                        'speaker': current_speaker,
                        'text': ' '.join(current_text)
                    })
                current_text = [line]
            elif not line.startswith('('):
                if any(char.isdigit() for char in line):
                    continue
                if ':' in line:
                    if current_speaker and current_text:
                        segments.append({
                            'speaker': current_speaker,
                            'text': ' '.join(current_text)
                        })
                    current_speaker = line.split(':')[0].strip()
                    current_text = []
                else:
                    current_text.append(line)
                    
        if current_speaker and current_text:
            segments.append({
                'speaker': current_speaker,
                'text': ' '.join(current_text)
            })
            
        return segments
        
    def analyze_discourse_markers(self, segments, marker="right"):
        """Analyze the usage of discourse markers in the transcript."""
        results = defaultdict(lambda: {
            'total_count': 0,
            'end_sentence_count': 0,
            'instances': []
        })
        
        for segment in segments:
            speaker = segment['speaker']
            text = segment['text']
            
            # Find all instances of the marker
            pattern = re.compile(rf'\b{marker}\b', re.IGNORECASE)
            matches = list(pattern.finditer(text))
            
            for match in matches:
                start = match.start()
                end = match.end()
                
                # Get context (50 chars before and after)
                context_start = max(0, start - 50)
                context_end = min(len(text), end + 50)
                context = text[context_start:context_end]
                
                # Check if marker is at end of sentence
                next_char_pos = min(len(text) - 1, end + 1)
                is_end = end == len(text) or text[end:next_char_pos+1].strip().startswith(('.', '?', '!'))
                
                results[speaker]['total_count'] += 1
                if is_end:
                    results[speaker]['end_sentence_count'] += 1
                results[speaker]['instances'].append({
                    'context': context,
                    'is_end': is_end
                })
                
        return results
        
    def plot_results(self, results, marker="right"):
        """Create visualizations of the analysis results."""
        speakers = list(results.keys())
        total_counts = [results[s]['total_count'] for s in speakers]
        end_counts = [results[s]['end_sentence_count'] for s in speakers]
        
        # Create bar plot
        fig, ax = plt.subplots(figsize=(10, 6))
        x = range(len(speakers))
        width = 0.35
        
        ax.bar([i - width/2 for i in x], total_counts, width, label='Total Usage')
        ax.bar([i + width/2 for i in x], end_counts, width, label='End of Sentence')
        
        ax.set_ylabel(f'Count of "{marker}"')
        ax.set_title(f'Usage of "{marker}" by Speaker')
        ax.set_xticks(x)
        ax.set_xticklabels(speakers, rotation=45)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(self.raw_dir / 'discourse_analysis.png')
        plt.close()
        
    def print_summary(self, results, marker="right"):
        """Print a summary of the analysis results."""
        total_instances = sum(r['total_count'] for r in results.values())
        total_end = sum(r['end_sentence_count'] for r in results.values())
        
        print(f"\nAnalysis of '{marker}' usage in transcript:")
        print(f"Total instances: {total_instances}")
        print(f"Instances at end of sentence: {total_end} ({total_end/total_instances*100:.1f}%)\n")
        
        print("By speaker:")
        for speaker, data in results.items():
            total = data['total_count']
            end = data['end_sentence_count']
            print(f"\n{speaker}:")
            print(f"  Total usage: {total}")
            print(f"  End of sentence: {end} ({end/total*100:.1f}% of their usage)")
            
            print("\n  Example contexts:")
            for i, instance in enumerate(data['instances'][:3], 1):
                print(f"    {i}. ...{instance['context']}...")
                print(f"       {'[End of sentence]' if instance['is_end'] else '[Mid-sentence]'}")

def main():
    analyzer = DiscourseAnalyzer()
    
    # Load and parse transcript
    transcript = analyzer.load_transcript()
    segments = analyzer.parse_segments(transcript)
    
    # Analyze discourse markers
    results = analyzer.analyze_discourse_markers(segments, marker="right")
    
    # Generate outputs
    analyzer.print_summary(results)
    analyzer.plot_results(results)

if __name__ == "__main__":
    main() 