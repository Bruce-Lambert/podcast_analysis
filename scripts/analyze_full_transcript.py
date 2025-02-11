#!/usr/bin/env python3

import json
import logging
from pathlib import Path
from analyze_discourse import DiscourseAnalyzer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('analysis.log'),
        logging.StreamHandler()
    ]
)

def main():
    # Initialize analyzer with the full podcast paths
    analyzer = DiscourseAnalyzer(
        raw_dir="data/raw/full_podcast",
        processed_dir="data/processed/full_podcast"
    )
    
    try:
        # Load the existing right analysis
        logging.info("Loading existing right analysis...")
        with open("data/processed/full_podcast/right_analysis.json") as f:
            instances = json.load(f)
            
        logging.info(f"Loaded {len(instances)} instances of 'right' from the full transcript")
        
        # Create visualizations
        logging.info("Creating visualizations...")
        analyzer.plot_results(instances)
        
        # Print summary
        print(f"\nAnalysis complete! Processed {len(instances)} instances of 'right'")
        print("Check data/processed/full_podcast/discourse_analysis.png for visualizations")
        
    except Exception as e:
        logging.error(f"Error during analysis: {str(e)}")
        raise

if __name__ == "__main__":
    main() 