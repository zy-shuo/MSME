"""
Script to run the stance label generation process for datasets.
"""

import os
import argparse
import logging
from knowledge_preparation.stance_label_generator import process_dataset_targets

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("stance_label_generation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Generate explicit stance labels for targets in datasets")
    parser.add_argument("--datasets", nargs="+", default=["data/SEM16.xlsx", "data/P-Stance.xlsx", "data/Weibo-SD.xlsx"],
                      help="Paths to dataset files")
    parser.add_argument("--output_dir", default="stance_labels_output",
                      help="Directory to save output files")
    parser.add_argument("--model", default="gpt-3.5-turbo",
                      help="OpenAI model to use (default: gpt-3.5-turbo)")
    parser.add_argument("--api_key", default=None,
                      help="OpenAI API key (if not provided, will try to get from environment variable)")
    
    args = parser.parse_args()
    
    logger.info("Starting stance label generation")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process each dataset
    for dataset_path in args.datasets:
        if not os.path.exists(dataset_path):
            logger.error(f"Dataset file not found: {dataset_path}")
            continue
            
        # Get dataset name
        dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
        output_path = os.path.join(args.output_dir, f"{dataset_name}_stance_labels.json")
        
        logger.info(f"Processing dataset: {dataset_path}")
        process_dataset_targets(dataset_path, output_path, args.model, args.api_key)
    
    logger.info("Stance label generation completed")

if __name__ == "__main__":
    main()