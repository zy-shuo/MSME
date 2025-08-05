"""
Main script to run the knowledge preparation phase.
This script processes datasets, retrieves knowledge, and selects relevant chunks.
"""

import os
import json
import argparse
import pandas as pd
from typing import List, Dict, Any
import logging
from knowledge_preparation.retriever import search_target
from knowledge_preparation.text_processor import process_text
from knowledge_preparation.knowledge_selector import select_knowledge

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("knowledge_preparation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def detect_language(text: str) -> str:
    """
    Detect language of text (simple heuristic).
    
    Args:
        text: Text to detect language
        
    Returns:
        Language code ('en' or 'zh')
    """
    # Simple heuristic: check for Chinese characters
    for char in text:
        if '\u4e00' <= char <= '\u9fff':
            return 'zh'
    return 'en'

def process_dataset(dataset_path: str, output_dir: str, api_key: str = None) -> None:
    """
    Process a dataset file, retrieve knowledge, and select relevant chunks.
    
    Args:
        dataset_path: Path to the dataset file
        output_dir: Directory to save output files
        api_key: SerpAPI API key
    """
    logger.info(f"Processing dataset: {dataset_path}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get dataset name
    dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
    
    try:
        # Load dataset
        df = pd.read_excel(dataset_path)
        
        if "target" not in df.columns:
            logger.error(f"No 'target' column found in {dataset_path}")
            return
        
        # Get unique targets
        targets = df["target"].dropna().unique()
        logger.info(f"Found {len(targets)} unique targets in dataset")
        
        # Process each target
        all_results = {}
        for i, target in enumerate(targets):
            target_str = str(target).strip()
            if not target_str:
                continue
                
            logger.info(f"Processing target {i+1}/{len(targets)}: {target_str}")
            
            # Detect language
            language = detect_language(target_str)
            logger.info(f"Detected language: {language}")
            
            # Search for target
            search_results = search_target(target_str, api_key)
            
            if not search_results:
                logger.warning(f"No search results found for target: {target_str}")
                continue
                
            # Process text
            processed_chunks = process_text(search_results, language=language)
            
            # Select knowledge
            selected_chunks = select_knowledge(processed_chunks, target_str, language=language)
            
            # Store results
            all_results[target_str] = {
                "search_results": search_results,
                "selected_chunks": selected_chunks
            }
            
            # Save intermediate results
            intermediate_path = os.path.join(output_dir, f"{dataset_name}_intermediate_{i}.json")
            with open(intermediate_path, "w", encoding="utf-8") as f:
                json.dump({target_str: all_results[target_str]}, f, ensure_ascii=False, indent=2)
        
        # Save final results
        output_path = os.path.join(output_dir, f"{dataset_name}_knowledge.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
            
        logger.info(f"Saved results to {output_path}")
        
    except Exception as e:
        logger.error(f"Error processing dataset: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="Run knowledge preparation phase")
    parser.add_argument("--datasets", nargs="+", default=["data/SEM16.xlsx", "data/P-Stance.xlsx", "data/Weibo-SD.xlsx"],
                      help="Paths to dataset files")
    parser.add_argument("--output_dir", default="knowledge_output",
                      help="Directory to save output files")
    parser.add_argument("--api_key", default=None,
                      help="SerpAPI API key (if not provided, will try to get from environment variable)")
    
    args = parser.parse_args()
    
    logger.info("Starting knowledge preparation phase")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process each dataset
    for dataset_path in args.datasets:
        if not os.path.exists(dataset_path):
            logger.error(f"Dataset file not found: {dataset_path}")
            continue
            
        process_dataset(dataset_path, args.output_dir, args.api_key)
    
    logger.info("Knowledge preparation phase completed")

if __name__ == "__main__":
    main()