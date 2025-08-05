"""
Script to update datasets with retrieved knowledge and explicit stance labels.
This script processes datasets, retrieves knowledge and generates stance labels,
then updates the original datasets with new columns.
"""

import os
import argparse
import pandas as pd
import json
import logging
from typing import Dict, List, Any, Optional
from knowledge_preparation.retriever import search_target
from knowledge_preparation.text_processor import process_text
from knowledge_preparation.knowledge_selector import select_knowledge
from knowledge_preparation.stance_label_generator import generate_stance_labels, setup_openai_api

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("update_dataset.log"),
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

def process_target_for_dataset(target: str, 
                             serpapi_key: Optional[str] = None,
                             openai_key: Optional[str] = None,
                             openai_model: str = "gpt-3.5-turbo") -> Dict[str, Any]:
    """
    Process a single target to get knowledge and stance labels.
    
    Args:
        target: Target keyword to process
        serpapi_key: SerpAPI API key
        openai_key: OpenAI API key
        openai_model: OpenAI model to use
        
    Returns:
        Dictionary containing raw knowledge and ESL
    """
    if not target or not isinstance(target, str) or not target.strip():
        logger.warning(f"Invalid target: {target}")
        return {"raw_knowledge": "", "ESL": ""}
        
    target = target.strip()
    language = detect_language(target)
    result = {"raw_knowledge": "", "ESL": ""}
    
    # Step 1: Knowledge Retrieval
    try:
        # Search for target
        search_results = search_target(target, serpapi_key)
        
        if search_results:
            # Process text
            processed_chunks = process_text(search_results, language=language)
            
            # Select knowledge
            selected_chunks = select_knowledge(processed_chunks, target, language=language)
            
            # Combine selected chunks into raw knowledge
            if selected_chunks:
                raw_knowledge = "\n\n".join([chunk["chunk"] for chunk in selected_chunks])
                result["raw_knowledge"] = raw_knowledge
                logger.info(f"Retrieved knowledge for target: {target}")
            else:
                logger.warning(f"No knowledge chunks selected for target: {target}")
        else:
            logger.warning(f"No search results found for target: {target}")
    except Exception as e:
        logger.error(f"Error retrieving knowledge for target {target}: {str(e)}")
    
    # Step 2: Stance Label Generation
    try:
        # Setup OpenAI API
        setup_openai_api(openai_key)
        
        # Generate stance labels
        stance_result = generate_stance_labels(target, openai_model)
        
        if "labels" in stance_result and stance_result["labels"]:
            result["ESL"] = stance_result["labels"]
            logger.info(f"Generated stance labels for target: {target}")
        else:
            logger.warning(f"Failed to generate stance labels for target: {target}")
    except Exception as e:
        logger.error(f"Error generating stance labels for target {target}: {str(e)}")
    
    return result

def update_dataset(dataset_path: str, output_path: Optional[str] = None,
                 serpapi_key: Optional[str] = None,
                 openai_key: Optional[str] = None,
                 openai_model: str = "gpt-3.5-turbo") -> None:
    """
    Update a dataset with raw knowledge and explicit stance labels.
    
    Args:
        dataset_path: Path to the dataset file
        output_path: Path to save the updated dataset
        serpapi_key: SerpAPI API key
        openai_key: OpenAI API key
        openai_model: OpenAI model to use
    """
    logger.info(f"Processing dataset: {dataset_path}")
    
    try:
        # Load dataset
        df = pd.read_excel(dataset_path)
        
        if "target" not in df.columns:
            logger.error(f"No 'target' column found in {dataset_path}")
            return
        
        # Add new columns if they don't exist
        if "raw_knowledge" not in df.columns:
            df["raw_knowledge"] = ""
        if "ESL" not in df.columns:
            df["ESL"] = ""
        
        # Get unique targets to avoid redundant processing
        unique_targets = df["target"].dropna().unique()
        logger.info(f"Found {len(unique_targets)} unique targets in dataset")
        
        # Process each unique target
        target_results = {}
        for i, target in enumerate(unique_targets):
            if pd.isna(target) or not str(target).strip():
                continue
                
            target_str = str(target).strip()
            logger.info(f"Processing target {i+1}/{len(unique_targets)}: {target_str}")
            
            # Process target
            result = process_target_for_dataset(target_str, serpapi_key, openai_key, openai_model)
            target_results[target_str] = result
        
        # Update dataset with results
        for idx, row in df.iterrows():
            target = row["target"]
            if pd.isna(target) or not str(target).strip():
                continue
                
            target_str = str(target).strip()
            if target_str in target_results:
                df.at[idx, "raw_knowledge"] = target_results[target_str]["raw_knowledge"]
                df.at[idx, "ESL"] = target_results[target_str]["ESL"]
        
        # Save updated dataset
        if output_path is None:
            # Create output path based on input path
            filename = os.path.basename(dataset_path)
            name, ext = os.path.splitext(filename)
            output_path = os.path.join(os.path.dirname(dataset_path), f"{name}_updated{ext}")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Save the updated dataset
        df.to_excel(output_path, index=False)
        logger.info(f"Updated dataset saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Error updating dataset: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="Update datasets with retrieved knowledge and explicit stance labels")
    parser.add_argument("--datasets", nargs="+", default=["data/SEM16.xlsx", "data/P-Stance.xlsx", "data/Weibo-SD.xlsx"],
                      help="Paths to dataset files")
    parser.add_argument("--output_dir", default=None,
                      help="Directory to save updated datasets (default: same directory as input)")
    parser.add_argument("--serpapi_key", default=None,
                      help="SerpAPI API key (if not provided, will try to get from environment variable)")
    parser.add_argument("--openai_key", default=None,
                      help="OpenAI API key (if not provided, will try to get from environment variable)")
    parser.add_argument("--openai_model", default="gpt-3.5-turbo",
                      help="OpenAI model to use (default: gpt-3.5-turbo)")
    
    args = parser.parse_args()
    
    logger.info("Starting dataset update process")
    
    # Process each dataset
    for dataset_path in args.datasets:
        if not os.path.exists(dataset_path):
            logger.error(f"Dataset file not found: {dataset_path}")
            continue
            
        # Create output path if output directory is specified
        output_path = None
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            filename = os.path.basename(dataset_path)
            output_path = os.path.join(args.output_dir, filename)
        
        # Update dataset
        update_dataset(dataset_path, output_path, args.serpapi_key, args.openai_key, args.openai_model)
    
    logger.info("Dataset update process completed")

if __name__ == "__main__":
    main()