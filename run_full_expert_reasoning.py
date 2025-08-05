"""
Script to run the complete expert reasoning process on datasets.
This script processes datasets and applies multi-expert reasoning to each entry,
extracting refined knowledge and expert responses.
"""

import os
import argparse
import pandas as pd
import logging
import re
from tqdm import tqdm
from typing import Dict, List, Any, Optional
from expert_reasoning.expert_coordinator import coordinate_experts

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("expert_reasoning_full.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def process_dataset(dataset_path: str, output_path: str, 
                  model: str = "gpt-3.5-turbo", 
                  api_key: Optional[str] = None,
                  base_url: Optional[str] = None,
                  batch_size: int = 10,
                  save_intermediate: bool = True) -> None:
    """
    Process a dataset with expert reasoning and save all expert responses.
    
    Args:
        dataset_path: Path to the dataset file
        output_path: Path to save the output file
        model: OpenAI model to use
        api_key: OpenAI API key
        batch_size: Number of entries to process before saving intermediate results
        save_intermediate: Whether to save intermediate results
    """
    logger.info(f"Processing dataset: {dataset_path}")
    
    try:
        # Load dataset
        df = pd.read_excel(dataset_path)
        
        required_columns = ["target", "text", "raw_knowledge", "ESL"]
        for col in required_columns:
            if col not in df.columns:
                logger.error(f"Required column '{col}' not found in {dataset_path}")
                return
        
        # Create output dataframe with original columns
        output_df = df.copy()
        
        # Add new columns for expert reasoning results
        output_df["refined_knowledge"] = ""
        output_df["knowledge_expert_response"] = ""
        output_df["label_expert_response"] = ""
        output_df["pragmatic_expert_response"] = ""
        output_df["meta_judge_response"] = ""
        
        # Process each entry
        total_entries = len(df)
        logger.info(f"Processing {total_entries} entries")
        
        for i in tqdm(range(total_entries)):
            row = df.iloc[i]
            
            # Skip entries with missing data
            if pd.isna(row["target"]) or pd.isna(row["text"]) or pd.isna(row["ESL"]):
                logger.warning(f"Skipping entry {i+1} due to missing data")
                continue
                
            target = str(row["target"]).strip()
            text = str(row["text"]).strip()
            raw_knowledge = "" if pd.isna(row["raw_knowledge"]) else str(row["raw_knowledge"]).strip()
            esl = str(row["ESL"]).strip()
            
            logger.info(f"Processing entry {i+1}/{total_entries}: {target}")
            
            try:
                # Run expert reasoning
                results = coordinate_experts(
                    target=target,
                    raw_knowledge=raw_knowledge,
                    esl=esl,
                    text=text,
                    model=model,
                    api_key=api_key,
                    base_url=base_url
                )
                
                # Update output dataframe
                output_df.at[i, "refined_knowledge"] = results.get("refined_knowledge", "")
                
                if "knowledge_expert" in results and "response" in results["knowledge_expert"]:
                    output_df.at[i, "knowledge_expert_response"] = results["knowledge_expert"]["response"]
                
                if "label_expert" in results and "response" in results["label_expert"]:
                    output_df.at[i, "label_expert_response"] = results["label_expert"]["response"]
                
                if "pragmatic_expert" in results and "response" in results["pragmatic_expert"]:
                    output_df.at[i, "pragmatic_expert_response"] = results["pragmatic_expert"]["response"]
                
                if "meta_judge" in results and "response" in results["meta_judge"]:
                    output_df.at[i, "meta_judge_response"] = results["meta_judge"]["response"]
                
            except Exception as e:
                logger.error(f"Error processing entry {i+1}: {str(e)}")
            
            # Save intermediate results
            if save_intermediate and (i + 1) % batch_size == 0:
                intermediate_path = f"{os.path.splitext(output_path)[0]}_intermediate_{i+1}.xlsx"
                output_df.to_excel(intermediate_path, index=False)
                logger.info(f"Saved intermediate results to {intermediate_path}")
        
        # Save final results
        output_df.to_excel(output_path, index=False)
        logger.info(f"Saved final results to {output_path}")
        
    except Exception as e:
        logger.error(f"Error processing dataset: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="Run full expert reasoning on datasets")
    parser.add_argument("--dataset", required=True,
                      help="Path to the dataset file")
    parser.add_argument("--output", required=True,
                      help="Path to save the output file")
    parser.add_argument("--model", default="gpt-3.5-turbo",
                      help="OpenAI model to use (default: gpt-3.5-turbo)")
    parser.add_argument("--api_key", default=None,
                      help="OpenAI API key (if not provided, will try to get from environment variable)")
    parser.add_argument("--base_url", default=None,
                      help="OpenAI API base URL (if not provided, will try to get from environment variable)")
    parser.add_argument("--batch_size", type=int, default=10,
                      help="Number of entries to process before saving intermediate results (default: 10)")
    parser.add_argument("--no_intermediate", action="store_true",
                      help="Disable saving intermediate results")
    
    args = parser.parse_args()
    
    logger.info("Starting full expert reasoning process")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    
    # Process dataset
    process_dataset(
        dataset_path=args.dataset,
        output_path=args.output,
        model=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
        batch_size=args.batch_size,
        save_intermediate=not args.no_intermediate
    )
    
    logger.info("Full expert reasoning process completed")

if __name__ == "__main__":
    main()