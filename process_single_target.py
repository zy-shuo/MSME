"""
Script to process a single target for both knowledge retrieval and stance label generation.
This allows direct processing without needing to read from a dataset.
"""

import os
import json
import argparse
import logging
from typing import Dict, Any, Optional
from knowledge_preparation.retriever import search_target
from knowledge_preparation.text_processor import process_text
from knowledge_preparation.knowledge_selector import select_knowledge
from knowledge_preparation.stance_label_generator import generate_stance_labels, setup_openai_api

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
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

def process_target(target: str, 
                 output_dir: str = "output", 
                 serpapi_key: Optional[str] = None,
                 openai_key: Optional[str] = None,
                 openai_model: str = "gpt-3.5-turbo") -> Dict[str, Any]:
    """
    Process a single target for both knowledge retrieval and stance label generation.
    
    Args:
        target: Target keyword to process
        output_dir: Directory to save output files
        serpapi_key: SerpAPI API key
        openai_key: OpenAI API key
        openai_model: OpenAI model to use
        
    Returns:
        Dictionary containing the results
    """
    if not target or not isinstance(target, str) or not target.strip():
        logger.error("Invalid target provided")
        return {"error": "Invalid target provided"}
        
    target = target.strip()
    logger.info(f"Processing target: {target}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Detect language
    language = detect_language(target)
    logger.info(f"Detected language: {language}")
    
    results = {"target": target, "language": language}
    
    # Step 1: Knowledge Retrieval
    logger.info("Step 1: Knowledge Retrieval")
    try:
        # Search for target
        search_results = search_target(target, serpapi_key)
        
        if not search_results:
            logger.warning(f"No search results found for target: {target}")
            results["knowledge_retrieval"] = {"status": "error", "message": "No search results found"}
        else:
            # Process text
            processed_chunks = process_text(search_results, language=language)
            
            # Select knowledge
            selected_chunks = select_knowledge(processed_chunks, target, language=language)
            
            # Store results
            results["knowledge_retrieval"] = {
                "status": "success",
                "search_results_count": len(search_results),
                "processed_chunks_count": len(processed_chunks),
                "selected_chunks": selected_chunks
            }
            
            logger.info(f"Retrieved and processed {len(selected_chunks)} knowledge chunks")
    except Exception as e:
        logger.error(f"Error in knowledge retrieval: {str(e)}")
        results["knowledge_retrieval"] = {"status": "error", "message": str(e)}
    
    # Step 2: Stance Label Generation
    logger.info("Step 2: Stance Label Generation")
    try:
        # Setup OpenAI API
        setup_openai_api(openai_key)
        
        # Generate stance labels
        stance_result = generate_stance_labels(target, openai_model)
        
        if "error" in stance_result:
            logger.warning(f"Error generating stance labels: {stance_result['error']}")
            results["stance_label_generation"] = {"status": "error", "message": stance_result["error"]}
        else:
            results["stance_label_generation"] = {
                "status": "success",
                "labels": stance_result["labels"],
                "model": stance_result["model"],
                "tokens": stance_result.get("tokens", {})
            }
            
            logger.info("Generated stance labels successfully")
    except Exception as e:
        logger.error(f"Error in stance label generation: {str(e)}")
        results["stance_label_generation"] = {"status": "error", "message": str(e)}
    
    # Save results to file
    output_file = os.path.join(output_dir, f"{target.replace(' ', '_')}_results.json")
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"Results saved to {output_file}")
    except Exception as e:
        logger.error(f"Error saving results to file: {str(e)}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Process a single target for knowledge retrieval and stance label generation")
    parser.add_argument("--target", required=True,
                      help="Target keyword to process")
    parser.add_argument("--output_dir", default="output",
                      help="Directory to save output files")
    parser.add_argument("--serpapi_key", default=None,
                      help="SerpAPI API key (if not provided, will try to get from environment variable)")
    parser.add_argument("--openai_key", default=None,
                      help="OpenAI API key (if not provided, will try to get from environment variable)")
    parser.add_argument("--openai_model", default="gpt-3.5-turbo",
                      help="OpenAI model to use (default: gpt-3.5-turbo)")
    parser.add_argument("--print_results", action="store_true",
                      help="Print results to console")
    
    args = parser.parse_args()
    
    logger.info("Starting target processing")
    
    results = process_target(
        args.target,
        args.output_dir,
        args.serpapi_key,
        args.openai_key,
        args.openai_model
    )
    
    if args.print_results:
        # Print knowledge retrieval results
        if results.get("knowledge_retrieval", {}).get("status") == "success":
            print("\n=== Knowledge Retrieval Results ===")
            selected_chunks = results["knowledge_retrieval"]["selected_chunks"]
            for i, chunk in enumerate(selected_chunks):
                print(f"\nChunk {i+1} (Score: {chunk['relevance_score']:.4f}):")
                print(f"Source: {chunk['metadata']['url']}")
                print(f"Content: {chunk['chunk'][:200]}...")
        
        # Print stance label results
        if results.get("stance_label_generation", {}).get("status") == "success":
            print("\n=== Stance Label Generation Results ===")
            print(results["stance_label_generation"]["labels"])
    
    logger.info("Target processing completed")

if __name__ == "__main__":
    main()