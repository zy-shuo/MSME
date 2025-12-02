"""
Full stance detection pipeline
Includes knowledge preparation and expert reasoning phases
"""

import os
import sys
import json
import argparse
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from knowledge_preparation.retriever import search_target
from knowledge_preparation.text_processor import chunk_text, load_embedding_model, compute_embeddings
from knowledge_preparation.knowledge_selector import select_knowledge
from knowledge_preparation.stance_label_generator import generate_stance_labels
from expert_reasoning.expert_coordinator import coordinate_experts

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def detect_language(text: str) -> str:
    """Detect text language"""
    chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    if chinese_chars > len(text) * 0.3:
        return "zh"
    return "en"


def prepare_knowledge_for_target(target: str, language: str = "en", 
                                 num_results: int = 3, top_k: int = 3) -> tuple:
    """
    Prepare knowledge for target (retrieve, chunk, select)
    
    Returns:
        (raw_knowledge, ESL)
    """
    logger.info(f"Preparing knowledge for target: {target}")
    
    # Step 1: Retrieve
    logger.info("Step 1: Retrieve relevant web pages")
    search_results = search_target(target, num_results=num_results, location="China" if language == "zh" else "United States")
    
    if not search_results:
        logger.warning(f"No search results found for target: {target}")
        return "", ""
    
    # Step 2: Chunk and embed
    logger.info("Step 2: Chunk text and compute embeddings")
    all_chunks = []
    model = load_embedding_model(language)
    
    for result in search_results:
        text = result.get("extracted_text", "")
        if not text:
            continue
        
        chunks = chunk_text(text, chunk_size=300, overlap=30, language=language)
        embeddings = compute_embeddings(chunks, model, language)
        
        for chunk, embedding in zip(chunks, embeddings):
            all_chunks.append({
                "chunk": chunk,
                "embedding": embedding,
                "source": result.get("link", ""),
                "metadata": {
                    "url": result.get("url", result.get("link", "")),
                    "title": result.get("title", ""),
                    "source": result.get("source", "")
                }
            })
    
    logger.info(f"Generated {len(all_chunks)} chunks")
    
    # Step 3: Select relevant knowledge
    logger.info("Step 3: Select most relevant knowledge")
    selected_chunks = select_knowledge(all_chunks, target, top_k=top_k, language=language)
    
    raw_knowledge = " ".join([chunk["chunk"] for chunk in selected_chunks])
    logger.info(f"Raw knowledge length: {len(raw_knowledge)} characters")
    
    # Step 4: Generate ESL
    logger.info("Step 4: Generate explicit stance labels")
    esl_result = generate_stance_labels(target)
    
    if "error" in esl_result:
        logger.error(f"ESL generation failed: {esl_result['error']}")
        esl = ""
    else:
        esl = esl_result.get("labels", "")
    
    logger.info(f"Knowledge preparation completed for target: {target}")
    return raw_knowledge, esl


def prepare_knowledge_for_dataset(df: pd.DataFrame, num_search_results: int = 3, 
                                  top_k_chunks: int = 3) -> pd.DataFrame:
    """
    Prepare knowledge for all unique targets in dataset
    
    Args:
        df: Dataset DataFrame
        num_search_results: Number of search results
        top_k_chunks: Number of knowledge chunks to select
    
    Returns:
        Updated DataFrame
    """
    logger.info("\n" + "=" * 80)
    logger.info("Phase 0: Prepare knowledge for unique targets")
    logger.info("=" * 80)
    
    # Ensure necessary columns exist
    if "raw_knowledge" not in df.columns:
        df["raw_knowledge"] = ""
    if "ESL" not in df.columns:
        df["ESL"] = ""
    
    unique_targets = df["target"].unique()
    logger.info(f"Found {len(unique_targets)} unique targets")
    
    target_knowledge_map = {}
    
    for target in tqdm(unique_targets, desc="Preparing knowledge for unique targets"):
        target = str(target).strip()
        
        existing_rows = df[df["target"] == target]
        has_knowledge = False
        
        for _, row in existing_rows.iterrows():
            if pd.notna(row.get("raw_knowledge")) and pd.notna(row.get("ESL")) and \
               str(row.get("raw_knowledge", "")).strip() and str(row.get("ESL", "")).strip():
                target_knowledge_map[target] = {
                    "raw_knowledge": str(row["raw_knowledge"]),
                    "ESL": str(row["ESL"])
                }
                has_knowledge = True
                logger.info(f"Target '{target}' already has knowledge, skipping")
                break
        
        if not has_knowledge:
            logger.info(f"\nGenerating knowledge for target '{target}'...")
            
            sample_text = str(existing_rows.iloc[0]["text"])
            language = detect_language(sample_text)
            logger.info(f"Detected language: {language}")
            
            try:
                raw_knowledge, esl = prepare_knowledge_for_target(
                    target, language, num_search_results, top_k_chunks
                )
                
                target_knowledge_map[target] = {
                    "raw_knowledge": raw_knowledge,
                    "ESL": esl
                }
                logger.info(f"Knowledge generation completed for target '{target}'")
            except Exception as e:
                logger.error(f"Error generating knowledge for target '{target}': {str(e)}")
                target_knowledge_map[target] = {
                    "raw_knowledge": "",
                    "ESL": ""
                }
    
    # Update knowledge in dataset
    logger.info("\nUpdating knowledge in dataset...")
    for target, knowledge in target_knowledge_map.items():
        mask = df["target"] == target
        df.loc[mask, "raw_knowledge"] = knowledge["raw_knowledge"]
        df.loc[mask, "ESL"] = knowledge["ESL"]
        logger.info(f"Updated {mask.sum()} rows (target: {target})")
    
    logger.info("Knowledge preparation phase completed!")
    return df


def process_dataset(dataset_path: str, output_dir: str = "outputs",
                   start_idx: int = 0, end_idx: int = None,
                   num_search_results: int = 3,
                   top_k_chunks: int = 3,
                   skip_knowledge_preparation: bool = False):
    """
    Process complete dataset
    
    Args:
        dataset_path: Dataset path
        output_dir: Output directory
        start_idx: Start index (inclusive)
        end_idx: End index (exclusive), None means process to the end
        num_search_results: Number of search results
        top_k_chunks: Number of knowledge chunks to select
        skip_knowledge_preparation: Skip knowledge preparation if dataset already has complete knowledge
    """
    logger.info("=" * 80)
    logger.info("Starting full stance detection pipeline")
    logger.info("=" * 80)
    
    # Load dataset
    logger.info(f"Loading dataset: {dataset_path}")
    df = pd.read_excel(dataset_path)
    logger.info(f"Dataset size: {df.shape}")
    logger.info(f"Columns: {df.columns.tolist()}")
    
    # Check required columns
    required_cols = ["target", "text"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Dataset missing required column: {col}")
    
    # Phase 0: Knowledge preparation
    if not skip_knowledge_preparation:
        df = prepare_knowledge_for_dataset(df, num_search_results, top_k_chunks)
        
        # Save updated dataset
        dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
        updated_dataset_file = os.path.join(output_dir, f"{dataset_name}_with_knowledge.xlsx")
        os.makedirs(output_dir, exist_ok=True)
        df.to_excel(updated_dataset_file, index=False)
        logger.info(f"\nDataset with knowledge saved: {updated_dataset_file}")
    else:
        logger.info("Skipping knowledge preparation (using existing knowledge in dataset)")
    
    # Determine processing range
    total_rows = len(df)
    if end_idx is None:
        end_idx = total_rows
    else:
        end_idx = min(end_idx, total_rows)
    
    logger.info(f"\nProcessing range: [{start_idx}, {end_idx}), total {end_idx - start_idx} items")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate output filenames
    dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
    output_file = os.path.join(output_dir, f"{dataset_name}_results.xlsx")
    output_json = os.path.join(output_dir, f"{dataset_name}_results.json")
    
    # Load existing results if available
    existing_results = []
    existing_indices = set()
    
    if os.path.exists(output_json):
        logger.info(f"Found existing results file: {output_json}")
        try:
            with open(output_json, "r", encoding="utf-8") as f:
                existing_results = json.load(f)
            existing_indices = {r["index"] for r in existing_results if "index" in r}
            logger.info(f"Loaded {len(existing_results)} existing results, index range: {min(existing_indices) if existing_indices else 'N/A'} - {max(existing_indices) if existing_indices else 'N/A'}")
        except Exception as e:
            logger.warning(f"Failed to load existing results: {str(e)}, will create new file")
            existing_results = []
            existing_indices = set()
    
    results = existing_results.copy()
    
    # Phase 1: Expert reasoning
    logger.info("\n" + "=" * 80)
    logger.info("Phase 1: Expert reasoning")
    logger.info("=" * 80)
    
    # Process each data item
    for idx in tqdm(range(start_idx, end_idx), desc="Expert reasoning"):
        # Check if already processed
        if idx in existing_indices:
            logger.info(f"Index {idx} already processed, skipping")
            continue
        
        row = df.iloc[idx]
        target = str(row["target"]).strip()
        text = str(row["text"]).strip()
        label =str(row["label"]).strip()
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing item {idx}")
        logger.info(f"Target: {target}")
        logger.info(f"Text: {text[:100]}...")
        
        try:
            # Get knowledge from dataset (prepared in Phase 0)
            raw_knowledge = str(row.get("raw_knowledge", ""))
            esl = str(row.get("ESL", ""))
            
            # Expert reasoning
            if not raw_knowledge or not esl:
                logger.warning(f"Index {idx} missing required knowledge, skipping expert reasoning")
                result = {
                    "index": idx,
                    "target": target,
                    "text": text,
                    "raw_knowledge": raw_knowledge,
                    "ESL": esl,
                    "error": "Missing required knowledge"
                }
            else:
                expert_results = coordinate_experts(
                    target=target,
                    raw_knowledge=raw_knowledge,
                    esl=esl,
                    text=text
                )
                
                # Extract results
                result = {
                    "index": idx,
                    "target": target,
                    "text": text,
                    "label": label,
                    "raw_knowledge": raw_knowledge,
                    "ESL": esl,
                    "refined_knowledge": expert_results.get("refined_knowledge", ""),
                    "knowledge_expert_response": expert_results.get("knowledge_expert", {}).get("response", ""),
                    "knowledge_expert_stance": expert_results.get("knowledge_expert", {}).get("stance", "UNKNOWN"),
                    "label_expert_response": expert_results.get("label_expert", {}).get("response", ""),
                    "label_expert_stance": expert_results.get("label_expert", {}).get("stance", "UNKNOWN"),
                    "pragmatic_expert_response": expert_results.get("pragmatic_expert", {}).get("response", ""),
                    "pragmatic_expert_stance": expert_results.get("pragmatic_expert", {}).get("stance", "UNKNOWN"),
                    "meta_judge_response": expert_results.get("meta_judge", {}).get("response", ""),
                    "final_stance": expert_results.get("final_stance", "UNKNOWN"),
                }
                
                logger.info(f"Final stance: {result['final_stance']}")
            
            results.append(result)
            
            # Save every 10 items
            if (idx - start_idx + 1) % 10 == 0:
                logger.info(f"Intermediate save (processed {idx - start_idx + 1} items)")
                save_results(results, output_file, output_json)
        
        except Exception as e:
            logger.error(f"Error processing index {idx}: {str(e)}")
            import traceback
            traceback.print_exc()
            
            result = {
                "index": idx,
                "target": target,
                "text": text,
                "error": str(e)
            }
            results.append(result)
    
    # Final save
    logger.info("\n" + "=" * 80)
    logger.info("Saving final results")
    save_results(results, output_file, output_json)
    
    logger.info("=" * 80)
    logger.info("Processing completed!")
    logger.info(f"Results saved to:")
    logger.info(f"  - Excel: {output_file}")
    logger.info(f"  - JSON: {output_json}")
    logger.info("=" * 80)


def save_results(results: list, excel_file: str, json_file: str):
    """Save results to Excel and JSON"""
    sorted_results = sorted(results, key=lambda x: x.get("index", 0))
    
    # Save to Excel
    df_results = pd.DataFrame(sorted_results)
    df_results.to_excel(excel_file, index=False)
    logger.info(f"Results saved to: {excel_file} ({len(sorted_results)} items)")
    
    # Save to JSON
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(sorted_results, f, ensure_ascii=False, indent=2)
    logger.info(f"Detailed results saved to: {json_file}")


def main():
    parser = argparse.ArgumentParser(description="Full stance detection pipeline")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset path")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--start", type=int, default=0, help="Start index (inclusive)")
    parser.add_argument("--end", type=int, default=None, help="End index (exclusive)")
    parser.add_argument("--num_search_results", type=int, default=3, 
                       help="Number of search results")
    parser.add_argument("--top_k_chunks", type=int, default=3, 
                       help="Number of knowledge chunks to select")
    parser.add_argument("--skip_knowledge_preparation", action="store_true",
                       help="Skip knowledge preparation if dataset already has complete knowledge")
    
    args = parser.parse_args()
    
    process_dataset(
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        start_idx=args.start,
        end_idx=args.end,
        num_search_results=args.num_search_results,
        top_k_chunks=args.top_k_chunks,
        skip_knowledge_preparation=args.skip_knowledge_preparation
    )


if __name__ == "__main__":
    main()

