"""
Knowledge selector module for the knowledge preparation phase.
This module selects the most relevant knowledge chunks based on similarity to target keywords.
"""

import numpy as np
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
import logging
from .text_processor import load_embedding_model, compute_embeddings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def compute_relevance_scores(chunks: List[Dict[str, Any]], 
                           target: str, 
                           model: Optional[SentenceTransformer] = None,
                           language: str = "en") -> List[float]:
    """
    Compute relevance scores between chunks and target keyword.
    
    Args:
        chunks: List of processed chunk dictionaries
        target: Target keyword
        model: SentenceTransformer model (if None, will load based on language)
        language: Language code for model selection if model is None
        
    Returns:
        List of relevance scores
    """
    if not chunks:
        return []
        
    if model is None:
        model = load_embedding_model(language)
    
    # Extract chunk texts
    chunk_texts = [chunk["chunk"] for chunk in chunks]
    
    # Get embeddings for chunks if not already computed
    chunk_embeddings = []
    for chunk in chunks:
        if "embedding" in chunk and chunk["embedding"] is not None:
            chunk_embeddings.append(chunk["embedding"])
        else:
            # We'll compute all embeddings at once below
            chunk_embeddings = None
            break
            
    if chunk_embeddings is None:
        chunk_embeddings = compute_embeddings(chunk_texts, model, language)
    
    # Get embedding for target
    target_embedding = compute_embeddings([target], model, language)[0]
    
    # Compute similarity scores
    # Normalize embeddings
    target_norm = np.linalg.norm(target_embedding)
    normalized_target = target_embedding / target_norm
    
    scores = []
    for emb in chunk_embeddings:
        chunk_norm = np.linalg.norm(emb)
        normalized_chunk = emb / chunk_norm
        similarity = np.dot(normalized_target, normalized_chunk)
        scores.append(float(similarity))
    
    return scores

def select_knowledge(processed_chunks: List[Dict[str, Any]], 
                    target: str,
                    top_k: int = 3,
                    language: str = "en") -> List[Dict[str, Any]]:
    """
    Select the most relevant knowledge chunks based on similarity to target.
    
    Args:
        processed_chunks: List of processed chunk dictionaries
        target: Target keyword
        top_k: Number of top chunks to select
        language: Language code for embedding model selection
        
    Returns:
        List of selected knowledge chunks with relevance scores
    """
    if not processed_chunks:
        return []
        
    logger.info(f"Selecting top {top_k} chunks for target: {target}")
    
    # Load model
    model = load_embedding_model(language)
    
    # Compute relevance scores
    relevance_scores = compute_relevance_scores(processed_chunks, target, model, language)
    
    # Sort chunks by relevance
    chunk_scores = list(zip(processed_chunks, relevance_scores))
    sorted_chunks = sorted(chunk_scores, key=lambda x: x[1], reverse=True)
    
    # Select top-k chunks
    selected_chunks = []
    for chunk, score in sorted_chunks[:top_k]:
        selected_chunk = {
            "chunk": chunk["chunk"],
            "relevance_score": score,
            "metadata": chunk["metadata"]
        }
        selected_chunks.append(selected_chunk)
    
    logger.info(f"Selected {len(selected_chunks)} most relevant chunks")
    return selected_chunks

def process_and_select_knowledge(search_results: List[Dict[str, Any]],
                               target: str,
                               top_k: int = 3,
                               language: str = "en") -> List[Dict[str, Any]]:
    """
    Process search results and select the most relevant knowledge chunks.
    
    Args:
        search_results: List of search result dictionaries
        target: Target keyword
        top_k: Number of top chunks to select
        language: Language code
        
    Returns:
        List of selected knowledge chunks
    """
    from .text_processor import process_text
    
    # Process text
    processed_chunks = process_text(search_results, language=language)
    
    # Select knowledge
    selected_chunks = select_knowledge(processed_chunks, target, top_k, language)
    
    return selected_chunks