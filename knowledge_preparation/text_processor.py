"""
Text processor module for the knowledge preparation phase.
This module handles text chunking and embedding.
"""

import os
import re
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from sentence_transformers import SentenceTransformer
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_embedding_model(language: str = "en") -> SentenceTransformer:
    """
    Load the appropriate embedding model based on language.
    
    Args:
        language: Language code ('en' for English, 'zh' for Chinese)
        
    Returns:
        SentenceTransformer model
    """
    base_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "embedding_models")
    
    if language.lower() == "zh":
        model_path = os.path.join(base_path, "bge-base-zh-v1.5")
        logger.info(f"Loading Chinese embedding model from {model_path}")
    else:
        model_path = os.path.join(base_path, "bge-base-en-v1.5")
        logger.info(f"Loading English embedding model from {model_path}")
    
    try:
        model = SentenceTransformer(model_path)
        return model
    except Exception as e:
        logger.error(f"Error loading embedding model: {str(e)}")
        logger.info("Falling back to default model from Hugging Face")
        
        # Fallback to downloading model from Hugging Face
        if language.lower() == "zh":
            model_name = "BAAI/bge-base-zh-v1.5"
        else:
            model_name = "BAAI/bge-base-en-v1.5"
            
        return SentenceTransformer(model_name)

def chunk_text(text: str, chunk_size: int = 200, overlap: int = 20) -> List[str]:
    """
    Split text into chunks of approximately chunk_size tokens with overlap.
    
    Args:
        text: Text to chunk
        chunk_size: Approximate number of tokens per chunk
        overlap: Number of tokens to overlap between chunks
        
    Returns:
        List of text chunks
    """
    # Simple approximation: assume average token is ~5 characters
    char_size = chunk_size * 5
    overlap_chars = overlap * 5
    
    # Clean text
    text = re.sub(r'\s+', ' ', text).strip()
    
    # If text is shorter than chunk size, return as is
    if len(text) <= char_size:
        return [text] if text else []
    
    chunks = []
    start = 0
    
    while start < len(text):
        # Find a good breaking point near the end of the chunk
        end = start + char_size
        if end >= len(text):
            chunks.append(text[start:])
            break
            
        # Try to break at sentence end (period, question mark, exclamation point)
        sentence_break = re.search(r'[.!?]\s+', text[end-100:end+100])
        if sentence_break:
            end = end - 100 + sentence_break.end()
        else:
            # If no sentence break, try to break at space
            space_break = text.rfind(' ', end-50, end+50)
            if space_break > start:
                end = space_break
        
        chunks.append(text[start:end])
        start = end - overlap_chars if end - overlap_chars > start else start + 1
    
    return chunks

def compute_embeddings(chunks: List[str], model: Optional[SentenceTransformer] = None, 
                      language: str = "en") -> np.ndarray:
    """
    Compute embeddings for text chunks.
    
    Args:
        chunks: List of text chunks
        model: SentenceTransformer model (if None, will load based on language)
        language: Language code for model selection if model is None
        
    Returns:
        Array of embeddings
    """
    if not chunks:
        return np.array([])
        
    if model is None:
        model = load_embedding_model(language)
    
    try:
        embeddings = model.encode(chunks, convert_to_numpy=True, show_progress_bar=False)
        return embeddings
    except Exception as e:
        logger.error(f"Error computing embeddings: {str(e)}")
        return np.array([])

def compute_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity matrix between embeddings.
    
    Args:
        embeddings: Array of embeddings
        
    Returns:
        Similarity matrix
    """
    if len(embeddings) == 0:
        return np.array([])
        
    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized_embeddings = embeddings / norms
    
    # Compute similarity matrix
    similarity_matrix = np.dot(normalized_embeddings, normalized_embeddings.T)
    return similarity_matrix

def deduplicate_chunks(chunks: List[str], embeddings: np.ndarray, 
                      similarity_threshold: float = 0.85) -> Tuple[List[str], np.ndarray]:
    """
    Remove duplicate or highly similar chunks based on embedding similarity.
    
    Args:
        chunks: List of text chunks
        embeddings: Array of embeddings corresponding to chunks
        similarity_threshold: Threshold for considering chunks as duplicates
        
    Returns:
        Tuple of (deduplicated chunks, corresponding embeddings)
    """
    if not chunks or len(embeddings) == 0:
        return [], np.array([])
        
    # Compute similarity matrix
    similarity_matrix = compute_similarity_matrix(embeddings)
    
    # Find duplicates
    to_keep = []
    for i in range(len(chunks)):
        # Check if this chunk is too similar to any chunk we've already decided to keep
        is_duplicate = False
        for j in to_keep:
            if i != j and similarity_matrix[i, j] > similarity_threshold:
                is_duplicate = True
                break
                
        if not is_duplicate:
            to_keep.append(i)
    
    # Keep only non-duplicate chunks and their embeddings
    unique_chunks = [chunks[i] for i in to_keep]
    unique_embeddings = embeddings[to_keep]
    
    logger.info(f"Deduplicated {len(chunks)} chunks to {len(unique_chunks)} unique chunks")
    return unique_chunks, unique_embeddings

def process_text(search_results: List[Dict[str, Any]], 
                chunk_size: int = 500,
                overlap: int = 50,
                language: str = "en") -> List[Dict[str, Any]]:
    """
    Process text from search results: chunk, embed, and deduplicate.
    
    Args:
        search_results: List of search result dictionaries with 'extracted_text' field
        chunk_size: Approximate number of tokens per chunk
        overlap: Number of tokens to overlap between chunks
        language: Language code for embedding model selection
        
    Returns:
        List of dictionaries containing processed text chunks and metadata
    """
    if not search_results:
        return []
    
    logger.info(f"Processing {len(search_results)} search results")
    
    # Load embedding model
    model = load_embedding_model(language)
    
    all_chunks = []
    all_metadata = []
    
    # Process each search result
    for result in search_results:
        text = result.get("extracted_text", "")
        if not text:
            continue
            
        # Chunk text
        chunks = chunk_text(text, chunk_size, overlap)
        
        # Store metadata for each chunk
        for chunk in chunks:
            all_chunks.append(chunk)
            all_metadata.append({
                "url": result.get("url", ""),
                "title": result.get("title", ""),
                "source_text": text[:100] + "..." if len(text) > 100 else text
            })
    
    if not all_chunks:
        return []
        
    # Compute embeddings
    embeddings = compute_embeddings(all_chunks, model, language)
    
    # Deduplicate chunks
    unique_chunks, unique_embeddings = deduplicate_chunks(all_chunks, embeddings)
    
    # Create final processed results
    processed_results = []
    for i, (chunk, embedding) in enumerate(zip(unique_chunks, unique_embeddings)):
        # Find original metadata for this chunk
        original_idx = all_chunks.index(chunk)
        metadata = all_metadata[original_idx]
        
        processed_results.append({
            "chunk": chunk,
            "embedding": embedding,
            "metadata": metadata
        })
    
    logger.info(f"Created {len(processed_results)} processed chunks")
    return processed_results