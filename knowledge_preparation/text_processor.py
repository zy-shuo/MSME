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

def estimate_tokens(text: str, language: str = "en") -> int:
    """
    Estimate the number of tokens in a text.
    
    Args:
        text: Text to estimate
        language: Language code ('en' for English, 'zh' for Chinese)
        
    Returns:
        Estimated number of tokens
    """
    if language.lower() == "zh":
        # For Chinese: count characters (each Chinese character ≈ 2-3 tokens)
        # English words within Chinese text
        chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        other_chars = len(text) - chinese_chars
        return chinese_chars
    else:
        # For English: ~4 characters per token on average
        return len(text) // 4


def chunk_text(text: str, chunk_size: int = 300, overlap: int = 30, language: str = "en") -> List[str]:
    """
    Split text into chunks of approximately chunk_size tokens with overlap.
    
    Args:
        text: Text to chunk
        chunk_size: Approximate number of tokens per chunk
        overlap: Number of tokens to overlap between chunks
        language: Language code for better token estimation
        
    Returns:
        List of text chunks
    """
    # Clean text
    text = re.sub(r'\s+', ' ', text).strip()
    
    if not text:
        return []
    
    # Estimate tokens in the full text
    total_tokens = estimate_tokens(text, language)
    
    # If text is shorter than chunk size, return as is
    if total_tokens <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        # Find approximate end position based on token estimation
        remaining_text = text[start:]
        
        # Binary search for the right character position that gives us chunk_size tokens
        if language.lower() == "zh":
            # For Chinese, estimate character position more carefully
            target_chars = chunk_size  # 1 char = 1 token (user's setting)
        else:
            target_chars = chunk_size * 4  # For English (4 chars = 1 token)
        
        end = min(start + target_chars, len(text))
        
        # Adjust to get closer to exact token count
        chunk_text_temp = text[start:end]
        chunk_tokens = estimate_tokens(chunk_text_temp, language)
        
        # Fine-tune the end position
        while chunk_tokens > chunk_size * 1.1 and end > start + 100:
            end = int(end * 0.9)
            chunk_text_temp = text[start:end]
            chunk_tokens = estimate_tokens(chunk_text_temp, language)
        
        while chunk_tokens < chunk_size * 0.9 and end < len(text):
            end = int(end * 1.1)
            if end > len(text):
                end = len(text)
                break
            chunk_text_temp = text[start:end]
            chunk_tokens = estimate_tokens(chunk_text_temp, language)
        
        # If we've reached the end of text
        if end >= len(text):
            chunks.append(text[start:])
            break
            
        # Try to break at sentence end (period, question mark, exclamation point)
        # For Chinese, also include Chinese punctuation
        if language.lower() == "zh":
            sentence_pattern = r'[.!?。！？]\s*'
        else:
            sentence_pattern = r'[.!?]\s+'
            
        sentence_break = re.search(sentence_pattern, text[max(start, end-200):min(len(text), end+200)])
        if sentence_break:
            end = max(start, end-200) + sentence_break.end()
        else:
            # If no sentence break, try to break at space
            space_break = text.rfind(' ', max(start, end-100), min(len(text), end+100))
            if space_break > start:
                end = space_break
        
        chunks.append(text[start:end].strip())
        
        # Calculate overlap in characters
        if language.lower() == "zh":
            overlap_chars = overlap  # 1 char = 1 token (user's setting)
        else:
            overlap_chars = overlap * 5  # For English (5 chars = 1 token)
            
        start = max(start + 1, end - overlap_chars)
    
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
        chunks = chunk_text(text, chunk_size, overlap, language)
        
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