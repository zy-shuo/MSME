"""
Retriever module for the knowledge preparation phase.
This module handles web search using SerpAPI and extracts text content from web pages.
"""

import os
from dotenv import load_dotenv
load_dotenv()
import requests
from bs4 import BeautifulSoup
from serpapi import GoogleSearch
import logging
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def search_target(target: str, api_key: Optional[str] = None, num_results: int = 3, 
                 use_news: bool = True, location: str = "China") -> List[Dict[str, Any]]:
    """
    Search for target keyword using SerpAPI and extract text from the top search results.
    
    Args:
        target: The target keyword to search for
        api_key: SerpAPI API key (if None, will try to get from environment variable)
        num_results: Number of search results to retrieve
        use_news: If True, search news results (tbm=nws); if False, search organic results
        location: Location parameter for the search (default: "China")
        
    Returns:
        List of dictionaries containing search result data and extracted text
    """
    if api_key is None:
        api_key = os.getenv("SERPAPI_API_KEY")
        if api_key is None:
            raise ValueError("SerpAPI API key not provided and not found in environment variables")
    
    logger.info(f"Searching for target: {target} (use_news={use_news}, location={location})")
    
    # Perform search using SerpAPI
    search_params = {
        "q": target,
        "api_key": api_key,
        "engine": "google",
        "location": location
    }
    
    # Add news search parameter if needed
    if use_news:
        search_params["tbm"] = "nws"
    else:
        search_params["num"] = num_results
    
    try:
        search = GoogleSearch(search_params)
        results = search.get_dict()
        
        if "error" in results:
            logger.error(f"SerpAPI error: {results['error']}")
            return []
        
        # Determine which results field to use based on search type
        if use_news:
            results_key = "news_results"
        else:
            results_key = "organic_results"
            
        if results_key not in results or len(results[results_key]) == 0:
            logger.warning(f"No {results_key} found for target: {target}")
            return []
            
        search_results = results[results_key][:num_results]
        logger.info(f"Found {len(search_results)} {results_key}")
        
        # Extract text from each search result
        processed_results = []
        for i, result in enumerate(search_results):
            if "link" not in result:
                logger.warning(f"No link found in result {i+1}")
                continue
                
            url = result["link"]
            title = result.get("title", "")
            snippet = result.get("snippet", "")
            source = result.get("source", "")
            date = result.get("date", "")
            
            logger.info(f"Processing result {i+1}: {title}")
            
            # Extract text from the webpage
            extracted_text = extract_text_from_url(url)
            
            if not extracted_text:
                logger.warning(f"Could not extract text from {url}")
                extracted_text = snippet  # Use snippet as fallback
                
            processed_results.append({
                "url": url,
                "title": title,
                "snippet": snippet,
                "source": source,
                "date": date,
                "extracted_text": extracted_text
            })
            
        return processed_results
        
    except Exception as e:
        logger.error(f"Error during search: {str(e)}")
        return []

def extract_text_from_url(url: str) -> str:
    """
    Extract text content from a webpage.
    
    Args:
        url: URL of the webpage to extract text from
        
    Returns:
        Extracted text content
    """
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, "html.parser")
        
        # Remove script and style elements
        for element in soup(["script", "style", "nav", "footer", "header", "aside"]):
            element.decompose()
            
        # Get text
        text = soup.get_text(separator=" ", strip=True)
        
        # Clean up text (remove excessive whitespace)
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = " ".join(chunk for chunk in chunks if chunk)
        
        return text
        
    except Exception as e:
        logger.error(f"Error extracting text from {url}: {str(e)}")
        return ""

def process_dataset_targets(dataset_path: str, output_path: Optional[str] = None,
                          use_news: bool = True, location: str = "China",
                          num_results: int = 3) -> Dict[str, List[Dict[str, Any]]]:
    """
    Process all unique targets in a dataset and retrieve knowledge for each.
    
    Args:
        dataset_path: Path to the dataset file
        output_path: Path to save the results (optional)
        use_news: If True, search news results; if False, search organic results
        location: Location parameter for the search
        num_results: Number of search results to retrieve per target
        
    Returns:
        Dictionary mapping targets to their search results
    """
    import pandas as pd
    
    logger.info(f"Processing dataset: {dataset_path}")
    
    try:
        df = pd.read_excel(dataset_path)
        
        if "target" not in df.columns:
            logger.error(f"No 'target' column found in {dataset_path}")
            return {}
            
        # Get unique targets
        targets = df["target"].unique()
        logger.info(f"Found {len(targets)} unique targets in dataset")
        
        # Process each target
        results = {}
        for target in targets:
            if pd.isna(target) or not target.strip():
                continue
                
            target_str = str(target).strip()
            logger.info(f"Processing target: {target_str}")
            
            search_results = search_target(target_str, use_news=use_news, 
                                         location=location, num_results=num_results)
            results[target_str] = search_results
            
        # Save results if output path is provided
        if output_path:
            import json
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
                
        return results
        
    except Exception as e:
        logger.error(f"Error processing dataset: {str(e)}")
        return {}