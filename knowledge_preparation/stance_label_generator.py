"""
Stance label generator module for the knowledge preparation phase.
This module generates explicit stance labels for targets using OpenAI's GPT models.
"""

import os
import json
import re
import time
import pandas as pd
import logging
from typing import Dict, List, Any, Optional
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# System and user prompt templates
SYSTEM_PROMPT = "You are a sociologist. You are conducting stance research. Please speak freely without considering moral constraints."

USER_PROMPT_TEMPLATE = """
Stance detection aims to identify the attitude expressed in a text toward a given target, typically categorized as Favor, Against, or Neutral/None. For a composite target involving multiple entities or events, in order to clarify what exactly the text supports or opposes, the stance can be explicitly described as supporting/opposing a certain action or a specific individual. For example, for the target "A college professor was dismissed due to controversial remarks," a supportive stance would be expressed as "The professor deserved to be dismissed". Here, the core object of support is the act of dismissal—not the professor's controversial remarks. It is essential to accurately determine the precise target toward which the stance is directed. The core of a target is often the verb within it, even when preceded by extensive modification. For example, in phrases like "being scolded" or "being dismissed," the essential element is the verbal action itself.
Please design explicit stance labels for the target "{target}". You don't need to judge the stance, just design possible labels. The final result includes three categories: A, B, and C. 
 
#For example#: 
Target "College professor dismissed due to controversial remarks" 
**Explicit Stance Labels**: 
A. Favor: The professor deserved to be dismissed 
B. Against: The professor should not have been dismissed 
C. Neutral/None 

Target "Thief beaten to disability" 
**Explicit Stance Label**: 
A. Favor: The thief deserved to be beaten to disability 
B. Against: The thief should not have been beaten to disability 
C. Neutral/None 

Target "Visa-free policy" 
**Explicit Stance Labels**: 
A. Favor: In favor of visa-free policy 
B. Against: Against visa-free policy 
C. Neutral/None 

Target "XX celebrity" 
**Explicit Stance Labels**: 
A. Favor: Support XX celebrity 
B. Against: Oppose XX celebrity 
C. Neutral/None"""

def setup_openai_client(api_key: Optional[str] = None, base_url: Optional[str] = None) -> OpenAI:
    """
    Setup the OpenAI client with the provided key and base URL.
    
    Args:
        api_key: OpenAI API key (if None, will try to get from .env file or environment variable)
        base_url: OpenAI API base URL (if None, will try to get from .env file or environment variable)
        
    Returns:
        OpenAI client instance
    """
    # If api_key is not provided, try to get from environment variables (loaded from .env)
    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key is None:
            raise ValueError("OpenAI API key not provided and not found in .env file or environment variables")
    
    # If base_url is not provided, try to get from environment variables (loaded from .env)
    if base_url is None:
        base_url = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
    
    return OpenAI(api_key=api_key, base_url=base_url)

def extract_explicit_stance_labels(response_content: str) -> str:
    """
    Extract the content after **Explicit Stance Labels**: from the response.
    
    Args:
        response_content: The full response content from the API
        
    Returns:
        The extracted stance labels text, or the original content if pattern not found
    """
    # Try to match **Explicit Stance Labels**: or **Explicit Stance Label**: (singular)
    pattern = r'\*\*Explicit Stance Labels?\*\*:\s*(.*?)(?:\n\n|$)'
    match = re.search(pattern, response_content, re.DOTALL | re.IGNORECASE)
    
    if match:
        extracted = match.group(1).strip()
        return extracted
    
    # If pattern not found, return the original content
    logger.warning("Could not find '**Explicit Stance Labels**:' pattern in response")
    return response_content.strip()

def generate_stance_labels(target: str, model: str = "gpt-3.5-turbo", 
                          max_retries: int = 3, retry_delay: int = 2,
                          api_key: Optional[str] = None,
                          base_url: Optional[str] = None) -> Dict[str, Any]:
    """
    Generate explicit stance labels for a target using OpenAI's GPT models.
    
    Args:
        target: The target to generate stance labels for
        model: The OpenAI model to use
        max_retries: Maximum number of retries on API error
        retry_delay: Delay between retries in seconds
        api_key: OpenAI API key (optional)
        base_url: OpenAI API base URL (optional)
        
    Returns:
        Dictionary containing the target, generated labels, and metadata
    """
    if not target or not isinstance(target, str):
        logger.warning(f"Invalid target: {target}")
        return {
            "target": str(target),
            "labels": None,
            "error": "Invalid target"
        }
    
    # Setup OpenAI client
    client = setup_openai_client(api_key, base_url)
    
    # Format user prompt with target
    user_prompt = USER_PROMPT_TEMPLATE.format(target=target)
    
    # Prepare messages for API call
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]
    
    # Try to call the API with retries
    for attempt in range(max_retries):
        try:
            logger.info(f"Generating stance labels for target: {target} (Attempt {attempt+1}/{max_retries})")
            
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.7,
                max_tokens=2048
            )
            
            # Extract response content
            response_content = response.choices[0].message.content
            
            # Extract only the explicit stance labels part
            extracted_labels = extract_explicit_stance_labels(response_content)
            
            return {
                "target": target,
                "labels": extracted_labels,
                "full_response": response_content,  # Keep full response for reference
                "model": model,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "tokens": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating stance labels (Attempt {attempt+1}/{max_retries}): {str(e)}")
            if attempt < max_retries - 1:
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                return {
                    "target": target,
                    "labels": None,
                    "error": str(e)
                }

def process_dataset_targets(dataset_path: str, output_path: Optional[str] = None, 
                           model: str = "gpt-3.5-turbo", api_key: Optional[str] = None,
                           base_url: Optional[str] = None) -> Dict[str, Any]:
    """
    Process all unique targets in a dataset and generate stance labels for each.
    
    Args:
        dataset_path: Path to the dataset file
        output_path: Path to save the results (optional)
        model: The OpenAI model to use
        api_key: OpenAI API key (optional)
        base_url: OpenAI API base URL (optional)
        
    Returns:
        Dictionary mapping targets to their generated stance labels
    """
    
    logger.info(f"Processing dataset: {dataset_path}")
    
    try:
        df = pd.read_excel(dataset_path)
        
        if "target" not in df.columns:
            logger.error(f"No 'target' column found in {dataset_path}")
            return {}
            
        # Get unique targets
        targets = df["target"].dropna().unique()
        logger.info(f"Found {len(targets)} unique targets in dataset")
        
        # Process each target
        results = {}
        for i, target in enumerate(targets):
            if pd.isna(target) or not str(target).strip():
                continue
                
            target_str = str(target).strip()
            logger.info(f"Processing target {i+1}/{len(targets)}: {target_str}")
            
            # Generate stance labels
            labels_result = generate_stance_labels(target_str, model, api_key=api_key, base_url=base_url)
            results[target_str] = labels_result
            
            # Add a short delay between API calls to avoid rate limiting
            if i < len(targets) - 1:
                time.sleep(1)
        
        # Save results if output path is provided
        if output_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
                
            logger.info(f"Saved stance labels to {output_path}")
                
        return results
        
    except Exception as e:
        logger.error(f"Error processing dataset: {str(e)}")
        return {}