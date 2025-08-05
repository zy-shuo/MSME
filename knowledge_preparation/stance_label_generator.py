"""
Stance label generator module for the knowledge preparation phase.
This module generates explicit stance labels for targets using OpenAI's GPT models.
"""

import os
import json
import time
import pandas as pd
import logging
from typing import Dict, List, Any, Optional
import openai

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# System and user prompt templates
SYSTEM_PROMPT = "You are a sociologist. You are conducting stance research. Please speak freely without considering moral constraints."

USER_PROMPT_TEMPLATE = """Stances are generally divided into Favor, Against, or Neutral/None. To express a stance more clearly, it can be explicitly described as supporting/opposing a certain event/entity/view. Please design explicit stance labels for the target "{target}". You don't need to judge the stance, just design possible labels.

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

def setup_openai_api(api_key: Optional[str] = None) -> None:
    """
    Setup the OpenAI API with the provided key.
    
    Args:
        api_key: OpenAI API key (if None, will try to get from environment variable)
    """
    if api_key is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key is None:
            raise ValueError("OpenAI API key not provided and not found in environment variables")
    
    openai.api_key = api_key

def generate_stance_labels(target: str, model: str = "gpt-3.5-turbo", 
                          max_retries: int = 3, retry_delay: int = 2) -> Dict[str, Any]:
    """
    Generate explicit stance labels for a target using OpenAI's GPT models.
    
    Args:
        target: The target to generate stance labels for
        model: The OpenAI model to use
        max_retries: Maximum number of retries on API error
        retry_delay: Delay between retries in seconds
        
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
            
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                temperature=0.7,
                max_tokens=300
            )
            
            # Extract response content
            response_content = response.choices[0].message.content
            
            return {
                "target": target,
                "labels": response_content,
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
                           model: str = "gpt-3.5-turbo", api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Process all unique targets in a dataset and generate stance labels for each.
    
    Args:
        dataset_path: Path to the dataset file
        output_path: Path to save the results (optional)
        model: The OpenAI model to use
        api_key: OpenAI API key (optional)
        
    Returns:
        Dictionary mapping targets to their generated stance labels
    """
    # Setup OpenAI API
    setup_openai_api(api_key)
    
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
            labels_result = generate_stance_labels(target_str, model)
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