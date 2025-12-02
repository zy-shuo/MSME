"""
Meta Judge module for the decision aggregation phase.
This module combines analyses from multiple experts and makes a final stance judgment.
"""

import os
import re
import logging
from openai import OpenAI
from typing import Dict, Any, Optional
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
SYSTEM_PROMPT = """You are the ultimate meta-judge of the opinion stance, adept at refining and summarizing content, and making your own decisions."""

USER_PROMPT_TEMPLATE = """Three experts have analyzed the stance of the comment from three perspectives: ① knowledge, ② labels, and ③ rhetorical devices. Please combine the analysis of the three experts, extract useful information, make your own analysis, and ultimately determine a stance of the comment on the target "{target}".  Choose from options A, B, and C.
{ESL}
#Background#: {refined_knowledge}
#Comment#: {text}

#Knowledge Expert#:
{knowledge_expert_response}

#Label Expert#：
{label_expert_response}

#pragmatic expert#:
{pragmatic_expert_response}

#Answer format as follows#:
**Analysis**:
**Stance Judgment**:"""

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

def remove_stance_judgments(response: str) -> str:
    """
    Remove the Stance Judgment sections from expert responses to avoid biasing the meta-judge.
    
    Args:
        response: The expert response text
        
    Returns:
        Response text with stance judgment sections removed
    """
    # Remove "Stance Judgment" sections
    cleaned_response = re.sub(r'\*\*Stance Judgment\*\*:.*?(?=\n\n|\Z)', '', response, flags=re.DOTALL)
    return cleaned_response


def extract_stance_from_response(response: str, max_retries: int = 3) -> str:
    """
    Extract stance from any expert response with retry mechanism.
    Looks for keywords in the "Stance Judgment" section.
    
    Args:
        response: The expert response text
        max_retries: Maximum number of extraction attempts with different strategies
        
    Returns:
        Extracted stance: "FAVOR", "AGAINST", "NONE", or "UNKNOWN"
    """
    if not response:
        return "UNKNOWN"
    
    for attempt in range(max_retries):
        try:
            # Strategy 1: Try to find the Stance Judgment section
            stance_match = re.search(r'\*\*Stance Judgment\*\*:\s*(.+?)(?:\n\n|\Z)', response, re.DOTALL | re.IGNORECASE)
            
            if stance_match:
                stance_text = stance_match.group(1).strip()
            else:
                # Strategy 2: If no explicit section found, use the last paragraph
                paragraphs = response.strip().split('\n\n')
                stance_text = paragraphs[-1] if paragraphs else response
            
            # Convert to lowercase for case-insensitive matching
            stance_text_lower = stance_text.lower()
            
            # Check for stance keywords (order matters: check more specific patterns first)
            # Check for explicit option markers first (A, B, C)
            if re.search(r'\b(option\s*)?a\b', stance_text_lower) or 'favor' in stance_text_lower:
                logger.info(f"Extracted stance: FAVOR (attempt {attempt+1})")
                return "FAVOR"
            elif re.search(r'\b(option\s*)?b\b', stance_text_lower) or 'against' in stance_text_lower:
                logger.info(f"Extracted stance: AGAINST (attempt {attempt+1})")
                return "AGAINST"
            elif re.search(r'\b(option\s*)?c\b', stance_text_lower) or 'none' in stance_text_lower or 'neutral' in stance_text_lower:
                logger.info(f"Extracted stance: NONE (attempt {attempt+1})")
                return "NONE"
            
            # Strategy 3: Try broader search in the entire response
            if attempt == 1:
                stance_text = response
                stance_text_lower = response.lower()
                continue
            
            # If still no match, log warning and return UNKNOWN
            if attempt == max_retries - 1:
                logger.warning(f"Could not extract clear stance after {max_retries} attempts. Text: {stance_text[:100]}...")
                return "UNKNOWN"
                
        except Exception as e:
            logger.error(f"Error extracting stance (attempt {attempt+1}/{max_retries}): {str(e)}")
            if attempt == max_retries - 1:
                return "UNKNOWN"
    
    return "UNKNOWN"


def extract_final_stance(response: str) -> str:
    """
    Extract the final stance from the meta-judge response.
    This is a wrapper for backward compatibility.
    
    Args:
        response: The meta-judge response text
        
    Returns:
        Extracted stance: "FAVOR", "AGAINST", "NONE", or "UNKNOWN"
    """
    return extract_stance_from_response(response, max_retries=3)

def aggregate_decisions(target: str, refined_knowledge: str, esl: str, text: str,
                      knowledge_expert_response: str, label_expert_response: str, 
                      pragmatic_expert_response: str,
                      model: str = "gpt-3.5-turbo", 
                      max_retries: int = 3, 
                      api_key: Optional[str] = None,
                      base_url: Optional[str] = None) -> Dict[str, Any]:
    """
    Aggregate analyses from multiple experts and make a final stance judgment.
    
    Args:
        target: The target entity/topic
        refined_knowledge: Refined background knowledge
        esl: Explicit stance labels
        text: The comment text to analyze
        knowledge_expert_response: Response from knowledge expert
        label_expert_response: Response from label expert
        pragmatic_expert_response: Response from pragmatic expert
        model: OpenAI model to use
        max_retries: Maximum number of retries on API error
        api_key: OpenAI API key
        
    Returns:
        Dictionary containing the response and metadata
    """
    # Setup OpenAI client
    client = setup_openai_client(api_key, base_url)
    
    # Clean expert responses to remove stance judgments
    clean_knowledge_response = remove_stance_judgments(knowledge_expert_response)
    clean_label_response = remove_stance_judgments(label_expert_response)
    clean_pragmatic_response = remove_stance_judgments(pragmatic_expert_response)
    
    # Format user prompt
    user_prompt = USER_PROMPT_TEMPLATE.format(
        target=target,
        refined_knowledge=refined_knowledge,
        ESL=esl,
        text=text,
        knowledge_expert_response=clean_knowledge_response,
        label_expert_response=clean_label_response,
        pragmatic_expert_response=clean_pragmatic_response
    )
    
    # Prepare messages for API call
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]
    
    # Try to call the API with retries
    for attempt in range(max_retries):
        try:
            logger.info(f"Aggregating decisions for target: {target} (Attempt {attempt+1}/{max_retries})")
            
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.0,
                timeout=1000,
                max_tokens=2048,
            )
            
            # Extract response content
            response_content = response.choices[0].message.content
            
            # Create response dictionary
            result = {
                "target": target,
                "expert": "meta_judge",
                "response": response_content,
                "model": model,
                "tokens": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in decision aggregation (Attempt {attempt+1}/{max_retries}): {str(e)}")
            if attempt < max_retries - 1:
                logger.info(f"Retrying in 2 seconds...")
                import time
                time.sleep(2)
            else:
                logger.error("Max retries reached, returning error")
                return {"error": str(e), "expert": "meta_judge", "target": target}