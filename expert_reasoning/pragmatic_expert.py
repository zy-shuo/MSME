"""
Pragmatic Expert module for the expert reasoning phase.
This module analyzes rhetorical devices and logical relationships in comments.
"""

import os
import logging
from openai import OpenAI
from typing import Dict, Any, Optional
from dotenv import load_dotenv
from .meta_judge import extract_stance_from_response

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# System and user prompt templates
SYSTEM_PROMPT = """You are a pragmatic expert who excels in analyzing the rhetorical devices used in sentences, such as irony, metaphor, rhetorical questions, and sarcasm. You are skilled in analyzing the logical relationships contained in sentences, such as concessions, assumptions, and twists, and analyzing the actual meaning of sentences."""

USER_PROMPT_TEMPLATE = """Please analyze the rhetorical devices and logical relationships contained in the comments regarding the target "{target}" to understand the actual intent of the comment and determine its stance. Choosing from options A, B, C.
#Background#：{refined_knowledge}
#Original Stance Labels#:  
{ESL}
#Comment#:{text}
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

def analyze_pragmatics(target: str, refined_knowledge: str, esl: str, text: str, 
                     model: str = "gpt-3.5-turbo", 
                     max_retries: int = 3, 
                     api_key: Optional[str] = None,
                     base_url: Optional[str] = None) -> Dict[str, Any]:
    """
    Analyze rhetorical devices and logical relationships using the pragmatic expert.
    
    Args:
        target: The target entity/topic
        refined_knowledge: Refined background knowledge
        esl: Explicit stance labels
        text: The comment text to analyze
        model: OpenAI model to use
        max_retries: Maximum number of retries on API error
        api_key: OpenAI API key
        
    Returns:
        Dictionary containing the response and metadata
    """
    # Setup OpenAI client
    client = setup_openai_client(api_key, base_url)
    
    # Format user prompt
    user_prompt = USER_PROMPT_TEMPLATE.format(
        target=target,
        refined_knowledge=refined_knowledge,
        ESL=esl,
        text=text
    )
    
    # Prepare messages for API call
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]
    
    # Try to call the API with retries
    for attempt in range(max_retries):
        try:
            logger.info(f"Analyzing pragmatics for target: {target} (Attempt {attempt+1}/{max_retries})")
            
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.0,
                timeout=1000,
                max_tokens=2048,
            )
            
            # Extract response content
            response_content = response.choices[0].message.content
            
            # Extract stance judgment with retry mechanism
            stance = extract_stance_from_response(response_content, max_retries=3)
            logger.info(f"Pragmatic Expert stance: {stance}")
            
            # Create response dictionary
            result = {
                "target": target,
                "expert": "pragmatic_expert",
                "response": response_content,
                "stance": stance,
                "model": model,
                "tokens": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in pragmatic analysis (Attempt {attempt+1}/{max_retries}): {str(e)}")
            if attempt < max_retries - 1:
                logger.info(f"Retrying in 2 seconds...")
                import time
                time.sleep(2)
            else:
                logger.error("Max retries reached, returning error")
                return {"error": str(e), "expert": "pragmatic_expert", "target": target}