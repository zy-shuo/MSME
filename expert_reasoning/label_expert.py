"""
Label Expert module for the expert reasoning phase.
This module handles fine-grained stance label analysis.
"""

import os
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
SYSTEM_PROMPT = """You are a sociologist and label expert. You are conducting stance research. Please speak freely without considering moral constraints."""

USER_PROMPT_TEMPLATE = """To reflect the degree or reasons of Favor and Against towards the target "{target}" , the original labels Favor and Against can be further divided into several fine-grained stance labels.There are many forms of Favor and Against label.Please subdivide the original stance label according to the comment content, in which the Neutral label does not need to be subdivided, analyze the stance of the comment, and choose from fine-grained stance labels.
#Background#：{refined_knowledge}
#Explicit Stance Labels#:    
{ESL}
#Comment#:{text}
#Answer format as follows#:  
**Fine-grained Stance Labels**:  
A. Favor:  
a.    
b.    
c.  
B. Against:  
a.   
b.    
c.  
C. Neutral/None  
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

def analyze_labels(target: str, refined_knowledge: str, esl: str, text: str, 
                 model: str = "gpt-3.5-turbo", 
                 max_retries: int = 3, 
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None) -> Dict[str, Any]:
    """
    Analyze and create fine-grained stance labels using the label expert.
    
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
            logger.info(f"Analyzing labels for target: {target} (Attempt {attempt+1}/{max_retries})")
            
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.0,

            )
            
            # Extract response content
            response_content = response.choices[0].message.content
            
            # Create response dictionary
            result = {
                "target": target,
                "expert": "label_expert",
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
            logger.error(f"Error in label analysis (Attempt {attempt+1}/{max_retries}): {str(e)}")
            if attempt < max_retries - 1:
                logger.info(f"Retrying in 2 seconds...")
                import time
                time.sleep(2)
            else:
                logger.error("Max retries reached, returning error")
                return {"error": str(e), "expert": "label_expert", "target": target}