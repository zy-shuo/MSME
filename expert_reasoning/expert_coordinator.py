"""
Expert Coordinator module for the expert reasoning phase.
This module coordinates the analysis from multiple experts and aggregates their decisions.
"""

import logging
from typing import Dict, Any, List, Optional
from .knowledge_expert import analyze_knowledge
from .label_expert import analyze_labels
from .pragmatic_expert import analyze_pragmatics
from .meta_judge import aggregate_decisions, extract_final_stance

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def coordinate_experts(target: str, raw_knowledge: str, esl: str, text: str, 
                     model: str = "gpt-3.5-turbo", 
                     api_key: Optional[str] = None,
                     base_url: Optional[str] = None) -> Dict[str, Any]:
    """
    Coordinate analysis from multiple experts for stance detection.
    
    Args:
        target: The target entity/topic
        raw_knowledge: Raw background knowledge
        esl: Explicit stance labels
        text: The comment text to analyze
        model: OpenAI model to use
        api_key: OpenAI API key
        
    Returns:
        Dictionary containing results from all experts
    """
    logger.info(f"Starting expert reasoning for target: {target}")
    
    results = {
        "target": target,
        "text": text,
        "esl": esl,
        "raw_knowledge": raw_knowledge
    }
    
    # Step 1: Knowledge Expert Analysis
    try:
        logger.info("Running Knowledge Expert analysis")
        refined_knowledge, knowledge_response = analyze_knowledge(
            target=target,
            raw_knowledge=raw_knowledge,
            esl=esl,
            text=text,
            model=model,
            api_key=api_key,
            base_url=base_url
        )
        
        results["knowledge_expert"] = knowledge_response
        results["refined_knowledge"] = refined_knowledge
        
        # If knowledge expert fails, we can't continue
        if "error" in knowledge_response:
            logger.error("Knowledge Expert analysis failed, skipping other experts")
            return results
            
    except Exception as e:
        logger.error(f"Error in Knowledge Expert coordination: {str(e)}")
        results["knowledge_expert"] = {"error": str(e)}
        results["refined_knowledge"] = ""
        return results
    
    # Step 2: Label Expert Analysis
    try:
        logger.info("Running Label Expert analysis")
        label_response = analyze_labels(
            target=target,
            refined_knowledge=refined_knowledge,
            esl=esl,
            text=text,
            model=model,
            api_key=api_key,
            base_url=base_url
        )
        
        results["label_expert"] = label_response
        
    except Exception as e:
        logger.error(f"Error in Label Expert coordination: {str(e)}")
        results["label_expert"] = {"error": str(e)}
    
    # Step 3: Pragmatic Expert Analysis
    try:
        logger.info("Running Pragmatic Expert analysis")
        pragmatic_response = analyze_pragmatics(
            target=target,
            refined_knowledge=refined_knowledge,
            esl=esl,
            text=text,
            model=model,
            api_key=api_key,
            base_url=base_url
        )
        
        results["pragmatic_expert"] = pragmatic_response
        
    except Exception as e:
        logger.error(f"Error in Pragmatic Expert coordination: {str(e)}")
        results["pragmatic_expert"] = {"error": str(e)}
    
    # Step 4: Meta Judge Decision Aggregation
    try:
        # Only proceed if we have all three expert analyses
        if ("knowledge_expert" in results and "error" not in results["knowledge_expert"] and
            "label_expert" in results and "error" not in results["label_expert"] and
            "pragmatic_expert" in results and "error" not in results["pragmatic_expert"]):
            
            logger.info("Running Meta Judge decision aggregation")
            meta_judge_response = aggregate_decisions(
                target=target,
                refined_knowledge=refined_knowledge,
                esl=esl,
                text=text,
                knowledge_expert_response=results["knowledge_expert"]["response"],
                label_expert_response=results["label_expert"]["response"],
                pragmatic_expert_response=results["pragmatic_expert"]["response"],
                model=model,
                api_key=api_key,
                base_url=base_url
            )
            
            results["meta_judge"] = meta_judge_response
            
            # Extract final stance from meta judge response
            if "response" in meta_judge_response:
                final_stance = extract_final_stance(meta_judge_response["response"])
                results["final_stance"] = final_stance
                logger.info(f"Final stance extracted: {final_stance}")
            else:
                results["final_stance"] = "UNKNOWN"
                logger.warning("No response from meta judge, setting final_stance to UNKNOWN")
        else:
            logger.warning("Skipping Meta Judge due to missing expert analyses")
            results["final_stance"] = "UNKNOWN"
            
    except Exception as e:
        logger.error(f"Error in Meta Judge coordination: {str(e)}")
        results["meta_judge"] = {"error": str(e)}
        results["final_stance"] = "UNKNOWN"
    
    logger.info("Expert reasoning and decision aggregation completed successfully")
    return results