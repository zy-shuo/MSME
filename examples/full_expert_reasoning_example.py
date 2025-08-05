"""
Example script demonstrating how to use the full expert reasoning process on a dataset.
"""

import os
import sys
import pandas as pd
import json
from tqdm import tqdm

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from expert_reasoning.expert_coordinator import coordinate_experts

def process_sample_data():
    """
    Process a small sample dataset with the expert reasoning framework.
    """
    print("=== Full Expert Reasoning Example ===\n")
    
    # Setup OpenAI API key
    # Either set it in environment variable or pass it directly
    # os.environ["OPENAI_API_KEY"] = "your_api_key_here"
    
    # Create a sample dataset
    sample_data = [
        {
            "target": "Atheism",
            "text": "He who exalts himself shall be humbled; and he who humbles himself shall be exalted.Matt 23:12.  ",
            "raw_knowledge": "Atheism, in the broadest sense, is an absence of belief in the existence of deities. Less broadly, atheism is a rejection of the belief that any deities exist. In an even narrower sense, atheism is specifically the position that there are no deities. Atheism is contrasted with theism, which is the belief that at least one deity exists. Historically, evidence of atheistic viewpoints can be traced back to classical antiquity and early Indian philosophy. In the Western world, atheism declined after Christianity gained prominence. The 16th century and the Age of Enlightenment marked the resurgence of atheistic thought in Europe. Atheism achieved a significant position worldwide in the 20th century. Estimates of those who have an absence of belief in a god range from 500 million to 1.1 billion people.Atheist organizations have defended the autonomy of science, freedom of thought, secular ethics and secularism.",
            "ESL": "A. Favor: Support atheism as a valid worldview.\nB. Against: Oppose atheism and advocate for theism.\nC. Neutral/None"
        },

    ]
    
    # Create a DataFrame
    df = pd.DataFrame(sample_data)
    
    # Save sample dataset to Excel
    sample_file = "sample_dataset.xlsx"
    df.to_excel(sample_file, index=False)
    print(f"Created sample dataset: {sample_file}")
    
    try:
        # Process each entry
        results = []
        
        for i, row in tqdm(df.iterrows(), total=len(df), desc="Processing entries"):
            target = row["target"]
            text = row["text"]
            raw_knowledge = row["raw_knowledge"]
            esl = row["ESL"]
            
            print(f"\nProcessing entry {i+1}: {target}")
            
            # Run expert reasoning
            entry_results = coordinate_experts(
                target=target,
                raw_knowledge=raw_knowledge,
                esl=esl,
                text=text
            )
            
            # Extract results
            refined_knowledge = entry_results.get("refined_knowledge", "")
            knowledge_response = entry_results.get("knowledge_expert", {}).get("response", "")
            label_response = entry_results.get("label_expert", {}).get("response", "")
            pragmatic_response = entry_results.get("pragmatic_expert", {}).get("response", "")
            
            # Print results
            print(f"\n=== Results for {target} ===")
            print("Refined Knowledge:")
            print(refined_knowledge)
            
            print("\nKnowledge Expert Response:")
            print(knowledge_response)
            
            print("\nLabel Expert Response:")
            print(label_response)
            
            print("\nPragmatic Expert Response:")
            print(pragmatic_response)
            
            print("\nMeta Judge Response:")
            meta_judge_response = entry_results.get("meta_judge", {}).get("response", "")
            print(meta_judge_response)
            
            # Add to results
            results.append({
                "target": target,
                "text": text,
                "raw_knowledge": raw_knowledge,
                "esl": esl,
                "refined_knowledge": refined_knowledge,
                "knowledge_expert_response": knowledge_response,
                "label_expert_response": label_response,
                "pragmatic_expert_response": pragmatic_response,
                "meta_judge_response": meta_judge_response
            })
        
        # Create output DataFrame
        output_df = pd.DataFrame(results)
        
        # Save results to Excel
        output_file = "expert_reasoning_results.xlsx"
        output_df.to_excel(output_file, index=False)
        print(f"\nResults saved to {output_file}")
        
        # Save detailed results to JSON
        json_file = "expert_reasoning_detailed_results.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Detailed results saved to {json_file}")
        
        # How to use the run_full_expert_reasoning.py script
        print("\n=== Using run_full_expert_reasoning.py ===")
        print("To process a dataset using the command line script:")
        print("python run_full_expert_reasoning.py --dataset sample_dataset.xlsx --output expert_reasoning_output.xlsx")
        
    except ValueError as e:
        print(f"Error: {e}")
        print("Please set your OpenAI API key using:")
        print("  os.environ[\"OPENAI_API_KEY\"] = \"your_api_key_here\"")

if __name__ == "__main__":
    process_sample_data()