"""
Example script demonstrating how to use the stance label generator module.
"""

import os
import sys
import json

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from knowledge_preparation.stance_label_generator import generate_stance_labels, setup_openai_api

def main():
    # Setup OpenAI API key
    # Either set it in environment variable or pass it directly
    # os.environ["OPENAI_API_KEY"] = "your_api_key_here"
    try:
        setup_openai_api()
    except ValueError as e:
        print(f"Error: {e}")
        print("Please set your OpenAI API key using:")
        print("  os.environ[\"OPENAI_API_KEY\"] = \"your_api_key_here\"")
        return
    
    # Example targets
    example_targets = [
        "Atheism",

    ]
    
    print("Generating explicit stance labels for example targets...\n")
    
    results = {}
    for target in example_targets:
        print(f"Processing target: {target}")
        
        # Generate stance labels
        result = generate_stance_labels(target)
        results[target] = result
        
        # Print results
        if result.get("error"):
            print(f"Error: {result['error']}")
        else:
            print("\nGenerated Stance Labels:")
            print(result["labels"])
            print(f"Tokens used: {result['tokens']['total_tokens']}")
        
        print("\n" + "-"*50 + "\n")
    
    # Save results to file
    output_file = "stance_labels_example_output.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
        
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()