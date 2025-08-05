"""
Quick command-line tool to process a single target.
This is a simplified version of process_single_target.py with fewer options.
"""

import sys
import os
import json
from knowledge_preparation.retriever import search_target
from knowledge_preparation.text_processor import process_text
from knowledge_preparation.knowledge_selector import select_knowledge
from knowledge_preparation.stance_label_generator import generate_stance_labels, setup_openai_api

def detect_language(text):
    """Simple language detection"""
    for char in text:
        if '\u4e00' <= char <= '\u9fff':
            return 'zh'
    return 'en'

def print_usage():
    print("Usage: python quick_process.py <target> [--knowledge|--stance|--both]")
    print("  <target>: The target keyword to process")
    print("  --knowledge: Only perform knowledge retrieval (default if not specified)")
    print("  --stance: Only perform stance label generation")
    print("  --both: Perform both knowledge retrieval and stance label generation")
    print("\nExample: python quick_process.py \"Climate Change\" --both")

def main():
    # Parse arguments
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)
        
    target = sys.argv[1]
    
    # Default mode is knowledge retrieval
    mode = "knowledge"
    if len(sys.argv) >= 3:
        if sys.argv[2] == "--knowledge":
            mode = "knowledge"
        elif sys.argv[2] == "--stance":
            mode = "stance"
        elif sys.argv[2] == "--both":
            mode = "both"
        else:
            print(f"Unknown option: {sys.argv[2]}")
            print_usage()
            sys.exit(1)
    
    print(f"Processing target: {target}")
    language = detect_language(target)
    print(f"Detected language: {language}")
    
    # Check for API keys
    serpapi_key = os.environ.get("SERPAPI_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")
    
    if (mode == "knowledge" or mode == "both") and not serpapi_key:
        print("Warning: SERPAPI_API_KEY not found in environment variables")
        print("Knowledge retrieval may fail without a valid API key")
    
    if (mode == "stance" or mode == "both") and not openai_key:
        print("Warning: OPENAI_API_KEY not found in environment variables")
        print("Stance label generation may fail without a valid API key")
    
    # Process according to mode
    if mode == "knowledge" or mode == "both":
        print("\n=== Knowledge Retrieval ===")
        try:
            # Search for target
            print("Searching for target...")
            search_results = search_target(target)
            
            if not search_results:
                print("No search results found")
            else:
                print(f"Found {len(search_results)} search results")
                
                # Process text
                print("Processing text...")
                processed_chunks = process_text(search_results, language=language)
                
                # Select knowledge
                print("Selecting relevant knowledge...")
                selected_chunks = select_knowledge(processed_chunks, target, language=language)
                
                # Print results
                print(f"\nSelected {len(selected_chunks)} most relevant chunks:")
                for i, chunk in enumerate(selected_chunks):
                    print(f"\nChunk {i+1} (Score: {chunk['relevance_score']:.4f}):")
                    print(f"Source: {chunk['metadata']['url']}")
                    print(f"Content: {chunk['chunk'][:200]}...")
                    
                # Save results
                filename = f"{target.replace(' ', '_')}_knowledge.json"
                with open(filename, "w", encoding="utf-8") as f:
                    json.dump(selected_chunks, f, ensure_ascii=False, indent=2)
                print(f"\nSaved knowledge results to {filename}")
        except Exception as e:
            print(f"Error in knowledge retrieval: {str(e)}")
    
    if mode == "stance" or mode == "both":
        print("\n=== Stance Label Generation ===")
        try:
            # Setup OpenAI API
            setup_openai_api()
            
            # Generate stance labels
            print("Generating stance labels...")
            stance_result = generate_stance_labels(target)
            
            if "error" in stance_result:
                print(f"Error: {stance_result['error']}")
            else:
                # Print results
                print("\nGenerated Stance Labels:")
                print(stance_result["labels"])
                
                # Save results
                filename = f"{target.replace(' ', '_')}_stance_labels.json"
                with open(filename, "w", encoding="utf-8") as f:
                    json.dump(stance_result, f, ensure_ascii=False, indent=2)
                print(f"\nSaved stance labels to {filename}")
        except Exception as e:
            print(f"Error in stance label generation: {str(e)}")

if __name__ == "__main__":
    main()