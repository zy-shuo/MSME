"""
Example script demonstrating how to use the knowledge preparation module.
"""

import os
import sys
import json

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from knowledge_preparation.retriever import search_target
from knowledge_preparation.text_processor import process_text
from knowledge_preparation.knowledge_selector import select_knowledge

def main():
    # Example target
    target = "Atheism"
    print(f"Processing target: {target}")
    
    # Step 1: Search for target and retrieve web content
    print("\n=== Step 1: Web Search ===")
    # Note: You need to set your SerpAPI API key as an environment variable or pass it directly
    # os.environ["SERPAPI_API_KEY"] = "your_api_key_here"
    search_results = search_target(target)
    
    if not search_results:
        print("No search results found. Please check your API key and internet connection.")
        return
        
    print(f"Retrieved {len(search_results)} search results")
    
    # Print first result snippet
    print("\nFirst result snippet:")
    print(f"Title: {search_results[0]['title']}")
    print(f"URL: {search_results[0]['url']}")
    print(f"Snippet: {search_results[0]['snippet']}")
    print(f"Extracted text length: {len(search_results[0]['extracted_text'])} characters")
    
    # Step 2: Process text (chunk, embed, deduplicate)
    print("\n=== Step 2: Text Processing ===")
    processed_chunks = process_text(search_results)
    
    print(f"Created {len(processed_chunks)} processed chunks after deduplication")
    
    # Print first chunk
    if processed_chunks:
        print("\nFirst chunk:")
        print(f"Source: {processed_chunks[0]['metadata']['url']}")
        print(f"Content: {processed_chunks[0]['chunk'][:200]}...")
    
    # Step 3: Select relevant knowledge
    print("\n=== Step 3: Knowledge Selection ===")
    selected_chunks = select_knowledge(processed_chunks, target)
    
    print(f"Selected {len(selected_chunks)} most relevant chunks")
    
    # Print selected chunks with relevance scores
    for i, chunk in enumerate(selected_chunks):
        print(f"\nSelected Chunk {i+1} (Score: {chunk['relevance_score']:.4f}):")
        print(f"Source: {chunk['metadata']['url']}")
        print(f"Content: {chunk['chunk'][:200]}...")
    
    # Save results to file
    output_file = "knowledge_example_output.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "target": target,
            "selected_chunks": selected_chunks
        }, f, ensure_ascii=False, indent=2)
        
    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    main()