#!/usr/bin/env python3
"""
Semantic Search Demo for CoCoDe

This script demonstrates the vector embedding-based semantic search capability of CoCoDe.
It can be used to search a codebase for files and code symbols that are semantically 
related to a search query, rather than just relying on exact keyword matches.
"""

import os
import sys
import argparse
from pprint import pprint

# Add the parent directory to the path so we can import the cocode module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import CoCoDe modules
from cocode.core.model_connector import ModelConnector
from cocode.core.file_indexer import FileIndexer

def setup_parser():
    """Set up the command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Semantic Search Demo for CoCoDe"
    )
    
    parser.add_argument(
        "--work-dir", 
        type=str, 
        default=os.getcwd(),
        help="Working directory (default: current directory)"
    )
    
    parser.add_argument(
        "--api-key", 
        type=str, 
        default=os.environ.get("OPENAI_API_KEY"),
        help="OpenAI API key (default: OPENAI_API_KEY environment variable)"
    )
    
    parser.add_argument(
        "query",
        type=str,
        nargs="?",
        default=None,
        help="Search query (if not provided, will prompt for input)"
    )
    
    return parser

def main():
    """Run the semantic search demo."""
    # Parse command line arguments
    parser = setup_parser()
    args = parser.parse_args()
    
    # Check for API key
    if not args.api_key:
        print("Error: No OpenAI API key provided. Set OPENAI_API_KEY environment variable or use --api-key.")
        sys.exit(1)
    
    # Initialize model connector
    print("Initializing model connector...")
    model_connector = ModelConnector(
        api_key=args.api_key,
        model="gpt-4o",
        temperature=0.2
    )
    
    # Initialize file indexer with model connector for embeddings
    print(f"Initializing file indexer for {args.work_dir}...")
    file_indexer = FileIndexer(
        work_dir=args.work_dir,
        model_connector=model_connector
    )
    
    # Run initial indexing
    print("Indexing codebase and generating embeddings (this may take a while)...")
    result = file_indexer.index(generate_embeddings=True)
    print(f"Indexing complete: {result['file_count']} files, {result['embeddings_count']} embeddings")
    
    # If no query was provided, prompt for one
    query = args.query
    while True:
        if not query:
            query = input("\nEnter a search query (or 'q' to quit): ")
        
        if query.lower() in ['q', 'quit', 'exit']:
            break
        
        # Run semantic search
        print("\n======= Semantic File Search =======")
        file_results = file_indexer.semantic_search_files(query, max_results=5)
        if file_results:
            print(f"Found {len(file_results)} semantically relevant files:")
            for i, result in enumerate(file_results):
                print(f"{i+1}. {result['path']} (relevance: {result.get('relevance', 0.0):.3f})")
        else:
            print("No semantically relevant files found.")
        
        # Run symbol search
        print("\n======= Semantic Symbol Search =======")
        symbol_results = file_indexer.semantic_search_symbols(query, max_results=5)
        if symbol_results:
            print(f"Found {len(symbol_results)} semantically relevant symbols:")
            for i, result in enumerate(symbol_results):
                print(f"{i+1}. {result['symbol']} ({result['type']}) in {result['file']}:{result['line']} (relevance: {result.get('relevance', 0.0):.3f})")
        else:
            print("No semantically relevant symbols found.")
        
        # Find similar files
        print("\n======= Would you like to find similar files? =======")
        show_similar = input("Enter a file path to find similar files (or press Enter to skip): ")
        if show_similar:
            similar_results = file_indexer.find_similar_files(show_similar, max_results=5)
            if isinstance(similar_results, list) and not any('error' in r for r in similar_results):
                print(f"Found {len(similar_results)} files similar to {show_similar}:")
                for i, result in enumerate(similar_results):
                    print(f"{i+1}. {result['path']} (similarity: {result.get('similarity', 0.0):.3f})")
            else:
                print("Error finding similar files:", similar_results)
        
        # Reset for next query
        query = None

if __name__ == "__main__":
    main()