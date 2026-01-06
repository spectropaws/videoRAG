#!/usr/bin/env python3
"""Query the RAG system with questions about video content."""

import click
import yaml
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag_pipeline import RAGPipeline, load_rag_from_config
from src.utils.logging_config import setup_logging


@click.command()
@click.option('--config', '-c', default="config/config.yaml",
              help="Path to configuration file")
@click.option('--question', '-q', default=None,
              help="Single question to ask (non-interactive mode)")
@click.option('--top-k', '-k', default=5, type=int,
              help="Number of documents to retrieve")
@click.option('--score-threshold', '-t', default=0.5, type=float,
              help="Minimum similarity score for retrieval")
@click.option('--no-sources', is_flag=True,
              help="Don't include source information")
@click.option('--interactive', '-i', is_flag=True, default=False,
              help="Start interactive chat mode")
@click.option('--log-level', default="WARNING",
              type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR']),
              help="Logging level")
@click.option('--log-file', default=None,
              help="Log file path (optional)")
def main(
    config: str,
    question: str,
    top_k: int,
    score_threshold: float,
    no_sources: bool,
    interactive: bool,
    log_level: str,
    log_file: str
):
    """Query the RAG system with questions about video content."""
    
    # Setup logging
    logger = setup_logging(log_level, log_file)
    
    try:
        # Load RAG pipeline
        logger.info("Loading RAG pipeline...")
        rag_pipeline = load_rag_from_config(config)
        logger.info("RAG pipeline loaded successfully")
        
        # Display system info
        collection_info = rag_pipeline.vector_db.get_collection_info()
        print("=== Video RAG System ===")
        print(f"Collection: {collection_info['collection_name']}")
        print(f"Documents: {collection_info['document_count']}")
        print(f"Embedding Model: {collection_info['embedding_model']}")
        print("=" * 40)
        
        if collection_info['document_count'] == 0:
            print("⚠️  No documents found in the vector database!")
            print("Please run process_video.py first to add video transcriptions.")
            return
        
        # Interactive mode
        if interactive or question is None:
            rag_pipeline.chat(
                top_k=top_k,
                score_threshold=score_threshold,
                include_sources=not no_sources
            )
        
        # Single question mode
        else:
            print(f"\nQuestion: {question}")
            print("-" * 50)
            
            result = rag_pipeline.generate_response(
                question=question,
                top_k=top_k,
                score_threshold=score_threshold,
                include_sources=not no_sources
            )
            
            print(f"Answer: {result['answer']}")
            print(f"\nConfidence: {result['confidence']:.3f}")
            print(f"Retrieved Documents: {result['retrieved_documents']}")
            
            if result.get('sources') and not no_sources:
                print(f"\nSources ({len(result['sources'])}):")
                for i, source in enumerate(result['sources'], 1):
                    print(f"{i}. {source['source']} (Score: {source['similarity_score']:.3f})")
                    if 'timestamp' in source:
                        ts = source['timestamp']
                        print(f"   Time: {ts['start']:.1f}s - {ts['end']:.1f}s")
                    if 'speakers' in source:
                        print(f"   Speakers: {', '.join(source['speakers'])}")
    
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure the configuration file exists and the model path is correct.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Query failed: {e}")
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
