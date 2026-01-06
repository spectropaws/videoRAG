"""RAG pipeline using Llama CPP for inference."""

import logging
import yaml
from pathlib import Path
from typing import List, Dict, Any, Optional
from llama_cpp import Llama
from .vector_database import VectorDatabase


logger = logging.getLogger("video_rag.rag_pipeline")


class RAGPipeline:
    """Retrieval-Augmented Generation pipeline."""
    
    def __init__(
        self,
        model_path: str,
        vector_db: VectorDatabase,
        context_length: int = 4096,
        max_tokens: int = 512,
        temperature: float = 0.1,
        top_p: float = 0.95,
        n_threads: int = 4,
        system_prompt_template: Optional[str] = None
    ):
        """Initialize RAG pipeline.
        
        Args:
            model_path: Path to GGUF model file
            vector_db: Vector database instance
            context_length: Maximum context length
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            n_threads: Number of threads for inference
            system_prompt_template: Template for system prompt
        """
        self.model_path = Path(model_path)
        self.vector_db = vector_db
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        
        # Validate model path
        if not self.model_path.exists():
            raise ValueError(f"Model file does not exist: {model_path}")
        
        # Default system prompt template
        if system_prompt_template is None:
            system_prompt_template = """You are a helpful assistant that answers questions based on the provided context from video transcripts.
Use only the information from the context to answer questions. If the answer is not in the context, say so clearly.
Be concise and accurate in your responses."""
        
        self.system_prompt_template = system_prompt_template
        
        # Initialize Llama model
        logger.info(f"Loading Llama model: {model_path}")
        self.llm = Llama(
            model_path=str(model_path),
            n_ctx=context_length,
            n_threads=n_threads,
            n_gpu_layers=35,
            verbose=True
        )
        
        logger.info("RAG pipeline initialized successfully")
    
    def generate_response(
        self,
        question: str,
        top_k: int = 5,
        score_threshold: float = 0.5,
        include_sources: bool = True
    ) -> Dict[str, Any]:
        """Generate response to a question using RAG.
        
        Args:
            question: User question
            top_k: Number of documents to retrieve
            score_threshold: Minimum similarity score for retrieval
            include_sources: Whether to include source information
            
        Returns:
            Dictionary containing answer and metadata
        """
        logger.info(f"Processing question: {question}")
        
        # Retrieve relevant documents
        retrieved_docs = self.vector_db.search(
            query=question,
            n_results=top_k,
            score_threshold=score_threshold
        )
        
        """
        if not retrieved_docs:
            return {
                "answer": "I couldn't find any relevant information in the video transcripts to answer your question.",
                "sources": [],
                "retrieved_documents": 0,
                "confidence": 0.0
            }
        """
        
        # Prepare context
        context = self._prepare_context(retrieved_docs)
        
        # Generate prompt
        user_prompt = """
Context: {context}

Question: {question}"""
        user_prompt = user_prompt.format(
            context=context,
            question=question
        )

        prompt_template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_prompt_template}<|eot_id|>
<|start_header_id|>user<|end_header_id|>

{user_prompt}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>"""

        phi_prompt_template = """<|system|>
{system_prompt_template}<|end|>
<|user|>
{user_prompt}<|end|>
<|assistant|>"""

        prompt = prompt_template.format(
            system_prompt_template=self.system_prompt_template,
            user_prompt=user_prompt
        )
        
        # Generate response
        logger.info("Generating response with Llama")
        response = self.llm(
            prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            stop=["Question:", "\n\nQuestion:", "Context:"],
            echo=False
        )
        
        answer = response["choices"][0]["text"].strip()
        
        # Prepare result
        result = {
            "answer": answer,
            "retrieved_documents": len(retrieved_docs),
            "confidence": self._calculate_confidence(retrieved_docs)
        }
        
        if include_sources:
            result["sources"] = self._format_sources(retrieved_docs)
        
        logger.info("Response generated successfully")
        return result
    
    def _prepare_context(self, retrieved_docs: List[Dict[str, Any]]) -> str:
        """Prepare context from retrieved documents."""
        context_parts = []
        
        for i, doc in enumerate(retrieved_docs, 1):
            document = doc["document"]
            metadata = doc["metadata"]
            score = doc["similarity_score"]
            
            # Add document with metadata
            context_part = f"[Document {i} - Score: {score:.3f}]"
            
            if "start_time" in metadata and "end_time" in metadata:
                start_time = metadata["start_time"]
                end_time = metadata["end_time"]
                context_part += f" [Time: {start_time:.1f}s - {end_time:.1f}s]"
            
            if "chunk_speakers" in metadata:
                speakers = metadata["chunk_speakers"]
                context_part += f" [Speakers: {', '.join(speakers)}]"
            
            context_part += f"\n{document}\n"
            context_parts.append(context_part)
        
        return "\n".join(context_parts)
    
    def _calculate_confidence(self, retrieved_docs: List[Dict[str, Any]]) -> float:
        """Calculate confidence score based on retrieved documents."""
        if not retrieved_docs:
            return 0.0
        
        # Use average similarity score as confidence
        scores = [doc["similarity_score"] for doc in retrieved_docs]
        return sum(scores) / len(scores)
    
    def _format_sources(self, retrieved_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format source information."""
        sources = []
        
        for doc in retrieved_docs:
            metadata = doc["metadata"]
            source_info = {
                "source": metadata.get("source", "Unknown"),
                "audio_file": metadata.get("audio_file", "Unknown"),
                "similarity_score": doc["similarity_score"]
            }
            
            if "start_time" in metadata and "end_time" in metadata:
                source_info["timestamp"] = {
                    "start": metadata["start_time"],
                    "end": metadata["end_time"]
                }
            
            if "chunk_speakers" in metadata:
                source_info["speakers"] = metadata["chunk_speakers"]
            
            sources.append(source_info)
        
        return sources
    
    def chat(
        self,
        interactive: bool = True,
        **generation_kwargs
    ):
        """Interactive chat interface."""
        print("=== Video RAG Chat Interface ===")
        print("Ask questions about the video content. Type 'quit' or 'exit' to stop.")
        print(f"Vector DB Info: {self.vector_db.get_collection_info()}")
        print("=" * 50)
        
        while True:
            try:
                question = input("\nYour question: ").strip()
                
                if question.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                
                if not question:
                    continue
                
                # Generate response
                result = self.generate_response(question, **generation_kwargs)
                
                # Display response
                print(f"\nAnswer: {result['answer']}")
                print(f"Confidence: {result['confidence']:.3f}")
                print(f"Retrieved Documents: {result['retrieved_documents']}")
                
                if result.get('sources') and input("\nShow sources? (y/n): ").lower() == 'y':
                    print("\nSources:")
                    for i, source in enumerate(result['sources'], 1):
                        print(f"{i}. {source['source']} (Score: {source['similarity_score']:.3f})")
                        if 'timestamp' in source:
                            ts = source['timestamp']
                            print(f"   Time: {ts['start']:.1f}s - {ts['end']:.1f}s")
                        if 'speakers' in source:
                            print(f"   Speakers: {', '.join(source['speakers'])}")
                
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                logger.error(f"Error processing question: {e}")
                print(f"Error: {e}")


def load_rag_from_config(config_path: str) -> RAGPipeline:
    """Load RAG pipeline from configuration file.
    
    Args:
        config_path: Path to configuration YAML file
        
    Returns:
        Initialized RAG pipeline
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize vector database
    vector_db_config = config.get('vector_db', {})
    vector_db = VectorDatabase(
        persist_directory=vector_db_config.get('persist_directory', 'data/vector_db'),
        collection_name=vector_db_config.get('collection_name', 'video_transcripts'),
        embedding_model=vector_db_config.get('embedding_model', 'all-MiniLM-L6-v2'),
        chunk_size=vector_db_config.get('chunk_size', 1000),
        chunk_overlap=vector_db_config.get('chunk_overlap', 200)
    )
    
    # Initialize RAG pipeline
    llm_config = config.get('llm', {})
    rag_config = config.get('rag', {})
    
    rag_pipeline = RAGPipeline(
        model_path=llm_config.get('model_path', 'data/models/model.gguf'),
        vector_db=vector_db,
        context_length=llm_config.get('context_length', 4096),
        max_tokens=llm_config.get('max_tokens', 512),
        temperature=llm_config.get('temperature', 0.1),
        top_p=llm_config.get('top_p', 0.95),
        n_threads=llm_config.get('n_threads', 4),
        system_prompt_template=rag_config.get('system_prompt')
    )
    
    return rag_pipeline

