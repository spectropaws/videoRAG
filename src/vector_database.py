"""Vector database management using ChromaDB."""

import logging
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from .utils.file_utils import ensure_directory


logger = logging.getLogger("video_rag.vector_database")


class VectorDatabase:
    """Vector database for storing and retrieving document embeddings."""
    
    def __init__(
        self,
        persist_directory: str = "data/vector_db",
        collection_name: str = "video_transcripts",
        embedding_model: str = "all-MiniLM-L6-v2",
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        """Initialize VectorDatabase.
        
        Args:
            persist_directory: Directory to persist the database
            collection_name: Name of the collection
            embedding_model: Sentence transformer model name
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        ensure_directory(self.persist_directory)
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(allow_reset=True)
        )
        
        # Initialize embedding model
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Get or create collection
        self.collection = self._get_or_create_collection()
    
    def _get_or_create_collection(self):
        """Get existing collection or create new one."""
        try:
            collection = self.client.get_collection(name=self.collection_name)
            logger.info(f"Using existing collection: {self.collection_name}")
        except Exception:
            collection = self.client.create_collection(name=self.collection_name)
            logger.info(f"Created new collection: {self.collection_name}")
        
        return collection
    
    def add_transcription(
        self,
        transcription: Dict[str, Any],
        source_id: Optional[str] = None
    ) -> int:
        """Add transcription to vector database.
        
        Args:
            transcription: Transcription results from TranscriptionService
            source_id: Optional source identifier
            
        Returns:
            Number of chunks added
        """
        if source_id is None:
            source_id = Path(transcription["audio_file"]).stem
        
        logger.info(f"Adding transcription to vector database: {source_id}")
        
        # Extract text content
        full_text = transcription["full_text"]
        segments = transcription.get("segments", [])
        
        # Create chunks from full text
        chunks = self.text_splitter.split_text(full_text)
        
        documents = []
        metadatas = []
        ids = []
        
        # Process each chunk
        for i, chunk in enumerate(chunks):
            chunk_id = f"{source_id}_chunk_{i}"
            
            # Find relevant segments for this chunk
            chunk_segments = self._find_segments_for_chunk(chunk, segments)
            
            # Create metadata
            metadata = {
                "source": source_id,
                "audio_file": transcription["audio_file"],
                "chunk_index": i,
                "total_chunks": len(chunks),
                "speakers": ",".join(transcription.get("speakers", [])),
                "segment_count": len(chunk_segments)
            }
            
            # Add timing information if available
            if chunk_segments:
                metadata["start_time"] = min(seg["start"] for seg in chunk_segments)
                metadata["end_time"] = max(seg["end"] for seg in chunk_segments)
                metadata["chunk_speakers"] = ",".join(set(seg["speaker"] for seg in chunk_segments))
            
            documents.append(chunk)
            metadatas.append(metadata)
            ids.append(chunk_id)
        
        # Generate embeddings
        logger.info(f"Generating embeddings for {len(chunks)} chunks")
        embeddings = self.embedding_model.encode(documents).tolist()
        
        # Add to collection
        self.collection.add(
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        
        logger.info(f"Added {len(chunks)} chunks to vector database")
        return len(chunks)
    
    def _find_segments_for_chunk(
        self,
        chunk: str,
        segments: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Find segments that are relevant to a text chunk."""
        relevant_segments = []
        
        for segment in segments:
            segment_text = segment.get("text", "").strip()
            if segment_text and segment_text in chunk:
                relevant_segments.append(segment)
        
        return relevant_segments
    
    def search(
        self,
        query: str,
        n_results: int = 5,
        score_threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """Search for relevant documents.
        
        Args:
            query: Search query
            n_results: Number of results to return
            score_threshold: Minimum similarity score
            
        Returns:
            List of search results with documents and metadata
        """
        logger.info(f"Searching vector database: '{query}'")
        print(query)
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query]).tolist()[0]
        
        # Search collection
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        
        # Format results
        formatted_results = []
        
        if results["documents"] and results["documents"][0]:
            for i in range(len(results["documents"][0])):
                # Calculate similarity score (ChromaDB returns distances)
                distance = results["distances"][0][i]
                # similarity = 1 - distance  # Convert distance to similarity
                similarity = 1 / (1 + distance)
                
                if similarity >= score_threshold:
                    result = {
                        "document": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "id": results["ids"][0][i],
                        "similarity_score": similarity
                    }
                    formatted_results.append(result)
        
        logger.info(f"Found {len(formatted_results)} relevant documents")
        print(formatted_results)
        return formatted_results
    
    
    def add_transcription_file(self, transcription_file: Union[str, Path]) -> int:
        """Add transcription from JSON file.
        
        Args:
            transcription_file: Path to transcription JSON file
            
        Returns:
            Number of chunks added
        """
        transcription_file = Path(transcription_file)
        
        if not transcription_file.exists():
            raise ValueError(f"Transcription file does not exist: {transcription_file}")
        
        with open(transcription_file, 'r', encoding='utf-8') as f:
            transcription = json.load(f)
        
        source_id = transcription_file.stem
        return self.add_transcription(transcription, source_id)
    
    def batch_add_transcriptions(
        self,
        transcription_files: List[Union[str, Path]]
    ) -> Dict[str, int]:
        """Add multiple transcription files to database.
        
        Args:
            transcription_files: List of transcription file paths
            
        Returns:
            Dictionary mapping source IDs to number of chunks added
        """
        results = {}
        
        for transcription_file in transcription_files:
            try:
                source_id = Path(transcription_file).stem
                chunks_added = self.add_transcription_file(transcription_file)
                results[source_id] = chunks_added
            except Exception as e:
                logger.error(f"Failed to add transcription {transcription_file}: {e}")
                results[str(transcription_file)] = 0
        
        return results
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection."""
        count = self.collection.count()
        
        return {
            "collection_name": self.collection_name,
            "document_count": count,
            "embedding_model": str(self.embedding_model),
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap
        }
    
    def reset_collection(self):
        """Reset the collection (delete all documents)."""
        logger.warning(f"Resetting collection: {self.collection_name}")
        self.client.delete_collection(name=self.collection_name)
        self.collection = self._get_or_create_collection()
