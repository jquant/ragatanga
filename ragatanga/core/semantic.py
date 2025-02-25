"""
Semantic search module for Ragatanga.

This module provides semantic search functionality using embeddings.
"""

import os
import asyncio
from typing import List, Tuple, Optional, Any
from loguru import logger
import numpy as np

# Import faiss properly
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("FAISS not available. Install it with 'pip install faiss-cpu' or 'pip install faiss-gpu'")

from ragatanga.utils.embeddings import EmbeddingProvider, build_faiss_index, save_faiss_index, load_faiss_index

class SemanticSearch:
    """
    Semantic search implementation for retrieving knowledge base entries.
    """
    
    def __init__(self, embed_provider=None, dimension: int = 768):
        """
        Initialize the semantic search.
        
        Args:
            embed_provider: Embedding provider to use
            dimension: Dimension of embeddings
        """
        self.embed_provider = embed_provider or EmbeddingProvider.get_provider()
        # Get dimension from provider or use default value
        self.dimension = getattr(self.embed_provider, 'dimension', 1536)  # Default to OpenAI's dimension
        self.kbase_entries = []
        self.kbase_index = None
        self.kbase_embeddings = None
    
    async def load_knowledge_base(self, kb_path: str, force_rebuild: bool = False) -> None:
        """
        Load knowledge base from a file.
        
        Args:
            kb_path: Path to the knowledge base file
            force_rebuild: Whether to force rebuild the index
        """
        try:
            if not os.path.exists(kb_path):
                logger.error(f"Knowledge base file not found: {kb_path}")
                raise FileNotFoundError(f"Knowledge base file not found: {kb_path}")
                
            # Create directory for index files if it doesn't exist
            index_dir = os.path.dirname(kb_path)
            os.makedirs(index_dir, exist_ok=True)
            
            # Define index and embedding file paths
            file_name = os.path.basename(kb_path).split('.')[0]
            index_file = os.path.join(index_dir, f"{file_name}_index.pkl")
            embed_file = os.path.join(index_dir, f"{file_name}_embeddings.npy")
            
            # Load knowledge base content
            with open(kb_path, 'r', encoding='utf-8') as f:
                kbase_content = f.read()
                
            # Process content based on file type
            if kb_path.endswith('.md'):
                # For markdown files, split by headers
                entries = []
                current_entry = []
                
                for line in kbase_content.split('\n'):
                    if line.startswith('#'):
                        if current_entry:
                            entries.append('\n'.join(current_entry))
                            current_entry = []
                        current_entry = [line]
                    else:
                        current_entry.append(line)
                
                # Add the last entry if there is one
                if current_entry:
                    entries.append('\n'.join(current_entry))
                
                self.kbase_entries = entries
            else:
                # For text files, just split by double newlines
                self.kbase_entries = [chunk.strip() for chunk in kbase_content.split("\n\n") if chunk.strip()]
            
            logger.info(f"Loaded {len(self.kbase_entries)} entries from knowledge base")
            
            # Build or load FAISS index
            if not force_rebuild and os.path.exists(index_file) and os.path.exists(embed_file):
                logger.info("Loading existing FAISS index and embeddings")
                try:
                    self.kbase_index, self.kbase_embeddings = load_faiss_index(index_file, embed_file)
                    # Verify the index is properly loaded
                    if self.kbase_index is None:
                        logger.warning("Failed to load existing index, rebuilding...")
                        raise Exception("Index loading failed")
                    else:
                        logger.info("FAISS index successfully loaded")
                except Exception as e:
                    logger.warning(f"Error loading index: {str(e)}. Rebuilding index...")
                    force_rebuild = True
            
            # If index should be rebuilt or loading failed
            if force_rebuild or self.kbase_index is None:
                logger.info("Building new FAISS index")
                if len(self.kbase_entries) == 0:
                    logger.warning("No entries to index! The knowledge base may be empty.")
                    # Initialize empty index to avoid None errors
                    self.kbase_embeddings = np.zeros((0, self.dimension), dtype=np.float32)
                    self.kbase_index = faiss.IndexFlatL2(self.dimension)
                else:
                    self.kbase_embeddings = await self.embed_provider.embed_texts(self.kbase_entries)
                    self.kbase_index, self.kbase_embeddings = build_faiss_index(self.kbase_embeddings, self.dimension)
                    save_faiss_index(self.kbase_index, index_file, self.kbase_embeddings, embed_file)
                    logger.info("FAISS index successfully built and saved")
            
            # Final verification
            if self.kbase_index is None:
                logger.error("Failed to initialize knowledge base index")
            
        except Exception as e:
            logger.error(f"Error loading knowledge base: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            # Initialize empty index to avoid None errors
            self.kbase_entries = []
            self.kbase_embeddings = np.zeros((0, self.dimension), dtype=np.float32)
            self.kbase_index = faiss.IndexFlatL2(self.dimension)
    
    async def search(self, query: str, k: int = 10) -> List[str]:
        """
        Search the knowledge base for entries similar to the query.
        
        Args:
            query: The query string
            k: Number of results to return
            
        Returns:
            List of matching entries
        """
        results, _ = await self.search_with_scores(query, k)
        return results
    
    async def search_with_scores(self, query: str, k: int = 10) -> Tuple[List[str], List[float]]:
        """
        Search the knowledge base and return entries with similarity scores.
        
        Args:
            query: The query string
            k: Number of results to return
            
        Returns:
            Tuple of (results, similarity_scores)
        """
        if self.kbase_index is None:
            logger.warning("Knowledge base index not initialized")
            return [], []
        
        if k > len(self.kbase_entries):
            k = len(self.kbase_entries)
            
        # Get query embedding
        q_emb = await self.embed_provider.embed_query(query)
        q_emb = q_emb.reshape(1, -1)
        
        # Search the index
        if not FAISS_AVAILABLE:
            raise ImportError("FAISS is required. Install it with 'pip install faiss-cpu' or 'pip install faiss-gpu'")
            
        def search_index():
            # Ensure index is not None before searching
            if self.kbase_index is None:
                raise ValueError("Knowledge base index is None")
                
            # FAISS search method returns distances and indices
            # We're ignoring linter errors since FAISS API doesn't match what the linter expects
            # distances, labels are return values, not parameters
            # type: ignore
            return self.kbase_index.search(q_emb, k)  # type: ignore
        
        distances, indices = await asyncio.to_thread(search_index)
        
        # Convert distances to similarity scores
        similarity_scores = distances[0].tolist()
        results = [self.kbase_entries[i] for i in indices[0].tolist()]
        
        return results, similarity_scores

async def retrieve_top_k(query: str, k: int, semantic_search: Optional[SemanticSearch] = None) -> List[str]:
    """
    Use semantic search to find the top-k most similar knowledge base entries.
    
    Args:
        query: The query string
        k: Number of results to retrieve
        semantic_search: SemanticSearch instance (if None, uses global instance)
        
    Returns:
        List of matching entries
    """
    global _semantic_search
    if semantic_search is None:
        # Try to get the properly initialized instance from the FastAPI app
        try:
            from ragatanga.api.app import app
            if hasattr(app.state, "semantic_search") and app.state.semantic_search is not None:
                semantic_search = app.state.semantic_search
                # Update the global instance to the app instance
                _semantic_search = app.state.semantic_search
            else:
                if '_semantic_search' not in globals() or _semantic_search is None:
                    _semantic_search = SemanticSearch()
                semantic_search = _semantic_search
        except ImportError:
            # If we can't import app (e.g., in tests), use the global instance
            if '_semantic_search' not in globals() or _semantic_search is None:
                _semantic_search = SemanticSearch()
            semantic_search = _semantic_search
    
    # Ensure we have a valid semantic search instance
    if semantic_search is None:
        logger.error("Failed to obtain a valid semantic search instance")
        return []
        
    return await semantic_search.search(query, k)

async def retrieve_top_k_with_scores(query: str, k: int, semantic_search: Optional[SemanticSearch] = None) -> Tuple[List[str], List[float]]:
    """
    Use semantic search to find the top-k most similar knowledge base entries with scores.
    
    Args:
        query: The query string
        k: Number of results to retrieve
        semantic_search: SemanticSearch instance (if None, uses global instance)
        
    Returns:
        Tuple of (results, similarity_scores)
    """
    global _semantic_search
    if semantic_search is None:
        # Try to get the properly initialized instance from the FastAPI app
        try:
            from ragatanga.api.app import app
            if hasattr(app.state, "semantic_search") and app.state.semantic_search is not None:
                semantic_search = app.state.semantic_search
                # Update the global instance to the app instance
                _semantic_search = app.state.semantic_search
            else:
                if '_semantic_search' not in globals() or _semantic_search is None:
                    _semantic_search = SemanticSearch()
                semantic_search = _semantic_search
        except ImportError:
            # If we can't import app (e.g., in tests), use the global instance
            if '_semantic_search' not in globals() or _semantic_search is None:
                _semantic_search = SemanticSearch()
            semantic_search = _semantic_search
    
    # Ensure we have a valid semantic search instance
    if semantic_search is None:
        logger.error("Failed to obtain a valid semantic search instance")
        return [], []
    
    return await semantic_search.search_with_scores(query, k)

# Initialize global semantic search instance
_semantic_search = None

def build_faiss_index(embeddings: np.ndarray, dimension: int) -> Tuple[Any, np.ndarray]:
    """
    Build a FAISS index from embeddings.
    
    Args:
        embeddings: Embeddings to index
        dimension: Dimension of embeddings
        
    Returns:
        Tuple of (index, embeddings)
    """
    if not FAISS_AVAILABLE:
        raise ImportError("FAISS is required. Install it with 'pip install faiss-cpu' or 'pip install faiss-gpu'")
    
    # Create an index
    if len(embeddings) == 0:
        index = faiss.IndexFlatL2(dimension)
        return index, embeddings
    
    # Ensure embeddings are in the correct format
    if not isinstance(embeddings, np.ndarray):
        embeddings = np.array(embeddings, dtype=np.float32)
    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype(np.float32)
    
    # Create and train the index
    index = faiss.IndexFlatL2(dimension)
    
    # Add embeddings to the index - ignoring linter error as FAISS API doesn't match linter expectations
    index.add(embeddings)  # type: ignore
    
    # Test that the index works - ignoring linter error as FAISS API doesn't match linter expectations
    _, _ = index.search(embeddings[:1], min(1, len(embeddings)))  # type: ignore
    
    return index, embeddings