"""
Semantic search module for Ragatanga.

This module provides semantic search functionality using embeddings.
"""

import os
import asyncio
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from loguru import logger

from ragatanga.utils.embeddings import EmbeddingProvider, build_faiss_index, save_faiss_index, load_faiss_index

class SemanticSearch:
    """
    Semantic search implementation for retrieving knowledge base entries.
    """
    
    def __init__(self, embed_provider: Optional[EmbeddingProvider] = None):
        """
        Initialize the semantic search.
        
        Args:
            embed_provider: Embedding provider to use (if None, uses default)
        """
        self.embed_provider = embed_provider or EmbeddingProvider.get_provider()
        self.dimension = self.embed_provider.dimension
        self.kbase_entries = []
        self.kbase_index = None
        self.kbase_embeddings = None
    
    async def load_knowledge_base(self, kb_file: str, index_file: str = None, embed_file: str = None, force_rebuild: bool = False):
        """
        Load knowledge base and build/load index.
        
        Args:
            kb_file: Path to the knowledge base file
            index_file: Path to the index file (if None, derived from kb_file)
            embed_file: Path to the embeddings file (if None, derived from kb_file)
            force_rebuild: Whether to force rebuilding the index
        """
        # Set default paths if not provided
        if index_file is None:
            index_file = kb_file.replace(".md", "_faiss.index").replace(".txt", "_faiss.index")
        if embed_file is None:
            embed_file = kb_file.replace(".md", "_embeddings.npy").replace(".txt", "_embeddings.npy")
        
        # Load knowledge base content
        if not os.path.exists(kb_file):
            logger.warning(f"Knowledge base file {kb_file} not found")
            self.kbase_entries = []
            return
        
        with open(kb_file, "r", encoding="utf-8") as file:
            kbase_content = file.read()
        
        self.kbase_entries = [chunk.strip() for chunk in kbase_content.split("\n\n") if chunk.strip()]
        logger.info(f"Loaded {len(self.kbase_entries)} entries from knowledge base")
        
        # Build or load FAISS index
        if not force_rebuild and os.path.exists(index_file) and os.path.exists(embed_file):
            logger.info("Loading existing FAISS index and embeddings")
            self.kbase_index, self.kbase_embeddings = load_faiss_index(index_file, embed_file)
        else:
            logger.info("Building new FAISS index")
            self.kbase_embeddings = await self.embed_provider.embed_texts(self.kbase_entries)
            self.kbase_index, self.kbase_embeddings = build_faiss_index(self.kbase_embeddings, self.dimension)
            save_faiss_index(self.kbase_index, index_file, self.kbase_embeddings, embed_file)
    
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
        try:
            import faiss
        except ImportError:
            raise ImportError("FAISS is required. Install it with 'pip install faiss-cpu' or 'pip install faiss-gpu'")
            
        def search_index():
            return self.kbase_index.search(q_emb, k)
        
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
        if '_semantic_search' not in globals() or _semantic_search is None:
            _semantic_search = SemanticSearch()
        semantic_search = _semantic_search
    
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
        if '_semantic_search' not in globals() or _semantic_search is None:
            _semantic_search = SemanticSearch()
        semantic_search = _semantic_search
    
    return await semantic_search.search_with_scores(query, k)

# Initialize global semantic search instance
_semantic_search = None