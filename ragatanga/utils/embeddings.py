"""
Embedding providers module that abstracts different embedding backends.
"""

import os
import abc
from typing import List, Union, Dict, Any, Optional
import numpy as np
import asyncio
from loguru import logger

class EmbeddingProvider(abc.ABC):
    """Abstract base class for embedding providers."""
    
    @abc.abstractmethod
    async def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single query string.
        
        Args:
            query: The query string to embed
            
        Returns:
            A numpy array containing the embedding
        """
        pass
    
    @abc.abstractmethod
    async def embed_texts(self, texts: List[str], batch_size: int = 16) -> np.ndarray:
        """
        Embed a list of texts.
        
        Args:
            texts: List of texts to embed
            batch_size: Size of batches for processing
            
        Returns:
            A numpy array containing embeddings for all texts
        """
        pass
    
    @staticmethod
    def get_provider(provider_name: str = None, **kwargs) -> "EmbeddingProvider":
        """
        Factory method to get an embedding provider based on configuration.
        
        Args:
            provider_name: Name of the provider (openai, huggingface, sentence-transformers)
            **kwargs: Additional configuration parameters
            
        Returns:
            An instance of EmbeddingProvider
        """
        # Default to environment variable or fallback to OpenAI
        if provider_name is None:
            provider_name = os.getenv("EMBEDDING_PROVIDER", "openai")
            
        if provider_name == "openai":
            return OpenAIEmbeddingProvider(**kwargs)
        elif provider_name == "huggingface":
            return HuggingFaceEmbeddingProvider(**kwargs)
        elif provider_name == "sentence-transformers":
            return SentenceTransformersEmbeddingProvider(**kwargs)
        else:
            raise ValueError(f"Unsupported embedding provider: {provider_name}")

class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI-based embedding provider."""
    
    def __init__(self, model: str = "text-embedding-3-large", api_key: str = None, **kwargs):
        """
        Initialize the OpenAI embedding provider.
        
        Args:
            model: The OpenAI embedding model to use
            api_key: OpenAI API key (if None, uses OPENAI_API_KEY environment variable)
            **kwargs: Additional parameters for the OpenAI client
        """
        try:
            import openai
            from openai import OpenAI
        except ImportError:
            raise ImportError("OpenAI package is required. Install it with 'pip install openai'")
        
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is not set")
            
        self.client = OpenAI(api_key=self.api_key)
        self.dimension = 3072 if model == "text-embedding-3-large" else 1536
    
    async def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string using OpenAI."""
        import asyncio
        
        response = await asyncio.to_thread(
            self.client.embeddings.create,
            input=[query],
            model=self.model
        )
        
        emb = np.array(response.data[0].embedding, dtype=np.float32)
        # Normalize for cosine similarity
        norm = np.linalg.norm(emb)
        return emb / (norm + 1e-10)
    
    async def embed_texts(self, texts: List[str], batch_size: int = 16) -> np.ndarray:
        """Embed a list of texts in batches using OpenAI."""
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = await asyncio.to_thread(
                self.client.embeddings.create,
                input=batch,
                model=self.model
            )
            
            for j in range(len(batch)):
                emb = response.data[j].embedding
                all_embeddings.append(emb)
                
        embeddings = np.array(all_embeddings, dtype=np.float32)
        
        # Normalize for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / (norms + 1e-10)

class HuggingFaceEmbeddingProvider(EmbeddingProvider):
    """HuggingFace-based embedding provider."""
    
    def __init__(self, model: str = "sentence-transformers/all-mpnet-base-v2", **kwargs):
        """
        Initialize the HuggingFace embedding provider.
        
        Args:
            model: The HuggingFace model to use
            **kwargs: Additional parameters for the model
        """
        # Lazy import to avoid dependency if not used
        try:
            from transformers import AutoTokenizer, AutoModel
            self.tokenizer = AutoTokenizer.from_pretrained(model)
            self.model = AutoModel.from_pretrained(model)
            self.device = kwargs.get("device", "cuda" if self._is_cuda_available() else "cpu")
            self.model.to(self.device)
            self.dimension = self.model.config.hidden_size
        except ImportError:
            raise ImportError(
                "HuggingFace transformers is required for HuggingFaceEmbeddingProvider. "
                "Install it with 'pip install transformers torch'"
            )
    
    def _is_cuda_available(self) -> bool:
        """Check if CUDA is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    async def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string using HuggingFace."""
        import torch
        
        def _embed():
            with torch.no_grad():
                inputs = self.tokenizer(query, return_tensors="pt", padding=True, truncation=True).to(self.device)
                outputs = self.model(**inputs)
                # Use mean pooling of the last hidden state
                embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
                # Normalize
                norm = np.linalg.norm(embeddings[0])
                return embeddings[0] / (norm + 1e-10)
        
        return await asyncio.to_thread(_embed)
    
    async def embed_texts(self, texts: List[str], batch_size: int = 16) -> np.ndarray:
        """Embed a list of texts in batches using HuggingFace."""
        import torch
        
        all_embeddings = []
        
        def _embed_batch(batch):
            with torch.no_grad():
                inputs = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(self.device)
                outputs = self.model(**inputs)
                # Use mean pooling of the last hidden state
                embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
                return embeddings
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            batch_embeddings = await asyncio.to_thread(_embed_batch, batch)
            all_embeddings.append(batch_embeddings)
            
        # Concatenate all batch embeddings
        embeddings = np.vstack(all_embeddings)
        
        # Normalize for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / (norms + 1e-10)

class SentenceTransformersEmbeddingProvider(EmbeddingProvider):
    """Sentence Transformers embedding provider."""
    
    def __init__(self, model: str = "all-mpnet-base-v2", **kwargs):
        """
        Initialize the SentenceTransformers embedding provider.
        
        Args:
            model: The SentenceTransformers model to use
            **kwargs: Additional parameters for the model
        """
        # Lazy import to avoid dependency if not used
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model)
            self.device = kwargs.get("device", "cuda" if self._is_cuda_available() else "cpu")
            self.model.to(self.device)
            self.dimension = self.model.get_sentence_embedding_dimension()
        except ImportError:
            raise ImportError(
                "SentenceTransformers is required for SentenceTransformersEmbeddingProvider. "
                "Install it with 'pip install sentence-transformers'"
            )
    
    def _is_cuda_available(self) -> bool:
        """Check if CUDA is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    async def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string using SentenceTransformers."""
        def _embed():
            embedding = self.model.encode(query, normalize_embeddings=True)
            return embedding
        
        return await asyncio.to_thread(_embed)
    
    async def embed_texts(self, texts: List[str], batch_size: int = 16) -> np.ndarray:
        """Embed a list of texts in batches using SentenceTransformers."""
        def _embed_batches():
            return self.model.encode(texts, batch_size=batch_size, normalize_embeddings=True)
        
        return await asyncio.to_thread(_embed_batches)

def build_faiss_index(embeddings: np.ndarray, dimension: int) -> tuple:
    """
    Build a FAISS index for inner product (cosine similarity if vectors are normalized).
    
    Args:
        embeddings: Numpy array of embeddings
        dimension: Dimensionality of the embeddings
        
    Returns:
        Tuple of (faiss_index, normalized_embeddings)
    """
    try:
        import faiss
    except ImportError:
        raise ImportError("FAISS is required. Install it with 'pip install faiss-cpu' or 'pip install faiss-gpu'")
    
    # Normalize embeddings for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings_norm = embeddings / (norms + 1e-10)

    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings_norm)
    return index, embeddings_norm

def save_faiss_index(index, index_file: str, embeddings: np.ndarray, embed_file: str):
    """
    Save FAISS index and embeddings to disk.
    
    Args:
        index: FAISS index
        index_file: Path to save the index
        embeddings: Embeddings numpy array
        embed_file: Path to save the embeddings
    """
    try:
        import faiss
    except ImportError:
        raise ImportError("FAISS is required. Install it with 'pip install faiss-cpu' or 'pip install faiss-gpu'")
    
    faiss.write_index(index, index_file)
    np.save(embed_file, embeddings)
    logger.info(f"FAISS index saved to {index_file}, embeddings to {embed_file}")

def load_faiss_index(index_file: str, embed_file: str):
    """
    Load FAISS index and embeddings from disk.
    
    Args:
        index_file: Path to the index file
        embed_file: Path to the embeddings file
        
    Returns:
        Tuple of (faiss_index, embeddings)
    """
    try:
        import faiss
    except ImportError:
        raise ImportError("FAISS is required. Install it with 'pip install faiss-cpu' or 'pip install faiss-gpu'")
    
    index = faiss.read_index(index_file)
    embeddings = np.load(embed_file)
    return index, embeddings