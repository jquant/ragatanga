"""
FastAPI application for the Ragatanga API.
"""

import os
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI
from loguru import logger

from ragatanga import config
from ragatanga.core.ontology import OntologyManager
from ragatanga.core.retrieval import AdaptiveRetriever
from ragatanga.core.semantic import SemanticSearch
from ragatanga.api.routes import router

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI app.
    
    Args:
        app: FastAPI application
    """
    # Set up file paths
    app.state.data_dir = config.DATA_DIR
    app.state.ontology_path = config.ONTOLOGY_PATH
    app.state.kb_file = os.path.join(config.DATA_DIR, "knowledge_base.md")
    
    # Ensure data directory exists
    os.makedirs(config.DATA_DIR, exist_ok=True)
    
    # Initialize ontology manager
    logger.info(f"Initializing ontology manager with {app.state.ontology_path}")
    app.state.ontology_manager = OntologyManager(app.state.ontology_path)
    await app.state.ontology_manager.load_and_materialize()
    
    # Initialize semantic search
    logger.info(f"Initializing semantic search with {app.state.kb_file}")
    app.state.semantic_search = SemanticSearch()
    
    # Check if knowledge base file exists
    if os.path.exists(app.state.kb_file):
        logger.info("Loading knowledge base...")
        await app.state.semantic_search.load_knowledge_base(app.state.kb_file)
    else:
        logger.warning(f"Knowledge base file {app.state.kb_file} not found. Starting with empty knowledge base.")
    
    # Initialize adaptive retriever
    logger.info("Initializing adaptive retriever")
    app.state.retriever = AdaptiveRetriever(app.state.ontology_manager)
    
    logger.info("System initialization complete!")
    yield
    
    # Cleanup code if needed
    logger.info("Shutting down...")

# Create FastAPI app
app = FastAPI(
    title="Ragatanga API",
    description="Hybrid Retrieval API combining SPARQL queries and semantic search",
    version=config.VERSION,
    lifespan=lifespan
)

# Include routers
app.include_router(router)

@app.get("/")
async def root() -> Dict[str, Any]:
    """
    Root endpoint for the API.
    
    Returns:
        Basic API information
    """
    return {
        "name": "Ragatanga API",
        "version": config.VERSION,
        "description": "Hybrid Retrieval API combining SPARQL queries and semantic search"
    }