"""
FastAPI routes for the Ragatanga API.
"""

import os
import traceback

from fastapi import APIRouter, HTTPException, UploadFile, Depends
from fastapi.responses import FileResponse

import aiofiles
from loguru import logger

from ragatanga.api.models import QueryRequest, QueryResponse, StatusResponse, OntologyStatistics
from ragatanga.core.ontology import OntologyManager
from ragatanga.core.retrieval import AdaptiveRetriever
from ragatanga.core.query import generate_structured_answer
from ragatanga.core.semantic import SemanticSearch

router = APIRouter()

def get_ontology_manager() -> OntologyManager:
    """Get the ontology manager from app state."""
    from ragatanga.api.app import app
    return app.state.ontology_manager

def get_retriever() -> AdaptiveRetriever:
    """Get the adaptive retriever from app state."""
    from ragatanga.api.app import app
    return app.state.retriever

def get_semantic_search() -> SemanticSearch:
    """Get the semantic search from app state."""
    from ragatanga.api.app import app
    return app.state.semantic_search

@router.post("/query", response_model=QueryResponse)
async def handle_query(
    req: QueryRequest,
    ontology_manager: OntologyManager = Depends(get_ontology_manager),
    retriever: AdaptiveRetriever = Depends(get_retriever)
):
    """
    Enhanced query endpoint with improved retrieval and response generation.
    
    Args:
        req: Query request
        ontology_manager: Ontology manager dependency
        retriever: Adaptive retriever dependency
        
    Returns:
        Query response with answer and retrieved facts
    """
    user_query = req.query
    logger.info(f"Processing query: {user_query}")
    
    try:
        # Debug the environment
        logger.debug(f"OpenAI API Key exists: {bool(os.environ.get('OPENAI_API_KEY'))}")
        logger.debug(f"OWL_FILE_PATH value: {os.environ.get('OWL_FILE_PATH')}")
        
        # Use adaptive retrieval
        logger.debug("Starting adaptive retrieval")
        retrieved_texts, confidence_scores = await retriever.retrieve(user_query)
        logger.debug(f"Retrieved {len(retrieved_texts)} results with adaptive parameters")
        
        # Generate structured answer
        logger.debug("Starting answer generation")
        answer = await generate_structured_answer(user_query, retrieved_texts, confidence_scores)
        
        # Log success
        logger.info(f"Successfully generated answer for query: {user_query}")
        
        return answer
    except Exception as e:
        # Capture and log the full exception
        error_detail = traceback.format_exc()
        logger.error(f"Error processing query: {str(e)}\n{error_detail}")
        
        # Return a graceful error response with more details
        return QueryResponse(
            retrieved_facts=[],
            retrieved_facts_sparql=[],
            retrieved_facts_semantic=[],
            answer=f"Error processing query. Details: {str(e)}. Type: {type(e).__name__}"
        )

@router.post("/upload/ontology", response_model=StatusResponse)
async def upload_ontology(
    file: UploadFile,
    ontology_manager: OntologyManager = Depends(get_ontology_manager)
):
    """
    Upload a new ontology file (.ttl or .owl).
    
    Args:
        file: Uploaded file
        ontology_manager: Ontology manager dependency
        
    Returns:
        Status response
    """
    if not file.filename or not file.filename.endswith(('.ttl', '.owl')):
        raise HTTPException(status_code=400, detail="File must be .ttl or .owl")
    
    try:
        contents = await file.read()
        try:
            decoded_contents = contents.decode('utf-8')
        except UnicodeDecodeError:
            decoded_contents = contents.decode('latin-1')
        
        async with aiofiles.open(ontology_manager.owl_file_path, "w", encoding='utf-8') as out_file:
            await out_file.write(decoded_contents)
            
        # Reload ontology
        await ontology_manager.load_and_materialize(force_rebuild=True)
        
        return StatusResponse(
            message="Ontology uploaded and loaded successfully",
            success=True,
            details={"file_name": file.filename}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading ontology: {str(e)}")

@router.get("/download/ontology")
async def download_ontology(
    ontology_manager: OntologyManager = Depends(get_ontology_manager)
):
    """
    Download the current ontology file.
    
    Args:
        ontology_manager: Ontology manager dependency
        
    Returns:
        Ontology file for download
    """
    if not os.path.exists(ontology_manager.owl_file_path):
        raise HTTPException(status_code=404, detail="Ontology file not found")
    return FileResponse(ontology_manager.owl_file_path)

@router.post("/upload/kb", response_model=StatusResponse)
async def upload_knowledge_base(
    file: UploadFile,
    semantic_search: SemanticSearch = Depends(get_semantic_search)
):
    """
    Upload a new knowledge base markdown file.
    
    Args:
        file: Uploaded file
        semantic_search: Semantic search dependency
        
    Returns:
        Status response
    """
    if not file.filename or not file.filename.endswith('.md'):
        raise HTTPException(status_code=400, detail="File must be .md")
    
    try:
        contents = await file.read()
        try:
            decoded_contents = contents.decode('utf-8')
        except UnicodeDecodeError:
            decoded_contents = contents.decode('latin-1')
        
        # Get the knowledge base file path from app state
        from ragatanga.api.app import app
        kb_file = app.state.kb_file
        
        async with aiofiles.open(kb_file, "w", encoding='utf-8') as out_file:
            await out_file.write(decoded_contents)
            
        # Reload and reindex knowledge base
        await semantic_search.load_knowledge_base(kb_file, force_rebuild=True)
        
        return StatusResponse(
            message="Knowledge base uploaded and indexed successfully",
            success=True,
            details={"file_name": file.filename}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading knowledge base: {str(e)}")

@router.get("/download/kb")
async def download_knowledge_base():
    """
    Download the current knowledge base markdown file.
    
    Returns:
        Knowledge base file for download
    """
    from ragatanga.api.app import app
    kb_file = app.state.kb_file
    
    if not os.path.exists(kb_file):
        raise HTTPException(status_code=404, detail="Knowledge base file not found")
    return FileResponse(kb_file)

@router.get("/describe_ontology", response_model=OntologyStatistics)
async def describe_ontology(
    ontology_manager: OntologyManager = Depends(get_ontology_manager)
):
    """
    Get a comprehensive description of the loaded ontology.
    
    Args:
        ontology_manager: Ontology manager dependency
        
    Returns:
        Detailed ontology statistics
    """
    try:
        stats = ontology_manager.get_ontology_statistics()
        return stats
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error describing ontology: {str(e)}"
        )