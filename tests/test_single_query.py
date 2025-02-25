import asyncio
import json
import time
import os
import sys
import numpy as np

# Add parent directory to path so we can import our modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import required modules
from ragatanga.utils.sparql import generate_sparql_query
from ragatanga.core.ontology import OntologyManager, extract_relevant_schema
from ragatanga.core.retrieval import AdaptiveRetriever
from ragatanga.core.query import generate_structured_answer, analyze_query_type
from ragatanga.utils.embeddings import build_faiss_index, load_faiss_index, save_faiss_index
from ragatanga.config import BATCH_SIZE, DATA_DIR, KNOWLEDGE_BASE_PATH

# Configure logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ragatanga-single-test")

# Get the absolute path to the ontology file
ONTOLOGY_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'ragatanga', 'data', 'ontology.ttl'))
logger.info(f"Using ontology file at: {ONTOLOGY_PATH}")

# Single test query
TEST_QUERY = "Which units are in Belo Horizonte?"

# Define constants for testing
KBASE_FILE = KNOWLEDGE_BASE_PATH
KBASE_FAISS_INDEX_FILE = os.path.join(DATA_DIR, "sample_knowledge_base_index.pkl")
KBASE_EMBEDDINGS_FILE = os.path.join(DATA_DIR, "sample_knowledge_base_embeddings.npy")

# Create placeholders for global variables
kbase_entries = []
kbase_index = None
kbase_embeddings_np = None

# Define the embed_texts_in_batches function for testing
async def embed_texts_in_batches(texts, batch_size=BATCH_SIZE):
    """
    Simplified version for testing - just create random embeddings.
    """
    from ragatanga.utils.embeddings import EmbeddingProvider
    
    embed_provider = EmbeddingProvider.get_provider()
    return await embed_provider.embed_texts(texts)

# Create a simplified execute_sparql_query function for testing
async def execute_sparql_query(query, endpoint=None):
    """
    Simplified version for testing.
    """
    # Use the ontology manager to execute the query
    ontology_manager = OntologyManager(ONTOLOGY_PATH)
    await ontology_manager.load_and_materialize()
    return await ontology_manager.execute_sparql(query)

async def initialize_knowledge_base():
    """Initialize the knowledge base index."""
    logger.info("Initializing knowledge base...")
    
    # Access our global variables
    global kbase_entries, kbase_index, kbase_embeddings_np
    
    # Check if knowledge base file exists
    if not os.path.exists(KBASE_FILE):
        logger.warning(f"Knowledge base file {KBASE_FILE} not found. Tests may fail.")
        return False
    
    # Load knowledge base content
    with open(KBASE_FILE, "r", encoding="utf-8") as file:
        kbase_content = file.read()
    
    # Update global kbase_entries
    kbase_entries = [chunk.strip() for chunk in kbase_content.split("\n\n") if chunk.strip()]
    
    # Build or load FAISS index
    if os.path.exists(KBASE_FAISS_INDEX_FILE) and os.path.exists(KBASE_EMBEDDINGS_FILE):
        logger.info("Loading existing FAISS index...")
        kbase_index, kbase_embeddings_np = load_faiss_index(KBASE_FAISS_INDEX_FILE, KBASE_EMBEDDINGS_FILE)
    else:
        logger.info("Building new FAISS index...")
        kbase_embeddings_np = await embed_texts_in_batches(kbase_entries)
        dimension = 1536  # Default dimension for OpenAI embeddings
        kbase_index, kbase_embeddings_np = build_faiss_index(kbase_embeddings_np, dimension)
        save_faiss_index(kbase_index, KBASE_FAISS_INDEX_FILE, kbase_embeddings_np, KBASE_EMBEDDINGS_FILE)
    
    logger.info("Knowledge base initialization complete!")
    return True

async def test_single_query():
    """Test a single query end-to-end."""
    logger.info("Testing a single query...")
    
    # Initialize knowledge base first
    kb_initialized = await initialize_knowledge_base()
    if not kb_initialized:
        logger.warning("Knowledge base initialization failed. Test may fail.")
    
    # Initialize ontology manager with absolute path
    ontology_manager = OntologyManager(ONTOLOGY_PATH)
    await ontology_manager.load_and_materialize()
    
    # Initialize adaptive retriever
    retriever = AdaptiveRetriever(ontology_manager)
    
    query_result = {
        "query": TEST_QUERY,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "success": True
    }
    
    try:
        # Extract relevant schema for SPARQL query generation
        schema = await extract_relevant_schema(query=TEST_QUERY, owl_path=ONTOLOGY_PATH)
        
        # Step 1: Generate SPARQL query
        start_time = time.time()
        sparql_query = await generate_sparql_query(TEST_QUERY, filtered_schema=schema)
        query_result["sparql_query"] = sparql_query
        logger.info(f"Generated SPARQL query: {sparql_query}")
        
        # Step 2: Execute SPARQL query directly
        sparql_results = await execute_sparql_query(sparql_query)
        query_result["sparql_results"] = sparql_results
        logger.info(f"SPARQL results: {sparql_results}")
        
        # Step 3: Retrieve texts with confidence scores
        retrieved_texts, confidence_scores = await retriever.retrieve(TEST_QUERY)
        query_result["retrieved_texts"] = retrieved_texts
        query_result["confidence_scores"] = confidence_scores
        logger.info(f"Retrieved {len(retrieved_texts)} texts with confidence scores")
        
        # Step 4: Generate structured answer
        answer = await generate_structured_answer(TEST_QUERY, retrieved_texts, confidence_scores)
        query_result["answer"] = answer.answer
        end_time = time.time()
        
        query_result["execution_time"] = end_time - start_time
        logger.info(f"Successfully processed query with answer length: {len(answer.answer)}")
        
    except Exception as e:
        query_result["success"] = False
        query_result["error"] = str(e)
        logger.error(f"Failed processing: {str(e)}")
    
    # Save result to file
    with open("single_query_result.json", "w", encoding="utf-8") as f:
        json.dump(query_result, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Single query test completed: {'Success' if query_result['success'] else 'Failed'}")
    return query_result

async def main():
    """Run the single query test."""
    try:
        await test_single_query()
        logger.info("Single query test completed successfully!")
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main()) 