import asyncio
import json
import time
import os
import sys

# Add parent directory to path so we can import our modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import required modules
from ragatanga.main import (
    generate_sparql_query, 
    OntologyManager, 
    AdaptiveRetriever,
    generate_structured_answer,
    execute_sparql_query,
    embed_texts_in_batches,
    build_faiss_index,
    load_faiss_index,
    save_faiss_index,
    BATCH_SIZE,
    KBASE_FILE,
    KBASE_FAISS_INDEX_FILE,
    KBASE_EMBEDDINGS_FILE
)

# Configure logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ragatanga-single-test")

# Get the absolute path to the ontology file
ONTOLOGY_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'ragatanga', 'data', 'ontology.ttl'))
logger.info(f"Using ontology file at: {ONTOLOGY_PATH}")

# Single test query
TEST_QUERY = "Which units are in Belo Horizonte?"

async def initialize_knowledge_base():
    """Initialize the knowledge base index."""
    logger.info("Initializing knowledge base...")
    
    # Import necessary globals
    from ragatanga.main import kbase_entries, kbase_index, kbase_embeddings_np
    import numpy as np
    
    # Check if knowledge base file exists
    if not os.path.exists(KBASE_FILE):
        logger.warning(f"Knowledge base file {KBASE_FILE} not found. Tests may fail.")
        return False
    
    # Load knowledge base content
    with open(KBASE_FILE, "r", encoding="utf-8") as file:
        kbase_content = file.read()
    
    # Update global kbase_entries
    global_kbase_entries = [chunk.strip() for chunk in kbase_content.split("\n\n") if chunk.strip()]
    
    # Set the global variable in the main module
    import ragatanga.main
    ragatanga.main.kbase_entries = global_kbase_entries
    
    # Build or load FAISS index
    if os.path.exists(KBASE_FAISS_INDEX_FILE) and os.path.exists(KBASE_EMBEDDINGS_FILE):
        logger.info("Loading existing FAISS index...")
        index, embeddings = load_faiss_index(KBASE_FAISS_INDEX_FILE, KBASE_EMBEDDINGS_FILE)
        
        # Set the global variables in the main module
        ragatanga.main.kbase_index = index
        ragatanga.main.kbase_embeddings_np = embeddings
    else:
        logger.info("Building new FAISS index...")
        embeddings = await embed_texts_in_batches(global_kbase_entries, BATCH_SIZE)
        index, embeddings = build_faiss_index(np.asarray(embeddings))
        save_faiss_index(index, KBASE_FAISS_INDEX_FILE, embeddings, KBASE_EMBEDDINGS_FILE)
        
        # Set the global variables in the main module
        ragatanga.main.kbase_index = index
        ragatanga.main.kbase_embeddings_np = embeddings
    
    logger.info("Knowledge base initialization complete!")
    return True

async def test_single_query():
    """Test with a single query."""
    logger.info(f"Testing single query: {TEST_QUERY}")
    
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
        # Step 1: Generate SPARQL query
        start_time = time.time()
        sparql_query = await generate_sparql_query(TEST_QUERY)
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