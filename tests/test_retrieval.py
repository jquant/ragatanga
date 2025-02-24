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
    analyze_query_type
)

# Configure logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ragatanga-test")

# Get the absolute path to the ontology file
ONTOLOGY_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'ragatanga', 'data', 'ontology.ttl'))
logger.info(f"Using ontology file at: {ONTOLOGY_PATH}")

# Test queries
TEST_QUERIES = [
    # Simple factual queries
    "What is the price of the Plus plan?",
    "Which units are in Belo Horizonte?",
    "List all the benefits of the Prime plan",
    
    # Descriptive queries
    "Tell me about the São Bento unit",
    "What features does the Premium plan have?",
    "Describe the swimming pool at Guarani",
    
    # Comparative queries
    "What's the difference between Slim and Premium units?",
    "Compare the Plus plan and Combo Saúde plan",
    "How does the São Bento unit compare to Mangabeiras?",
    
    # Procedural queries
    "How do I cancel my subscription?",
    "What's the process for getting a bioimpedance exam?",
    "How can I sign up for swimming classes?",
    
    # Exploratory queries
    "What activities are available at the academies?",
    "Tell me about the different types of units",
    "What services does Pratique Fitness offer?"
]

async def test_sparql_generation():
    """Test improved SPARQL query generation."""
    logger.info("Testing SPARQL query generation...")
    
    # Initialize ontology manager with absolute path
    ontology_manager = OntologyManager(ONTOLOGY_PATH)
    await ontology_manager.load_and_materialize()
    
    results = []
    
    for query in TEST_QUERIES[:5]:  # Test first 5 queries for speed
        logger.info(f"Generating SPARQL for: {query}")
        try:
            start_time = time.time()
            sparql_query = await generate_sparql_query(query)
            end_time = time.time()
            
            results.append({
                "query": query,
                "sparql_query": sparql_query,
                "execution_time": end_time - start_time,
                "success": True
            })
            logger.info(f"Successfully generated SPARQL: {sparql_query[:100]}...")
        except Exception as e:
            results.append({
                "query": query,
                "error": str(e),
                "success": False
            })
            logger.error(f"Failed to generate SPARQL: {str(e)}")
    
    # Save results to file
    with open("sparql_generation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"SPARQL generation test completed: {sum(1 for r in results if r['success'])} succeeded, {sum(1 for r in results if not r['success'])} failed")
    return results

async def test_adaptive_retrieval():
    """Test adaptive retrieval system."""
    logger.info("Testing adaptive retrieval...")
    
    # Initialize ontology manager with absolute path
    ontology_manager = OntologyManager(ONTOLOGY_PATH)
    await ontology_manager.load_and_materialize()
    
    # Initialize adaptive retriever
    retriever = AdaptiveRetriever(ontology_manager)
    
    results = []
    
    for query in TEST_QUERIES[:5]:  # Test first 5 queries for speed
        logger.info(f"Retrieving results for: {query}")
        try:
            start_time = time.time()
            retrieved_texts, confidence_scores = await retriever.retrieve(query)
            end_time = time.time()
            
            query_type = await analyze_query_type(query)
            
            results.append({
                "query": query,
                "query_type": query_type,
                "num_results": len(retrieved_texts),
                "execution_time": end_time - start_time,
                "top_confidence_scores": confidence_scores[:3] if confidence_scores else [],
                "success": True
            })
            logger.info(f"Successfully retrieved {len(retrieved_texts)} results")
        except Exception as e:
            results.append({
                "query": query,
                "error": str(e),
                "success": False
            })
            logger.error(f"Failed retrieval: {str(e)}")
    
    # Save results to file
    with open("adaptive_retrieval_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Adaptive retrieval test completed: {sum(1 for r in results if r['success'])} succeeded, {sum(1 for r in results if not r['success'])} failed")
    return results

async def test_end_to_end():
    """Test the full end-to-end process."""
    logger.info("Testing end-to-end query processing...")
    
    # Initialize ontology manager with absolute path
    ontology_manager = OntologyManager(ONTOLOGY_PATH)
    await ontology_manager.load_and_materialize()
    
    # Initialize adaptive retriever
    retriever = AdaptiveRetriever(ontology_manager)
    
    results = []
    
    for query in TEST_QUERIES[:3]:  # Test first 3 queries for speed
        logger.info(f"Processing end-to-end query: {query}")
        try:
            # Step 1: Retrieve texts with confidence scores
            start_time = time.time()
            retrieved_texts, confidence_scores = await retriever.retrieve(query)
            
            # Step 2: Generate structured answer
            answer = await generate_structured_answer(query, retrieved_texts, confidence_scores)
            end_time = time.time()
            
            results.append({
                "query": query,
                "answer_length": len(answer.answer),
                "total_execution_time": end_time - start_time,
                "success": True
            })
            logger.info(f"Successfully generated answer of length {len(answer.answer)}")
        except Exception as e:
            results.append({
                "query": query,
                "error": str(e),
                "success": False
            })
            logger.error(f"Failed end-to-end processing: {str(e)}")
    
    # Save results to file
    with open("end_to_end_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"End-to-end test completed: {sum(1 for r in results if r['success'])} succeeded, {sum(1 for r in results if not r['success'])} failed")
    return results

async def main():
    """Run all tests."""
    try:
        await test_sparql_generation()
        await test_adaptive_retrieval()
        await test_end_to_end()
        logger.info("All tests completed successfully!")
    except Exception as e:
        logger.error(f"Test suite failed: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())