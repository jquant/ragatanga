import os
import asyncio
import json
from loguru import logger

# Set up environment variables
os.environ["ONTOLOGY_PATH"] = os.path.join(os.path.dirname(__file__), "ragatanga/data/sample_ontology.ttl")
os.environ["KNOWLEDGE_BASE_PATH"] = os.path.join(os.path.dirname(__file__), "ragatanga/data/sample_knowledge_base.md")

# Print environment variables for debugging
logger.debug(f"OpenAI API Key exists: {bool(os.environ.get('OPENAI_API_KEY'))}")
logger.debug(f"ONTOLOGY_PATH: {os.environ.get('ONTOLOGY_PATH')}")
logger.debug(f"KNOWLEDGE_BASE_PATH: {os.environ.get('KNOWLEDGE_BASE_PATH')}")

async def test_query():
    # Import necessary modules
    from ragatanga.core.ontology import OntologyManager
    from ragatanga.core.retrieval import AdaptiveRetriever
    from ragatanga.core.semantic import SemanticSearch
    from ragatanga.core.query import generate_structured_answer
    
    # Initialize ontology manager
    logger.info(f"Initializing ontology manager with {os.environ.get('ONTOLOGY_PATH')}")
    ontology_path = os.environ.get('ONTOLOGY_PATH')
    if not ontology_path:
        raise ValueError("ONTOLOGY_PATH environment variable is not set")
    ontology_manager = OntologyManager(ontology_path)
    await ontology_manager.load_and_materialize()
    
    # Initialize semantic search
    logger.info(f"Initializing semantic search with {os.environ.get('KNOWLEDGE_BASE_PATH')}")
    kb_path = os.environ.get('KNOWLEDGE_BASE_PATH')
    if not kb_path:
        raise ValueError("KNOWLEDGE_BASE_PATH environment variable is not set")
    semantic_search = SemanticSearch()
    await semantic_search.load_knowledge_base(kb_path)
    
    # Initialize adaptive retriever
    logger.info("Initializing adaptive retriever")
    retriever = AdaptiveRetriever(ontology_manager)
    
    # Test query
    query = "What is Ragatanga?"
    logger.info(f"Processing query: {query}")
    
    # Use adaptive retrieval
    logger.debug("Starting adaptive retrieval")
    retrieved_texts, confidence_scores = await retriever.retrieve(query)
    logger.debug(f"Retrieved {len(retrieved_texts)} results with adaptive parameters")
    
    # Print retrieved texts and confidence scores
    for i, (text, score) in enumerate(zip(retrieved_texts, confidence_scores)):
        logger.info(f"Result {i+1}: {text} (Confidence: {score:.2f})")
    
    # Generate structured answer
    logger.debug("Starting answer generation")
    try:
        # Get the LLM provider
        from ragatanga.core.llm import LLMProvider
        llm_provider = LLMProvider.get_provider()
        logger.info(f"Using LLM provider: {type(llm_provider).__name__}")
        
        # Generate the answer
        answer = await generate_structured_answer(query, retrieved_texts, confidence_scores, llm_provider=llm_provider)
        
        # Print the answer
        logger.info("Answer:")
        logger.info(json.dumps(answer.model_dump(), indent=2))
        
        # If the answer is empty, try to generate it directly
        if not answer.answer:
            logger.warning("Answer is empty, trying direct text generation")
            
            # Prepare a simple prompt
            prompt = f"Query: {query}\n\nFacts:\n"
            for i, (text, score) in enumerate(zip(retrieved_texts, confidence_scores)):
                prompt += f"{i+1}. {text} (Confidence: {score:.2f})\n"
            prompt += "\nPlease provide a comprehensive answer based on these facts."
            
            # Generate text directly
            direct_answer = await llm_provider.generate_text(
                prompt=prompt,
                system_prompt="You are a helpful assistant that answers questions based on provided facts.",
                temperature=0.7,
                max_tokens=1000
            )
            
            logger.info("Direct answer:")
            logger.info(direct_answer)
    except Exception as e:
        logger.error(f"Error during answer generation: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
    
    return answer

if __name__ == "__main__":
    asyncio.run(test_query()) 