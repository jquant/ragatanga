# Real-World Examples

This page provides in-depth examples of using Ragatanga in real-world scenarios.

## 1. Building a Custom Knowledge Base

This example demonstrates how to build a custom knowledge base from various sources including Markdown files, websites, and structured data.

```python
import asyncio
from ragatanga.core.ontology import OntologyManager
from ragatanga.core.knowledge import KnowledgeBaseBuilder
from ragatanga.utils.web import extract_content

async def build_custom_kb():
    # Initialize ontology
    ontology_path = "company_ontology.ttl"
    ontology_manager = OntologyManager(ontology_path)
    await ontology_manager.load_and_materialize()
    
    # Create knowledge base builder
    kb_builder = KnowledgeBaseBuilder(ontology_manager)
    
    # Add content from Markdown files
    await kb_builder.add_markdown_file("company_handbook.md")
    await kb_builder.add_markdown_file("product_documentation.md")
    
    # Add content from websites
    company_blog_content = await extract_content("https://example.com/blog")
    await kb_builder.add_text(company_blog_content, source="Company Blog")
    
    # Add content from structured data
    with open("product_catalog.json", "r") as f:
        import json
        products = json.load(f)
        
    for product in products:
        product_text = f"""
        # {product['name']}
        
        {product['description']}
        
        - Price: ${product['price']}
        - Category: {product['category']}
        - Features: {', '.join(product['features'])}
        """
        await kb_builder.add_text(product_text, source=f"Product: {product['name']}")
    
    # Save the knowledge base
    await kb_builder.save("company_knowledge_base.md")
    
    # Generate embeddings
    from ragatanga.core.embeddings import EmbeddingManager
    embedding_manager = EmbeddingManager()
    await embedding_manager.generate_embeddings("company_knowledge_base.md")
    
    print("Knowledge base built successfully!")

if __name__ == "__main__":
    asyncio.run(build_custom_kb())
```

## 2. Building a Question-Answering System

This example shows how to build a question-answering system using Ragatanga:

```python
import asyncio
from ragatanga.core.ontology import OntologyManager
from ragatanga.core.retrieval import AdaptiveRetriever
from ragatanga.core.query import generate_structured_answer
from ragatanga.core.config import RetrievalConfig, LLMConfig

async def initialize_qa_system():
    # Initialize ontology
    ontology_path = "knowledge_ontology.ttl"
    ontology_manager = OntologyManager(ontology_path)
    await ontology_manager.load_and_materialize()
    
    # Configure retrieval parameters
    retrieval_config = RetrievalConfig(
        semantic_search_weight=0.7,
        ontology_search_weight=0.3,
        max_results=5,
        confidence_threshold=0.6,
        use_query_enhancement=True
    )
    
    # Configure LLM
    llm_config = LLMConfig(
        provider="openai",
        model_name="gpt-4",
        temperature=0.2,
        max_tokens=500,
        system_prompt="You are an expert assistant that provides accurate and concise answers based on the provided context."
    )
    
    # Initialize retriever
    from ragatanga.core.llm import LLMManager
    llm_manager = LLMManager(config=llm_config)
    
    retriever = AdaptiveRetriever(
        ontology_manager,
        llm_manager=llm_manager,
        config=retrieval_config
    )
    
    return retriever

async def answer_question(retriever, question):
    # Retrieve information
    results = await retriever.retrieve(question)
    
    # Generate structured answer
    answer = await generate_structured_answer(question, results)
    
    # Format output
    if not results:
        return {
            "question": question,
            "answer": "I don't have enough information to answer this question.",
            "confidence": 0.0,
            "sources": []
        }
    
    sources = [
        {"content": r.content[:100] + "...", "confidence": r.confidence}
        for r in results
    ]
    
    return {
        "question": question,
        "answer": answer,
        "confidence": max(r.confidence for r in results),
        "sources": sources
    }

async def main():
    retriever = await initialize_qa_system()
    
    questions = [
        "What is Ragatanga's architecture?",
        "How do I configure the embedding provider?",
        "What are the key features of the adaptive retriever?"
    ]
    
    for question in questions:
        response = await answer_question(retriever, question)
        print(f"\nQ: {response['question']}")
        print(f"A: {response['answer']}")
        print(f"Confidence: {response['confidence']:.2f}")
        print("Sources:")
        for i, source in enumerate(response['sources']):
            print(f"  {i+1}. {source['content']} (Confidence: {source['confidence']:.2f})")

if __name__ == "__main__":
    asyncio.run(main())
```

## 3. Building a REST API with FastAPI

This example demonstrates how to create a RESTful API using Ragatanga and FastAPI:

```python
import asyncio
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager

from ragatanga.core.ontology import OntologyManager
from ragatanga.core.retrieval import AdaptiveRetriever

# Define models
class QueryRequest(BaseModel):
    query: str
    max_results: Optional[int] = 5
    confidence_threshold: Optional[float] = 0.6

class ResultItem(BaseModel):
    content: str
    confidence: float
    source: str

class QueryResponse(BaseModel):
    query: str
    answer: str
    results: List[ResultItem]
    execution_time_ms: float

# Global variables
ontology_manager = None
retriever = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize on startup
    global ontology_manager, retriever
    
    print("Initializing Ragatanga...")
    
    ontology_path = "knowledge_ontology.ttl"
    ontology_manager = OntologyManager(ontology_path)
    await ontology_manager.load_and_materialize()
    
    retriever = AdaptiveRetriever(ontology_manager)
    
    print("Ragatanga initialized successfully!")
    
    yield
    
    # Cleanup on shutdown
    print("Shutting down Ragatanga...")

app = FastAPI(lifespan=lifespan, title="Ragatanga API", description="API for querying knowledge using Ragatanga")

async def get_retriever():
    if retriever is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    return retriever

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest, retriever: AdaptiveRetriever = Depends(get_retriever)):
    import time
    start_time = time.time()
    
    try:
        # Process query
        results = await retriever.retrieve(
            request.query,
            max_results=request.max_results,
            confidence_threshold=request.confidence_threshold
        )
        
        # Generate answer
        from ragatanga.core.query import generate_structured_answer
        answer = await generate_structured_answer(request.query, results)
        
        # Format response
        result_items = [
            ResultItem(
                content=result.content,
                confidence=result.confidence,
                source=result.source
            )
            for result in results
        ]
        
        execution_time = (time.time() - start_time) * 1000  # convert to ms
        
        return QueryResponse(
            query=request.query,
            answer=answer,
            results=result_items,
            execution_time_ms=execution_time
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

## 4. Command-Line Interface

This example shows how to create a command-line interface for Ragatanga:

```python
import asyncio
import argparse
import sys
import json
from ragatanga.core.ontology import OntologyManager
from ragatanga.core.retrieval import AdaptiveRetriever
from ragatanga.core.config import RetrievalConfig

async def setup_ragatanga(ontology_path, kb_path=None):
    # Initialize ontology manager
    ontology_manager = OntologyManager(ontology_path)
    await ontology_manager.load_and_materialize()
    
    # Initialize adaptive retriever
    retriever = AdaptiveRetriever(ontology_manager)
    
    return retriever

async def process_query(retriever, query, output_format="text", max_results=5):
    # Retrieve information
    results = await retriever.retrieve(query, max_results=max_results)
    
    # Generate structured answer
    from ragatanga.core.query import generate_structured_answer
    answer = await generate_structured_answer(query, results)
    
    # Format output
    if output_format == "json":
        output = {
            "query": query,
            "answer": answer,
            "results": [
                {
                    "content": r.content,
                    "confidence": r.confidence,
                    "source": r.source
                }
                for r in results
            ]
        }
        print(json.dumps(output, indent=2))
    else:
        print(f"Query: {query}")
        print(f"Answer: {answer}")
        print("\nSupporting information:")
        for i, result in enumerate(results):
            print(f"\n--- Result {i+1} (Confidence: {result.confidence:.2f}) ---")
            print(f"Source: {result.source}")
            print(f"{result.content[:300]}...")

async def main():
    parser = argparse.ArgumentParser(description="Ragatanga CLI")
    parser.add_argument("--ontology", required=True, help="Path to ontology file")
    parser.add_argument("--knowledge-base", help="Path to knowledge base file")
    parser.add_argument("--query", help="Query to process")
    parser.add_argument("--format", choices=["text", "json"], default="text", help="Output format")
    parser.add_argument("--max-results", type=int, default=5, help="Maximum number of results")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    
    args = parser.parse_args()
    
    # Set up Ragatanga
    retriever = await setup_ragatanga(args.ontology, args.knowledge_base)
    
    if args.interactive:
        print("Ragatanga Interactive Mode (Ctrl+C to exit)")
        print("-------------------------------------------")
        try:
            while True:
                query = input("\nEnter query: ")
                if not query:
                    continue
                await process_query(retriever, query, args.format, args.max_results)
        except KeyboardInterrupt:
            print("\nExiting...")
    elif args.query:
        await process_query(retriever, args.query, args.format, args.max_results)
    else:
        parser.print_help()

if __name__ == "__main__":
    asyncio.run(main())
```

## 5. Integration with Streamlit

This example shows how to create a simple web interface using Streamlit:

```python
import streamlit as st
import asyncio
from ragatanga.core.ontology import OntologyManager
from ragatanga.core.retrieval import AdaptiveRetriever
from ragatanga.core.query import generate_structured_answer

@st.cache_resource
def load_ragatanga():
    # Create an async wrapper for initialization
    async def init():
        ontology_path = "knowledge_ontology.ttl"
        ontology_manager = OntologyManager(ontology_path)
        await ontology_manager.load_and_materialize()
        return AdaptiveRetriever(ontology_manager)
    
    # Run the async init function
    return asyncio.run(init())

# Initialize Ragatanga
retriever = load_ragatanga()

# Streamlit UI
st.title("Ragatanga Knowledge Assistant")
st.write("Ask questions about Ragatanga and get answers from its knowledge base.")

query = st.text_input("Your question:", placeholder="e.g., What is Ragatanga's architecture?")

if st.button("Submit") or query:
    if query:
        st.write("Processing query...")
        
        # Create an async function for the query
        async def process_query():
            results = await retriever.retrieve(query, max_results=5)
            answer = await generate_structured_answer(query, results)
            return results, answer
        
        # Run the async function
        results, answer = asyncio.run(process_query())
        
        # Display the answer
        st.write("## Answer")
        st.write(answer)
        
        # Display the source information
        st.write("## Sources")
        for i, result in enumerate(results):
            with st.expander(f"Source {i+1} (Confidence: {result.confidence:.2f})"):
                st.write(f"**Source:** {result.source}")
                st.write(result.content)
    else:
        st.write("Please enter a question.")
```

## Running the Examples

To run these examples, you'll need to:

1. Install Ragatanga and its dependencies:

```bash
pip install ragatanga
```

2. Install additional dependencies for specific examples:

```bash
# For API example
pip install fastapi uvicorn

# For Streamlit example
pip install streamlit
```

3. Prepare your ontology and knowledge base files as needed.

4. Run the Python scripts for the examples you want to try. 