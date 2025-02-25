#!/usr/bin/env python3
"""
Benchmarking tool for Ragatanga performance evaluation.

This script measures and compares the performance of different configurations
and components of the Ragatanga system.
"""
# Import Ragatanga components
from ragatanga.core.ontology import OntologyManager
from ragatanga.core.query import generate_structured_answer
from ragatanga.core.retrieval import AdaptiveRetriever
from ragatanga.utils.embeddings import EmbeddingProvider
from ragatanga.utils.sparql import generate_sparql_query

import argparse
import asyncio
import json
import logging
import os
import time
from datetime import datetime
from typing import Any, Awaitable, Callable, Dict, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ragatanga-benchmark")



# Test queries for benchmarking
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

class BenchmarkResult:
    """Class to store and report benchmark results."""
    
    def __init__(self, name: str):
        """
        Initialize a benchmark result.
        
        Args:
            name: Name of the benchmark
        """
        self.name = name
        self.iterations: List[Dict[str, Any]] = []
        self.start_time = time.time()
        self.end_time: float = 0
    
    def add_iteration(self, **kwargs):
        """
        Add a benchmark iteration.
        
        Args:
            **kwargs: Iteration data
        """
        self.iterations.append(kwargs)
    
    def finish(self):
        """Finish the benchmark and calculate final metrics."""
        self.end_time = time.time()
    
    def get_total_time(self) -> float:
        """Get the total time taken for all iterations."""
        if self.end_time == 0:
            self.end_time = time.time()
        return self.end_time - self.start_time
    
    def get_average_time(self, key: str = "execution_time") -> float:
        """
        Get the average time for a specific metric.
        
        Args:
            key: The metric key to average
            
        Returns:
            Average time in seconds
        """
        values = [it[key] for it in self.iterations if key in it]
        return sum(values) / len(values) if values else 0
    
    def get_success_rate(self) -> float:
        """Get the success rate as a percentage."""
        if not self.iterations:
            return 0.0
        successes = sum(1 for it in self.iterations if it.get("success", False))
        return (successes / len(self.iterations)) * 100
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the results to a dictionary."""
        return {
            "name": self.name,
            "total_time": self.get_total_time(),
            "average_time": self.get_average_time(),
            "success_rate": self.get_success_rate(),
            "iterations": len(self.iterations),
            "iteration_details": self.iterations
        }
    
    def __str__(self) -> str:
        """String representation of the results."""
        return (
            f"Benchmark: {self.name}\n"
            f"Total time: {self.get_total_time():.2f}s\n"
            f"Average time: {self.get_average_time():.2f}s\n"
            f"Success rate: {self.get_success_rate():.1f}%\n"
            f"Iterations: {len(self.iterations)}"
        )

async def run_benchmark(
    name: str,
    func: Callable[..., Awaitable[Any]],
    inputs: List[Any],
    iterations: int = 1,
    **kwargs
) -> BenchmarkResult:
    """
    Run a benchmark for a specific function.
    
    Args:
        name: Name of the benchmark
        func: Async function to benchmark
        inputs: List of inputs to the function
        iterations: Number of iterations per input
        **kwargs: Additional arguments to pass to the function
        
    Returns:
        Benchmark results
    """
    result = BenchmarkResult(name)
    
    logger.info(f"Starting benchmark: {name}")
    
    for i, input_data in enumerate(inputs):
        logger.info(f"Input {i+1}/{len(inputs)}: {input_data}")
        
        for j in range(iterations):
            try:
                start_time = time.time()
                output = await func(input_data, **kwargs)
                end_time = time.time()
                
                execution_time = end_time - start_time
                
                iteration_result = {
                    "input": input_data,
                    "execution_time": execution_time,
                    "success": True,
                    "iteration": j + 1,
                    "input_index": i
                }
                
                # Add output size metrics if applicable
                if isinstance(output, list):
                    iteration_result["output_size"] = len(output)
                elif hasattr(output, "__len__"):
                    iteration_result["output_size"] = len(output)
                
                # Add custom metrics based on output type
                if hasattr(output, "answer") and not isinstance(output, list):
                    iteration_result["answer_length"] = len(output.answer)
                
                result.add_iteration(**iteration_result)
                logger.info(f"  Iteration {j+1}/{iterations}: {execution_time:.2f}s")
                
            except Exception as e:
                logger.error(f"  Iteration {j+1}/{iterations} failed: {str(e)}")
                result.add_iteration(
                    input=input_data,
                    execution_time=time.time() - start_time,
                    success=False,
                    error=str(e),
                    iteration=j + 1,
                    input_index=i
                )
    
    result.finish()
    logger.info(f"Benchmark complete: {result}")
    return result

async def benchmark_sparql_generation(ontology_path: str) -> BenchmarkResult:
    """
    Benchmark SPARQL query generation.
    
    Args:
        ontology_path: Path to the ontology file
        
    Returns:
        Benchmark results
    """
    async def wrapped_func(query: str):
        ontology_manager = OntologyManager(ontology_path)
        await ontology_manager.load_and_materialize()
        schema = await ontology_manager.load_ontology_schema()
        return await generate_sparql_query(query, schema)
    
    return await run_benchmark(
        name="SPARQL Generation",
        func=wrapped_func,
        inputs=TEST_QUERIES[:5]  # Use first 5 queries for speed
    )

async def benchmark_ontology_loading(ontology_path: str) -> BenchmarkResult:
    """
    Benchmark ontology loading and materialization.
    
    Args:
        ontology_path: Path to the ontology file
        
    Returns:
        Benchmark results
    """
    async def load_ontology(iteration_number):
        # Create a new manager each time to test loading
        manager = OntologyManager(ontology_path)
        await manager.load_and_materialize(force_rebuild=True)
        return manager
    
    return await run_benchmark(
        name="Ontology Loading",
        func=load_ontology,
        inputs=list(range(3))  # Run 3 iterations
    )

async def benchmark_embedding_providers(
    num_queries: int = 5,
    providers: List[str] = ["openai", "huggingface", "sentence-transformers"]
) -> Dict[str, BenchmarkResult]:
    """
    Benchmark different embedding providers.
    
    Args:
        num_queries: Number of queries to test
        providers: List of provider names to benchmark
        
    Returns:
        Dictionary of benchmark results by provider
    """
    results = {}
    
    for provider_name in providers:
        try:
            logger.info(f"Testing provider: {provider_name}")
            
            # Try to get the provider
            try:
                provider = EmbeddingProvider.get_provider(provider_name)
            except Exception as e:
                logger.error(f"Failed to initialize {provider_name}: {str(e)}")
                continue
            
            # Benchmark single query embedding
            single_result = await run_benchmark(
                name=f"Single Embedding ({provider_name})",
                func=provider.embed_query,
                inputs=TEST_QUERIES[:num_queries]
            )
            results[f"single_{provider_name}"] = single_result
            
            # Benchmark batch embedding
            async def test_batch_embed(iteration_number):
                return await provider.embed_texts(TEST_QUERIES[:num_queries], batch_size=5)
            
            batch_result = await run_benchmark(
                name=f"Batch Embedding ({provider_name})",
                func=test_batch_embed,
                inputs=list(range(3))  # Run 3 iterations
            )
            results[f"batch_{provider_name}"] = batch_result
            
        except Exception as e:
            logger.error(f"Failed to benchmark {provider_name}: {str(e)}")
    
    return results

async def benchmark_adaptive_retrieval(ontology_path: str) -> BenchmarkResult:
    """
    Benchmark adaptive retrieval system.
    
    Args:
        ontology_path: Path to the ontology file
        
    Returns:
        Benchmark results
    """
    # Initialize ontology manager
    manager = OntologyManager(ontology_path)
    await manager.load_and_materialize()
    
    # Initialize adaptive retriever
    retriever = AdaptiveRetriever(manager)
    
    return await run_benchmark(
        name="Adaptive Retrieval",
        func=retriever.retrieve,
        inputs=TEST_QUERIES[:8]  # Use 8 queries for more thorough testing
    )

async def benchmark_end_to_end(ontology_path: str) -> BenchmarkResult:
    """
    Benchmark end-to-end query processing.
    
    Args:
        ontology_path: Path to the ontology file
        
    Returns:
        Benchmark results
    """
    # Initialize ontology manager
    manager = OntologyManager(ontology_path)
    await manager.load_and_materialize()
    
    # Initialize adaptive retriever
    retriever = AdaptiveRetriever(manager)
    
    async def process_query(query):
        # Step 1: Retrieve texts with confidence scores
        retrieved_texts, confidence_scores = await retriever.retrieve(query)
        
        # Step 2: Generate structured answer
        return await generate_structured_answer(query, retrieved_texts, confidence_scores)
    
    return await run_benchmark(
        name="End-to-End Query Processing",
        func=process_query,
        inputs=TEST_QUERIES[:3]  # Use first 3 queries for speed
    )

async def run_all_benchmarks(ontology_path: str) -> Dict[str, Any]:
    """
    Run all benchmarks and collect results.
    
    Args:
        ontology_path: Path to the ontology file
        
    Returns:
        Dictionary of all benchmark results
    """
    all_results = {}
    
    # 1. Ontology loading benchmark
    all_results["ontology_loading"] = await benchmark_ontology_loading(ontology_path)
    
    # 2. SPARQL generation benchmark
    all_results["sparql_generation"] = await benchmark_sparql_generation(ontology_path)
    
    # 3. Embedding providers benchmark
    embedding_results = await benchmark_embedding_providers()
    all_results.update(embedding_results)
    
    # 4. Adaptive retrieval benchmark
    all_results["adaptive_retrieval"] = await benchmark_adaptive_retrieval(ontology_path)
    
    # 5. End-to-end benchmark
    all_results["end_to_end"] = await benchmark_end_to_end(ontology_path)
    
    return all_results

async def main():
    """Main entry point for the benchmark tool."""
    parser = argparse.ArgumentParser(description="Ragatanga benchmark tool")
    parser.add_argument(
        "--ontology",
        default=os.path.join("ragatanga", "data", "ontology.ttl"),
        help="Path to the ontology file"
    )
    parser.add_argument(
        "--output",
        default=f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        help="Output file for benchmark results"
    )
    parser.add_argument(
        "--benchmark",
        choices=["all", "ontology", "sparql", "embedding", "retrieval", "end_to_end"],
        default="all",
        help="Which benchmark to run"
    )
    parser.add_argument(
        "--providers",
        nargs="+",
        default=["openai", "sentence-transformers"],
        help="Embedding providers to benchmark"
    )
    
    args = parser.parse_args()
    
    # Check if ontology file exists
    if not os.path.exists(args.ontology):
        logger.error(f"Ontology file not found: {args.ontology}")
        return
    
    logger.info(f"Starting benchmarks with ontology: {args.ontology}")
    
    results = {}
    
    if args.benchmark == "all" or args.benchmark == "ontology":
        results["ontology_loading"] = await benchmark_ontology_loading(args.ontology)
        
    if args.benchmark == "all" or args.benchmark == "sparql":
        results["sparql_generation"] = await benchmark_sparql_generation(args.ontology)
        
    if args.benchmark == "all" or args.benchmark == "embedding":
        embedding_results = await benchmark_embedding_providers(providers=args.providers)
        results.update(embedding_results)
        
    if args.benchmark == "all" or args.benchmark == "retrieval":
        results["adaptive_retrieval"] = await benchmark_adaptive_retrieval(args.ontology)
        
    if args.benchmark == "all" or args.benchmark == "end_to_end":
        results["end_to_end"] = await benchmark_end_to_end(args.ontology)
    
    # Convert results to dictionary
    results_dict = {name: result.to_dict() for name, result in results.items()}
    
    # Add metadata
    results_dict["metadata"] = {
        "timestamp": datetime.now().isoformat(),
        "ontology_path": args.ontology,
        "benchmark_type": args.benchmark
    }
    
    # Save results to file
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results_dict, f, indent=2)
    
    logger.info(f"Benchmark results saved to: {args.output}")
    
    # Print summary
    print("\n===== BENCHMARK SUMMARY =====")
    for name, result in results.items():
        print(f"\n{result}")

if __name__ == "__main__":
    asyncio.run(main())