"""
SPARQL utilities module for Ragatanga.

This module provides utilities for generating and executing SPARQL queries.
"""

import asyncio
import re
from typing import List, Dict, Any, Optional, Type, TypeVar, Generic
from pydantic import BaseModel, Field, field_validator
from rdflib.plugins.sparql.parser import parseQuery
from loguru import logger

from ragatanga.utils.embeddings import EmbeddingProvider

T = TypeVar('T')

class Query(BaseModel):
    """
    A model for handling SPARQL query generation with validation.
    """
    user_query: str = Field(..., description="The user's original query")
    reasoning_about_schema: str = Field(..., description="Reasoning about the schema and how it relates to the user's query")
    valid_sparql_query: str = Field(..., description="A valid SPARQL query")

    @field_validator("valid_sparql_query")
    def check_sparql_validity(cls, value):
        try:
            parseQuery(value)
        except Exception as e:
            raise ValueError(
                f"Invalid SPARQL query: {e}. Please prompt the LLM to generate a correct SPARQL query."
            ) from e
        return value

class SPARQLQueryGenerator(BaseModel):
    """
    A model for generating SPARQL queries using a plan-and-solve approach.
    """
    query_analysis: str = Field(..., description="Analysis of the natural language query and relevant ontology concepts")
    query_plan: str = Field(..., description="Step-by-step plan for constructing the SPARQL query")
    sparql_query: str = Field(..., description="The final SPARQL query with proper PREFIX declarations")

    @field_validator("sparql_query")
    def validate_sparql(cls, value):
        if "PREFIX" not in value:
            prefixes = """PREFIX : <http://example.org/ontology#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
"""
            value = prefixes + value
        try:
            parseQuery(value)
        except Exception as e:
            raise ValueError(f"Invalid SPARQL query: {e}")
        return value

async def generate_sparql_query(query: str, filtered_schema: str, llm_provider=None) -> str:
    """
    Generate a SPARQL query from a natural language query with improved prompting.
    
    Args:
        query: The natural language query
        filtered_schema: Filtered ontology schema relevant to the query
        llm_provider: LLM provider to use (if None, uses default)
        
    Returns:
        Generated SPARQL query
    """
    # Import here to avoid circular imports
    from ragatanga.core.llm import LLMProvider
    
    if llm_provider is None:
        llm_provider = LLMProvider.get_provider()
    
    system_prompt = """You are a SPARQL query expert specializing in ontology querying.
Your task is to translate natural language questions into precise SPARQL queries.

IMPORTANT GUIDELINES:
1. Always include necessary PREFIX declarations
2. Use DISTINCT to avoid duplicate results
3. Include rdfs:label when available for human-readable results
4. Use OPTIONAL for potentially missing properties
5. Include FILTER when appropriate to narrow results
6. Return a reasonably limited number of results (use LIMIT if needed)

ONTOLOGY SCHEMA:
The following schema shows the classes and properties relevant to the user's query:

{schema}

EXAMPLES:
User: "What unidades are in Belo Horizonte?"
SPARQL:
```

PREFIX : <http://www.semanticweb.org/ontologies/pratique-fitness/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT DISTINCT ?unidade ?label ?address
WHERE {{
  ?unidade a :Unidade ;
           rdfs:label ?label ;
           :hasCity "Belo Horizonte" .
  OPTIONAL {{ ?unidade :hasAddress ?address }}
}}
ORDER BY ?label
```

User: "What are the planos available at unidade São Bento?"
SPARQL:
```

PREFIX : <http://www.semanticweb.org/ontologies/pratique-fitness/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT DISTINCT ?plano ?planoLabel ?price
WHERE {{
  :unidade_sao_bento :hasPlan ?plano .
  ?plano rdfs:label ?planoLabel ;
         :hasPrice ?price .
}}
ORDER BY ?price
```

Now, generate a valid SPARQL query for the following user question:"""

    user_message = f"User question: {query}\n\nPlease generate a SPARQL query to answer this question based on the provided ontology schema."
    
    try:
        response = await llm_provider.generate_structured(
            prompt=user_message,
            response_model=Query,
            system_prompt=system_prompt.format(schema=filtered_schema)
        )
        generated_query = response.valid_sparql_query
        
        # Validate the generated query
        try:
            parseQuery(generated_query)
        except Exception as e:
            logger.warning(f"Generated query validation failed: {e}. Falling back.")
            return await generate_fallback_query(query, filtered_schema)
            
    except Exception as e:
        logger.warning(f"Failed to generate query: {e}. Using fallback strategy.")
        generated_query = await generate_fallback_query(query, filtered_schema)
    
    logger.debug(f"Generated SPARQL Query:\n{generated_query}")
    return generated_query

async def generate_fallback_query(query: str, schema: str) -> str:
    """
    Generate a fallback query if the main generation fails.
    
    Args:
        query: The natural language query
        schema: The ontology schema
        
    Returns:
        A fallback SPARQL query
    """
    # Extract potential entity names from the query
    words = re.findall(r'\b\w+\b', query.lower())
    
    # Create a simple but more targeted query than the default one
    fallback_query = """
PREFIX : <http://www.semanticweb.org/ontologies/pratique-fitness/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX owl: <http://www.w3.org/2002/07/owl#>

SELECT DISTINCT ?subject ?predicate ?object ?label
WHERE {
    ?subject ?predicate ?object .
"""
    
    # Add filters for words that might match entities
    filters = []
    for word in words:
        if len(word) > 3:  # Only consider words with at least 4 characters
            filters.append(f"""
    OPTIONAL {{
        ?subject rdfs:label ?label .
        FILTER(CONTAINS(LCASE(STR(?label)), "{word}"))
    }}
    OPTIONAL {{
        ?object rdfs:label ?objLabel .
        FILTER(CONTAINS(LCASE(STR(?objLabel)), "{word}"))
    }}""")
    
    if filters:
        fallback_query += "\n".join(filters)
    
    fallback_query += """
}
LIMIT 30
"""
    
    return fallback_query

def text_similarity(text1: str, text2: str) -> float:
    """
    Calculate simple text similarity using Jaccard similarity of words.
    
    Args:
        text1: First text
        text2: Second text
        
    Returns:
        Similarity score between 0 and 1
    """
    # Tokenize and convert to sets
    words1 = set(re.findall(r'\b\w+\b', text1.lower()))
    words2 = set(re.findall(r'\b\w+\b', text2.lower()))
    
    # Calculate Jaccard similarity
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    
    return intersection / union if union > 0 else 0.0