"""
Query generation and processing module for Ragatanga.

This module handles query analysis, processing, and answer generation.
"""

import re
import asyncio
from typing import List, Tuple, Dict, Any, Optional, Union
from pydantic import BaseModel, Field
from loguru import logger

from ragatanga.core.llm import LLMProvider

async def analyze_query_type(query: str, llm_provider=None) -> str:
    """
    Analyze the query to determine its type for better answer generation.
    
    Args:
        query: The query string
        llm_provider: LLM provider to use (if None, uses default)
        
    Returns:
        Query type: 'factual', 'descriptive', 'comparative', 'procedural', or 'exploratory'
    """
    if llm_provider is None:
        llm_provider = LLMProvider.get_provider()
    
    system_prompt = """
    You are a query analysis expert. Determine the type of the given query.
    Categorize it as one of the following:
    
    - factual: Simple questions asking for specific facts (e.g., "What is the price of Plan X?")
    - descriptive: Questions asking for descriptions (e.g., "Tell me about the São Bento unit")
    - comparative: Questions asking for comparisons (e.g., "What's the difference between Plan X and Plan Y?")
    - procedural: Questions about how to do something (e.g., "How do I cancel my subscription?")
    - exploratory: Open-ended questions seeking broader information (e.g., "What services are available?")
    
    Return only the category name, nothing else.
    """
    
    try:
        response = await llm_provider.generate_text(
            prompt=query,
            system_prompt=system_prompt,
            temperature=0.0,  # Use deterministic output
            max_tokens=50     # Very short response needed
        )
        
        query_type = response.strip().lower()
        
        # Validate and default to 'factual' if not recognized
        valid_types = {'factual', 'descriptive', 'comparative', 'procedural', 'exploratory'}
        if query_type not in valid_types:
            return 'factual'
            
        return query_type
    except Exception as e:
        logger.error(f"Error analyzing query type: {str(e)}")
        # Default to factual if analysis fails
        return 'factual'

def generate_system_prompt_by_query_type(query_type: str) -> str:
    """
    Generate a system prompt tailored to the query type.
    
    Args:
        query_type: Type of the query
        
    Returns:
        System prompt for the LLM
    """
    base_prompt = """
    You are an intelligent assistant specializing in fitness academy information.
    Your task is to provide accurate, helpful answers based on the provided facts.
    
    IMPORTANT GUIDELINES:
    1. Focus primarily on high-confidence facts
    2. When facts from different sources conflict, prefer SPARQL results over semantic search
    3. Clearly indicate any uncertainty or incomplete information
    4. Structure your response with appropriate markdown formatting
    5. Be concise but comprehensive
    6. Only include information that is directly relevant to the query
    """
    
    if query_type == 'factual':
        return base_prompt + """
        For this factual query:
        - Start with a direct answer to the specific question
        - Present the facts in a structured, easy-to-read format
        - Use bullet points for multiple facts
        - Include numerical data where available
        - Be precise and avoid unnecessary elaboration
        """
    
    elif query_type == 'descriptive':
        return base_prompt + """
        For this descriptive query:
        - Begin with a brief overview summary
        - Organize information into logical sections with headers
        - Include all relevant details from high-confidence sources
        - Use rich, descriptive language while maintaining accuracy
        - Include any context that helps understand the subject better
        """
    
    elif query_type == 'comparative':
        return base_prompt + """
        For this comparative query:
        - First identify the key entities being compared
        - Use a structured comparison format (table if appropriate)
        - Highlight similarities and differences explicitly
        - Include quantitative comparisons where possible
        - Provide a balanced assessment of advantages/disadvantages
        - End with a summary of key differences
        """
    
    elif query_type == 'procedural':
        return base_prompt + """
        For this procedural query:
        - Present instructions in a clear, step-by-step format
        - Number the steps sequentially
        - Include any prerequisites or requirements first
        - Highlight important cautions or notes
        - Provide alternative approaches if available
        - End with any follow-up actions that might be needed
        """
    
    elif query_type == 'exploratory':
        return base_prompt + """
        For this exploratory query:
        - Start with a broad overview of the topic
        - Organize information into major categories
        - Present a variety of relevant facts to give a comprehensive picture
        - Highlight particularly interesting or unusual information
        - Suggest related areas the user might want to explore further
        - Structure the response to help the user discover new information
        """
    
    else:
        return base_prompt  # Default prompt

def generate_fallback_answer(query: str, sparql_facts: List[str], semantic_facts: List[str]) -> str:
    """
    Generate a simple fallback answer when LLM generation fails.
    
    Args:
        query: The original query
        sparql_facts: List of facts from SPARQL
        semantic_facts: List of facts from semantic search
        
    Returns:
        Formatted fallback answer
    """
    answer = f"## Answer to: {query}\n\n"
    
    if not sparql_facts and not semantic_facts:
        return answer + "I don't have enough information to answer this question. Please try rephrasing your query or ask about a different topic."
    
    if sparql_facts:
        answer += "### Information from Structured Data:\n\n"
        for fact in sparql_facts[:5]:  # Limit to top 5
            answer += f"- {fact}\n"
        
    if semantic_facts:
        answer += "\n### Additional Information:\n\n"
        for fact in semantic_facts[:5]:  # Limit to top 5
            answer += f"- {fact}\n"
    
    return answer

class QueryResponse(BaseModel):
    """Model for query response information."""
    retrieved_facts_sparql: List[str] = Field(default_factory=list)
    retrieved_facts_semantic: List[str] = Field(default_factory=list)
    retrieved_facts: List[str] = Field(default_factory=list)
    answer: str = ""

    class Config:
        allow_mutation = True

async def generate_structured_answer(
    query: str,
    retrieved_texts: List[str],
    confidence_scores: List[float],
    llm_provider=None,
    temperature: float = 0.7,
    max_tokens: int = 8000
) -> QueryResponse:
    """
    Generate a structured, comprehensive answer using retrieved facts with intelligent organization.
    
    Args:
        query: The user's query
        retrieved_texts: List of retrieved text snippets
        confidence_scores: Confidence scores for each snippet
        llm_provider: LLM provider to use (if None, uses default)
        temperature: Temperature for text generation
        max_tokens: Maximum tokens for the answer
        
    Returns:
        A QueryResponse object with the structured answer
    """
    if llm_provider is None:
        llm_provider = LLMProvider.get_provider()
    
    # Split retrieved texts by source
    sparql_facts = []
    semantic_facts = []
    
    for fact, score in zip(retrieved_texts, confidence_scores):
        if fact.startswith("SPARQL:"):
            sparql_facts.append((fact.replace("SPARQL: ", ""), score))
        elif fact.startswith("Semantic:"):
            semantic_facts.append((fact.replace("Semantic: ", ""), score))
    
    # Analyze the user query to determine what information is most relevant
    query_type = await analyze_query_type(query, llm_provider)
    
    # Prepare prompt based on query type
    system_prompt = generate_system_prompt_by_query_type(query_type)
    
    # Build context for LLM
    # Structure the context to emphasize more confident results
    high_confidence_threshold = 0.7
    medium_confidence_threshold = 0.4
    
    high_confidence_facts = []
    medium_confidence_facts = []
    low_confidence_facts = []
    
    for i, (fact, score) in enumerate(sparql_facts + semantic_facts):
        source = "SPARQL" if i < len(sparql_facts) else "Semantic"
        confidence_indicator = f"[{source} | Confidence: {score:.2f}]"
        
        fact_with_confidence = f"{confidence_indicator} {fact}"
        if score >= high_confidence_threshold:
            high_confidence_facts.append(fact_with_confidence)
        elif score >= medium_confidence_threshold:
            medium_confidence_facts.append(fact_with_confidence)
        else:
            low_confidence_facts.append(fact_with_confidence)
    
    # Build the context with facts grouped by confidence
    context_block = "HIGH CONFIDENCE FACTS:\n" + "\n".join(high_confidence_facts) if high_confidence_facts else "HIGH CONFIDENCE FACTS: None"
    
    if medium_confidence_facts:
        context_block += "\n\nMEDIUM CONFIDENCE FACTS:\n" + "\n".join(medium_confidence_facts)
    else:
        context_block += "\n\nMEDIUM CONFIDENCE FACTS: None"
        
    if low_confidence_facts:
        context_block += "\n\nLOW CONFIDENCE FACTS:\n" + "\n".join(low_confidence_facts)
    else:
        context_block += "\n\nLOW CONFIDENCE FACTS: None"
    
    user_message = (
        f"User Query: {query}\n\n"
        f"Retrieved Facts:\n{context_block}\n\n"
        f"Please provide a comprehensive answer based on the above facts. "
        f"Focus primarily on high confidence facts. "
        f"Clearly indicate if information might be uncertain. "
        f"Use markdown formatting for better readability."
    )
    
    # Split facts for QueryResponse object
    sparql_fact_texts = [fact for fact, _ in sparql_facts]
    semantic_fact_texts = [fact for fact, _ in semantic_facts]
    all_fact_texts = retrieved_texts
    
    # Create initial response object
    response = QueryResponse(
        retrieved_facts=all_fact_texts,
        retrieved_facts_sparql=sparql_fact_texts,
        retrieved_facts_semantic=semantic_fact_texts,
        answer=""  # Will be filled by LLM response
    )
    
    # Generate the answer using the LLM
    try:
        try:
            # Try structured generation first (will work with OpenAI provider)
            llm_response = await llm_provider.generate_structured(
                prompt=user_message,
                response_model=QueryResponse,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            response = llm_response
            
        except NotImplementedError:
            # Fallback to text generation and manual construction
            answer_text = await llm_provider.generate_text(
                prompt=user_message,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            response.answer = answer_text
            
    except Exception as e:
        logger.error(f"Error generating answer: {str(e)}")
        # Provide a fallback answer using simple template
        response.answer = generate_fallback_answer(query, sparql_fact_texts, semantic_fact_texts)
    
    return response