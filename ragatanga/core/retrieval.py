"""
Hybrid retrieval module for Ragatanga.

This module implements the hybrid retrieval approach combining ontology-based
and semantic search-based retrieval with adaptive parameter tuning.
"""

import re
from typing import List, Tuple
from loguru import logger

from ragatanga.core.ontology import OntologyManager, extract_relevant_schema
from ragatanga.core.semantic import retrieve_top_k_with_scores
from ragatanga.utils.sparql import text_similarity, generate_sparql_query

class AdaptiveRetriever:
    """
    Implements adaptive retrieval with dynamic parameters based on query complexity and type.
    """
    
    def __init__(self, ontology_manager, base_top_k=10):
        """
        Initialize the adaptive retriever.
        
        Args:
            ontology_manager: OntologyManager instance
            base_top_k: Base number of results to retrieve
        """
        self.ontology_manager = ontology_manager
        self.base_top_k = base_top_k
        self.query_cache = {}  # Simple cache of previous queries and their parameters
    
    async def retrieve(self, query: str) -> Tuple[List[str], List[float]]:
        """
        Perform adaptive retrieval with parameters tailored to the query.
        
        Args:
            query: The user's query
            
        Returns:
            Tuple of (merged_results, confidence_scores)
        """
        # Check cache for similar queries
        cache_hit, cached_params = self._check_cache(query)
        
        if cache_hit:
            logger.info(f"Using cached parameters for similar query: {cached_params}")
            sparql_weight, semantic_weight, top_k = cached_params
        else:
            # Analyze query complexity and type
            query_complexity = await self._analyze_query_complexity(query)
            query_specificity = await self._analyze_query_specificity(query)
            
            # Import here to avoid circular imports
            from ragatanga.core.query import analyze_query_type
            query_type = await analyze_query_type(query)
            
            # Adjust parameters based on analysis
            sparql_weight, semantic_weight, top_k = self._calculate_parameters(
                query_complexity, query_specificity, query_type)
            
            # Cache the parameters
            self._update_cache(query, (sparql_weight, semantic_weight, top_k))
        
        logger.info(f"Adaptive retrieval parameters: SPARQL weight={sparql_weight}, " +
                    f"Semantic weight={semantic_weight}, top_k={top_k}")
        
        # Execute retrieval with the calculated parameters
        results, scores = await self._execute_retrieval(query, sparql_weight, semantic_weight, top_k)
        
        return results, scores
    
    async def _analyze_query_complexity(self, query: str) -> float:
        """
        Analyze the complexity of the query on a scale from 0 (simple) to 1 (complex).
        Complexity is based on query length, number of entities mentioned, etc.
        
        Args:
            query: The query string
            
        Returns:
            Complexity score between 0 and 1
        """
        # Simple complexity based on length and structure
        words = query.split()
        length_factor = min(len(words) / 20, 1.0)  # Cap at 1.0
        
        # Check for complex linguistic structures
        complex_indicators = ['compare', 'difference', 'versus', 'relationship', 'between']
        structure_factor = 0.5 * sum(1 for word in words if any(ind in word.lower() for ind in complex_indicators)) / len(words)
        
        # Count potential entity mentions
        potential_entities = await self._extract_potential_entities(query)
        entity_factor = min(len(potential_entities) / 3, 1.0)  # Cap at 1.0
        
        # Combined complexity score
        complexity = 0.4 * length_factor + 0.3 * structure_factor + 0.3 * entity_factor
        return min(complexity, 1.0)  # Ensure it's in range [0,1]
    
    async def _analyze_query_specificity(self, query: str) -> float:
        """
        Analyze how specific the query is to ontology entities on a scale from 0 to 1.
        Higher values indicate queries that likely need more SPARQL focus.
        
        Args:
            query: The query string
            
        Returns:
            Specificity score between 0 and 1
        """
        potential_entities = await self._extract_potential_entities(query)
        
        if not potential_entities:
            return 0.3  # Default low-medium specificity
        
        # Check how many potential entities match actual ontology entities
        matches = await self._match_entities_to_ontology(potential_entities)
        match_ratio = len(matches) / len(potential_entities) if potential_entities else 0
        
        # Check for specific ontology keywords
        specificity_keywords = ['unidade', 'plano', 'modalidade', 'benefício', 'piscina', 'tipo']
        keyword_factor = 0.0
        for keyword in specificity_keywords:
            if keyword in query.lower():
                keyword_factor += 0.2  # Increase by 0.2 for each keyword
        
        specificity = 0.6 * match_ratio + 0.4 * min(keyword_factor, 1.0)
        return min(specificity, 1.0)  # Ensure it's in range [0,1]
    
    def _calculate_parameters(self, complexity: float, specificity: float, query_type: str) -> Tuple[float, float, int]:
        """
        Calculate retrieval parameters based on query analysis.
        
        Args:
            complexity: Complexity score
            specificity: Specificity score
            query_type: Type of query
            
        Returns:
            Tuple of (sparql_weight, semantic_weight, top_k)
        """
        # Base values
        base_sparql_weight = 0.6
        base_semantic_weight = 0.5
        
        # Adjust weights based on specificity
        sparql_weight = base_sparql_weight + 0.3 * specificity
        semantic_weight = base_semantic_weight + 0.2 * (1 - specificity)
        
        # Adjust top_k based on complexity
        top_k = int(self.base_top_k * (1 + complexity))
        
        # Further adjustments based on query type
        if query_type == 'factual':
            sparql_weight += 0.1  # Boost SPARQL for factual queries
        elif query_type == 'descriptive':
            semantic_weight += 0.1  # Boost semantic for descriptive queries
        elif query_type == 'comparative':
            top_k += 5  # Get more results for comparative queries
        elif query_type == 'exploratory':
            top_k += 10  # Get even more results for exploratory queries
        
        # Ensure parameters are in valid ranges
        sparql_weight = min(max(sparql_weight, 0.3), 1.0)
        semantic_weight = min(max(semantic_weight, 0.3), 1.0)
        top_k = min(max(top_k, 5), 50)  # Minimum 5, maximum 50
        
        return sparql_weight, semantic_weight, top_k
    
    async def _execute_retrieval(self, query: str, sparql_weight: float, semantic_weight: float, top_k: int) -> Tuple[List[str], List[float]]:
        """
        Execute the hybrid retrieval with the given parameters.
        
        Args:
            query: The query string
            sparql_weight: Weight for SPARQL results
            semantic_weight: Weight for semantic search results
            top_k: Number of top results to retrieve
            
        Returns:
            Tuple of (results, confidence_scores)
        """
        # Extract relevant schema for SPARQL query generation
        schema = await extract_relevant_schema(query, self.ontology_manager.owl_file_path)
        
        # Generate and execute SPARQL query
        try:
            sparql_query = await generate_sparql_query(query, schema)
            sparql_results = await self.ontology_manager.execute_sparql(sparql_query)
            sparql_success = len(sparql_results) > 0 and "No matching results found" not in sparql_results[0]
        except Exception as e:
            logger.error(f"SPARQL query error: {str(e)}")
            sparql_results = []
            sparql_success = False
        
        # If SPARQL failed, increase emphasis on semantic search
        if not sparql_success:
            semantic_weight += 0.2
        
        # Perform semantic search with similarity scores
        try:
            semantic_results, semantic_scores = await retrieve_top_k_with_scores(query, top_k)
        except Exception as e:
            logger.error(f"Semantic search error: {str(e)}")
            semantic_results = []
            semantic_scores = []
        
        # Merge results with weights
        merged_results = []
        confidence_scores = []
        
        # Process SPARQL results
        for i, result in enumerate(sparql_results):
            # Skip error messages or empty results
            if "error" in result.lower() or "no matching results" in result.lower():
                continue
                
            # Calculate position-based weight (earlier results get higher weight)
            position_weight = 1.0 - (i / max(len(sparql_results), 1)) * 0.5
            final_weight = sparql_weight * position_weight
            
            # Add source prefix
            merged_results.append(f"SPARQL: {result}")
            confidence_scores.append(final_weight)
        
        # Process semantic results
        for i, (result, score) in enumerate(zip(semantic_results, semantic_scores)):
            # Calculate combined weight from semantic similarity and position
            position_weight = 1.0 - (i / max(len(semantic_results), 1)) * 0.5
            final_weight = semantic_weight * score * position_weight
            
            # Add source prefix
            merged_results.append(f"Semantic: {result}")
            confidence_scores.append(final_weight)
        
        # Sort by confidence score
        sorted_pairs = sorted(zip(confidence_scores, merged_results), key=lambda pair: pair[0], reverse=True)
        sorted_results = [x for _, x in sorted_pairs]
        sorted_scores = [s for s, _ in sorted_pairs]
        
        # Remove duplicate information
        unique_results = []
        unique_scores = []
        for i, result in enumerate(sorted_results):
            # Extract result text without source prefix
            result_text = result.split(':', 1)[1].strip() if ':' in result else result
            
            # Check for similar content in higher-ranked results
            is_duplicate = False
            for prev_result in unique_results:
                prev_text = prev_result.split(':', 1)[1].strip() if ':' in prev_result else prev_result
                
                # Check similarity
                if text_similarity(result_text, prev_text) > 0.8:  # High similarity threshold
                    is_duplicate = True
                    break
                    
            if not is_duplicate:
                unique_results.append(result)
                unique_scores.append(sorted_scores[i])
        
        return unique_results, unique_scores
    
    async def _extract_potential_entities(self, query: str) -> List[str]:
        """
        Extract potential entity mentions from the query.
        
        Args:
            query: The query string
            
        Returns:
            List of potential entity mentions
        """
        words = query.split()
        
        # Extract capitalized words and multi-word phrases
        potential_entities = []
        
        # Single capitalized words
        potential_entities.extend([word for word in words if word and word[0].isupper()])
        
        # Look for multi-word entities (simple approach)
        for i in range(len(words) - 1):
            if words[i] and words[i][0].isupper() and words[i+1] and words[i+1][0].isupper():
                potential_entities.append(f"{words[i]} {words[i+1]}")
        
        # Extract quoted phrases
        quoted_phrases = re.findall(r'"([^"]*)"', query)
        potential_entities.extend(quoted_phrases)
        
        return list(set(potential_entities))  # Remove duplicates
    
    async def _match_entities_to_ontology(self, potential_entities: List[str]) -> List[str]:
        """
        Match potential entities to actual entities in the ontology.
        
        Args:
            potential_entities: List of potential entity mentions
            
        Returns:
            List of matched entities
        """
        # Simple implementation: check if entities appear in ontology labels
        if self.ontology_manager.onto is None:
            return []
            
        matched_entities = []
        
        # Get all labels from the ontology
        all_labels = []
        for entity in list(self.ontology_manager.onto.classes()) + list(self.ontology_manager.get_all_individuals()):
            if hasattr(entity, 'label') and entity.label:
                all_labels.extend([str(label).lower() for label in entity.label])
            all_labels.append(entity.name.lower())
        
        # Check if potential entities match any labels
        for entity in potential_entities:
            entity_lower = entity.lower()
            for label in all_labels:
                if entity_lower in label or label in entity_lower:
                    matched_entities.append(entity)
                    break
        
        return matched_entities
    
    def _check_cache(self, query: str) -> Tuple[bool, Tuple[float, float, int]]:
        """
        Check if a similar query exists in the cache.
        
        Args:
            query: The query string
            
        Returns:
            Tuple of (cache_hit, parameters)
        """
        # Simple implementation - check for exact match or very similar queries
        if query in self.query_cache:
            return True, self.query_cache[query]
            
        # Check for similar queries (very basic implementation)
        for cached_query, params in self.query_cache.items():
            if text_similarity(query, cached_query) > 0.8:
                return True, params
                
        # Default parameters if not found
        return False, (0.6, 0.5, self.base_top_k)
    
    def _update_cache(self, query: str, parameters: Tuple[float, float, int]):
        """
        Update the query cache with the calculated parameters.
        
        Args:
            query: The query string
            parameters: Tuple of (sparql_weight, semantic_weight, top_k)
        """
        # Simple implementation - just store the query and parameters
        # Limit cache size to prevent memory issues
        if len(self.query_cache) >= 100:
            # Remove oldest entry (not efficient but simple)
            self.query_cache.pop(next(iter(self.query_cache)))
            
        self.query_cache[query] = parameters

async def hybrid_retrieve_weighted(query: str, ontology_manager: OntologyManager, top_k: int = 30) -> Tuple[List[str], List[float]]:
    """
    Enhanced hybrid retrieval with weighted ranking of results based on relevance.
    
    Args:
        query: The natural language query
        ontology_manager: Ontology manager instance
        top_k: Number of top results to retrieve
        
    Returns:
        Tuple of (merged_results, confidence_scores)
    """
    # Parameters for weighting
    SPARQL_BASE_WEIGHT = 0.7  # SPARQL results generally have higher precision
    SEMANTIC_BASE_WEIGHT = 0.5  # Semantic results may be more noisy
    
    # Extract relevant schema for SPARQL query generation
    schema = await extract_relevant_schema(query, ontology_manager.owl_file_path)
    
    # Generate and execute SPARQL query
    try:
        sparql_query = await generate_sparql_query(query, schema)
        sparql_results = await ontology_manager.execute_sparql(sparql_query)
        sparql_success = len(sparql_results) > 0 and "No matching results found" not in sparql_results[0]
    except Exception as e:
        logger.error(f"SPARQL query error: {str(e)}")
        sparql_results = []
        sparql_success = False
    
    # Adjust weights based on SPARQL success
    if not sparql_success:
        # If SPARQL failed, rely more on semantic search
        SPARQL_BASE_WEIGHT = 0.3
        SEMANTIC_BASE_WEIGHT = 0.8
    
    # Perform semantic search with similarity scores
    try:
        semantic_results, semantic_scores = await retrieve_top_k_with_scores(query, top_k)
    except Exception as e:
        logger.error(f"Semantic search error: {str(e)}")
        semantic_results = []
        semantic_scores = []
    
    # Analyze query to adjust weights further
    query_keywords = set(re.findall(r'\b\w+\b', query.lower()))
    
    # If query contains terms likely to be in the ontology, boost SPARQL weight
    ontology_terms = {'unidade', 'plano', 'tipo', 'modalidade', 'benefício', 'piscina', 'prime', 'slim'}
    if any(term in query_keywords for term in ontology_terms):
        SPARQL_BASE_WEIGHT += 0.1
    
    # If query is asking about specific facts, boost semantic weight
    fact_indicators = {'quantos', 'quando', 'onde', 'como', 'quem', 'por que', 'qual'}
    if any(term in query_keywords for term in fact_indicators):
        SEMANTIC_BASE_WEIGHT += 0.1
    
    # Merge results with weights
    merged_results = []
    confidence_scores = []
    
    # Process SPARQL results
    for i, result in enumerate(sparql_results):
        # Skip error messages or empty results
        if "error" in result.lower() or "no matching results" in result.lower():
            continue
            
        # Calculate position-based weight (earlier results get higher weight)
        position_weight = 1.0 - (i / max(len(sparql_results), 1)) * 0.5
        final_weight = SPARQL_BASE_WEIGHT * position_weight
        
        # Add source prefix
        merged_results.append(f"SPARQL: {result}")
        confidence_scores.append(final_weight)
    
    # Process semantic results
    for i, (result, score) in enumerate(zip(semantic_results, semantic_scores)):
        # Calculate combined weight from semantic similarity and position
        position_weight = 1.0 - (i / max(len(semantic_results), 1)) * 0.5
        final_weight = SEMANTIC_BASE_WEIGHT * score * position_weight
        
        # Add source prefix
        merged_results.append(f"Semantic: {result}")
        confidence_scores.append(final_weight)
    
    # Sort by confidence score
    sorted_results = [x for _, x in sorted(zip(confidence_scores, merged_results), key=lambda pair: pair[0], reverse=True)]
    sorted_scores = sorted(confidence_scores, reverse=True)
    
    # Remove duplicate information
    unique_results = []
    unique_scores = []
    for i, result in enumerate(sorted_results):
        # Check for similar content in higher-ranked results
        is_duplicate = False
        result_text = result.split(':', 1)[1].strip()  # Remove source prefix
        
        for prev_result in unique_results:
            prev_text = prev_result.split(':', 1)[1].strip()
            # Check similarity
            if text_similarity(result_text, prev_text) > 0.8:  # High similarity threshold
                is_duplicate = True
                break
                
        if not is_duplicate:
            unique_results.append(result)
            unique_scores.append(sorted_scores[i])
    
    # Return normalized confidence scores
    max_score = max(unique_scores) if unique_scores else 1.0
    normalized_scores = [score/max_score for score in unique_scores]
    
    return unique_results, normalized_scores