import asyncio
import os
import re
import tempfile
from datetime import datetime
from typing import List, Tuple

import aiofiles
import faiss
import instructor
import numpy as np
import openai
import owlready2
import rdflib
import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse
from openai import OpenAI
from pydantic import BaseModel, Field, field_validator
from rdflib.plugins.sparql.parser import parseQuery
from contextlib import asynccontextmanager
from rdflib.plugins.sparql import prepareQuery
from loguru import logger
from owlready2 import sync_reasoner_pellet, get_ontology

# Import configuration from config.py
from ragatanga import config

###############################################################################
# CONFIGURATION
###############################################################################
# File paths (using config.py)
BASE_DIR = config.BASE_DIR
DATA_DIR = config.DATA_DIR
os.makedirs(DATA_DIR, exist_ok=True)  # Create data directory if it doesn't exist

# Define file paths based on config
OWL_FILE_PATH = config.ONTOLOGY_PATH
KBASE_FILE = os.path.join(DATA_DIR, "knowledge_base.md")  # Keep .md extension for backward compatibility
KBASE_FAISS_INDEX_FILE = os.path.join(DATA_DIR, "knowledge_base_faiss.index")
KBASE_EMBEDDINGS_FILE = os.path.join(DATA_DIR, "knowledge_base_embeddings.npy")
# Use config path for index if needed
KBASE_INDEX_PATH = config.KBASE_INDEX_PATH

# Initialize global variables
kbase_entries = []
kbase_index = None
kbase_embeddings_np = None

# OpenAI client configuration
openai_client = OpenAI()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY is not set")
openai_client.api_key = api_key

# Create a patched client using instructor.from_openai
client = instructor.from_openai(openai_client)

# Embedding + Model configuration
EMBED_MODEL      = "text-embedding-3-large"   # OpenAI embedding model
GPT_MODEL        = "gpt-4o"                   # or "gpt-4" if you have access
BATCH_SIZE       = 16                         # Batch size for embedding calls
TOP_K            = 30                         # Retrieve top-K entries
DIMENSIONS       = 3072                       # Dimensionality for embedding model
# Use config values for semantic search and question answering
CHUNK_SIZE       = config.CHUNK_SIZE
CHUNK_OVERLAP    = config.CHUNK_OVERLAP
MAX_TOKENS       = config.MAX_TOKENS
TEMPERATURE      = config.TEMPERATURE

###############################################################################
# FASTAPI MODELS
###############################################################################
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    retrieved_facts_sparql: List[str]
    retrieved_facts_semantic: List[str]
    retrieved_facts: List[str]
    answer: str

    class Config:
        allow_mutation = True    

class AdaptiveRetriever:
    """
    Implements adaptive retrieval with dynamic parameters based on query complexity and type.
    """
    
    def __init__(self, ontology_manager, base_top_k=10):
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
        """
        # Generate and execute SPARQL query
        try:
            sparql_query = await generate_sparql_query(query)
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
        """
        words = query.split()
        
        # Extract capitalized words and multi-word phrases
        potential_entities = []
        
        # Single capitalized words
        potential_entities.extend([word for word in words if word[0].isupper()])
        
        # Look for multi-word entities (simple approach)
        for i in range(len(words) - 1):
            if words[i][0].isupper() and words[i+1][0].isupper():
                potential_entities.append(f"{words[i]} {words[i+1]}")
        
        # Extract quoted phrases
        import re
        quoted_phrases = re.findall(r'"([^"]*)"', query)
        potential_entities.extend(quoted_phrases)
        
        return list(set(potential_entities))  # Remove duplicates
    
    async def _match_entities_to_ontology(self, potential_entities: List[str]) -> List[str]:
        """
        Match potential entities to actual entities in the ontology.
        """
        # This is a placeholder implementation
        # In a real system, this would query the ontology for matching entities
        return potential_entities[:2]  # Mock implementation
    
    def _check_cache(self, query: str) -> Tuple[bool, Tuple[float, float, int]]:
        """
        Check if a similar query exists in the cache.
        
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
        """
        # Simple implementation - just store the query and parameters
        # Limit cache size to prevent memory issues
        if len(self.query_cache) >= 100:
            # Remove oldest entry (not efficient but simple)
            self.query_cache.pop(next(iter(self.query_cache)))
            
        self.query_cache[query] = parameters

class OntologyManager:
    """Manages ontology loading and reasoning with incremental updates."""
    
    def __init__(self, owl_file_path: str):
        self.owl_file_path = owl_file_path
        self.materialized_file = owl_file_path.replace(".ttl", "_materialized.ttl")
        self.onto = None
        self.last_modified_time = None
        self.graph = None
    
    async def load_and_materialize(self, force_rebuild=False):
        """Load ontology and materialize inferences, with cached handling."""
        current_modified_time = os.path.getmtime(self.owl_file_path)
        
        # Check if we need to rebuild
        rebuild_needed = (
            force_rebuild or 
            self.onto is None or 
            self.last_modified_time != current_modified_time or
            not os.path.exists(self.materialized_file)
        )
        
        if rebuild_needed:
            logger.info("Rebuilding materialized ontology...")
            self.onto = await self._materialize_inferences()
            self.last_modified_time = current_modified_time
            self.graph = rdflib.Graph()
            self.graph.parse(self.materialized_file, format='turtle')
        elif self.graph is None:
            logger.info("Loading cached materialized ontology...")
            self.graph = rdflib.Graph()
            self.graph.parse(self.materialized_file, format='turtle')
            # Load the onto object too for API methods that need it
            self.onto = await asyncio.to_thread(load_ontology, self.owl_file_path)
            
        return self.onto
    
    async def _materialize_inferences(self):
        """Materialize inferences with improved error handling."""
        try:
            # Load with rdflib first
            g = rdflib.Graph()
            g.parse(self.owl_file_path, format='turtle')
            
            # Validate basic ontology consistency
            validation_query = """
            ASK {
                ?s ?p ?o .
                FILTER (!isIRI(?s) || !isIRI(?p))
            }
            """
            has_invalid_triples = g.query(validation_query).askAnswer
            
            if has_invalid_triples:
                logger.warning("Ontology contains potentially invalid triples")
            
            # Save as RDF/XML temporarily
            temp = tempfile.NamedTemporaryFile(suffix='.owl', delete=False)
            g.serialize(destination=temp.name, format='xml')
            temp.close()
            
            # Now load with Owlready2
            onto_path = "file://" + temp.name
            onto = await asyncio.to_thread(get_ontology, onto_path)
            await asyncio.to_thread(onto.load)
            
            # Run reasoner with timeout protection
            try:
                # Define a synchronous function for the reasoner
                def run_reasoner_sync():
                    with onto:
                        sync_reasoner_pellet(infer_property_values=True)
                
                # Run with timeout
                await asyncio.wait_for(asyncio.to_thread(run_reasoner_sync), timeout=300)  # 5-minute timeout
            except asyncio.TimeoutError:
                logger.error("Reasoning timed out after 5 minutes, using partial results")
            except Exception as e:
                logger.error(f"Reasoning error: {str(e)}, using ontology without inferences")
            
            # Save materialized version
            materialized_file = self.materialized_file
            await asyncio.to_thread(g.serialize, destination=materialized_file, format='turtle')
            
            # Clean up temp file
            os.unlink(temp.name)
            
            return onto
                
        except Exception as e:
            logger.error(f"Failed to load ontology: {str(e)}")
            raise
    
    async def execute_sparql(self, sparql_query: str) -> List[str]:
        """Execute a SPARQL query with improved error handling."""
        if self.graph is None:
            await self.load_and_materialize()
            
        try:
            prepared_query = prepareQuery(sparql_query)
        except Exception as e:
            return [f"Invalid SPARQL syntax: {str(e)}"]

        try:
            # Use a separate thread for query execution
            def run_query():
                assert self.graph is not None
                results = self.graph.query(prepared_query)
                
                output_texts = []
                for row in results:
                    if isinstance(row, bool):
                        output_texts.append(str(row))
                    elif hasattr(row, '__iter__'):
                        row_values = []
                        for val in row:
                            if isinstance(val, rdflib.URIRef):
                                val_str = str(val).split('#')[-1].split('/')[-1]
                            else:
                                val_str = str(val)
                            if val_str.strip():
                                row_values.append(val_str)
                        if row_values:
                            output_texts.append(", ".join(row_values))
                    else:
                        output_texts.append(str(row))

                return output_texts if output_texts else ["No matching results found in the ontology"]

            return await asyncio.to_thread(run_query)
        except Exception as e:
            logger.error(f"SPARQL query execution error: {str(e)}")
            return [f"SPARQL query execution error: {str(e)}"]
    
    async def get_individual_properties(self, individual_uri: str) -> List[str]:
        """Get all properties of an individual with their values."""
        query = f"""
        SELECT ?prop ?value
        WHERE {{
            <{individual_uri}> ?prop ?value .
        }}
        """
        results = await self.execute_sparql(query)
        return results

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

###############################################################################
# ONTOLOGY LOADING
###############################################################################
def materialize_inferences(onto):
    """Load the ontology with rdflib first, then convert to a format Owlready2 can read."""
    try:
        # Load with rdflib first
        g = rdflib.Graph()
        g.parse(OWL_FILE_PATH, format='turtle')
        
        # Save as RDF/XML temporarily
        temp = tempfile.NamedTemporaryFile(suffix='.owl', delete=False)
        g.serialize(destination=temp.name, format='xml')
        temp.close()
        
        # Now load with Owlready2
        onto_path = "file://" + temp.name
        onto = get_ontology(onto_path).load()
        
        # Run reasoner
        with onto:
            sync_reasoner_pellet(infer_property_values=True)
        
        # Save materialized version
        materialized_file = OWL_FILE_PATH.replace(".ttl", "_materialized.ttl")
        g.serialize(destination=materialized_file, format='turtle')
        
        # Clean up temp file
        os.unlink(temp.name)
        
        return onto
        
    except Exception as e:
        logger.error(f"Failed to load ontology: {str(e)}")
        raise

async def load_ontology_schema() -> str:
    """
    Return the filtered ontology schema without individual declarations.
    Uses the materialized file if it exists and is nonempty.
    """
    materialized_file = OWL_FILE_PATH.replace(".ttl", "_materialized.ttl")
    
    # Use the materialized file if it exists and is non-empty
    if os.path.exists(materialized_file):
        async with aiofiles.open(materialized_file, "r", encoding="utf-8") as file:
            contents = await file.read()
        if not contents.strip():
            logger.warning("Materialized ontology is empty, falling back to original file.")
            async with aiofiles.open(OWL_FILE_PATH, "r", encoding="utf-8") as file:
                contents = await file.read()
    else:
        async with aiofiles.open(OWL_FILE_PATH, "r", encoding="utf-8") as file:
            contents = await file.read()
    
    logger.debug(f"Raw ontology schema length: {len(contents)}")
    
    # Define patterns to keep (schema-related)
    keep_patterns = [
        r'^@prefix',  # Prefix declarations
        r'^\s*:\w+\s+a\s+owl:Class\s*;',  # Class declarations
        r'^\s*:\w+\s+a\s+owl:(Object|Datatype)Property\s*;',  # Property declarations
        r'^\s*rdfs:domain\s+:',  # Property domains
        r'^\s*rdfs:range\s+:',  # Property ranges
        r'^\s*rdfs:subClassOf\s+:',  # Class hierarchy
    ]
    
    import re
    pattern = re.compile('|'.join(keep_patterns))
    
    # Keep only schema-related lines and their associated labels/comments
    schema_lines = []
    current_block = []
    in_relevant_block = False
    
    for line in contents.splitlines():
        if pattern.search(line):
            if current_block:  # Save previous block if it was relevant
                if in_relevant_block:
                    schema_lines.extend(current_block)
                current_block = []
            in_relevant_block = True
            current_block.append(line)
        elif line.strip().startswith('rdfs:label') or line.strip().startswith('rdfs:comment'):
            if in_relevant_block:
                current_block.append(line)
        elif not line.strip():  # Empty line
            if in_relevant_block and current_block:
                schema_lines.extend(current_block)
                schema_lines.append('')
            current_block = []
            in_relevant_block = False
        elif line.strip().endswith(';') or line.strip().endswith('.'):
            if in_relevant_block:
                current_block.append(line)
    
    # Add any remaining block
    if in_relevant_block and current_block:
        schema_lines.extend(current_block)
    
    filtered_schema = '\n'.join(schema_lines)
    
    logger.debug(f"Filtered ontology schema length: {len(filtered_schema)}")
    
    if not schema_lines:
        raise ValueError("Filtered schema is empty - check keep patterns.")
    
    return filtered_schema

def load_ontology(owl_path: str):
    """Loads the ontology with rdflib first, then converts to a format Owlready2 can read."""
    try:
        # Load with rdflib first
        g = rdflib.Graph()
        g.parse(owl_path, format='turtle')
        
        # Save as RDF/XML temporarily
        temp = tempfile.NamedTemporaryFile(suffix='.owl', delete=False)
        g.serialize(destination=temp.name, format='xml')
        temp.close()
        
        # Now load with Owlready2
        onto_path = "file://" + temp.name
        onto = owlready2.get_ontology(onto_path).load()
        
        # Clean up
        os.unlink(temp.name)
        
        return onto
        
    except Exception as e:
        logger.error(f"Failed to load ontology: {str(e)}")
        raise

###############################################################################
# BUILD TEXT ENTRIES FROM ONTOLOGY
###############################################################################
def get_all_individuals(onto):
    """Collect all individuals, including those available as instances of a class."""
    inds = set(onto.individuals())
    for cls in onto.classes():
        for instance in cls.instances():
            inds.add(instance)
    return list(inds)

def build_enhanced_ontology_entries(onto, include_inferred=True):
    """
    Build enhanced text representations of ontology elements with more context and relationships.
    
    Args:
        onto: The loaded ontology
        include_inferred: Whether to include inferred statements
        
    Returns:
        List of dictionaries with id and text fields
    """
    entries = []
    idx = 0
    
    # Create a mapping of entities to their labels for better context
    label_map = {}
    for entity in list(onto.classes()) + list(onto.properties()) + list(get_all_individuals(onto)):
        if hasattr(entity, 'label') and entity.label:
            label_map[entity] = ', '.join(entity.label)
        else:
            label_map[entity] = entity.name
    
    # 1) Classes with enhanced context
    print("\nBuilding enhanced class representations...")
    for cls in onto.classes():
        # Skip owl:Thing and other built-in classes
        if cls.name in ['Thing', 'Nothing'] or cls.name.startswith('owl_'):
            continue
            
        label = ', '.join(cls.label) if cls.label else cls.name
        doc = f"[CLASS]\nName: {cls.name}\nLabel: {label}\n"
        
        # Add description
        if cls.comment:
            doc += f"Description: {', '.join(cls.comment)}\n"
        
        # Add parent classes with their labels
        parents = [p for p in cls.is_a if p is not owlready2.Thing and hasattr(p, "name")]
        if parents:
            parent_texts = [f"{p.name} ({label_map.get(p, p.name)})" for p in parents]
            doc += f"Parent Classes: {', '.join(parent_texts)}\n"
        
        # Add subclasses with their labels
        subclasses = cls.subclasses()
        if subclasses:
            subclass_texts = [f"{s.name} ({label_map.get(s, s.name)})" for s in subclasses]
            doc += f"Subclasses: {', '.join(subclass_texts)}\n"
        
        # Add properties that have this class in their domain
        domain_props = [p for p in onto.properties() if cls in p.domain]
        if domain_props:
            prop_texts = [f"{p.name} ({label_map.get(p, p.name)})" for p in domain_props]
            doc += f"Properties with this domain: {', '.join(prop_texts)}\n"
        
        # Add instances count
        instances = list(cls.instances())
        doc += f"Number of instances: {len(instances)}\n"
        
        # Add a few example instances if available
        if instances:
            sample_size = min(5, len(instances))
            sample_instances = instances[:sample_size]
            instance_texts = [f"{i.name} ({label_map.get(i, i.name)})" for i in sample_instances]
            doc += f"Example instances: {', '.join(instance_texts)}"
            if len(instances) > sample_size:
                doc += f" (and {len(instances) - sample_size} more)"
            doc += "\n"
        
        print(f"✓ Enhanced Class: {cls.name}")
        entries.append({"id": idx, "text": doc.strip()})
        idx += 1
    
    # 2) Individuals with enhanced context
    print("\nBuilding enhanced individual representations...")
    for indiv in get_all_individuals(onto):
        doc = f"[INDIVIDUAL]\nName: {indiv.name}\n"
        
        # Add label and description
        if hasattr(indiv, 'label') and indiv.label:
            doc += f"Label: {', '.join(indiv.label)}\n"
        
        if hasattr(indiv, 'comment') and indiv.comment:
            doc += f"Description: {', '.join(indiv.comment)}\n"
        
        # Add types with their labels
        types = [t for t in indiv.is_a if hasattr(t, "name")]
        if types:
            type_texts = [f"{t.name} ({label_map.get(t, t.name)})" for t in types]
            doc += f"Types: {', '.join(type_texts)}\n"
        
        # Add properties and their values with better formatting and context
        props_dict = {}
        for prop in onto.properties():
            try:
                if prop in indiv.get_properties():
                    values = prop[indiv]
                    if values:
                        # Format and contextualize the values
                        if isinstance(values, list):
                            formatted_values = []
                            for v in values:
                                if hasattr(v, 'name') and v in label_map:
                                    formatted_values.append(f"{v.name} ({label_map.get(v, v.name)})")
                                else:
                                    formatted_values.append(str(v))
                            props_dict[prop.name] = ', '.join(formatted_values)
                        else:
                            if hasattr(values, 'name') and values in label_map:
                                props_dict[prop.name] = f"{values.name} ({label_map.get(values, values.name)})"
                            else:
                                props_dict[prop.name] = str(values)
            except Exception:
                continue
        
        # Add properties section if there are any
        if props_dict:
            doc += "Properties:\n"
            for prop_name, prop_value in props_dict.items():
                # Get the property object to add its label
                prop_obj = next((p for p in onto.properties() if p.name == prop_name), None)
                prop_label = ', '.join(prop_obj.label) if prop_obj and prop_obj.label else prop_name
                doc += f"  - {prop_name} ({prop_label}): {prop_value}\n"
        
        # Add inverse relationships
        inverse_relations = []
        for prop in onto.properties():
            if not hasattr(prop, 'range') or not prop.range:
                continue
                
            # Check if this individual could be in the range of this property
            if any(isinstance(indiv, r) for r in prop.range if hasattr(r, 'instances')):
                # Find subjects that have this individual as the value for this property
                for subj in get_all_individuals(onto):
                    try:
                        if prop in subj.get_properties():
                            values = prop[subj]
                            if isinstance(values, list) and indiv in values:
                                inverse_relations.append((prop, subj))
                            elif values == indiv:
                                inverse_relations.append((prop, subj))
                    except Exception:
                        continue
        
        # Add inverse relations section if there are any
        if inverse_relations:
            doc += "Referenced by:\n"
            for prop, subj in inverse_relations:
                prop_label = ', '.join(prop.label) if prop.label else prop.name
                subj_label = label_map.get(subj, subj.name)
                doc += f"  - {subj.name} ({subj_label}) via property {prop.name} ({prop_label})\n"
        
        print(f"✓ Enhanced Individual: {indiv.name}")
        entries.append({"id": idx, "text": doc.strip()})
        idx += 1
    
    # 3) Properties with enhanced context
    print("\nBuilding enhanced property representations...")
    for prop in onto.properties():
        doc = f"[PROPERTY]\nName: {prop.name}\n"
        
        # Add label and description
        if prop.label:
            doc += f"Label: {', '.join(prop.label)}\n"
        
        if prop.comment:
            doc += f"Description: {', '.join(prop.comment)}\n"
        
        # Property type
        prop_type = prop.__class__.__name__
        doc += f"Type: {prop_type}\n"
        
        # Domain and range with labels
        if prop.domain:
            domain_texts = []
            for d in prop.domain:
                if hasattr(d, "name"):
                    domain_texts.append(f"{d.name} ({label_map.get(d, d.name)})")
            if domain_texts:
                doc += f"Domain: {', '.join(domain_texts)}\n"
        
        if prop.range:
            range_texts = []
            for r in prop.range:
                if hasattr(r, "name"):
                    range_texts.append(f"{r.name} ({label_map.get(r, r.name)})")
            if range_texts:
                doc += f"Range: {', '.join(range_texts)}\n"
        
        # Inverse properties
        if hasattr(prop, 'inverse') and prop.inverse:
            inverse_props = prop.inverse
            inverse_texts = [f"{p.name} ({label_map.get(p, p.name)})" for p in inverse_props if hasattr(p, "name")]
            if inverse_texts:
                doc += f"Inverse properties: {', '.join(inverse_texts)}\n"
        
        # Usage examples
        usage_examples = []
        try:
            # Find a few examples of this property's usage
            for subj in get_all_individuals(onto):
                try:
                    if prop in subj.get_properties():
                        values = prop[subj]
                        if values:
                            subj_label = label_map.get(subj, subj.name)
                            if isinstance(values, list):
                                for val in values[:2]:  # Limit to 2 values per subject
                                    if hasattr(val, 'name'):
                                        val_label = label_map.get(val, val.name)
                                        usage_examples.append(f"{subj.name} ({subj_label}) → {val.name} ({val_label})")
                                    else:
                                        usage_examples.append(f"{subj.name} ({subj_label}) → {val}")
                            else:
                                if hasattr(values, 'name'):
                                    val_label = label_map.get(values, values.name)
                                    usage_examples.append(f"{subj.name} ({subj_label}) → {values.name} ({val_label})")
                                else:
                                    usage_examples.append(f"{subj.name} ({subj_label}) → {values}")
                        
                        # Limit to 5 examples maximum
                        if len(usage_examples) >= 5:
                            break
                except Exception:
                    continue
        except Exception:
            pass
        
        if usage_examples:
            doc += "Usage examples:\n"
            for example in usage_examples:
                doc += f"  - {example}\n"
        
        print(f"✓ Enhanced Property: {prop.name}")
        entries.append({"id": idx, "text": doc.strip()})
        idx += 1
    
    print(f"✓ Total enhanced entries: {len(entries)}")
    return entries

###############################################################################
# EMBEDDING UTILS (Async versions)
###############################################################################
async def embed_texts_in_batches(texts: List[str], batch_size: int = 16) -> np.ndarray:
    """
    Embed a list of strings in batches. Returns a numpy array of shape (N, D).
    Uses the synchronous OpenAI embeddings call wrapped in asyncio.to_thread.
    """
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = await asyncio.to_thread(openai.embeddings.create, input=batch, model=EMBED_MODEL)
        for j in range(len(batch)):
            emb = response.data[j].embedding
            all_embeddings.append(emb)
    return np.array(all_embeddings, dtype=np.float32)

def build_faiss_index(embeddings: np.ndarray) -> Tuple[faiss.IndexFlatIP, np.ndarray]:
    """
    Build a FAISS index for inner product (cosine similarity if vectors are normalized).
    """
    # Normalize embeddings for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings_norm = embeddings / (norms + 1e-10)

    index = faiss.IndexFlatIP(DIMENSIONS)
    index.add(x=embeddings_norm)  # type: ignore
    return index, embeddings_norm

def save_faiss_index(index, index_file: str, embeddings: np.ndarray, embed_file: str):
    faiss.write_index(index, index_file)
    np.save(embed_file, embeddings)
    print(f"FAISS index saved to {index_file}, embeddings to {embed_file}")

def load_faiss_index(index_file: str, embed_file: str):
    index = faiss.read_index(index_file)
    embeddings = np.load(embed_file)
    return index, embeddings

###############################################################################
# INITIALIZATION
###############################################################################
print("=== Initializing Ontology Retrieval Tool ===")

def debug_print_individuals(onto):
    """Print all individuals in the ontology to verify parsing."""
    print("\nVerifying ontology individuals...")
    for individual in onto.individuals():
        print(f"✓ {individual.name}")

def debug_print_classes(onto):
    """Print all classes and their instances in the ontology."""
    print("\nVerifying ontology classes...")
    for cls in onto.classes():
        print(f"✓ {cls.name}")

# Load the ontology (for SPARQL queries, debugging, etc.)
try:
    # First try to materialize the ontology
    onto = materialize_inferences(OWL_FILE_PATH)
    with onto:
        sync_reasoner_pellet(infer_property_values=True)
    debug_print_individuals(onto)
    debug_print_classes(onto)
except Exception as e:
    print(f"Warning: Could not load ontology file {OWL_FILE_PATH}. Error: {e}")
    onto = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Materialize inferences first
    materialize_inferences(onto)
    
    global kbase_entries, kbase_index, kbase_embeddings_np
    
    if not os.path.exists(KBASE_FILE):
        print(f"Warning: Knowledge base file {KBASE_FILE} not found. Starting with empty knowledge base.")
        kbase_entries = []
        kbase_index = None
        kbase_embeddings_np = None
        yield
    else:
        # Load knowledge base content
        with open(KBASE_FILE, "r", encoding="utf-8") as file:
            kbase_content = file.read()
        kbase_entries = [chunk.strip() for chunk in kbase_content.split("\n\n") if chunk.strip()]

        # Build or load FAISS index
        if os.path.exists(KBASE_FAISS_INDEX_FILE) and os.path.exists(KBASE_EMBEDDINGS_FILE):
            print("Loading existing FAISS index and embeddings for knowledge base...")
            kbase_index, kbase_embeddings_np = load_faiss_index(KBASE_FAISS_INDEX_FILE, KBASE_EMBEDDINGS_FILE)
        else:
            print("Embedding knowledge base entries in batches...")
            kbase_embeddings_np = await embed_texts_in_batches(kbase_entries, BATCH_SIZE)
            kbase_index, kbase_embeddings_np = build_faiss_index(np.asarray(kbase_embeddings_np))
            save_faiss_index(kbase_index, KBASE_FAISS_INDEX_FILE, kbase_embeddings_np, KBASE_EMBEDDINGS_FILE)

        print(f"=== Ontology Retrieval System ready with SPARQL on {OWL_FILE_PATH} and semantic search on {KBASE_FILE} ===")
        yield

    # Optional shutdown code if needed

app = FastAPI(
    title="Ontology Retrieval Tool",
    description="Hybrid Retrieval API combining SPARQL queries and semantic search",
    version="1.0.0",
    lifespan=lifespan
)

###############################################################################
# QUERY FUNCTIONS (Async versions)
###############################################################################
async def get_query_embedding(query: str) -> np.ndarray:
    """Embed and normalize a single query string."""
    response = await asyncio.to_thread(openai.embeddings.create, input=[query], model=EMBED_MODEL)
    emb = np.array(response.data[0].embedding, dtype=np.float32)
    norm = np.linalg.norm(emb)
    return emb / (norm + 1e-10)

async def retrieve_top_k(query: str, k: int) -> List[str]:
    """
    Use FAISS to find the top-k most similar knowledge base entries.
    """
    q_emb = (await get_query_embedding(query)).reshape(1, -1)
    def search_index():
        return kbase_index.search(q_emb, k)  # type: ignore
    distances, indices = await asyncio.to_thread(search_index)
    return [kbase_entries[i] for i in indices[0].tolist()]

async def generate_sparql_query(query: str) -> str:
    """Generate a SPARQL query from a natural language query with improved prompting."""
    # Extract relevant schema parts
    filtered_schema = await extract_relevant_schema(query, OWL_FILE_PATH)
    
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
        response = await asyncio.to_thread(
            client.create,
            max_retries=3,
            response_model=Query,
            model=GPT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt.format(schema=filtered_schema)},
                {"role": "user", "content": user_message}
            ],
            temperature=TEMPERATURE,
        )
        generated_query = response.valid_sparql_query
        
        # Validate the generated query
        try:
            prepareQuery(generated_query)
        except Exception as e:
            logger.warning(f"Generated query validation failed: {e}. Falling back.")
            return await generate_fallback_query(query, filtered_schema)
            
    except Exception as e:
        logger.warning(f"Failed to generate query: {e}. Using fallback strategy.")
        generated_query = await generate_fallback_query(query, filtered_schema)
    
    logger.debug(f"Generated SPARQL Query:\n{generated_query}")
    return generated_query

async def generate_fallback_query(query: str, schema: str) -> str:
    """Generate a fallback query if the main generation fails."""
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

async def execute_sparql_query(sparql_query: str) -> List[str]:
    """Execute a SPARQL query against the materialized ontology."""
    try:
        prepared_query = prepareQuery(sparql_query)
    except Exception as e:
        return [f"Invalid SPARQL syntax: {str(e)}"]

    def run_query():
        g = rdflib.Graph()
        try:
            materialized_file = OWL_FILE_PATH.replace(".ttl", "_materialized.ttl")
            g.parse(materialized_file, format='turtle')
            logger.debug(f"Using materialized ontology for query: {sparql_query}")
            
            results = g.query(prepared_query)
            logger.debug(f"Query results: {list(results)}")

            output_texts = []
            for row in results:
                if isinstance(row, bool):
                    output_texts.append(str(row))
                elif hasattr(row, '__iter__'):
                    row_values = []
                    for val in row:
                        if isinstance(val, rdflib.URIRef):
                            val_str = str(val).split('#')[-1].split('/')[-1]
                        else:
                            val_str = str(val)
                        if val_str.strip():
                            row_values.append(val_str)
                    if row_values:
                        output_texts.append(", ".join(row_values))
                else:
                    output_texts.append(str(row))

            return output_texts if output_texts else ["No matching results found in the ontology"]

        except Exception as e:
            logger.error(f"Error executing SPARQL query: {str(e)}")
            return [f"Error executing SPARQL query: {str(e)}"]

    try:
        return await asyncio.to_thread(run_query)
    except Exception as e:
        logger.error(f"SPARQL query execution error: {str(e)}")
        return [f"SPARQL query execution error: {str(e)}"]

async def hybrid_retrieve_weighted(query: str, top_k: int = TOP_K) -> Tuple[List[str], List[float]]:
    """
    Enhanced hybrid retrieval with weighted ranking of results based on relevance.
    
    Args:
        query: The natural language query
        top_k: Number of top results to retrieve
        
    Returns:
        Tuple of (merged_results, confidence_scores)
    """
    # Parameters for weighting
    SPARQL_BASE_WEIGHT = 0.7  # SPARQL results generally have higher precision
    SEMANTIC_BASE_WEIGHT = 0.5  # Semantic results may be more noisy
    
    # Generate and execute SPARQL query
    try:
        sparql_query = await generate_sparql_query(query)
        sparql_results = await execute_sparql_query(sparql_query)
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

async def retrieve_top_k_with_scores(query: str, k: int) -> Tuple[List[str], List[float]]:
    """
    Retrieve top-k most similar knowledge base entries with their similarity scores.
    
    Args:
        query: The query string
        k: Number of results to retrieve
        
    Returns:
        Tuple of (results, similarity_scores)
    """
    global kbase_index, kbase_entries
    
    if kbase_index is None:
        logger.warning("Knowledge base index not initialized, returning empty results")
        return [], []
        
    q_emb = (await get_query_embedding(query)).reshape(1, -1)
    
    def search_index():
        return kbase_index.search(q_emb, k) # type: ignore
    
    distances, indices = await asyncio.to_thread(search_index)
    
    # Convert distances to similarity scores (inner products are already similarities in range [0,1])
    similarity_scores = distances[0].tolist()
    results = [kbase_entries[i] for i in indices[0].tolist()]
    
    return results, similarity_scores

async def generate_structured_answer(
    query: str, 
    retrieved_texts: List[str], 
    confidence_scores: List[float]
) -> QueryResponse:
    """
    Generate a structured, comprehensive answer using retrieved facts with intelligent organization.
    
    Args:
        query: The user's query
        retrieved_texts: List of retrieved text snippets
        confidence_scores: Confidence scores for each snippet
        
    Returns:
        A QueryResponse object with the structured answer
    """
    # Split retrieved texts by source
    sparql_facts = []
    semantic_facts = []
    
    for fact, score in zip(retrieved_texts, confidence_scores):
        if fact.startswith("SPARQL:"):
            sparql_facts.append((fact.replace("SPARQL: ", ""), score))
        elif fact.startswith("Semantic:"):
            semantic_facts.append((fact.replace("Semantic: ", ""), score))
    
    # Analyze the user query to determine what information is most relevant
    query_type = await analyze_query_type(query)
    
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
    context_block = "HIGH CONFIDENCE FACTS:\n" + "\n".join(high_confidence_facts)
    if medium_confidence_facts:
        context_block += "\n\nMEDIUM CONFIDENCE FACTS:\n" + "\n".join(medium_confidence_facts)
    if low_confidence_facts:
        context_block += "\n\nLOW CONFIDENCE FACTS:\n" + "\n".join(low_confidence_facts)
    
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
        llm_response = await asyncio.to_thread(
            client.create,
            response_model=QueryResponse,
            max_retries=3,
            model=GPT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS
        )
        
        response = llm_response
    except Exception as e:
        logger.error(f"Error generating answer: {str(e)}")
        # Provide a fallback answer using simple template
        response.answer = generate_fallback_answer(query, sparql_fact_texts, semantic_fact_texts)
    
    return response

async def analyze_query_type(query: str) -> str:
    """
    Analyze the query to determine its type for better answer generation.
    
    Returns one of: 'factual', 'descriptive', 'comparative', 'procedural', 'exploratory'
    """
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
        response = await asyncio.to_thread(
            client.create,
            max_retries=3,
            response_model=QueryResponse,
            model=GPT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS
        )
        
        query_type = response.answer.strip().lower()
        
        # Validate and default to 'factual' if not recognized
        valid_types = {'factual', 'descriptive', 'comparative', 'procedural', 'exploratory'}
        if query_type not in valid_types:
            return 'factual'
            
        return query_type
    except Exception:
        # Default to factual if analysis fails
        return 'factual'

def generate_system_prompt_by_query_type(query_type: str) -> str:
    """Generate a system prompt tailored to the query type."""
    
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
    """Generate a simple fallback answer when LLM generation fails."""
    
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

###############################################################################
# FASTAPI ENDPOINTS
###############################################################################
@app.post("/query", response_model=QueryResponse)
async def handle_query(req: QueryRequest):
    """
    Enhanced query endpoint with improved retrieval and response generation.
    """
    user_query = req.query
    logger.info(f"Processing query: {user_query}")
    
    # Initialize ontology manager if not done already
    if not hasattr(app.state, "ontology_manager"):
        app.state.ontology_manager = OntologyManager(OWL_FILE_PATH)
        await app.state.ontology_manager.load_and_materialize()
    
    # Initialize adaptive retriever if not done already
    if not hasattr(app.state, "retriever"):
        app.state.retriever = AdaptiveRetriever(app.state.ontology_manager)
    
    try:
        # Use adaptive retrieval
        retrieved_texts, confidence_scores = await app.state.retriever.retrieve(user_query)
        logger.debug(f"Retrieved {len(retrieved_texts)} results with adaptive parameters")
        
        # Generate structured answer
        answer = await generate_structured_answer(user_query, retrieved_texts, confidence_scores)
        
        # Log success
        logger.info(f"Successfully generated answer for query: {user_query}")
        
        return answer
    except Exception as e:
        # Log the error
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        
        # Return a graceful error response
        return QueryResponse(
            retrieved_facts=[],
            retrieved_facts_sparql=[],
            retrieved_facts_semantic=[],
            answer=f"I encountered an error while processing your query. Please try again or rephrase your question. Error details: {str(e)}"
        )

@app.on_event("startup")
async def startup_event():
    """Initialize components on startup."""
    logger.info("Initializing ontology manager...")
    app.state.ontology_manager = OntologyManager(OWL_FILE_PATH)
    await app.state.ontology_manager.load_and_materialize()
    
    logger.info("Initializing adaptive retriever...")
    app.state.retriever = AdaptiveRetriever(app.state.ontology_manager)
    
    # Initialize knowledge base embeddings
    if os.path.exists(KBASE_FILE):
        logger.info("Loading knowledge base...")
        with open(KBASE_FILE, "r", encoding="utf-8") as file:
            kbase_content = file.read()
        
        global kbase_entries
        kbase_entries = [chunk.strip() for chunk in kbase_content.split("\n\n") if chunk.strip()]
        
        # Build or load FAISS index
        global kbase_index, kbase_embeddings_np
        if os.path.exists(KBASE_FAISS_INDEX_FILE) and os.path.exists(KBASE_EMBEDDINGS_FILE):
            logger.info("Loading existing FAISS index...")
            kbase_index, kbase_embeddings_np = load_faiss_index(KBASE_FAISS_INDEX_FILE, KBASE_EMBEDDINGS_FILE)
        else:
            logger.info("Building new FAISS index...")
            kbase_embeddings_np = await embed_texts_in_batches(kbase_entries, BATCH_SIZE)
            kbase_index, kbase_embeddings_np = build_faiss_index(np.asarray(kbase_embeddings_np))
            save_faiss_index(kbase_index, KBASE_FAISS_INDEX_FILE, kbase_embeddings_np, KBASE_EMBEDDINGS_FILE)
    
    logger.info("System initialization complete!")

@app.post("/upload/ontology")
async def upload_ontology(file: UploadFile):
    """Upload a new ontology file (.ttl or .owl)"""
    if not file.filename or not file.filename.endswith(('.ttl', '.owl')):
        raise HTTPException(status_code=400, detail="File must be .ttl or .owl")
    
    try:
        contents = await file.read()
        try:
            decoded_contents = contents.decode('utf-8')
        except UnicodeDecodeError:
            decoded_contents = contents.decode('latin-1')
        
        async with aiofiles.open(OWL_FILE_PATH, "w", encoding='utf-8') as out_file:
            await out_file.write(decoded_contents)
            
        global onto
        onto = await asyncio.to_thread(load_ontology, OWL_FILE_PATH)
        return {"message": "Ontology uploaded and loaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading ontology: {str(e)}")

@app.get("/download/ontology")
async def download_ontology():
    """Download the current ontology file"""
    if not os.path.exists(OWL_FILE_PATH):
        raise HTTPException(status_code=404, detail="Ontology file not found")
    return FileResponse(OWL_FILE_PATH)

@app.post("/upload/kb")
async def upload_knowledge_base(file: UploadFile):
    """Upload a new knowledge base markdown file"""
    if not file.filename or not file.filename.endswith('.md'):
        raise HTTPException(status_code=400, detail="File must be .md")
    
    try:
        contents = await file.read()
        try:
            decoded_contents = contents.decode('utf-8')
        except UnicodeDecodeError:
            decoded_contents = contents.decode('latin-1')
        
        async with aiofiles.open(KBASE_FILE, "w", encoding='utf-8') as out_file:
            await out_file.write(decoded_contents)
            
        global kbase_entries, kbase_index, kbase_embeddings_np
        kbase_entries = [chunk.strip() for chunk in decoded_contents.split("\n\n") if chunk.strip()]
        kbase_embeddings_np = await embed_texts_in_batches(kbase_entries, BATCH_SIZE)
        
        def build_index():
            return build_faiss_index(np.asarray(kbase_embeddings_np))
        kbase_index, kbase_embeddings_np = await asyncio.to_thread(build_index)
        await asyncio.to_thread(save_faiss_index, kbase_index, KBASE_FAISS_INDEX_FILE, kbase_embeddings_np, KBASE_EMBEDDINGS_FILE)
        
        return {"message": "Knowledge base uploaded and indexed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading knowledge base: {str(e)}")

@app.get("/download/kb")
async def download_knowledge_base():
    """Download the current knowledge base markdown file"""
    if not os.path.exists(KBASE_FILE):
        raise HTTPException(status_code=404, detail="Knowledge base file not found")
    return FileResponse(KBASE_FILE)

def get_ontology_statistics(onto) -> dict:
    """Gather comprehensive statistics about the ontology."""
    classes = list(onto.classes())
    individuals = list(get_all_individuals(onto))
    properties = list(onto.properties())
    
    class_instances = {}
    for cls in classes:
        instances = list(cls.instances())
        if instances:
            class_instances[cls.name] = len(instances)
    
    property_stats = {}
    for prop in properties:
        if hasattr(prop, 'name'):
            domain = [d.name for d in prop.domain if hasattr(d, "name")] if prop.domain else []
            range_vals = [r.name for r in prop.range if hasattr(r, "name")] if prop.range else []
            property_stats[prop.name] = {
                "type": prop.__class__.__name__,
                "domain": domain,
                "range": range_vals,
                "label": list(prop.label) if prop.label else [],
                "comment": list(prop.comment) if prop.comment else []
            }
    
    individual_properties = {}
    for ind in individuals:
        if hasattr(ind, 'name'):
            props = {}
            for prop in onto.properties():
                try:
                    if prop in ind.get_properties():
                        values = prop[ind]
                        if values:
                            props[prop.name] = [str(v) for v in values] if isinstance(values, list) else [str(values)]
                except Exception:
                    continue
            if props:
                individual_properties[ind.name] = props

    return {
        "statistics": {
            "total_classes": len(classes),
            "total_individuals": len(individuals),
            "total_properties": len(properties),
            "classes_with_instances": len(class_instances)
        },
        "classes": {
            cls.name: {
                "label": list(cls.label) if cls.label else [],
                "comment": list(cls.comment) if cls.comment else [],
                "instance_count": class_instances.get(cls.name, 0),
                "parents": [p.name for p in cls.is_a if p is not owlready2.Thing and hasattr(p, "name")]
            } for cls in classes if hasattr(cls, 'name')
        },
        "properties": property_stats,
        "individuals": {
            ind.name: {
                "types": [t.name for t in ind.is_a if hasattr(t, "name")],
                "label": list(ind.label) if hasattr(ind, 'label') and ind.label else [],
                "comment": list(ind.comment) if hasattr(ind, 'comment') and ind.comment else [],
                "properties": individual_properties.get(ind.name, {})
            } for ind in individuals if hasattr(ind, 'name')
        }
    }

@app.get("/describe_ontology")
async def describe_ontology():
    """
    Get a comprehensive description of the loaded ontology.
    """
    try:
        description = await asyncio.to_thread(get_ontology_statistics, onto)
        description["metadata"] = {
            "file_path": OWL_FILE_PATH,
            "file_size": os.path.getsize(OWL_FILE_PATH),
            "last_modified": datetime.fromtimestamp(os.path.getmtime(OWL_FILE_PATH)).isoformat(),
            "format": "Turtle" if OWL_FILE_PATH.endswith('.ttl') else "OWL"
        }
        return description
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error describing ontology: {str(e)}"
        )

async def extract_relevant_schema(query: str, owl_path: str) -> str:
    """
    Extract schema elements relevant to the query using semantic similarity.
    """
    logger.debug(f"Extracting schema elements relevant to: {query}")
    
    # Load the full schema
    with open(owl_path, "r", encoding="utf-8") as file:
        full_schema = file.read()
    
    # Parse the ontology with rdflib for proper traversal
    g = rdflib.Graph()
    g.parse(data=full_schema, format='turtle')
    
    # Extract classes
    classes = []
    for s, p, o in g.triples((None, rdflib.RDF.type, rdflib.OWL.Class)):
        class_triples = list(g.triples((s, None, None)))
        class_str = "\n".join([f"{s.n3(g.namespace_manager)} {p.n3(g.namespace_manager)} {o.n3(g.namespace_manager)}." 
                              for s, p, o in class_triples])
        # Get label if available
        label = None
        for _, _, label_val in g.triples((s, rdflib.RDFS.label, None)):
            label = str(label_val)
            break
        
        class_name = str(s).split('#')[-1] if '#' in str(s) else str(s).split('/')[-1]
        class_text = f"{class_name}" + (f" ({label})" if label else "")
        classes.append((class_text, class_str))
    
    # Extract properties
    properties = []
    for prop_type in [rdflib.OWL.ObjectProperty, rdflib.OWL.DatatypeProperty]:
        for s, p, o in g.triples((None, rdflib.RDF.type, prop_type)):
            prop_triples = list(g.triples((s, None, None)))
            prop_str = "\n".join([f"{s.n3(g.namespace_manager)} {p.n3(g.namespace_manager)} {o.n3(g.namespace_manager)}." 
                                for s, p, o in prop_triples])
            
            # Get label if available
            label = None
            for _, _, label_val in g.triples((s, rdflib.RDFS.label, None)):
                label = str(label_val)
                break
            
            prop_name = str(s).split('#')[-1] if '#' in str(s) else str(s).split('/')[-1]
            prop_text = f"{prop_name}" + (f" ({label})" if label else "")
            properties.append((prop_text, prop_str))
    
    # Get embeddings for query and schema elements
    query_embedding = await get_query_embedding(query)
    
    # Get embeddings for class and property names
    class_names = [name for name, _ in classes]
    property_names = [name for name, _ in properties]
    
    all_names = class_names + property_names
    
    if not all_names:
        logger.warning("No classes or properties found in the ontology")
        return full_schema
    
    logger.debug(f"Embedding {len(all_names)} ontology elements")
    all_embeddings = await embed_texts_in_batches(all_names, BATCH_SIZE)
    
    # Calculate similarities
    similarities = np.dot(all_embeddings, query_embedding)
    
    # Select top elements based on similarity
    num_to_select = min(20, len(all_names))  # Top 20 or fewer if not enough elements
    top_indices = np.argsort(similarities)[-num_to_select:]  # Top most relevant elements
    
    # Construct filtered schema with relevant elements
    filtered_parts = []
    
    # Add prefixes
    prefix_pattern = r'@prefix.*\n'
    import re
    prefixes = re.findall(prefix_pattern, full_schema)
    filtered_parts.extend(prefixes)
    
    # Add selected classes and properties
    for idx in top_indices:
        if idx < len(classes):
            filtered_parts.append(classes[idx][1])
        else:
            prop_idx = idx - len(classes)
            if prop_idx < len(properties):  # Safety check
                filtered_parts.append(properties[prop_idx][1])
    
    filtered_schema = "\n\n".join(filtered_parts)
    logger.debug(f"Extracted schema length: {len(filtered_schema)} characters")
    
    return filtered_schema

if __name__ == "__main__":
    # For local development only
    uvicorn.run(app, host="0.0.0.0", port=8000)
