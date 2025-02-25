"""
Ontology management module for Ragatanga.

This module handles ontology loading, materialization, and SPARQL query execution.
"""

import os
import tempfile
from datetime import datetime
from typing import List, Dict, Any, Optional, Set, Tuple

import asyncio
import aiofiles
import rdflib
from rdflib.plugins.sparql import prepareQuery
import owlready2
from owlready2 import sync_reasoner_pellet, get_ontology
from loguru import logger


class OntologyManager:
    """Manages ontology loading and reasoning with incremental updates."""
    
    def __init__(self, owl_file_path: str):
        """
        Initialize the ontology manager.
        
        Args:
            owl_file_path: Path to the ontology file (.ttl or .owl)
        """
        self.owl_file_path = owl_file_path
        self.materialized_file = owl_file_path.replace(".ttl", "_materialized.ttl")
        self.onto = None
        self.last_modified_time = None
        self.graph = None
    
    async def load_and_materialize(self, force_rebuild=False) -> owlready2.Ontology:
        """
        Load ontology and materialize inferences, with cached handling.
        
        Args:
            force_rebuild: Whether to force rebuilding the materialized ontology
            
        Returns:
            The loaded ontology object
        """
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
            self.onto = await asyncio.to_thread(self._load_ontology, self.owl_file_path)
            
        return self.onto
    
    async def _materialize_inferences(self) -> owlready2.Ontology:
        """
        Materialize inferences with improved error handling.
        
        Returns:
            The ontology with inferences materialized
        
        Raises:
            Exception: If materialization fails
        """
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
        """
        Execute a SPARQL query with improved error handling.
        
        Args:
            sparql_query: The SPARQL query string to execute
            
        Returns:
            List of result strings
        """
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
        """
        Get all properties of an individual with their values.
        
        Args:
            individual_uri: The URI of the individual
            
        Returns:
            List of property-value pairs as strings
        """
        query = f"""
        SELECT ?prop ?value
        WHERE {{
            <{individual_uri}> ?prop ?value .
        }}
        """
        results = await self.execute_sparql(query)
        return results
    
    @staticmethod
    def _load_ontology(owl_path: str) -> owlready2.Ontology:
        """
        Loads the ontology with rdflib first, then converts to a format Owlready2 can read.
        
        Args:
            owl_path: Path to the ontology file
            
        Returns:
            The loaded ontology
            
        Raises:
            Exception: If loading fails
        """
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

    def get_all_individuals(self) -> List[owlready2.Thing]:
        """
        Collect all individuals, including those available as instances of a class.
        
        Returns:
            List of all individuals in the ontology
        """
        if self.onto is None:
            return []
            
        inds = set(self.onto.individuals())
        for cls in self.onto.classes():
            for instance in cls.instances():
                inds.add(instance)
        return list(inds)
    
    async def load_ontology_schema(self) -> str:
        """
        Return the filtered ontology schema without individual declarations.
        Uses the materialized file if it exists and is nonempty.
        
        Returns:
            Filtered ontology schema
        """
        materialized_file = self.materialized_file
        
        # Use the materialized file if it exists and is non-empty
        if os.path.exists(materialized_file):
            async with aiofiles.open(materialized_file, "r", encoding="utf-8") as file:
                contents = await file.read()
            if not contents.strip():
                logger.warning("Materialized ontology is empty, falling back to original file.")
                async with aiofiles.open(self.owl_file_path, "r", encoding="utf-8") as file:
                    contents = await file.read()
        else:
            async with aiofiles.open(self.owl_file_path, "r", encoding="utf-8") as file:
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

    def build_enhanced_ontology_entries(self, include_inferred=True) -> List[Dict[str, Any]]:
        """
        Build enhanced text representations of ontology elements with more context and relationships.
        
        Args:
            include_inferred: Whether to include inferred statements
            
        Returns:
            List of dictionaries with id and text fields
        """
        entries = []
        idx = 0
        
        if self.onto is None:
            return entries
        
        # Create a mapping of entities to their labels for better context
        label_map = {}
        all_entities = list(self.onto.classes()) + list(self.onto.properties()) + list(self.get_all_individuals())
        for entity in all_entities:
            if hasattr(entity, 'label') and entity.label:
                label_map[entity] = ', '.join(entity.label)
            else:
                label_map[entity] = entity.name
        
        # 1) Classes with enhanced context
        print("\nBuilding enhanced class representations...")
        for cls in self.onto.classes():
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
            domain_props = [p for p in self.onto.properties() if cls in p.domain]
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
        for indiv in self.get_all_individuals():
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
            for prop in self.onto.properties():
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
                    prop_obj = next((p for p in self.onto.properties() if p.name == prop_name), None)
                    prop_label = ', '.join(prop_obj.label) if prop_obj and prop_obj.label else prop_name
                    doc += f"  - {prop_name} ({prop_label}): {prop_value}\n"
            
            # Add inverse relationships
            inverse_relations = []
            for prop in self.onto.properties():
                if not hasattr(prop, 'range') or not prop.range:
                    continue
                    
                # Check if this individual could be in the range of this property
                if any(isinstance(indiv, r) for r in prop.range if hasattr(r, 'instances')):
                    # Find subjects that have this individual as the value for this property
                    for subj in self.get_all_individuals():
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
        for prop in self.onto.properties():
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
                for subj in self.get_all_individuals():
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
    
    def get_ontology_statistics(self) -> Dict[str, Any]:
        """
        Gather comprehensive statistics about the ontology.
        
        Returns:
            Dictionary of ontology statistics
        """
        if self.onto is None:
            return {"error": "Ontology not loaded"}
            
        classes = list(self.onto.classes())
        individuals = list(self.get_all_individuals())
        properties = list(self.onto.properties())
        
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
                for prop in self.onto.properties():
                    try:
                        if prop in ind.get_properties():
                            values = prop[ind]
                            if values:
                                props[prop.name] = [str(v) for v in values] if isinstance(values, list) else [str(values)]
                    except Exception:
                        continue
                if props:
                    individual_properties[ind.name] = props

        result = {
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
        
        # Add metadata
        result["metadata"] = {
            "file_path": self.owl_file_path,
            "file_size": os.path.getsize(self.owl_file_path),
            "last_modified": datetime.fromtimestamp(os.path.getmtime(self.owl_file_path)).isoformat(),
            "format": "Turtle" if self.owl_file_path.endswith('.ttl') else "OWL"
        }
        
        return result

async def extract_relevant_schema(query: str, owl_path: str) -> str:
    """
    Extract schema elements relevant to the query using semantic similarity.
    
    Args:
        query: The query to extract relevant schema for
        owl_path: Path to the ontology file
        
    Returns:
        Filtered ontology schema relevant to the query
    """
    from ragatanga.utils.embeddings import EmbeddingProvider
    
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
    embed_provider = EmbeddingProvider.get_provider()
    query_embedding = await embed_provider.embed_query(query)
    
    # Get embeddings for class and property names
    class_names = [name for name, _ in classes]
    property_names = [name for name, _ in properties]
    
    all_names = class_names + property_names
    
    if not all_names:
        logger.warning("No classes or properties found in the ontology")
        return full_schema
    
    logger.debug(f"Embedding {len(all_names)} ontology elements")
    all_embeddings = await embed_provider.embed_texts(all_names)
    
    # Calculate similarities
    import numpy as np
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