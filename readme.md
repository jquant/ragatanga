# Ragatanga - Semantic Knowledge Base and Query System

A sophisticated system that combines ontology-based reasoning with semantic search capabilities to provide intelligent query responses. The system uses a hybrid approach, leveraging both SPARQL queries against an OWL ontology and semantic similarity search over a knowledge base.

## Features

- 🧠 Hybrid query processing combining ontological reasoning and semantic search
- 🔍 SPARQL query generation from natural language
- 📚 Knowledge base management with semantic embeddings
- 🌐 RESTful API endpoints for querying and knowledge management
- 🔄 Automatic inference materialization
- 📊 Vector similarity search using FAISS

## Prerequisites

- Python 3.8+
- OpenAI API key for semantic processing
- FastAPI for the web API
- FAISS for efficient similarity search
- Owlready2 for ontology management

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ragatanga.git
cd ragatanga
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your environment variables:
```bash
export OPENAI_API_KEY='your-api-key'
```

## Project Structure

- `main.py` - Core application with FastAPI endpoints and query processing
- `modal_deploy.py` - Modal deployment configuration
- `setup_volume.py` - Volume setup for deployment
- `ontology.ttl` - Main ontology file
- `knowledge_base.md` - Knowledge base content
- `dados_pratique_typesense/` - Data directory for Typesense integration

## API Endpoints

- `POST /query` - Process hybrid queries using ontology and semantic search
- `POST /upload/ontology` - Upload a new ontology file
- `GET /download/ontology` - Download the current ontology
- `POST /upload/kb` - Upload a new knowledge base
- `GET /download/kb` - Download the current knowledge base
- `GET /describe_ontology` - Get statistics and information about the ontology

## Usage

1. Start the server:
```bash
uvicorn main:app --reload
```

2. Send queries using the API:
```bash
curl -X POST "http://localhost:8000/query" \
     -H "Content-Type: application/json" \
     -d '{"query": "Find all exercises targeting biceps"}'
```

## Deployment

The project is configured for deployment using Modal. To deploy:

```bash
modal deploy modal_deploy.py
```

## Architecture

The system uses a hybrid approach for query processing:

1. **Ontological Reasoning**
   - Uses OWL ontology for structured knowledge
   - Generates SPARQL queries from natural language
   - Materializes inferences for complete reasoning

2. **Semantic Search**
   - Embeds knowledge base content using OpenAI embeddings
   - Uses FAISS for efficient similarity search
   - Combines results with ontological queries

3. **Result Generation**
   - Merges results from both approaches
   - Generates natural language responses
   - Provides context-aware answers

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[Add your license information here]

## Contact

[Add your contact information here]
