import modal
import os
import sys
import logging
import argparse

# Suppress verbose logging from HTTP/2 and gRPC related libraries
logging.getLogger('hpack.hpack').setLevel(logging.WARNING)
logging.getLogger('h2').setLevel(logging.WARNING)
logging.getLogger('grpc').setLevel(logging.WARNING)

# Create a Modal app
app = modal.App("ragatanga")

# Create an image with all dependencies
image = (modal.Image.debian_slim()
    .apt_install("default-jre")  # Add Java Runtime Environment
    .pip_install(
        "fastapi",
        "uvicorn",
        "aiofiles",
        "faiss-cpu",
        "instructor",
        "numpy",
        "openai",
        "owlready2",
        "rdflib",
        "pydantic",
        "loguru"
    )
    .add_local_dir("ragatanga", "/root/ragatanga")  # Copy entire ragatanga directory
)

# Create a Modal volume to persist data
volume = modal.Volume.from_name("kb_data")

@app.function(
    image=image,
    volumes={"/root/data": volume},  # Mount volume at /root/data
    secrets=[modal.Secret.from_name("openai-secret")],
    timeout=600
)
@modal.asgi_app()
def fastapi_app():
    from loguru import logger
    
    # Set up logging - suppress hpack debug messages
    logging.basicConfig(level=logging.INFO)  # Change from DEBUG to INFO
    
    # Specifically set hpack logger to WARNING level to suppress debug messages
    logging.getLogger('hpack.hpack').setLevel(logging.WARNING)
    
    # Keep loguru debug level for application logs
    logger.add(sys.stderr, level="DEBUG")
    
    # Use the sample files in ragatanga/data instead of trying to use files in /root/data
    # This ensures the environment variables don't override the default config
    os.environ["ONTOLOGY_PATH"] = "/root/ragatanga/data/sample_ontology.ttl"
    os.environ["KNOWLEDGE_BASE_PATH"] = "/root/ragatanga/data/sample_knowledge_base.md"
    
    # Remove these environment variables to avoid confusion with the config
    if "OWL_FILE_PATH" in os.environ:
        del os.environ["OWL_FILE_PATH"]
    if "KBASE_FILE" in os.environ:
        del os.environ["KBASE_FILE"]
    
    # Print environment variables for debugging
    logger.debug(f"OpenAI API Key exists: {bool(os.environ.get('OPENAI_API_KEY'))}")
    logger.debug(f"ONTOLOGY_PATH: {os.environ.get('ONTOLOGY_PATH')}")
    logger.debug(f"KNOWLEDGE_BASE_PATH: {os.environ.get('KNOWLEDGE_BASE_PATH')}")
    
    # Import the FastAPI app
    from ragatanga.main import app
    return app 

def run_local():
    """Run the application locally using uvicorn."""
    import uvicorn
    from loguru import logger
    import traceback
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger.add(sys.stderr, level="DEBUG")
    
    try:
        # Set environment variables for local development
        os.environ["ONTOLOGY_PATH"] = os.path.join(os.path.dirname(__file__), "ragatanga/data/sample_ontology.ttl")
        os.environ["KNOWLEDGE_BASE_PATH"] = os.path.join(os.path.dirname(__file__), "ragatanga/data/sample_knowledge_base.md")
        
        # Print environment variables for debugging
        logger.debug(f"OpenAI API Key exists: {bool(os.environ.get('OPENAI_API_KEY'))}")
        logger.debug(f"ONTOLOGY_PATH: {os.environ.get('ONTOLOGY_PATH')}")
        logger.debug(f"KNOWLEDGE_BASE_PATH: {os.environ.get('KNOWLEDGE_BASE_PATH')}")
        
        # Import and run the FastAPI app
        from ragatanga.main import run_server
        logger.info("Starting local server...")
        
        # Run the server directly with uvicorn instead of using the run_server function
        from ragatanga.api.app import app
        uvicorn.run(app, host="127.0.0.1", port=8001, log_level="debug")
    except Exception as e:
        logger.error(f"Error running local server: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deploy Ragatanga API")
    parser.add_argument("--local", action="store_true", help="Run locally instead of deploying to Modal")
    args = parser.parse_args()
    
    if args.local:
        print("Running locally...")
        run_local()
    else:
        print("Deploying to Modal...")
        # For Modal deployment, we just need to import the app
        # Modal CLI will handle the deployment 