"""
Main entry point for the Ragatanga API.

This module provides a simple way to start the Ragatanga API server.
"""

import os
import uvicorn
from loguru import logger

from ragatanga import config
from ragatanga.api.app import app

def run_server(host: str = "0.0.0.0", port: int = None):
    """
    Run the Ragatanga API server.
    
    Args:
        host: Host to bind to
        port: Port to bind to (defaults to config.DEFAULT_PORT or 8000)
    """
    port = port or config.DEFAULT_PORT or 8000
    logger.info(f"Starting Ragatanga API server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    # For local development only
    run_server()