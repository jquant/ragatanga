import modal

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
    secrets=[modal.Secret.from_name("openai-secret")]
)
@modal.asgi_app()
def fastapi_app():
    import os
    os.environ["OWL_FILE_PATH"] = "/root/data/ontology.ttl"
    os.environ["KBASE_FILE"] = "/root/data/knowledge_base.md"
    from ragatanga.main import app
    return app 