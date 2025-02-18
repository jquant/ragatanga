import modal

# Create a Modal app (instead of stub)
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
    .add_local_file("main.py", "/root/main.py")  # Copy main.py into the container
)

# Create a Modal volume to persist data
volume = modal.Volume.from_name("kb_data")

@app.function(
    image=image,
    volumes={"/data": volume},
    secrets=[modal.Secret.from_name("openai-secret")]
)
@modal.asgi_app()
def fastapi_app():
    from main import app
    return app 