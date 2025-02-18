from modal import Volume, Stub

stub = Stub("setup-volume")

@stub.local_entrypoint()
def setup():
    # Create the volume first
    volume = Volume.from_name("kb_data", create_if_missing=True)

    # Then upload files to it with force=True to allow overwriting
    with volume.batch_upload(force=True) as batch:
        batch.put_file("ragatanga/data/ontology.ttl", "/root/data/ontology.ttl")
        batch.put_file("ragatanga/data/knowledge_base.md", "/root/data/knowledge_base.md")