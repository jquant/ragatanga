from setuptools import setup, find_namespace_packages

with open("readme.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="ragatanga",
    version="0.2.0",
    author="José Saum",
    author_email="your.email@example.com",  # Replace with your email
    description="A hybrid semantic knowledge base and query system combining ontology-based reasoning with semantic search",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jquant/ragatanga",  # Replace with your repo URL
    packages=find_namespace_packages(include=['ragatanga*']),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    include_package_data=True,
    package_data={
        'ragatanga.data': ['*.md', '*.ttl'],
    },
) 