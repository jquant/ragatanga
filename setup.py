"""
Legacy setup.py file maintained for compatibility.
Actual package configuration is in pyproject.toml.
"""

from setuptools import setup, find_namespace_packages

if __name__ == "__main__":
    setup(
        version="0.3.2",
        packages=find_namespace_packages(include=["ragatanga", "ragatanga.*"]),
        # The rest of the configuration is in pyproject.toml
    ) 