#!/usr/bin/env python3
"""
Build and publish script for ragatanga package.
"""

import os
import sys
import glob
import subprocess
import shutil

def clean_build_directories():
    """Remove build artifacts."""
    print("Cleaning build directories...")
    directories = ['build', 'dist', 'ragatanga.egg-info']
    for directory in directories:
        if os.path.exists(directory):
            shutil.rmtree(directory)
            print(f"Removed {directory}")

def build_package():
    """Build the package."""
    print("Building package...")
    subprocess.run([sys.executable, "-m", "build"], check=True)

def test_upload():
    """Upload to Test PyPI."""
    print("Uploading to Test PyPI...")
    subprocess.run([
        "twine", "upload", "--verbose",
        "--repository-url", "https://test.pypi.org/legacy/",
        "dist/*"
    ], check=True)

def prod_upload():
    """Upload to PyPI."""
    print("Uploading to PyPI...")
    subprocess.run([
        "twine", "upload",
        "dist/*"
    ], check=True)

def main():
    """Main function."""
    if len(sys.argv) < 2:
        print("Usage: python build_and_publish.py [build|test|publish|all]")
        sys.exit(1)

    command = sys.argv[1]
    
    if command in ['build', 'all']:
        clean_build_directories()
        build_package()
    
    if command in ['test', 'all']:
        test_upload()
    
    if command in ['publish', 'all']:
        prod_upload()

if __name__ == "__main__":
    main() 