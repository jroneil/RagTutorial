#!/usr/bin/env python3
"""
Simple test script to verify the RAG setup is working correctly.
This script will check if all required packages are installed.
"""

import sys

def check_package(package_name):
    """Check if a package is installed and importable."""
    try:
        __import__(package_name)
        print(f"[OK] {package_name} is installed and importable")
        return True
    except ImportError as e:
        print(f"X {package_name} is NOT installed: {e}")
        return False

def main():
    print("Testing RAG application setup...")
    print("=" * 40)
    
    required_packages = [
        "llama_index",
        "pypdf",
        "llama_index.llms.ollama",
        "llama_index.embeddings.huggingface",
        "llama_index.vector_stores.chroma",
        "chromadb"
    ]
    
    all_installed = True
    for package in required_packages:
        if not check_package(package):
            all_installed = False
    
    print("=" * 40)
    if all_installed:
        print("[OK] All required packages are installed!")
        print("\nNext steps:")
        print("1. Place your PDF file in the 'data' folder")
        print("2. Create your RAG application script")
        print("3. Run your application to query the PDF")
    else:
        print("[ERROR] Some packages are missing. Please check the installation.")
    
    return 0 if all_installed else 1

if __name__ == "__main__":
    sys.exit(main())
