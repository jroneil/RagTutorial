#!/usr/bin/env python3
"""
Test script to check imports and identify the issue with response synthesizer
"""

try:
    from llama_index.core.response_synthesizers.factory import get_response_synthesizer
    print("SUCCESS: get_response_synthesizer import successful")
except ImportError as e:
    print(f"FAILED: get_response_synthesizer import failed: {e}")

try:
    from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
    print("SUCCESS: Core imports successful")
except ImportError as e:
    print(f"FAILED: Core imports failed: {e}")

try:
    from llama_index.core.storage.storage_context import StorageContext
    print("SUCCESS: StorageContext import successful")
except ImportError as e:
    print(f"FAILED: StorageContext import failed: {e}")

try:
    from llama_index.vector_stores.chroma import ChromaVectorStore
    print("SUCCESS: ChromaVectorStore import successful")
except ImportError as e:
    print(f"FAILED: ChromaVectorStore import failed: {e}")

try:
    from llama_index.llms.ollama import Ollama
    print("SUCCESS: Ollama import successful")
except ImportError as e:
    print(f"FAILED: Ollama import failed: {e}")

try:
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    print("SUCCESS: HuggingFaceEmbedding import successful")
except ImportError as e:
    print(f"FAILED: HuggingFaceEmbedding import failed: {e}")

print("\nTesting response synthesizer creation...")
try:
    from llama_index.core import get_response_synthesizer
    synthesizer = get_response_synthesizer()
    print("SUCCESS: Response synthesizer creation successful")
except Exception as e:
    print(f"FAILED: Response synthesizer creation failed: {e}")
    print(f"Error type: {type(e)}")
