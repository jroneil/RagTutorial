#!/usr/bin/env python3
"""
Simple test to verify the RAG pipeline works end-to-end
"""

import chromadb
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.storage.storage_context import StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import os
from pathlib import Path

def test_rag_pipeline():
    print("Testing RAG Pipeline...")
    
    # Check if data exists
    data_dir = Path("data")
    if not data_dir.exists():
        print("ERROR: 'data' folder does not exist")
        return False
    
    pdf_files = list(data_dir.glob("*.pdf"))
    if not pdf_files:
        print("No PDF files found")
        return False
    
    print(f"Found {len(pdf_files)} PDF file(s)")
    
    # Quick configuration
    Settings.llm = Ollama(model="llama3", base_url="http://localhost:11434")
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    
    # Test document loading
    try:
        documents = SimpleDirectoryReader("./data").load_data()
        print(f"SUCCESS: Loaded {len(documents)} document(s)")
    except Exception as e:
        print(f"FAILED: Failed to load documents: {e}")
        return False
    
    # Test index creation
    try:
        db = chromadb.PersistentClient(path="./chroma_db")
        chroma_collection = db.get_or_create_collection("quickstart")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        index = VectorStoreIndex.from_documents(
            documents, storage_context=storage_context
        )
        print("SUCCESS: Index created successfully")
    except Exception as e:
        print(f"FAILED: Failed to create index: {e}")
        return False
    
    # Test query engine creation
    try:
        query_engine = index.as_query_engine(llm=Settings.llm)
        print("SUCCESS: Query engine created successfully")
        
        # Test a simple query
        question = "What is this document about?"
        response = query_engine.query(question)
        print(f"SUCCESS: Query executed successfully")
        print(f"Question: {question}")
        print(f"Answer: {str(response)[:200]}...")  # Show first 200 chars
        
        return True
    except Exception as e:
        print(f"FAILED: Failed to create query engine or execute query: {e}")
        return False

if __name__ == "__main__":
    success = test_rag_pipeline()
    if success:
        print("\n🎉 RAG Pipeline Test: SUCCESS")
    else:
        print("\n❌ RAG Pipeline Test: FAILED")
