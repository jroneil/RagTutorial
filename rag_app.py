#!/usr/bin/env python3
"""
Simple RAG application for querying PDF documents.
This script implements a full RAG pipeline using ChromaDB, Ollama, and HuggingFace embeddings.
"""

import chromadb
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.storage.storage_context import StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import os
from pathlib import Path

def main():
    print("RAG Application")
    print("=" * 40)
    
    # Check if data folder exists and has PDF files
    data_dir = Path("data")
    if not data_dir.exists():
        print("ERROR: 'data' folder does not exist. Please create it and add your PDF files.")
        return
    
    pdf_files = list(data_dir.glob("*.pdf"))
    if not pdf_files:
        print("No PDF files found in the 'data' folder.")
        print("Please add your PDF files to the 'data' folder.")
        print("For example: enterprise_scaling_guide.pdf")
        return
    
    print(f"Found {len(pdf_files)} PDF file(s) in the data folder:")
    for pdf in pdf_files:
        print(f"  - {pdf.name}")
    
    print("\n--- Starting RAG Pipeline ---")
    
    # Block 1: Configuration & Setup
    # ==============================
    print("--- Configuration & Setup ---")
    
    # Configure the LLM. Point this to your Mac's IP address.
    # This is the "Client-Server" connection. Your app is the client, the Mac is the server.
    Settings.llm = Ollama(model="llama3", base_url="http://192.168.1.235:11434")
    
    # Configure the embedding model. This runs locally on your Windows PC.
    # "BAAI/bge-small-en-v1.5" is a great, lightweight default.
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    
    print("--- Configuration complete ---")
    
    # Block 2: The Indexing Phase (Loading knowledge into the "library")
    # =================================================================
    print("--- Indexing Phase ---")
    
    # Initialize ChromaDB client. It will store data in a folder named 'chroma_db'
    db = chromadb.PersistentClient(path="./chroma_db")
    chroma_collection = db.get_or_create_collection("quickstart")
    
    # Create a Vector Store object
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # Load the documents from the 'data' directory
    documents = SimpleDirectoryReader("./data").load_data()
    print(f"--- Loaded {len(documents)} document(s) ---")
    
    # Create the index. This is the magic step.
    # LlamaIndex handles the chunking, embedding, and storage for us.
    index = VectorStoreIndex.from_documents(
        documents, storage_context=storage_context
    )
    print("--- Indexing complete. Knowledge is now stored in ChromaDB. ---")
    
    # Block 3: The Querying Phase (The "Open Book Exam")
    # ===================================================
    print("\n--- Querying Phase ---")
    
    # Create a query engine from the index
    query_engine = index.as_query_engine()
    
    # Ask a question!
    # The query will be embedded, used to find relevant chunks,
    # and then passed to Llama 3 with the chunks as context.
    question = "What are the best practices for database scaling according to the document?"
    response = query_engine.query(question)
    
    print("\n--- Question ---")
    print(question)
    print("\n--- Answer ---")
    print(str(response))
    
    # Block 4: The Most Important Part - "Show Your Work"
    # ===================================================
    # For any enterprise app, you MUST know where the answer came from.
    # This is the "provenance" or "source" of the information.
    
    print("\n--- Sources ---")
    for node in response.source_nodes:
        # The 'node.metadata' contains info like the file name
        print(f"Source: {node.metadata['file_name']}, Score: {node.get_score():.2f}")
        # 'node.get_text()' shows the exact chunk of text used for the answer
        # print(f"Text: {node.get_text()}\n---") # Uncomment to see the full text
    
    print("\n--- RAG Pipeline Complete ---")

if __name__ == "__main__":
    main()
