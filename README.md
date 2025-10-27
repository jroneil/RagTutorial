# My RAG Application

A Retrieval-Augmented Generation (RAG) application built with LlamaIndex, ChromaDB, and Ollama for querying PDF documents with AI-powered responses.

## Overview

This project implements a RAG (Retrieval-Augmented Generation) system that allows you to:
- Load and process PDF documents
- Create semantic embeddings using HuggingFace models
- Store embeddings in ChromaDB vector database
- Query documents using Ollama LLM with source attribution
- Get AI-powered answers based on your document content

## Project Structure

```
my-rag-app/
├── app.py                 # Main RAG application
├── rag_app.py            # RAG application template
├── test_setup.py         # Setup verification script
├── requirements.txt      # Python dependencies
├── tutorial.md          # Detailed tutorial and explanation
├── README.md            # This file
└── data/                # Directory for PDF documents
    └── (your PDF files here)
```

## Features

- **Document Processing**: Automatically loads and chunks PDF documents
- **Semantic Search**: Uses vector embeddings for intelligent document retrieval
- **Source Attribution**: Shows exactly which parts of documents were used for answers
- **Configurable Models**: Supports different LLM and embedding models
- **Persistent Storage**: ChromaDB for efficient vector storage and retrieval

## Prerequisites

### Hardware Requirements
- **Windows PC**: Development machine running the application
- **MacBook (or alternative)**: AI server running Ollama
- **Network**: Both machines on the same network

### Software Requirements

**On Windows (Development Machine):**
- Python 3.10+
- Visual Studio Code (or preferred IDE)
- Virtual environment support

**On Mac (AI Server):**
- Homebrew package manager
- Ollama installed and running

## Installation

### Step 1: Set up Ollama Server (Mac)

1. Install Ollama:
   ```bash
   brew install ollama
   ```

2. Download and run Llama 3:
   ```bash
   ollama run llama3
   ```

3. Find your Mac's IP address (e.g., `192.168.1.235`)

### Step 2: Set up Development Environment (Windows)

**Important**: Use **Command Prompt (cmd)** for these commands, not PowerShell or bash.

**Option 1: Using setup script (Recommended)**
```cmd
setup.bat
```

**Option 2: Manual setup**
1. Create and activate virtual environment (with pip):
   ```cmd
   python -m venv .venv --upgrade-deps
   .venv\Scripts\activate
   ```

2. Install dependencies:
   ```cmd
   pip install -r requirements.txt
   ```

3. Verify setup:
   ```cmd
   python test_setup.py
   ```

### Step 3: Add Documents

1. Create a `data` folder in your project
2. Add PDF documents you want to query to the `data` folder

## Usage

### Running the Main Application

1. Update the IP address in `app.py` (line with `base_url`) to match your Mac's IP
2. Run the application:
   ```bash
   python app.py
   ```

The application will:
- Load and process documents from the `data` folder
- Create embeddings and store them in ChromaDB
- Query the documents and display results with source attribution

### Using the Template Application

For a simpler starting point:
```bash
python rag_app.py
```

### Testing Setup

To verify all dependencies are installed correctly:
```bash
python test_setup.py
```

## Configuration

### Model Configuration

The application uses two types of models:

1. **LLM (Reasoning Engine)**: Ollama with Llama 3
   - Configured in `app.py` with `Settings.llm`
   - Runs on your Mac server

2. **Embedding Model (Semantic Search)**: HuggingFace BAAI/bge-small-en-v1.5
   - Configured in `app.py` with `Settings.embed_model`
   - Runs locally on Windows

### Vector Database

- **ChromaDB**: Persistent vector storage
- **Location**: `./chroma_db` folder
- **Collection**: "quickstart"

## Key Components

### 1. Document Loading
- Uses `SimpleDirectoryReader` from LlamaIndex
- Automatically processes PDF files in the `data` folder
- Handles text extraction and chunking

### 2. Vector Storage
- ChromaDB for efficient similarity search
- Persistent storage for reusability
- Automatic embedding generation

### 3. Query Engine
- Semantic search for relevant document chunks
- Context-aware prompting to LLM
- Source attribution and scoring

## Customization

### Changing Models

**LLM Configuration:**
```python
Settings.llm = Ollama(model="your-model", base_url="http://your-ip:11434")
```

**Embedding Model:**
```python
Settings.embed_model = HuggingFaceEmbedding(model_name="your-embedding-model")
```

### Chunking Parameters

To customize document chunking:
```python
from llama_index.core.node_parser import SentenceSplitter

node_parser = SentenceSplitter(
    chunk_size=1024,
    chunk_overlap=200
)
```

## Troubleshooting

### Common Issues

1. **Connection Error**: Ensure Ollama is running on Mac and IP address is correct
2. **Missing Dependencies**: Run `python test_setup.py` to verify installations
3. **No PDF Files**: Add PDF files to the `data` folder
4. **Memory Issues**: Use smaller embedding models or reduce chunk sizes
5. **File Permission Errors**: If you encounter file permission errors during installation:
   - Close all Python processes: `taskkill /f /im python.exe`
   - Delete and recreate the virtual environment: `rmdir /s /q .venv` then `python -m venv .venv --upgrade-deps`
   - Try the installation again

### Debug Mode

Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Best Practices

1. **Source Verification**: Always check `response.source_nodes` for answer provenance
2. **Document Quality**: Ensure source documents are clean and well-structured
3. **Chunk Optimization**: Tune chunk sizes based on document content
4. **Model Selection**: Choose appropriate models for your use case and hardware
5. **Index Management**: Rebuild index when documents change significantly

## Architecture

The application follows a client-server architecture:

- **Client (Windows)**: Application logic, embedding generation, vector storage
- **Server (Mac)**: LLM inference via Ollama API
- **Vector Database**: Local ChromaDB for semantic search

## Dependencies

- `llama-index` >= 0.10.0
- `pypdf` - PDF processing
- `llama-index-llms-ollama` - Ollama integration
- `llama-index-embeddings-huggingface` - Embedding models
- `llama-index-vector-stores-chroma` - ChromaDB integration
- `chromadb` - Vector database

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is for educational and development purposes. Please ensure you comply with the licenses of all used components (LlamaIndex, Ollama, HuggingFace models, etc.).

## Support

For issues and questions:
1. Check the `tutorial.md` for detailed explanations
2. Verify setup with `test_setup.py`
3. Review common troubleshooting steps above

---

**Note**: This is a development prototype. For production use, consider implementing proper error handling, security measures, and performance optimizations.
