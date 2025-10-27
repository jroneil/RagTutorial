Create a new project folder, e.g., c:\dev\my-rag-app.
Open it in VS Code.
Open a terminal in VS Code (`Ctrl+``).
Create a Python virtual environment: python -m venv .venv
Activate it: .venv\Scripts\activate
Install the necessary libraries:
Bash

pip install llama-index pypdf llama-index-llms-ollama llama-index-embeddings-huggingface llama-index-vector-stores-chromadb chromadb
Create a folder named data inside your project.
Find a PDF you want to ask questions about. A technical manual, a spec document, anything. Copy it into the data folder. For this example, let's pretend it's named enterprise_scaling_guide.pdf.
Of course. Let's do this.

This is **RAG for the Experienced Engineer**. We'll skip the fluff and focus on the mechanics, the "why," and the critical "gotchas" that separate a demo from a real application.

### RAG for Dummies: The "Open Book Exam" Analogy

Imagine you ask a brilliant-but-forgetful student (the LLM) a question.

*   **Standard LLM Call (Closed Book):** The student answers from memory. The answer is eloquent but might be wrong, outdated, or completely made up (a "hallucination").
*   **RAG Call (Open Book):** Before you ask the question, you go to a library, find the *exact page* in a textbook that contains the answer, and give it to the student. You then say, "**Using only this page**, answer my question." The student's brilliance is now focused on summarizing and presenting information you *know* is correct.

**RAG is the system that finds the right page in the library for you.**

---

### The Big Picture: The Flow of Data

Here is the entire process. We will build this piece by piece.



### The Cast of Characters (Key Concepts Explained)

1.  **The LLM (The Brain):** In RAG, the LLM is a **Stateless Reasoning Engine**. Its job is *not* to know things, but to *reason* about the context you provide it. We use it for summarization, answering, and reformatting. (Your `llama3` on the Mac).
2.  **Documents (The Knowledge):** Your proprietary data. Code files, PDFs, Word docs, Confluence pages, etc. The "textbooks" in our library.
3.  **Chunking (The Librarian's Method):** You don't give the LLM an entire 500-page book. You break it down into small, digestible pieces, like paragraphs or pages ("chunks"). **This is a critical step.** If chunks are too big, the key info is diluted. Too small, and you lose context.
4.  **Embeddings (The Digital Dewey Decimal System):** This is the magic. An "embedding model" (a separate, smaller AI model) reads a chunk of text and converts it into a list of numbers (a "vector"). This vector represents the text's *semantic meaning*.
    *   **Analogy:** Imagine a giant 3D library where books about "cats" are physically close to books about "kittens" and "feline behavior," but far from books about "rocket science." The embedding vector is the `(x, y, z)` coordinate of a chunk of text in this "meaning library."
5.  **Vector Database (The Library Index):** This is a special database that stores all the embedding vectors and their corresponding text chunks. Its only job is to do one thing incredibly fast: given a new vector (from your question), find the vectors in the database that are "closest" to it mathematically. (We'll use `ChromaDB` for this).
6.  **The RAG Prompt (The Crucial Instruction):** This is the final, assembled instruction sent to the LLM. It's a template that looks like this:

    ```
    Context:
    [...insert the relevant text chunks found by the Vector DB...]

    ---
    Question:
    [...insert the user's original question...]

    ---
    Instruction:
    Based ONLY on the context provided above, answer the question.
    ```

---

## Tutorial: Build a Doc-Query App in Under an Hour

### Prerequisites

*   **Windows PC:** Your dev machine.
*   **MacBook:** Your (temporary) AI server.
*   **Software on Windows:**
    *   Python 3.10+
    *   VS Code (or your preferred IDE)
*   **Software on Mac:**
    *   Homebrew installed.

### Step 0: The Setup (5 minutes)

**On your MacBook:**

1.  Open Terminal.
2.  Install Ollama: `brew install ollama`
3.  Download and run Llama 3: `ollama run llama3`
    *   This will start the model and an API server. The first time, it will download several gigabytes.
4.  Keep this terminal window open. Find your Mac's IP address. Go to `System Settings > Wi-Fi > Details...` or `System Settings > Network` and look for the IP Address (e.g., `192.168.1.105`).

**On your Windows PC:**

1.  Create a new project folder, e.g., `c:\dev\my-rag-app`.
2.  Open it in VS Code.
3.  Open a terminal in VS Code (`Ctrl+``).
4.  Create a Python virtual environment: `python -m venv .venv`
5.  Activate it: `.venv\Scripts\activate`
6.  Install the necessary libraries:
    ```bash
    pip install llama-index pypdf llama-index-llms-ollama llama-index-embeddings-huggingface llama-index-vector-stores-chromadb chromadb
    ```
7.  Create a folder named `data` inside your project.
8.  Find a PDF you want to ask questions about. A technical manual, a spec document, anything. **Copy it into the `data` folder.** For this example, let's pretend it's named `enterprise_scaling_guide.pdf`.

### Step 1: The Code (`app.py`) (15 minutes)

Create a file named `app.py` in your project folder. We'll build it block by block.

```python
# app.py

# Block 1: Imports
# =================
import chromadb
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.storage.storage_context import StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

print("--- Imports complete ---")

# Block 2: Configuration & Setup
# ==============================
# Configure the LLM. Point this to your Mac's IP address.
# This is the "Client-Server" connection. Your app is the client, the Mac is the server.
Settings.llm = Ollama(model="llama3", base_url="http://192.168.1.105:11434")

# Configure the embedding model. This runs locally on your Windows PC.
# "BAAI/bge-small-en-v1.5" is a great, lightweight default.
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

print("--- Configuration complete ---")

# Block 3: The Indexing Phase (Loading knowledge into the "library")
# =================================================================
# This is where we load our documents, chunk them, embed them, and store them.

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


# Block 4: The Querying Phase (The "Open Book Exam")
# ===================================================
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


# Block 5: The Most Important Part - "Show Your Work"
# ===================================================
# For any enterprise app, you MUST know where the answer came from.
# This is the "provenance" or "source" of the information.

print("\n--- Sources ---")
for node in response.source_nodes:
    # The 'node.metadata' contains info like the file name
    print(f"Source: {node.metadata['file_name']}, Score: {node.get_score():.2f}")
    # 'node.get_text()' shows the exact chunk of text used for the answer
    # print(f"Text: {node.get_text()}\n---") # Uncomment to see the full text
```

### Step 2: Run It! (1 minute)

1.  Make sure your Mac is still running `ollama run llama3`.
2.  In your VS Code terminal on Windows (with the `.venv` activated), run the script:
    ```bash
    python app.py
    ```
3.  Watch the output. It will load the docs, create the index, and then print the question, the AI-generated answer, and critically, the source file it used!

---

### Gotchas & Pro Tips for the Enterprise Engineer

1.  **Gotcha: The Black Box is Dangerous.** Your code in `Block 5` is the cure. **Never** trust an LLM response without knowing its sources. `response.source_nodes` is your best friend for building trustworthy, debuggable AI systems.

2.  **Gotcha: Garbage In, Garbage Out.** The quality of your RAG system is 100% dependent on the quality of your source documents. If your docs are messy, outdated, or contradictory, your answers will be too. Data cleaning is still king.

3.  **Gotcha: The Chunky Problem.** `LlamaIndex` uses default settings for chunking. For real projects, you will need to tune `chunk_size` and `chunk_overlap`.
    *   **Chunk Size:** How many characters/tokens in each piece. A common starting point is 1024.
    *   **Chunk Overlap:** How many characters are repeated between consecutive chunks. This prevents losing context when a sentence is split across two chunks. A common starting point is 20% of the chunk size.

4.  **Gotcha: The Two-Model Shuffle.** Notice we used two different models: `llama3` (the LLM) and `BAAI/bge-small-en-v1.5` (the embedding model). **They have different jobs.** The embedding model's only job is to create the vectors. Don't use a huge, slow model for embeddings. Small, specialized embedding models are faster and often better.

5.  **Gotcha: My Index is Stale!** We just built a static index. What happens when the documents change? You need an indexing strategy.
    *   **Simple:** Delete the `chroma_db` folder and re-run the script periodically.
    *   **Advanced:** LlamaIndex has sophisticated ways to update/refresh the index, checking for modified files and only updating the necessary chunks.

You have now built the fundamental pattern for enterprise AI. You've separated your compute (Mac) from your application (Windows), used a vector database, and built a query engine that can cite its sources. This foundation is where 90% of the practical value of AI in the enterprise currently lies. Your next step is to take this pattern and scale it up on your Proxmox server.