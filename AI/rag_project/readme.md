A production-grade Retrieval-Augmented Generation (RAG) system , document processing, vector storage, and Streamlit UI 

### 1. High-Level Architecture
A professional RAG system is typically split into several layers:
* **Ingestion Layer:** Handles PDF parsing, text chunking, and embedding generation.
* **Storage Layer:** A vector database (like Pinecone, Weaviate, or ChromaDB) to store embeddings.
* **Retrieval Layer:** The logic that finds the most relevant chunks based on a user query.
* **Generation Layer:** The LLM (via Ollama, OpenAI, or Anthropic) that synthesizes the answer.
* **Application Layer:** Streamlit with session management and authentication.


**Basic implementation with `streamlit-authenticator`:**
1.  Define a YAML file for user credentials (passwords should be hashed).
2.  Use the `Authenticator` object to wrap your main app logic.

### 4. The Streamlit UI (`app.py`)



###  Structure
```text
rag_project/
├── .env                # API keys and secrets
├── app.py              # Main Streamlit UI
├── auth_config.yaml    # User credentials (hashed)
├── requirements.txt    # Dependencies
└── src/
    ├── __init__.py
    ├── engine.py       # RAG Logic (Chains, Retrieval)
    └── processor.py    # Document processing (Parsing, Chunking)
```



**`requirements.txt`**
```text
streamlit
streamlit-authenticator
langchain
langchain-community
chromadb
pypdf
unstructured
pyyaml
```

### Key "Production" Features :
1.  **State Management:** The PDF is only embedded once per upload via `st.session_state`.
2.  **Modular Logic:** The UI code is clean; the RAG logic is isolated in the `src/` directory.
3.  **Temporary Handling:** Uses `tempfile` to handle PDF bytes safely.
4.  **Security:** Implements a professional login/logout flow with cookie persistence.


Questions:

* “What diagnosis was given to the patient?”
* “List all medications prescribed.”
* “What symptoms were observed?”
* “What follow-up treatment was recommended?”
* “Summarize the patient condition in simple terms.”
* “Which lab tests were abnormal?”


# Demo 

https://github.com/user-attachments/assets/73d78c4c-ab41-45bc-bd3b-ed990e82b6a5


