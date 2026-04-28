from langchain_community.vectorstores import Chroma
# from langchain_community.embeddings import OllamaEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import os


load_dotenv(override=True)


def get_rag_chain(documents):
    # nomic-embed-text is highly recommended for local RAG
    # embeddings = OllamaEmbeddings(model="nomic-embed-text")
    
    # In-memory vectorstore for the session
    # vectorstore = Chroma.from_documents(documents, embeddings)
    embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
    
    #  Create the Vector Store
    vectorstore = Chroma.from_documents(
        documents=documents, 
        embedding=embeddings,
        persist_directory="./chroma_db" # Persists to disk on Windows
    )
    # Initialize the LLM (e.g., Llama3 or Qwen)
    # llm = Ollama(model="qwen3.5:0.8b", temperature=0)
    llm=ChatGroq(model="llama-3.3-70b-versatile",api_key=os.getenv("GROQ_API_KEY"))
    
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3})
    )