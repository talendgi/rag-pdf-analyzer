import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_classic.chains import RetrievalQA
import tempfile
import os
from dotenv import load_dotenv

# Load Environment Variables (Ensure GROQ_API_KEY is in your .env)
load_dotenv(override=True)

# --- PAGE CONFIG ---
st.set_page_config(page_title="Enterprise RAG", layout="wide")

# --- AUTHENTICATION SETUP ---
with open('auth_config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days']
)

# Render Login Widget
authenticator.login()

# Retrieve Auth Status
auth_status = st.session_state.get("authentication_status")
username = st.session_state.get("username")
name = st.session_state.get("name")

# --- MAIN APP LOGIC ---
if auth_status:
    # 1. Sidebar for User Info and File Upload
    with st.sidebar:
        st.title(f"Welcome, {name}!")
        authenticator.logout('Logout', 'sidebar')
        st.divider()
        uploaded_file = st.file_uploader("Upload Knowledge Base (PDF)", type="pdf")

    st.title("📑 Professional Document Assistant")
    
    # Initialize Chat History
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # 2. Check if a file has been uploaded
    if uploaded_file:
        # Processing Logic (Only runs once per upload)
        if "rag_chain" not in st.session_state:
            with st.spinner("Processing PDF with HuggingFace & Groq..."):
                # Save to temp file safely
                with tempfile.NamedTemporaryFile(delete=False) as tf:
                    tf.write(uploaded_file.getbuffer())
                    file_path = tf.name
                
                try:
                    # Load and Split
                    loader = PyPDFLoader(file_path)
                    data = loader.load()
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
                    chunks = text_splitter.split_documents(data)
                    
                    # Embeddings (Local HuggingFace - Lighter than Ollama)
                    embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
                    vector_db = Chroma.from_documents(chunks, embeddings)
                    
                    # Groq LLM Setup
                    llm = ChatGroq(
                        model="llama-3.3-70b-versatile",
                        temperature=0,
                        groq_api_key=os.getenv("GROQ_API_KEY")
                    )
                    
                    # Create RAG Chain
                    st.session_state.rag_chain = RetrievalQA.from_chain_type(
                        llm=llm, 
                        chain_type="stuff", 
                        retriever=vector_db.as_retriever(search_kwargs={"k": 3})
                    )
                    st.success("System Ready via Groq Cloud!")
                
                finally:
                    # Cleanup temp file even if it fails
                    if os.path.exists(file_path):
                        os.remove(file_path)

        # 3. Chat Interface Logic
        # Display existing messages
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Input for new questions
        if prompt := st.chat_input("What would you like to know?"):
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    # Use the chain stored in session state
                    response = st.session_state.rag_chain.invoke(prompt)
                    full_response = response["result"]
                    st.markdown(full_response)
                    st.session_state.chat_history.append({"role": "assistant", "content": full_response})
    else:
        st.info("Please upload a PDF document in the sidebar to begin your session.")

elif auth_status is False:
    st.error('Username/password is incorrect')
elif auth_status is None:
    st.warning('Please enter your credentials to access the RAG system.')