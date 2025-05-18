import os
import tempfile
import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Streamlit
st.set_page_config(page_title="PDF RAG with Gemini", layout="wide")
st.title("📄 PDF RAG with Google Gemini and FAISS")

# Sidebar for settings
with st.sidebar:
    st.header("Settings")
    google_api_key = st.text_input("Google API Key", type="password")
    if not google_api_key:
        google_api_key = os.getenv("GOOGLE_API_KEY")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7)
    max_tokens = st.number_input("Max Tokens", min_value=100, max_value=2000, value=1000)
    st.markdown("FAISS is used for efficient vector similarity search")

# Initialize components
@st.cache_resource
def load_rag_components(pdf_path, google_api_key):
    # Load and split PDF
    loader = PyPDFLoader(pdf_path)
    pages = loader.load_and_split()
    
    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    splits = text_splitter.split_documents(pages)
    
    # Create vectorstore with FAISS
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=google_api_key
    )
    
    # Create FAISS index
    vectorstore = FAISS.from_documents(
        documents=splits,
        embedding=embeddings
    )
    
    # Create LLM instance
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=google_api_key,
        temperature=temperature,
        max_output_tokens=max_tokens
    )
    
    # Create QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
        return_source_documents=True
    )
    
    return qa_chain, vectorstore  # Return both for potential reuse

# File upload
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name
    
    if google_api_key:
        try:
            qa_chain, _ = load_rag_components(tmp_file_path, google_api_key)
            st.success("PDF processed and ready for questions!")
            
            # Question input
            question = st.text_input("Ask a question about the PDF:")
            
            if question:
                with st.spinner("Searching and generating answer..."):
                    result = qa_chain({"query": question})
                    st.subheader("Answer")
                    st.write(result["result"])
                    
                    # Show source documents
                    with st.expander("See source documents"):
                        for i, doc in enumerate(result["source_documents"]):
                            st.write(f"Source {i+1} (Page {doc.metadata.get('page', 'N/A')}):")
                            st.text(doc.page_content[:500] + ("..." if len(doc.page_content) > 500 else ""))
                            st.divider()
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
        finally:
            os.unlink(tmp_file_path)
    else:
        st.warning("Please enter your Google API key")