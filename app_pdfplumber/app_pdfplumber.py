import streamlit as st
import logging
import os
import tempfile
import io
# Variant I - pdfplumber
# LlamaIndex core imports
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document
from llama_index.core.readers.base import BaseReader

# LLM and Embedding Model imports
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding as GoogleGenerativeAIEmbedding

# Import pdfplumber
import pdfplumber
from typing import List

from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---
st.set_page_config(
    page_title="Local Document Assistant",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded",
)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --- CHANGE 1: Create a Custom Reader for pdfplumber ---
class PdfPlumberReader(BaseReader):
    """
    A custom LlamaIndex reader that uses pdfplumber to extract text from PDFs.
    This reader processes each page separately to ensure accurate page number metadata.
    """
    def load_data(self, file_path: str, **kwargs) -> List[Document]:
        """Loads data from a PDF file using pdfplumber."""
        documents = []
        try:
            with pdfplumber.open(file_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    text = page.extract_text(x_tolerance=2, keep_v_dir=True) # Tolerances for better layout detection
                    if text:  # Ensure we don't create documents for empty pages
                        metadata = {
                            "page_label": str(i + 1),  # Page numbers are 1-based for users
                            "file_name": os.path.basename(file_path)
                        }
                        documents.append(Document(text=text, metadata=metadata))
            logger.info(f"Successfully extracted text from {len(documents)} pages.")
        except Exception as e:
            logger.error(f"Error reading PDF file {file_path}: {e}")
            raise
        return documents

# --- Core Logic ---

@st.cache_resource(show_spinner="Parsing and Indexing your document...")
def build_index_from_file(file_bytes, file_name):
    """
    Builds a LlamaIndex VectorStoreIndex using our custom PdfPlumberReader.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file_bytes)
        temp_file_path = temp_file.name

    try:
        # --- CHANGE 2: Use the new PdfPlumberReader ---
        # No API key is needed here anymore.
        parser = PdfPlumberReader()
        documents = parser.load_data(file_path=temp_file_path)

        if not documents:
            logger.error("No documents were extracted from the PDF. Aborting indexing.")
            return None

        index = VectorStoreIndex.from_documents(documents)
        logger.info("VectorStoreIndex created successfully.")
        return index

    finally:
        os.remove(temp_file_path)

def get_chat_engine(index):
    """
    Initializes a chat engine with memory and a custom system prompt.
    """
    system_prompt = (
        "You are a helpful assistant specialized in answering questions based ONLY on the provided document context. "
        "Your answers must be grounded in the information from the document. "
        "When you provide an answer, you MUST also cite the source page number(s).\n"
        "If the answer to a question is not available in the provided context, you MUST respond with: "
       "'No answer found and I could not find an answer to this question in the document.'\n"
        "Do not use any prior knowledge or make up information."
    )
    memory = ChatMemoryBuffer.from_defaults(token_limit=3000)

    chat_engine = index.as_chat_engine(
        chat_mode="context",
        memory=memory,
        system_prompt=system_prompt,
        verbose=True,
        streming=True,  
    )
    return chat_engine


# --- Streamlit UI ---
st.title("RAG - 📄 Document Chat Assistant (Local Parser - pdfplumber)")
st.caption("Powered by `pdfplumber` for free, local document processing.")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! Upload a document and I'll help you with it."}]
if "chat_engine" not in st.session_state:
    st.session_state.chat_engine = None

# --- Sidebar for API Keys and Upload ---
with st.sidebar:
    st.header("Configuration")
    # --- CHANGE 3: Removed LlamaCloud API Key Input ---
    #google_api_key = st.text_input("Enter your Google API Key", type="password", key="google_api_key")
    google_api_key = os.getenv("GOOGLE_API_KEY")

    if google_api_key:
        try:
            #Settings.llm = Gemini(model_name="models/gemini-1.5-flash-latest", api_key=google_api_key)
            Settings.llm = Gemini(model_name="models/gemini-2.5-flash", api_key=google_api_key)
            Settings.embed_model = GoogleGenerativeAIEmbedding(model_name="models/embedding-001", api_key=google_api_key)
            Settings.node_parser = SentenceSplitter(chunk_size=1024, chunk_overlap=100)
            st.success("API Key accepted. Ready to process documents.", icon="✅")
        except Exception as e:
            st.error(f"Failed to configure models: {e}", icon="⛔️")
            st.stop()

    st.header("Document Upload")
    uploaded_file = st.file_uploader(
        "Upload your document here.",
        type=["pdf"],
        disabled=not google_api_key
    )

    if uploaded_file:
        if st.button("Process Document"):
            st.session_state.messages = [{"role": "assistant", "content": f"Processing `{uploaded_file.name}`... Ask me anything!"}]
            file_bytes = uploaded_file.getvalue()
            # Pass only the necessary arguments
            index = build_index_from_file(file_bytes, uploaded_file.name)
            if index:
                st.session_state.chat_engine = get_chat_engine(index)
            else:
                st.error("Failed to process the document. Please check if it contains selectable text.")

# --- Main Chat Interface ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask a question about the document..."):
    if not st.session_state.chat_engine:
        st.warning("Please upload and process a document first.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.chat_engine.chat(prompt)
            answer = response.response
            citations = []
            if response.source_nodes:
                for node in response.source_nodes:
                    page_number = node.metadata.get('page_label', 'N/A')
                    #page_number = node.metadata.get('page_label')
                    file_name = node.metadata.get('file_name')
                    citations.append(f"Page {page_number}")
                    citations.append(f"File name {file_name}")

            unique_citations = sorted(list(set(citations)))
            citation_text = f"\n\n*Sources: {', '.join(unique_citations)}*" if unique_citations else ""

            full_response = f"{answer}{citation_text}"
            st.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})















