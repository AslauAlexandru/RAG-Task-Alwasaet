

import streamlit as st
import logging
import os
import tempfile
import io
# Variant II - LlamaParseReader
# LlamaIndex imports
from llama_index.core import VectorStoreIndex, Settings, SimpleDirectoryReader
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding as GoogleGenerativeAIEmbedding
#from llama_index.readers.file import LlamaParseReader # For LlamaParse
from llama_cloud_services import LlamaParse, EU_BASE_URL  # For LlamaParse
from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---
# Streamlit page configuration
st.set_page_config(
    page_title="Document Assistant Pro",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Helper Functions & Core Logic ---

@st.cache_resource(show_spinner="Parsing and Indexing your document...")
def build_index_from_file(file_bytes, file_name, llama_cloud_api_key):
    """
    Builds a LlamaIndex VectorStoreIndex from file bytes using LlamaParse.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file_bytes)
        temp_file_path = temp_file.name

    try:
        #parser = LlamaParseReader(api_key=llama_cloud_api_key, result_type="markdown")
        parser = LlamaParse(api_key=llama_cloud_api_key, base_url=EU_BASE_URL, result_type="markdown",
                                  user_prompt="If the input is not in English, translate the output into English.")
        documents = parser.load_data(file_path=temp_file_path)

        # Add filename to metadata for citations
        for doc in documents:
            doc.metadata["file_name"] = file_name

        logger.info(f"Successfully parsed {len(documents)} documents with LlamaParse.")

        # Create the index (will use global Settings for chunking, embedding)
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

    # Using a memory buffer to retain conversation history
    memory = ChatMemoryBuffer.from_defaults(token_limit=3000)

    chat_engine = index.as_chat_engine(
        chat_mode="context",
        memory=memory,
        system_prompt=system_prompt,
        verbose=True,
        streaming=True,
    )
    return chat_engine


# --- Streamlit UI ---
st.title("RAG - 📄 Document Chat Assistant (LlamaParse)")
st.caption("Upload a document, ask questions, and get cited answers.")

# Initialize session state for messages and the chat engine
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! Upload a document and I'll help you with it."}]
if "chat_engine" not in st.session_state:
    st.session_state.chat_engine = None

# --- Sidebar for API Keys and Upload ---  
with st.sidebar:
    st.header("Configuration")
    #google_api_key = st.text_input("Enter your Google API Key", type="password", key="google_api_key")
    google_api_key = os.getenv("GOOGLE_API_KEY")
    #llama_cloud_api_key = st.text_input("Enter your LlamaCloud API Key (for LlamaParse)", type="password", key="llama_cloud_api_key")
    llama_cloud_api_key = os.getenv("LLAMA_CLOUD_API_KEY")


    if google_api_key and llama_cloud_api_key:
        try:
            # Configure LlamaIndex settings globally
            #Settings.llm = Gemini(model_name="models/gemini-1.5-flash-latest", api_key=google_api_key)
            Settings.llm = Gemini(model_name="models/gemini-2.5-flash", api_key=google_api_key)
            Settings.embed_model = GoogleGenerativeAIEmbedding(model_name="models/embedding-001", api_key=google_api_key)
            Settings.node_parser = SentenceSplitter(chunk_size=1024, chunk_overlap=100)
            st.success("API Keys accepted. Ready to process documents.", icon="✅")
        except Exception as e:
            st.error(f"Failed to configure models: {e}", icon="⛔️")
            st.stop()

    st.header("Document Upload")
    uploaded_file = st.file_uploader(
        "Upload your document here.",
        type=["pdf"],
        disabled=not (google_api_key and llama_cloud_api_key)
    )

    if uploaded_file:
        if st.button("Process Document"):
            # Reset chat
            st.session_state.messages = [{"role": "assistant", "content": f"Processing `{uploaded_file.name}`... Ask me anything!"}]

            file_bytes = uploaded_file.getvalue()
            index = build_index_from_file(file_bytes, uploaded_file.name, llama_cloud_api_key)
            st.session_state.chat_engine = get_chat_engine(index)


# --- Main Chat Interface ---
# Display chat messages from history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Accept user input
if prompt := st.chat_input("Ask a question about the document..."):
    if not st.session_state.chat_engine:
        st.warning("Please upload and process a document first.")
        st.stop()

    # Add user message to display
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.chat_engine.chat(prompt)

            # Format the response with citations
            answer = response.response
            citations = []
            if response.source_nodes:
                for node in response.source_nodes:
                    page_number = node.metadata.get('page_label', 'N/A')
                    #page_number = node.metadata.get('page_label')
                    file_name = node.metadata.get('file_name')
                    citations.append(f"Page {page_number}")
                    citations.append(f"File name {file_name}")

            # Remove duplicates and format
            unique_citations = sorted(list(set(citations)))
            citation_text = f"\n\n*Sources: {', '.join(unique_citations)}*" if unique_citations else ""

            full_response = f"{answer}{citation_text}"
            st.markdown(full_response)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})










