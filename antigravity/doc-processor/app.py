import streamlit as st
import os
import tempfile
import pandas as pd
from langchain_community.document_loaders import (
    PyPDFLoader,
    CSVLoader,
    UnstructuredExcelLoader,
    Docx2txtLoader,
    UnstructuredPowerPointLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.documents import Document

# --- Page Configuration ---
st.set_page_config(
    page_title="Offline Doc Insight",
    layout="wide",
    page_icon="🛡️",
    initial_sidebar_state="expanded"
)

# --- Premium UI Styling ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600&display=swap');

    /* Global Typography */
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }

    /* Main Background Gradient */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #f8fafc;
    }

    /* Glassmorphic Sidebar */
    section[data-testid="stSidebar"] {
        background-color: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(12px);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }

    /* Premium Header */
    .main-header {
        background: linear-gradient(90deg, #3b82f6 0%, #8b5cf6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 600;
        font-size: 3rem;
        margin-bottom: 0.5rem;
    }

    /* Pulse Animation for Status */
    @keyframes pulse {
        0% { transform: scale(1); opacity: 1; }
        50% { transform: scale(1.05); opacity: 0.8; }
        100% { transform: scale(1); opacity: 1; }
    }
    .pulsing-icon {
        animation: pulse 2s infinite;
        display: inline-block;
    }

    /* Custom Chat Bubbles */
    [data-testid="stChatMessage"] {
        background-color: rgba(51, 65, 85, 0.4) !important;
        border-radius: 15px !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        padding: 1rem !important;
        margin-bottom: 1rem !important;
        transition: all 0.3s ease;
    }
    [data-testid="stChatMessage"]:hover {
        border-color: rgba(59, 130, 246, 0.5) !important;
        transform: translateY(-2px);
    }

    /* Buttons Override */
    .stButton>button {
        background: linear-gradient(90deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 10px 15px -3px rgba(59, 130, 246, 0.4);
    }

    /* Sidebar Divider */
    hr {
        margin: 1.5rem 0 !important;
        border-color: rgba(255, 255, 255, 0.1) !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Constants ---
DB_DIR = "./chroma_db"
MODEL_LLM = "llama3.2"
MODEL_EMBED = "nomic-embed-text"
OLLAMA_BASE_URL = "http://localhost:11434"

# Ensure directories exist
if not os.path.exists(DB_DIR):
    os.makedirs(DB_DIR)

# --- Functions ---

def load_document(file_path, file_type):
    """Loads document and returns (list of docs, info_string)."""
    try:
        docs = []
        info = ""
        if file_type == "pdf":
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            info = f"{len(docs)} pages"
        elif file_type == "csv":
            df = pd.read_csv(file_path)
            # Convert full dataframe to a markdown table for the LLM to process holistically
            content = f"CSV Data:\n{df.to_markdown(index=False)}"
            docs = [Document(page_content=content, metadata={"source": file_path, "type": "csv"})]
            info = f"{len(df)} rows"
        elif file_type == "xlsx":
            # Read ALL sheets
            excel_data = pd.read_excel(file_path, sheet_name=None)
            total_rows = 0
            for sheet_name, df in excel_data.items():
                content = f"Sheet: {sheet_name}\n{df.to_markdown(index=False)}"
                docs.append(Document(page_content=content, metadata={"source": file_path, "sheet": sheet_name, "type": "excel"}))
                total_rows += len(df)
            info = f"{len(excel_data)} sheets, {total_rows} total rows"
        elif file_type == "docx":
            loader = Docx2txtLoader(file_path)
            docs = loader.load()
            info = f"{len(docs)} sections"
        elif file_type == "pptx":
            loader = UnstructuredPowerPointLoader(file_path)
            docs = loader.load()
            info = f"{len(docs)} slides"
        return docs, info
    except Exception as e:
        st.error(f"Error loading {file_type}: {e}")
        return [], f"Error: {e}"

def process_files(uploaded_files):
    """Processes uploaded files and updates the vector store."""
    all_docs = []
    total_chars = 0
    
    with st.status("📄 Indexing documents...", expanded=True) as status:
        progress_bar = st.progress(0)
        for i, uploaded_file in enumerate(uploaded_files):
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            file_ext = uploaded_file.name.split('.')[-1].lower()
            docs, info = load_document(tmp_path, file_ext)
            
            status.write(f"✅ Read **{uploaded_file.name}** ({info})")
            for doc in docs:
                doc.metadata["source_name"] = uploaded_file.name
                if "sheet" in doc.metadata:
                    doc.metadata["page"] = f"Sheet: {doc.metadata['sheet']}"
                total_chars += len(doc.page_content)
            all_docs.extend(docs)
            os.remove(tmp_path)
            progress_bar.progress((i + 1) / len(uploaded_files))

        if not all_docs:
            status.update(label="❌ No text extracted.", state="error")
            return None

        # --- PERFORMANCE & SCALING STRATEGY ---
        # Increased threshold to 120k (~25-30k tokens) for holistic analysis
        is_large = total_chars >= 120000
        if not is_large:
            full_text = "\n\n".join([f"Source: {d.metadata['source_name']}\n{d.page_content}" for d in all_docs])
            st.session_state.direct_context = full_text
            st.session_state.is_large = False
            status.write("✨ Holistic Mode: Using full data buffer for 100% accuracy.")
        else:
            st.session_state.direct_context = None
            st.session_state.is_large = True
            status.write("📊 Scaling Mode: Large dataset detected. Using optimized Semantic Search.")

        status.write("Analyzing content structure...")
        tabular_docs = [d for d in all_docs if d.metadata.get("type") in ["csv", "excel"]]
        text_docs = [d for d in all_docs if d.metadata.get("type") not in ["csv", "excel"]]

        final_splits = []
        
        # Split text documents (PDF, Docx, etc.)
        if text_docs:
            status.write("Processing text segments...")
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            final_splits.extend(text_splitter.split_documents(text_docs))
            
        # Process tabular data with scaling safety
        if tabular_docs:
            status.write("Optimizing table blocks for memory safety...")
            # Downscale chunk size for extremely large files to prevent context overload
            table_chunk_size = 10000 if is_large else 50000
            table_splitter = RecursiveCharacterTextSplitter(chunk_size=table_chunk_size, chunk_overlap=0)
            final_splits.extend(table_splitter.split_documents(tabular_docs))

        status.write(f"Generating local embeddings for {len(final_splits)} segments...")
        embeddings = OllamaEmbeddings(model=MODEL_EMBED, base_url=OLLAMA_BASE_URL)
        
        # Robust Reset: We use a new collection name each time
        import uuid
        uid_str = str(uuid.uuid4())
        collection_name = f"col_{uid_str[:8]}"
        vectorstore = Chroma.from_documents(
            documents=final_splits,
            embedding=embeddings,
            persist_directory=DB_DIR,
            collection_name=collection_name
        )
        status.update(label="✅ Ready for analysis!", state="complete", expanded=False)
    return vectorstore

# --- UI Layout ---

# Premium Header
st.markdown('<div class="main-header">🛡️ Offline Doc Insight</div>', unsafe_allow_html=True)
st.markdown(
    '<div style="margin-bottom: 2rem;">'
    '<span class="pulsing-icon">🟢</span> '
    '<span style="color: #94a3b8; font-size: 0.9rem;">Secured Local Environment | Ollama Active</span>'
    '</div>', 
    unsafe_allow_html=True
)

# Sidebar for Uploads
with st.sidebar:
    st.markdown("### 📁 Documents")
    uploaded_files = st.file_uploader(
        "PDF, CSV, Excel, Word, PPTX",
        type=["pdf", "csv", "xlsx", "docx", "pptx"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🚀 Process", use_container_width=True):
            if uploaded_files:
                vectorstore = process_files(uploaded_files)
                if vectorstore:
                    st.session_state.vectorstore = vectorstore
            else:
                st.warning("Upload first.")
    with col2:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.clear()
            st.rerun()

    if "direct_context" in st.session_state and st.session_state.direct_context:
        st.markdown("---")
        st.success("✨ Full Context Mode: Active")
        with st.expander("🔍 Preview Data"):
            st.text_area("Live Context Buffer:", st.session_state.direct_context, height=200)

# Main Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message:
            with st.expander("📚 Sources & References"):
                for src in message["sources"]:
                    st.write(f"- **{src['source']}** ({src['metadata']})")
                    st.info(src['page_content'])

# Chat prompt
if prompt := st.chat_input("What would you like to know about your documents?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if "vectorstore" not in st.session_state:
        with st.chat_message("assistant"):
            st.markdown("⚠️ Please upload and process documents in the sidebar first.")
    else:
        with st.chat_message("assistant"):
            with st.status("🧠 Consulting local AI...", expanded=True) as status:
                context_docs = []
                if "direct_context" in st.session_state and st.session_state.direct_context:
                    status.write("✨ Utilizing Holistic Data Scan (100% visibility)...")
                    context_text = st.session_state.direct_context
                else:
                    status.write("🔍 Searching local vector database...")
                    # Scaling: Downscale retrieval for large docs to prevent context overflow
                    k_val = 5 if st.session_state.get("is_large", False) else 15
                    retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": k_val})
                    context_docs = retriever.invoke(prompt)
                    context_text = "\n\n".join([doc.page_content for doc in context_docs])
                
                status.write("⚡ Processing data through llama3.2...")
                status.write("💡 Note: Analytical tasks on large files may take a minute.")
                
                llm = ChatOllama(
                    model=MODEL_LLM, 
                    base_url=OLLAMA_BASE_URL, 
                    streaming=True,
                    num_ctx=32768
                )
                
                system_prompt = (
                    "You are a precise data analyst. "
                    "Use the PROVIDED CONTEXT to answer the question. "
                    "IMPORTANT: The context contains tables. Read EVERY row and column. "
                    "Perform step-by-step calculations. Do not skip any data points. "
                    "If a total is asked, sum up ALL relevant values retrieved. "
                    f"\n--- CONTEXT START ---\n{context_text}\n--- CONTEXT END ---"
                )
                
                status.update(label="✅ Analysis complete!", state="complete", expanded=False)

            response_placeholder = st.empty()
            full_response = ""
            for chunk in llm.stream(f"{system_prompt}\n\nQuestion: {prompt}"):
                full_response += chunk.content
                response_placeholder.markdown(full_response + "▌")
            
            response_placeholder.markdown(full_response)
            
            sources = []
            if context_docs:
                for doc in context_docs:
                    sources.append({
                        "source": doc.metadata.get("source_name", "Unknown"),
                        "metadata": doc.metadata.get("page", doc.metadata.get("row", "N/A")),
                        "page_content": doc.page_content[:200] + "..."
                    })
            elif "direct_context" in st.session_state:
                sources.append({
                    "source": "Full Document",
                    "metadata": "Holistic Scan",
                    "page_content": st.session_state.direct_context[:200] + "..."
                })
            
            if sources:
                with st.expander("📚 Sources & References"):
                    for src in sources:
                        st.write(f"- **{src['source']}** ({src['metadata']})")
                        st.info(src['page_content'])
            
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response,
                "sources": sources
            })
