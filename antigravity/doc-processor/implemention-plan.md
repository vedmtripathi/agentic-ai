Local Offline Document Analysis Dashboard Implementation Plan
The goal is to build a private, local document chat application using Python.

Proposed Changes
Configuration & Dependencies
[NEW] 
requirements.txt
Define essential packages: streamlit, langchain, langchain-community, langchain-ollama, chromadb, pypdf, pandas, openpyxl.

Application Logic
[NEW] 
app.py
This will be the main entry point, containing:

UI Components: Sidebar for file uploads, chat window, and citations.
Document Loading: logic for .pdf, .csv, .xlsx, .docx, and .pptx.
Vector DB Logic: Integrating Chroma with OllamaEmbeddings (nomic-embed-text).
Chat Chain: Using ChatOllama (llama3.2) with streaming support.
Verification Plan
Automated Tests
As this is a local setup with Ollama, manual verification will be needed for end-to-end functionality.
I will ensure the code logic for file loaders and chunking is correct.
Manual Verification
Dependency check: Ensure all packages install correctly.
UI check: Verify sidebar shows up and main panel has chat interface.
File processing: Test uploading a sample PDF/CSV/DOCX/PPTX.
Chat check: Confirm streaming output and "Sources" section appear correctly.
UI Feedback: Verify progress bars or status indicators show up during indexing and retrieval.
Accuracy Check: Ensure tabular data (Excel/CSV) retrieves enough rows for aggregate questions (Total/Sum).
Full Scan: Verify that small Excel files are read in their entirety (including all sheets) to prevent missing rows.
Context Retention: Increase num_ctx to 32k to ensure retrieved tables aren't truncated by the model.
Verification UI: Display "Rows indexed" for each file so the user can verify data completeness.
Smart Bypass: If total document size < 120,000 chars, bypass RAG and feed the entire text directly to the LLM for 100% accurate global analysis.
Reset Logic: Use a distinct Chroma collection index or temp directory to ensure fresh indexing when files change.
Performance & Scaling: Handling Large Tables
To prevent hangs during complex analysis of 1000+ row files, we will implement the following:

Aggressive Full Context: Increase total_chars threshold to 120,000 (~25k tokens). llama3.2 with 32k context can handle this easily, and it's much faster than RAG for global analysis.
RAG Downscaling: If files are > 120k chars:
Reduce k from 15 to 5.
Reduce table chunk_size from 50k to 10k.
This keeps the retrieval window within the reliable ~30k token range.
UI Heartbeats: Use st.empty() to show live status updates (e.g., "Step 1: Reading data...", "Step 2: Processing...") to provide visual feedback during long inference.
Verification Plan
Manual Verification
Stress Test: Upload a 1000+ row Excel file.
Answer Quality: Verify the AI can analyze duplicates across the whole file without hanging.
