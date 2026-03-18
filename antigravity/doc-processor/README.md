***

# 🛡️ Local Offline Document Analysis Dashboard

A 100% local, privacy-first document chat application. Analyze PDFs, Excel spreadsheets, Word documents, and PowerPoints without sending a single byte of data to the internet. 

This project uses a hybrid architecture, smartly switching between full-context processing for smaller files and Retrieval-Augmented Generation (RAG) for massive datasets to optimize performance on mid-tier hardware.


## 🚀 Key Features

* **Smart Context Bypass:** Automatically bypasses the vector database for documents under 120,000 characters, feeding the entire text directly to the LLM for 100% accurate, global analysis.
* **Dynamic RAG Scaling:** For massive files (>120k characters), the system automatically downscales chunk sizes (to 10k) and retrieval limits to prevent memory hangs and maintain speed.
* **Transparent Citations:** Every answer includes a "Sources" section detailing the exact text snippets, page numbers, or Excel rows used.
* **Live UI Heartbeats:** Streamlit progress indicators provide real-time status updates during document indexing and long inference periods.
* **Broad File Support:** Natively processes `.pdf`, `.csv`, `.xlsx`, `.docx`, and `.pptx` files.
* **Auto-Reset Indexing:** Utilizes a distinct temp directory or Chroma collection index to ensure fresh memory when new files are uploaded.

---

## 🛠️ Technology Stack

| Component | Technology | Purpose |
| :--- | :--- | :--- |
| **Frontend UI** | Streamlit | Chat interface, sidebar, and live status updates |
| **Orchestration** | LangChain | Managing the data pipelines and document loaders |
| **LLM Engine** | Ollama | Running open-weights AI models locally |
| **Chat Model** | Llama 3.2 (3B) | Conversational AI (configured with a 32k context window) |
| **Embeddings** | nomic-embed-text | Translating raw text into searchable vectors |
| **Vector DB** | ChromaDB | Local storage and retrieval of document embeddings |

---

## ⚙️ Installation & Setup

Because this application runs entirely locally, you must install the AI engine (Ollama) on your machine before running the Python code.

1.  **Install Ollama:** Download and install the engine from [ollama.com](https://ollama.com/).
2.  **Download Local Models:** Open your terminal or PowerShell and run the following commands to pull the required AI models to your hard drive:
    * `ollama run llama3.2`
    * `ollama pull nomic-embed-text`
3.  **Set Up Python Environment:** Clone this repository and create a virtual environment (recommended).
4.  **Install Dependencies:** Install the required Python packages from the requirements file.
    * `pip install -r requirements.txt`
5.  **Launch the Dashboard:** Start the Streamlit server.
    * `streamlit run app.py`

---

## 📈 Performance & Scaling Notes

This application is specifically tuned to run smoothly on mid-level hardware (e.g., laptops with 16GB of RAM). 

By utilizing the lightweight `llama3.2` model, the system requires minimal memory overhead. Furthermore, the `num_ctx` parameter is explicitly increased to 32,000 tokens in the application logic. This ensures that when the "Smart Bypass" feeds entire Excel sheets or documents directly to the model, the data is not truncated, allowing for accurate aggregate queries (like calculating total sums across hundreds of rows). 

---

## 🧪 Verification & Testing

If you are contributing to this repository, please manually verify the following after making changes:
* Verify the Streamlit sidebar renders correctly and accepts all supported file types.
* Check the UI for the "Rows indexed" metric to confirm data completeness upon upload.
* Test a file larger than 120,000 characters to ensure the Dynamic RAG scaling engages without hanging the system.
* Confirm the AI streams its output word-by-word and successfully appends the citation block at the end of the response.
