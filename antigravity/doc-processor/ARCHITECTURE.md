***

# 🛡️ Local Offline Document Analysis Dashboard

A 100% local, privacy-first document chat application designed to run smoothly on mid-tier hardware. Analyze PDFs, Excel spreadsheets, Word documents, and PowerPoints without sending a single byte of data to the internet. 

This project uses a hybrid architecture, smartly switching between full-context processing for smaller files and Retrieval-Augmented Generation (RAG) for massive datasets.

---

## 🧠 Why This Tech Stack? (Architecture Justification)

Building a local AI tool requires carefully balancing capability with hardware constraints (specifically targeting machines with ~16GB of RAM). Here is why these specific tools were chosen over the alternatives:

### 1. The LLM: Llama 3.2 (3B) via Ollama
* **Why not Llama 3 (8B) or Cloud APIs?** Cloud APIs (like OpenAI or Gemini) violate the strict 100% offline privacy constraint. However, running a standard 8B local model consumes ~5.5GB of RAM. On a 16GB laptop running an OS, a browser, and an IDE, an 8B model will force the system into "swap memory," causing severe system lag. 
* **The Solution:** **Llama 3.2 (3B)** takes up only ~2GB of RAM. It is incredibly fast and highly optimized for following strict instructions, making it the perfect lightweight engine for extracting answers from retrieved text (RAG) without freezing the host machine.

### 2. The Vector Database: ChromaDB 
* **Why not FAISS?** FAISS (by Meta) is a low-level, high-performance math library. It runs entirely in RAM (meaning it forgets your documents when you close the app) and it **does not natively store metadata**. To get page numbers or Excel row citations using FAISS, we would have to build and maintain a parallel SQLite database from scratch.
* **The Solution:** **ChromaDB** is a fully assembled vector database. It features built-in persistence (saving your embedded documents to a local `./chroma_db` folder so they don't need to be re-read every time) and seamlessly pairs vectors with their metadata (the exact page/row numbers required for our citation feature).

### 3. The Embedding Model: `nomic-embed-text`
* **Why this model?** AI cannot read English; it reads numbers. `nomic-embed-text` is a highly efficient, tiny (~300MB) model specifically trained to translate long-form documents into 768-dimensional mathematical vectors. It runs natively through Ollama, ensuring our text-to-math pipeline remains entirely offline.

### 4. The Frameworks: Streamlit & LangChain
* **Streamlit:** Chosen for its rapid frontend capabilities. It provides out-of-the-box support for file uploads, chat interfaces, and live UI heartbeats (`st.empty()`) to show users exactly what the local processor is doing.
* **LangChain:** Acts as the orchestration layer. It handles the complex logic of parsing messy Excel/PDF files, splitting them into overlapping chunks, and routing the data between ChromaDB and Ollama.

---

## ⚡ Smart Context Scaling

To prevent system hangs during complex analysis, this application implements a dynamic routing system:
* **Smart Bypass (< 120,000 chars):** Small documents bypass the RAG database entirely. The full text is fed directly into Llama 3.2's expanded 32k context window for 100% accurate global analysis (e.g., "Sum up all the rows in this Excel file").
* **RAG Downscaling (> 120,000 chars):** For massive files, the system automatically falls back to ChromaDB, reducing chunk sizes to 10k and retrieval limits to keep the memory footprint small and inference speeds high.

---

## 🛠️ Installation & Setup

Because this application runs entirely locally, you **must** install the AI engine (Ollama) on your machine before running the Python code.

### Step 1: Install the Local AI Engine
1. Download and install **Ollama** from [ollama.com](https://ollama.com/).
2. Open your terminal (or PowerShell on Windows) and download the required models to your hard drive:
   ```bash
   ollama run llama3.2
   ```
   *(Wait for the download to finish, type `/bye` to exit the prompt, then run:)*
   ```bash
   ollama pull nomic-embed-text
   ```

### Step 2: Set Up the Python Environment
1. Clone this repository to your local machine.
2. Navigate to the project folder and create a virtual environment (recommended to avoid dependency conflicts):
   ```bash
   python -m venv venv
   # On Windows use: venv\Scripts\activate
   # On Mac/Linux use: source venv/bin/activate
   ```
3. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

### Step 3: Launch the Dashboard
Start the Streamlit server. The dashboard will automatically open in your default web browser.
```bash
streamlit run app.py
```

---

## 🧪 Verification & Testing
Upon launching, verify the installation by uploading a small sample PDF. 
1. Ensure the UI displays the "Rows/Pages indexed" status.
2. Ask a question about the document in the chat.
3. Verify that the answer **streams** word-by-word and that the **Sources** citation block appears at the bottom of the response containing the correct metadata.
