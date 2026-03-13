# 🧠 Inbox Genius AI

We all face the "Inbox Infinity" problem—thousands of unread emails, newsletters, and notifications burying what actually matters. I decided to stop clicking 'Unsubscribe' one by one and built a smarter solution.

Introducing **Inbox Genius AI**: An Agentic Copilot that doesn't just sort your mail—it understands it.

## 🚀 Key Features I Architected

* **Agentic Categorization:** Uses LangGraph to orchestrate Gemini 2.0 Flash for semantic sorting (Newsletters vs. VIPs vs. Action Items).
* **Local RAG (Retrieval-Augmented Generation):** Integrated a FAISS vector database so I can "chat" with my entire inbox to find specific details without manual searching.
* **Cost-Aware Engineering:** Built a real-time token cost tracker to monitor API spend (averaging only ₹10 per 1,000 emails!).
* **Privacy-First:** All vector embeddings live in local RAM—data evaporates the moment the session ends.

## 🛠️ Tech Stack

* **Core Logic:** Python, Antigravity
* **AI & Orchestration:** Gemini API, LangGraph, LangChain
* **Vector Database:** FAISS
* **Frontend UI:** Streamlit

## 🔗 Explore the Project

This project is located within the `antigravity/mailbox-organizer` directory of this repository. 

Check out the code and run the demo to see how Agentic AI can transform productivity!
