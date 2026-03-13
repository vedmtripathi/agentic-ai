# 🤖 Agentic AI & Machine Learning Portfolio

[![Location](https://img.shields.io/badge/Location-Stockholm%20|%20Nordics%20|%20Europe-blue.svg)](#)
[![AWS Certified](https://img.shields.io/badge/Cloud-AWS-FF9900?logo=amazonaws&logoColor=white)](#)
[![Java 21](https://img.shields.io/badge/Backend-Java%2021-ED8B00?logo=java&logoColor=white)](#)
[![Python](https://img.shields.io/badge/AI-Python-3776AB?logo=python&logoColor=white)](#)

Welcome to my Agentic AI workspace. I am a **Backend & AI Engineer** bridging the gap between robust, traditional enterprise architecture (Java/AWS/Microservices) and modern autonomous AI systems (Python/LangGraph/RAG). 

My focus is on building **Stateful AI Agents**, highly efficient data pipelines, and Human-in-the-Loop automated workflows. I prioritize production-ready code, focusing heavily on API rate-limit management, token-cost optimization, and secure cloud architecture.

---

## 🛠️ Core Competencies & Tech Stack

### AI & Machine Learning
* **Agentic Frameworks:** LangGraph (State Machines & Actor/Worker models), LangChain, CrewAI.
* **RAG & Vector Search:** Local & Cloud Vector Databases (FAISS, Chroma), Semantic Search, Embeddings (Google Generative AI, OpenAI).
* **LLMs & Prompt Engineering:** Gemini API (2.0 Flash, 3.1 Pro), Claude 3.5 Sonnet, PydanticAI for strict JSON schema enforcement.
* **Development Workflows:** Agent-First IDEs (Cursor, Google Antigravity), Claude Code for CLI automation.

### Backend & Cloud Infrastructure
* **Languages:** Java 21, Python.
* **Architecture:** Microservices, Event-Driven Architecture, API Design.
* **Cloud & DevOps:** AWS, Cloud Cost Management (Budgets, API Quotas), CI/CD.
* **Frontend Integration:** Streamlit for rapid AI dashboard prototyping.

---

## 🚀 Featured Project: Inbox Genius AI (Agentic Copilot)
An advanced, stateful AI agent that connects to IMAP, semantically categorizes thousands of emails, and provides a RAG-powered chat interface to interact with your inbox locally.

**Architectural Highlights:**
* **Lazy Loading & Memory Protection:** Implemented a two-stage IMAP fetch strategy (`BODYSTRUCTURE` and headers first) to calculate metadata and group emails without downloading heavy attachments.
* **LangGraph State Machine:** Engineered a stateful workflow allowing the AI to categorize senders and propose bulk actions (Move/Delete) while enforcing a "Human-in-the-Loop" approval step.
* **Cost-Aware AI:** Hardcoded token-counting and cost-estimation tracking (INR/USD) into the UI prior to FAISS embedding execution. Kept processing costs strictly minimized by utilizing high-speed **Gemini 2.0 Flash** for heavy data ingestion and separating the reasoning engine from the data processing layer.
* **Resilient API Handling:** Implemented exponential backoff and batch-chunking to gracefully handle HTTP 429 (`ResourceExhausted`) rate limits across global API regions.

---

## 🧠 Engineering Philosophy
I believe the future of software isn't just "chatbots," but autonomous agents that act as digital employees. However, AI in production requires the same rigor as traditional backend engineering:
1. **Data over Magic:** An LLM is only as good as the context you provide it. Proper document chunking and metadata filtering in RAG pipelines are more important than the model itself.
2. **Cost is an Architecture Metric:** I design systems that separate expensive "reasoning" models from cheap "reading/summarization" models to protect cloud billing.
3. **Structured Outputs:** Production APIs cannot rely on raw text generation. I leverage tools like PydanticAI to force strict schema adherence.

---

## 📫 Let's Connect
I am actively exploring **AI Engineering, Backend, and MLOps roles** in **Stockholm, Sweden**, the broader **Nordics**, and **Europe**. 

If your team is looking to integrate scalable, cost-efficient Agentic AI into your enterprise architecture, let's talk.

* 💼 **LinkedIn:** [https://www.linkedin.com/in/manved/]
* 🐙 **GitHub:** [https://github.com/vedmtripathi/agentic-ai](https://github.com/vedmtripathi/agentic-ai)
