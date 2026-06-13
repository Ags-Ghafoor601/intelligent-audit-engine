# ⚡ Intelligent Enterprise Audit Engine

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B)
![LangChain](https://img.shields.io/badge/Framework-LangChain-green)
![Groq](https://img.shields.io/badge/LLM-Groq%20Llama%203.3-orange)

An advanced, fault-tolerant Retrieval-Augmented Generation (RAG) agent designed for mission-critical document auditing. This engine bypasses the limitations of standard semantic search by implementing a high-performance Hybrid Retrieval pipeline, Reciprocal Rank Fusion (RRF), and AI Cross-Encoder re-ranking.



## 🧠 Core Architecture
Standard RAG systems suffer from keyword blindness and contextual hallucinations. This engine solves those bottlenecks through a multi-stage data pipeline:
1. **Hybrid Retrieval:** Executes parallel searches using **FAISS** (for deep semantic meaning) and **BM25** (for exact keyword/ID matching).
2. **Mathematical Fusion (RRF):** Merges distinct search results using a custom Reciprocal Rank Fusion algorithm to guarantee zero data loss.
3. **Cross-Encoder Re-Ranking:** A secondary AI (`ms-marco-MiniLM-L-6-v2`) ruthlessly scores and filters the retrieved chunks, feeding only the highest-quality evidence to the primary LLM.
4. **Verifiable Citations:** PyMuPDF extracts strict layout metadata, forcing the model to explicitly cite the exact Source File and Page Number for every claim.

## ✨ Features
- **Dual-Engine LLM Routing:** Hot-swap between `llama-3.3-70b-versatile` (deep reasoning) and `llama-3.1-8b-instant` (high speed).
- **SaaS-Grade UI:** High-contrast dark mode, glassmorphism chat bubbles, CSS keyframe animations, and real-time system telemetry.
- **Graceful Degradation:** Built-in safeguards auto-generate required directories and catch API rate-limit timeouts without crashing.

## ⚙️ Installation & Setup

**1. Clone the repository**<br>
git clone [https://github.com/yourusername/intelligent-audit-engine.git](https://github.com/yourusername/intelligent-audit-engine.git)
<br>cd intelligent-audit-engine

**2. Create a virtual environment and install dependencies**

python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt

**3. Configure Environment Variables**

Create a .env file in the root directory and add your Groq API key:
GROQ_API_KEY=your_actual_api_key_here

**4. Run the Engine**

streamlit run app.py

📂 Usage
1. Launch the application.
2. The system will automatically create a documents/ folder in your root directory if one does not exist.
3. Drop your PDF files (invoices, legal contracts, compliance policies) into the documents/ folder.
4. Click "Initialize Pipeline" in the sidebar telemetry dashboard.
5. Enter your complex audit queries in the chat interface and expand the "View Verified Source Evidence" tab to verify the AI's citations.

System Architect: Abdul Ghafoor | Lead Data Engineer

Developed for the Softora: www.softorapk.com