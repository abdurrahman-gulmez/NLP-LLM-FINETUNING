# 🧠 Sectoral ReAct Agent: Python Data Science Assistant

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Framework-Streamlit-red)
![LLM](https://img.shields.io/badge/Model-Llama--3.1--8b-green)

## 📖 Overview
This project transforms a standard Large Language Model (LLM) into an **Autonomous ReAct Agent** specialized in **Python Data Science & Image Processing**.

Unlike standard RAG systems, this agent acts as an **orchestrator**. It follows a **Thought -> Action -> Observation** cycle to reason about user queries, utilizing official technical documentation (OpenCV, NumPy, Matplotlib) as tools to provide accurate, hallucination-free answers.

## 🏗 Architecture
The system uses **Agentic RAG** architecture:
1.  **The Brain:** Llama-3.1-8b-instant (via Groq API).
2.  **The Limbs (Tools):**
    * `search_docs`: Retrieves raw context from ChromaDB (Vector Store).
    * `web_search`: DuckDuckGo search for real-time info.
    * `calculator`: Safe mathematical operations.
3.  **The Knowledge:** Official PDF documentation chunks indexed in ChromaDB.

## 🚀 Features
* **ReAct Loop:** Displays the agent's reasoning process (Thoughts) in real-time.
* **Smart Loop Protection:** Prevents the agent from getting stuck in repetitive search cycles.
* **Multi-Hop Reasoning:** Can solve complex tasks combining retrieval and calculation.
* **Trace Logging:** Automatically saves the thought chain to `logs/agent_trace.log`.

## 🛠 Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/react-agent-project.git](https://github.com/YOUR_USERNAME/react-agent-project.git)
    cd react-agent-project
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Setup Environment:**
    Create a `.env` file in the root directory and add your Groq API key:
    ```env
    GROQ_API_KEY=gsk_your_api_key_here
    ```

4.  **Add Documents:**
    Place your PDF files (e.g., `opencv.pdf`, `numpy.pdf`) into the `docs/` folder.

## ▶️ Usage

Run the Streamlit application:
```bash
streamlit run app.py
