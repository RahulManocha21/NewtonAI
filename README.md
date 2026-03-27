# Newton AI

Newton is a personal AI assistant built for **Rahul Manocha's** portfolio. It answers questions about Rahul's background, experience, projects, and skills using a RAG (Retrieval-Augmented Generation) pipeline powered by Groq's LLaMA 3.3 model.

## Features

- Conversational AI assistant with chat history
- RAG pipeline — answers grounded in Rahul's resume, PDFs, and portfolio data
- Loads context from PDFs, CSVs, and live web pages (GitHub, LinkedIn, portfolio)
- Vector store cached locally with automatic refresh when content changes
- Built with Streamlit for an interactive chat UI

## Tech Stack

| Layer | Library |
|---|---|
| LLM | `langchain-groq` — LLaMA 3.3 70B via Groq |
| Embeddings | `langchain-huggingface` — `BAAI/bge-small-en-v1.5` |
| Vector Store | `FAISS` |
| Chain | LangChain LCEL (`RunnablePassthrough` + `StrOutputParser`) |
| Document Loaders | `PyPDFLoader`, `CSVLoader`, `WebBaseLoader` |
| UI | `Streamlit` |

## Project Structure

```
NewtonAI/
├── app.py               # Main Streamlit app
├── requirements.txt     # Python dependencies
├── Content/             # Source documents (PDFs, CSVs)
└── vector_store/        # Cached FAISS vector store (auto-generated)
```

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/RahulManocha21/NewtonAI.git
   cd NewtonAI
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate   # Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Add your API keys to `.streamlit/secrets.toml`:
   ```toml
   GROQ_API_KEY = "your_groq_api_key"
   HF_API_KEY = "your_huggingface_api_key"
   LANGCHAIN_TRACING_V2 = "true"
   LANGCHAIN_ENDPOINT = "https://api.smith.langchain.com"
   LANGCHAIN_API_KEY = "your_langsmith_api_key"
   LANGCHAIN_PROJECT = "your_project_name"
   ```

5. Run the app:
   ```bash
   streamlit run app.py
   ```

## How It Works

1. On startup, Newton loads documents from `Content/` (PDFs, CSVs) and live URLs (GitHub, LinkedIn, portfolio site)
2. Documents are split into chunks and embedded using `BAAI/bge-small-en-v1.5`
3. Embeddings are stored in a local FAISS vector store — regenerated automatically if content changes
4. User questions are answered using an LCEL retrieval chain: relevant chunks are fetched and passed as context to LLaMA 3.3 via Groq
