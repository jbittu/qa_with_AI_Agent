# RAG Agent Complete

Local Retrieval-Augmented Generation pipeline using:
- Document loaders (PDF/TXT/CSV/DOCX/XLSX)
- Chunking using LangChain text splitters
- Embeddings with SentenceTransformers (all-MiniLM)
- FAISS vectorstore for nearest-neighbor search
- Simple RAG agent with Plan → Retrieve → Answer → Reflect

## Setup

1. Create a Python virtual environment (recommended)
```bash
python -m venv venv
source venv/Scripts/activate       
pip install -r requirements.txt
```
