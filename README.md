# 🔍 RAG Agent — Retrieval Augmented Generation (Local + Gemini)

A complete RAG (Retrieval-Augmented Generation) pipeline built with:

✅ Multiple document loader support (PDF, TXT, CSV, DOCX, XLSX)  
✅ Chunking using LangChain text splitters  
✅ Local embeddings using `SentenceTransformer` (`all-MiniLM-L6-v2`)  
✅ Chroma Vector Database for semantic search  
✅ Gemini 2.5 Flash for answer generation  
✅ RAG Pipeline → **PLAN → RETRIEVE → ANSWER → REFLECT**  
✅ Works on **Windows / Linux / Mac** — No GPU needed

### 🌐 Live Demo (Hosted on Streamlit)

👉 **🔗 https://ragwithpdfapplication.streamlit.app/**  
```bash
Try asking:
- *"What are the drawbacks of climate change?"*
- *"What is machine learning and its features?"*
```

## 🚀 Architecture Flow
```bash
User Question
↓
[ PLAN ]
↓
[ RETRIEVE ] → (using Chroma + MiniLM embeddings)
↓
[ ANSWER ] → (Gemini 2.5 Flash LLM Output)
↓
[ REFLECT ] → (Relevance evaluation)
↓
Final Output (Answer + Reflection + Sources)
```

## 📁 Project Structure
```bash
qa_with_AI_Agent/
│
├── data/ # PDFs, TXT, CSV docs
├── chroma_store/ # Vector DB (auto generated)
├── src/
│ ├── data_loader.py # Loads docs
│ ├── embedding.py # Chunk + embedding generation
│ ├── vectorstore.py # Chroma DB wrapper
│ ├── search.py # Retrieval + LLM answering
│ ├── agent.py # PLAN → RETRIEVE → ANSWER → REFLECT
│
├── ingest.py # Build vector DB
├── app.py # CLI Interface
├── main_app.py # Streamlit UI
├── .env # Gemini API key (ignored in Git)
├── requirements.txt
└── README.md
```
## ⚙️ Setup

### Create Virtual Environment & Install Dependencies

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate

pip install -r requirements.txt
```
### Add API key
Create .env in project root:
```bash
GOOGLE_API_KEY=your_api_key_here
```
### Ingest Documents
Add files to the data/ folder (PDF/TXT/CSV/DOCX/XLSX), then run:
```bash
python ingest.py
```
You should see:
```bash
✅ Embeddings generated
✅ Chroma DB created at: chroma_store
4️⃣ Run CLI Version
```
### Test the working
```bash
python app.py
```
### Run Streamlit UI
```bash
streamlit run main_app.py
```
Or use the live deployed version:

👉 https://ragwithpdfapplication.streamlit.app/

### Features
Feature	Status
Load multiple document formats	
SentenceTransformer Embeddings	
Chroma Vector DB (persisted)	
Gemini 2.5 Flash LLM RAG Answering	
Reflection score on relevance	
Streamlit Web UI	