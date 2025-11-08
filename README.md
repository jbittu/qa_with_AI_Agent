# qa_with_AI_Agent — RAG (Retrieval-Augmented Generation) Pipeline

A complete RAG (Retrieval-Augmented Generation) pipeline built with:

* ✅ Multiple document loader support (PDF, TXT, CSV, DOCX, XLSX)
* ✅ Chunking using LangChain text splitters
* ✅ Local embeddings using `SentenceTransformer` (`all-MiniLM-L6-v2`)
* ✅ Chroma Vector Database for semantic search
* ✅ Gemini 2.5 Flash for answer generation
* ✅ RAG Pipeline → **PLAN → RETRIEVE → ANSWER → REFLECT**
* ✅ Works on **Windows / Linux / Mac** — No GPU needed


## Architecture Flow

```
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

```
qa_with_AI_Agent/
│
├── data/ # PDFs, TXT docs
├── chroma_store/
├── src/
│ ├── data_loader.py # Loads docs
│ ├── embedding.py # Chunk + embedding generation
│ ├── vectorstore.py # Chroma DB wrapper
│ ├── search.py # Retrieval + LLM answering
│ ├── agent.py # PLAN → RETRIEVE → ANSWER → REFLECT
│
├── ingest.py # Build vector DB
├── app.py
├── main_app.py
├── .env
├── requirements.txt
└── README.md
```

## ⚙️ Setup

### Prerequisites

* Python 3.9+
* `git` installed
* Internet access for downloading packages and for the Gemini API

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

Create `.env` in project root:

```bash
GOOGLE_API_KEY=your_api_key_here
```


## 🔁 Clone the repository

You can clone this repository using HTTPS or SSH. Replace the URL below with the repository URL if you have forked it.

**Using HTTPS**

```bash
git clone https://github.com/jbittu/qa_with_AI_Agent.git
cd qa_with_AI_Agent
```

**Using SSH** (requires SSH key setup in your GitHub account)

```bash
git clone git@github.com:jbittu/qa_with_AI_Agent.git
cd qa_with_AI_Agent
```

If you want to contribute, it's a good idea to create a feature branch:

```bash
git checkout -b feat/your-feature-name
```

## 🗂 Ingest Documents

Add files to the `data/` folder (PDF/TXT/CSV/DOCX/XLSX), then run:

```bash
python ingest.py
```

You should see:

```
✅ Embeddings generated
✅ Chroma DB created at: chroma_store
```

---

## 4️⃣ Run CLI Version

```bash
python app.py
```

## 🖥 Run Streamlit UI

```bash
streamlit run main_app.py
```

**Deployed Streamlit demo (if available):**

* Add your Streamlit deployment URL here (e.g. `https://ragwithpdfapplication.streamlit.app/`) to share a live demo.

## ✅ Try asking

```bash
- "What are the drawbacks of climate change?"
- "What is machine learning and its features?"
```

## 🔧 Features

| Feature                            | Status |
| ---------------------------------- | ------ |
| Load multiple document formats     | ✅      |
| SentenceTransformer Embeddings     | ✅      |
| Chroma Vector DB (persisted)       | ✅      |
| Gemini 2.5 Flash LLM RAG Answering | ✅      |
| Reflection score on relevance      | ✅      |
| Streamlit Web UI                   | ✅      |


##  Contributing

Contributions welcome! Please open issues or PRs. Suggested workflow:

1. Fork the repo
2. Create a feature branch (`git checkout -b feat/my-feature`)
3. Commit and push
4. Open a Pull Request


## 🔗 Links

* Repository: `https://github.com/jbittu/qa_with_AI_Agent`
