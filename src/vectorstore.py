from langchain_chroma import Chroma 
from langchain.embeddings.base import Embeddings
from sentence_transformers import SentenceTransformer


class HuggingFaceEmbedding(Embeddings):
    """Wrapper so Chroma can use SentenceTransformer for embedding queries"""

    def __init__(self, model_name="all-MiniLM-L6-v2"):
        print(f"[INFO] Loading embedding model for query stage: {model_name}")
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        return self.model.encode(texts).tolist()

    def embed_query(self, query):
        return self.model.encode([query]).tolist()[0]


class ChromaVectorStore:
    def __init__(self, persist_dir="chroma_store"):
        self.persist_dir = persist_dir
        self.embedding = HuggingFaceEmbedding()
        self.db = None

    def build(self, chunks, embeddings):
        print("[INFO] Creating Chroma DB using prepared chunks + embeddings...")

        # Initialize DB
        self.db = Chroma(
            persist_directory=self.persist_dir,
            embedding_function=self.embedding
        )

        # Use internal collection add() (correct API for Chroma ≥ 0.5)
        self.db._collection.add(
            embeddings=embeddings.tolist(),
            metadatas=[{"text": c.page_content} for c in chunks],
            documents=[c.page_content for c in chunks],
            ids=[str(i) for i in range(len(embeddings))]
        )

        print(f"[INFO] Chroma DB persisted at: {self.persist_dir}")

    def load(self):
        self.db = Chroma(
            persist_directory=self.persist_dir,
            embedding_function=self.embedding,
        )
        print(f"[INFO] Loaded existing Chroma DB from: {self.persist_dir}")

    def query(self, query_text: str, top_k: int = 5):
        results = self.db.similarity_search(query_text, k=top_k)

        return [{
            "metadata": {"text": r.page_content}
        } for r in results]
