from src.data_loader import load_all_documents
from src.embedding import EmbeddingPipeline
from src.vectorstore import ChromaVectorStore


def run_ingest():
    docs = load_all_documents("data")

    embedder = EmbeddingPipeline()
    chunks, embeddings = embedder.process(docs)

    store = ChromaVectorStore()
    store.build(chunks, embeddings)

    print("[OK] Ingestion completed successfully!")


if __name__ == "__main__":
    run_ingest()
