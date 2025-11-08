from typing import List, Any, Tuple
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np


class EmbeddingPipeline:
    def __init__(self, model_name="all-MiniLM-L6-v2", chunk_size=1000, chunk_overlap=200):
        print(f"[INFO] Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def process(self, documents: List[Any]) -> Tuple[List[Any], np.ndarray]:
        """Chunks documents and generates embeddings."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

        print(f"[INFO] Splitting {len(documents)} documents...")
        chunks = splitter.split_documents(documents)
        print(f"[INFO] Created {len(chunks)} document chunks")

        texts = [chunk.page_content for chunk in chunks]
        print(f"[INFO] Generating {len(texts)} embeddings...")

        embeddings = self.model.encode(texts, show_progress_bar=True)
        embeddings = np.array(embeddings)

        print(f"[INFO] Embeddings shape: {embeddings.shape}")
        return chunks, embeddings
