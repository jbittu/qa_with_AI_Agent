import google.generativeai as genai
from dotenv import load_dotenv
import os
from src.vectorstore import ChromaVectorStore


load_dotenv()


class RAGSearch:
    def __init__(self, persist_dir="chroma_store"):
        """
        Loads vector DB (Chroma) and initializes Gemini LLM
        """
        print(f"[INFO] Loading Chroma Vector Store from: {persist_dir}")
        self.vectorstore = ChromaVectorStore(persist_dir=persist_dir)
        self.vectorstore.load()    

        # Initialize Gemini
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        self.model = genai.GenerativeModel("gemini-2.5-flash")
        print("[INFO] Using Gemini 2.5 Flash as LLM")

    def _compose_context(self, docs):
        """Concatenate retrieved metadata text to form context"""
        return "\n\n".join([d["metadata"]["text"] for d in docs if d["metadata"].get("text")])

    def search_and_answer(self, query, top_k=5):
        """Retrieve context → send to Gemini → return structured answer"""
        docs = self.vectorstore.query(query, top_k=top_k)
        context = self._compose_context(docs)

        if not context:
            return {"answer": "No relevant data found in documents.", "sources": []}

        prompt = f"""
You are an AI assistant. Use the context below to answer the question.

If answer is not fully found in the context, you may use your general knowledge.

Context:
{context}

Question:
{query}
"""

        response = self.model.generate_content(prompt)

        # handle text output safely
        final_text = getattr(response, "text", None) or response.candidates[0].content.parts[0].text

        return {
            "answer": final_text,
            "sources": docs
        }
