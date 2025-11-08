from dataclasses import dataclass
from typing import List, Optional
from src.search import RAGSearch 


@dataclass
class AgentState:
    question: str
    retrieved: Optional[list] = None
    answer: Optional[str] = None
    reflection: Optional[str] = None


class RAGAgent:
    def __init__(self, persist_dir="chroma_store"):
        """
        Initializes the agent and loads Chroma vector DB + Gemini model.
        """
        print("\n Initializing RAG Agent (Gemini + Chroma + HuggingFace Embeddings)")
        self.rag = RAGSearch(persist_dir=persist_dir)  

    def plan(self, state: AgentState) -> AgentState:
        print("\n PLAN NODE — Deciding whether retrieval is required")
        print(f" User Question: {state.question}")
        return state

    def retrieve(self, state: AgentState) -> AgentState:
        print("\n RETRIEVE NODE — Fetching relevant chunks (Chroma)")

        retrieved_docs = self.rag.vectorstore.query(state.question, top_k=5)
        state.retrieved = retrieved_docs

        print(f"Retrieved {len(retrieved_docs)} Chunks from Vector DB")
        for idx, result in enumerate(retrieved_docs[:2], start=1):
            print(f" CHUNK {idx}: {result['metadata']['text'][:200]}...")
        return state

    def answer(self, state: AgentState) -> AgentState:
        print("\n ANSWER NODE — Generating response using Gemini (RAG)")

        result = self.rag.search_and_answer(state.question, top_k=5)
        state.answer = result.get("answer")

        print("\n Gemini Final Answer:")
        print(state.answer)
        return state

    def reflect(self, state: AgentState) -> AgentState:
        print("\n REFLECT NODE — Evaluating answer relevance")

        q_words = set(state.question.lower().split())
        a_words = set((state.answer or "").lower().split())

        overlap = q_words.intersection(a_words)
        score = len(overlap) / max(1, len(q_words))

        if score > 0.12:
            evaluation = f"[INFO] RELEVANT (score={score:.2f})"
        else:
            evaluation = f"[INFO] LOW RELEVANCE (score={score:.2f})"

        state.reflection = evaluation
        print(state.reflection)
        return state
