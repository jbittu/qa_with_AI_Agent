from src.agent import RAGAgent, AgentState

def main():
    agent = RAGAgent()
    print("Simple RAG Agent CLI. Type 'exit' to quit.")

    while True:
        q = input("\nQuestion> ").strip()
        if q.lower() in ("exit", "quit"):
            break

        state = AgentState(question=q)

        # pipeline
        state = agent.plan(state)
        state = agent.retrieve(state)
        state = agent.answer(state)
        state = agent.reflect(state)

        print("\n--- ANSWER ---")
        print(state.answer)

        print("\n--- REFLECTION ---")
        print(state.reflection)

        print("\n--- TOP SOURCES (first 3 retrieved chunks) ---")
        for i, chunk in enumerate((state.retrieved or [])[:3], start=1):
            text = chunk.get("metadata", {}).get("text", "")
            print(f"\nChunk {i}:")
            print(text[:200] + "..." if len(text) > 200 else text)

if __name__ == "__main__":
    main()
