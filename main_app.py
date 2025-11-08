import streamlit as st
from src.agent import RAGAgent, AgentState

st.set_page_config(page_title="RAG Agent", layout="wide")
st.title("RAG Agent")
st.write("Ask a question and the agent will retrieve context, generate an answer, and reflect on relevance.")

# input box
question = st.text_input("Enter your question:")

if st.button("Ask"):
    if not question.strip():
        st.warning("Please enter a question.")
        st.stop()

    agent = RAGAgent()
    state = AgentState(question=question)

    with st.spinner("Planning..."):
        state = agent.plan(state)

    with st.spinner("Retrieving relevant document chunks..."):
        state = agent.retrieve(state)

    with st.spinner("Generating answer using Gemini..."):
        state = agent.answer(state)

    with st.spinner("Evaluating answer relevance..."):
        state = agent.reflect(state)

    # Display results
    st.subheader("Answer")
    st.write(state.answer)

    st.subheader("Reflection (Self-Evaluation)")
    st.write(state.reflection)

    st.subheader("Top Retrieved Chunks")
    for i, chunk in enumerate((state.retrieved or [])[:5], start=1):
        text = chunk.get("metadata", {}).get("text", "")
        st.markdown(f"**Chunk {i}:**")
        st.write(text[:500] + ("..." if len(text) > 500 else ""))
        st.markdown("---")
