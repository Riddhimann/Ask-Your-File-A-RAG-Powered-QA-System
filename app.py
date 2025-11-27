import streamlit as st
import backend  # our RAG logic file

st.set_page_config(page_title="RAG Text QA", layout="wide")

st.title("📚 RAG Demo on Your Own Text File")
st.write(
    "Upload a `.txt` file, then ask questions about it. "
    "Answers are generated using retrieval-augmented generation with Ollama."
)

# ---------- 1. File upload ----------
uploaded_file = st.file_uploader("Upload a .txt file", type=["txt"])

if "db_loaded" not in st.session_state:
    st.session_state["db_loaded"] = False

if uploaded_file is not None:
    # Read file content as UTF-8 text
    raw_text = uploaded_file.read().decode("utf-8", errors="ignore")

    with st.spinner("Embedding and indexing your file..."):
        n_chunks = backend.load_text(raw_text)

    st.session_state["db_loaded"] = True
    st.success(f"Loaded {n_chunks} text chunks into the database.")

# ---------- 2. Question input ----------
st.subheader("Ask a question about the uploaded text")

question = st.text_input("Enter your question:")

if st.button("Ask"):
    if not st.session_state["db_loaded"]:
        st.error("Please upload a .txt file first.")
    elif not question.strip():
        st.error("Please type a question.")
    else:
        # 2a. Retrieve
        with st.spinner("Retrieving relevant chunks..."):
            retrieved = backend.retrieve(question, top_n=3)

        if not retrieved:
            st.warning("No data in the database. Did the file load correctly?")
        else:
            st.subheader("🔎 Retrieved Chunks")
            for chunk, sim in retrieved:
                st.markdown(
                    f"""
                    <div style="padding:10px;margin-bottom:8px;
                                border-radius:6px;border:1px solid #ccc;">
                        <b>Similarity:</b> {sim:.2f}<br>
                        <i>{chunk}</i>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            context_chunks = [c for c, _ in retrieved]

            # 2b. Generate answer
            with st.spinner("Generating answer from LLM..."):
                answer = backend.generate_answer(context_chunks, question)

            st.subheader("💬 Answer")
            st.write(answer)
