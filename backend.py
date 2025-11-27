import ollama
import numpy as np

# === Models ===
# If your hf.co model names work in your setup, you can keep them:
# EMBEDDING_MODEL = 'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf'
# LANGUAGE_MODEL  = 'hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF'

# Or use local names if you pulled them like this:
EMBEDDING_MODEL = 'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf'
LANGUAGE_MODEL = 'hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF'

# === In-memory "vector DB" ===
VECTOR_DB = []   # list of (chunk, embedding)


# ---------- Helpers ----------
def _get_embedding(text: str):
    """Call Ollama to get an embedding for a string."""
    resp = ollama.embed(model=EMBEDDING_MODEL, input=text)
    # Depending on Ollama version, key can be 'embeddings' or 'embedding'
    if "embeddings" in resp:
        return resp["embeddings"][0]
    return resp["embedding"]


# ---------- Load text from uploaded file ----------
def load_text(text: str):
    """
    Takes the full text from the uploaded .txt file,
    splits into lines/chunks, embeds them, and fills VECTOR_DB.
    """
    global VECTOR_DB
    VECTOR_DB = []  # reset DB each time you upload a new file

    # Simple "chunking": use non-empty lines
    lines = [line.strip() for line in text.split("\n") if line.strip()]

    for line in lines:
        emb = _get_embedding(line)
        VECTOR_DB.append((line, emb))

    return len(lines)


# ---------- Retrieval ----------
def _cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def retrieve(query: str, top_n: int = 3):
    """
    Retrieve top_n most similar chunks from VECTOR_DB given a query.
    """
    if not VECTOR_DB:
        return []

    q_emb = _get_embedding(query)
    scores = []
    for chunk, emb in VECTOR_DB:
        sim = _cosine_similarity(q_emb, emb)
        scores.append((chunk, sim))

    # sort desc by similarity
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_n]


# ---------- LLM Answer ----------
def generate_answer(context_chunks, query: str) -> str:
    """
    Given a list of chunks and the user query, call the LLM
    in the same way your current code does.
    """
    context_text = "\n".join([f" - {c}" for c in context_chunks])

    system_prompt = f"""You are a helpful chatbot.
Use only the following pieces of context to answer the question.
Don't make up any new information:

{context_text}
"""

    stream = ollama.chat(
        model=LANGUAGE_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ],
        stream=True,
    )

    answer = ""
    for part in stream:
        answer += part["message"]["content"]

    return answer
