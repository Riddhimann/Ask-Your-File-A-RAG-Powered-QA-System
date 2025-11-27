# Ask-Your-File-A-RAG-Powered-QA-System
A lightweight RAG system that answers questions directly from an uploaded text file. It retrieves the most relevant content and uses an LLM to generate grounded responses.

Type it in Compiler to run the code : python -m streamlit run app.py

STEP 1 — Uploading the text file
The user uploads a .txt file using the website.
Streamlit reads the file’s content and passes the full text to the backend.

STEP 2 — Splitting the file into chunks
The backend splits the text into lines or small chunks.

Example:
Line 1: "Cats sleep 16 hours."
Line 2: "Cats run at 31 mph."

STEP 3 — Embedding the chunks
Each chunk is converted into a vector (a list of numbers).
This is done using an embedding model (bge-m3) running locally in Ollama.

Why embeddings?
Embeddings capture semantic meaning
Semantically similar sentences have similar vectors

Example:
“Cat running speed” → [0.19, -0.03, 0.77, ...]
“How fast does a cat run?” → embedding close to the above
These vectors are stored in memory as the VECTOR_DB.

STEP 4 — User asks a question
Example:
"How fast can a cat run?"
This question is also converted into an embedding (vector).

STEP 5 — Similarity Search (Retrieval Step)
The system compares: The question vector with all document vectors
Using cosine similarity:

similarity= A.B / ∣∣A∣∣∣∣B∣∣

Higher score = more relevant chunk.

Example:
“cat speed” line → similarity = 0.78 (strong match)
“cat sleeps” line → similarity = 0.62
“cat purring” line → similarity = 0.59

The top 3 chunks are returned.

STEP 6 — Context is sent to the LLM
We combine the top 3 retrieved chunks into a prompt:

Use only the following context:
 - A cat can travel at 31 mph.
 - Cats sleep 16 hours.
 - Cats purr by vibrating vocal folds.

And send it to the LLM (llama3.2) running locally in Ollama, along with the user question.

The LLM is instructed:
“Use ONLY this information. Don’t invent facts.”
This stops hallucinations.

STEP 7 — LLM generates the final answer
Example output:
“Cats can reach a top speed of around 31 mph (49 km/h) for short bursts.”

This answer comes strictly from the text file you uploaded.
