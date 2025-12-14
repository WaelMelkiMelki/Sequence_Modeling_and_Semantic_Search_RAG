# Sequence Modeling & Semantic Search (RAG System)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![FAISS](https://img.shields.io/badge/Vector_DB-FAISS-orange)
![Transformers](https://img.shields.io/badge/Hugging_Face-Transformers-yellow)

## 🚀 Project Overview
This project implements an end-to-end **Retrieval-Augmented Generation (RAG)** system capable of processing sequential text data, generating dense vector embeddings, and performing high-speed semantic search.

Unlike simple keyword search, this system understands the *meaning* (semantics) of a query. It uses a **Sliding Window approach** for sequence alignment to ensure context is preserved across data chunks.

## 🔑 Key Features
*   **Semantic Search:** Retrieves answers based on meaning, not just keyword matching.
*   **Vector Database:** Utilizes **FAISS (Facebook AI Similarity Search)** for efficient similarity search in high-dimensional space ($R^{384}$).
*   **Context-Aware Chunking:** Implements a "Sliding Sentence Window" strategy to preserve semantic context between chunks.
*   **Dimensionality Reduction:** Visualizes the vector space using **PCA (Principal Component Analysis)** to map 384 dimensions down to 2D.

## 🛠️ Tech Stack
*   **Language:** Python
*   **Embeddings:** `sentence-transformers` (Model: *all-MiniLM-L6-v2*)
*   **Vector Indexing:** `faiss-cpu` (L2 Euclidean Distance)
*   **Data Processing:** `numpy`, `pandas`
*   **Visualization:** `matplotlib`, `scikit-learn` (PCA)

## ⚙️ How It Works

### 1. Sequence Modeling (Data Prep)
Raw text (or biological sequences) cannot be processed all at once. I implemented a **Sliding Window** logic:
*   **Logic:** Groups 3 sentences together with an overlap of 1 sentence.
*   **Benefit:** Ensures that the context of a subject (e.g., "Cas9") is linked to its action (e.g., "cuts DNA") even if they appear in different sentences.

### 2. Vector Embedding
The text chunks are passed through a pre-trained Transformer model, converting them into **384-dimensional dense vectors**.

### 3. Indexing & Retrieval
*   Vectors are stored in a **FAISS IndexFlatL2** structure.
*   When a user asks a question, the query is vectorized.
*   The system calculates the Euclidean distance between the query vector and all stored vectors to find the "Nearest Neighbors."

## 💻 Usage Example

```python
# 1. Initialize System
model = SentenceTransformer('all-MiniLM-L6-v2')
index = faiss.IndexFlatL2(384)

# 2. Search
query = "How does the enzyme cut the DNA?"
results = search(query)

# Output:
# "... an enzyme called Cas9. This acts as a pair of 'molecular scissors' that can cut..."
