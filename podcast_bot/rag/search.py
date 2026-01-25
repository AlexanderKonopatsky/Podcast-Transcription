"""
FAISS-based vector search for RAG.
"""

import json
import numpy as np
from pathlib import Path
from typing import Optional
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_DIR, SEARCH_TOP_K
from rag.embeddings import get_embedding


# Global index (loaded once)
_faiss_index = None
_chunks_metadata = None


def load_index():
    """Load FAISS index and chunks metadata."""
    global _faiss_index, _chunks_metadata

    embeddings_path = DATA_DIR / "embeddings.npy"
    chunks_path = DATA_DIR / "chunks.json"

    if not embeddings_path.exists() or not chunks_path.exists():
        raise FileNotFoundError(
            "Index not found. Run 'python scripts/index_podcasts.py' first."
        )

    # Load embeddings
    embeddings = np.load(embeddings_path).astype(np.float32)

    # Normalize for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms

    # Create simple index (brute force for small datasets)
    # For larger datasets, use faiss.IndexIVFFlat
    import faiss
    dimension = embeddings.shape[1]
    _faiss_index = faiss.IndexFlatIP(dimension)  # Inner product (cosine after normalization)
    _faiss_index.add(embeddings)

    # Load chunks metadata
    with open(chunks_path, 'r', encoding='utf-8') as f:
        _chunks_metadata = json.load(f)

    print(f"Loaded index: {len(_chunks_metadata)} chunks, dimension {dimension}")


def ensure_index_loaded():
    """Ensure index is loaded."""
    if _faiss_index is None:
        load_index()


def search(
    query: str,
    speaker: Optional[str] = None,
    top_k: int = SEARCH_TOP_K
) -> list[dict]:
    """
    Search for relevant chunks.

    Args:
        query: Search query
        speaker: Optional speaker filter
        top_k: Number of results

    Returns:
        List of chunks with scores
    """
    ensure_index_loaded()

    # Get query embedding
    query_embedding = np.array([get_embedding(query)], dtype=np.float32)

    # Normalize
    norm = np.linalg.norm(query_embedding)
    query_embedding = query_embedding / norm

    # Search (get more results if filtering by speaker)
    search_k = top_k * 3 if speaker else top_k
    scores, indices = _faiss_index.search(query_embedding, search_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:  # FAISS returns -1 for missing results
            continue

        chunk = _chunks_metadata[idx]

        # Filter by speaker if specified
        if speaker and speaker not in chunk["speakers"]:
            continue

        results.append({
            "chunk": chunk,
            "score": float(score),
            "speaker_text": chunk["speaker_texts"].get(speaker, []) if speaker else None
        })

        if len(results) >= top_k:
            break

    return results


def format_search_results(results: list[dict], speaker: Optional[str] = None) -> str:
    """Format search results for LLM context."""
    parts = []

    for i, result in enumerate(results, 1):
        chunk = result["chunk"]
        score = result["score"]

        # Header
        header = f"[{chunk['podcast_id']} | {chunk['timestamp_start']}-{chunk['timestamp_end']} | score: {score:.2f}]"

        # Text - either speaker-specific or full
        if speaker and speaker in chunk["speaker_texts"]:
            texts = chunk["speaker_texts"][speaker]
            text = "\n".join(f"  {speaker}: {t}" for t in texts)
        else:
            text = chunk["text"]

        parts.append(f"{header}\n{text}")

    return "\n\n---\n\n".join(parts)


# Test
if __name__ == "__main__":
    print("Testing search...")

    try:
        results = search("что думаешь про биткоин", top_k=3)
        print(f"\nFound {len(results)} results")

        for i, result in enumerate(results, 1):
            chunk = result["chunk"]
            print(f"\n--- Result {i} (score: {result['score']:.3f}) ---")
            print(f"Podcast: {chunk['podcast_id']}")
            print(f"Time: {chunk['timestamp_start']} - {chunk['timestamp_end']}")
            print(f"Speakers: {chunk['speakers']}")
            print(f"Text preview: {chunk['text'][:200]}...")

    except FileNotFoundError as e:
        print(f"Error: {e}")
