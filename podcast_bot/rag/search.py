"""
FAISS-based vector search for RAG.
"""

import json
import numpy as np
from pathlib import Path
from typing import Optional, Union, Literal
from datetime import date, datetime, timedelta
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_DIR, SEARCH_TOP_K
from rag.embeddings import get_embedding


# Типы фильтров
TimeFilterType = Union[
    Literal["recent"],      # Только свежие (2 недели)
    Literal["balanced"],    # Default: decay (70% свежие / 30% история)
    Literal["historical"],  # Вся история без фильтра
    dict                    # {"start": "2025-12-01", "end": "2026-01-01"}
]

# Константы для temporal filtering
TEMPORAL_DECAY_FACTOR = 0.93   # Экспоненциальный коэффициент затухания
RECENCY_WINDOW_DAYS = 14       # "Свежее" окно (2 недели)


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
    top_k: int = SEARCH_TOP_K,
    time_filter: TimeFilterType = "balanced"
) -> list[dict]:
    """
    Search for relevant chunks with temporal filtering.

    Args:
        query: Search query
        speaker: Optional speaker filter
        top_k: Number of results
        time_filter: Temporal filtering mode:
            - "recent": Only last 2 weeks (hard filter)
            - "balanced": Decay mode (default, 70% recent / 30% history)
            - "historical": All history, no decay
            - {"start": date, "end": date}: Custom date range

    Returns:
        List of chunks with scores
    """
    ensure_index_loaded()

    # Get query embedding
    query_embedding = np.array([get_embedding(query)], dtype=np.float32)

    # Normalize
    norm = np.linalg.norm(query_embedding)
    query_embedding = query_embedding / norm

    # Search (get more results for filtering)
    search_k = top_k * 3 if speaker else top_k * 2
    scores, indices = _faiss_index.search(query_embedding, search_k)

    # Collect results with metadata
    today = date.today()
    results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:  # FAISS returns -1 for missing results
            continue

        chunk = _chunks_metadata[idx]

        # Parse podcast date
        podcast_date_str = chunk.get("podcast_date") or chunk["podcast_id"]
        try:
            podcast_date = datetime.fromisoformat(podcast_date_str).date()
        except ValueError:
            # Fallback to today if can't parse
            podcast_date = today

        days_old = (today - podcast_date).days

        results.append({
            "chunk": chunk,
            "score": float(score),
            "original_score": float(score),
            "podcast_date": podcast_date,
            "days_old": days_old,
            "speaker_text": chunk["speaker_texts"].get(speaker, []) if speaker else None
        })

    # Apply temporal filter
    if time_filter == "recent":
        # Hard filter: only last 2 weeks
        cutoff_date = today - timedelta(days=RECENCY_WINDOW_DAYS)
        results = [r for r in results if r["podcast_date"] >= cutoff_date]

    elif time_filter == "balanced":
        # Temporal decay (default mode)
        for result in results:
            decay_multiplier = TEMPORAL_DECAY_FACTOR ** result["days_old"]
            result["score"] = result["score"] * decay_multiplier
            result["decay_multiplier"] = decay_multiplier

    elif time_filter == "historical":
        # No changes - use original scores
        pass

    elif isinstance(time_filter, dict):
        # Custom date range
        try:
            start_date = datetime.fromisoformat(time_filter["start"]).date()
            end_date = datetime.fromisoformat(time_filter["end"]).date()
            results = [r for r in results
                      if start_date <= r["podcast_date"] <= end_date]
        except (ValueError, KeyError):
            # Invalid date format, fall back to no filtering
            pass

    # Re-sort by (possibly modified) scores
    results.sort(key=lambda x: x["score"], reverse=True)

    # Filter by speaker if specified
    if speaker:
        results = [r for r in results if speaker in r["chunk"]["speakers"]]

    # Return top_k
    return results[:top_k]


def format_search_results(
    results: list[dict],
    speaker: Optional[str] = None,
    show_recency_indicator: bool = True
) -> str:
    """
    Format search results for LLM context.

    Args:
        results: Search results from search()
        speaker: Optional speaker filter
        show_recency_indicator: Show 🔥 for recent podcasts (default: True)
    """
    if not results:
        return "(контекст не найден)"

    today = date.today()
    recency_threshold = today - timedelta(days=RECENCY_WINDOW_DAYS)

    parts = []

    for i, result in enumerate(results, 1):
        chunk = result["chunk"]
        score = result["score"]

        # Determine recency marker
        recency_marker = ""
        if show_recency_indicator:
            podcast_date_str = chunk.get("podcast_date") or chunk["podcast_id"]
            try:
                podcast_date = datetime.fromisoformat(podcast_date_str).date()
                if podcast_date >= recency_threshold:
                    recency_marker = " 🔥"  # Fresh opinion
            except ValueError:
                pass

        # Header
        header = f"[{chunk['podcast_id']}{recency_marker} | {chunk['timestamp_start']}-{chunk['timestamp_end']} | score: {score:.2f}]"

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
