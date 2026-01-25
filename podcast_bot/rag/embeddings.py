"""
Embeddings via OpenRouter API.
"""

import httpx
import numpy as np
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, EMBEDDING_MODEL


def get_embedding(text: str) -> list[float]:
    """Get embedding for a single text."""
    return get_embeddings([text])[0]


def get_embeddings(texts: list[str], batch_size: int = 10) -> list[list[float]]:
    """
    Get embeddings for multiple texts.
    Batches requests to avoid rate limits.
    """
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]

        # Retry logic
        for attempt in range(3):
            try:
                batch_embeddings = _request_embeddings(batch)
                all_embeddings.extend(batch_embeddings)
                break
            except Exception as e:
                if attempt < 2:
                    print(f"  Retry {attempt + 1}/3 after error: {e}")
                    time.sleep(2)
                else:
                    raise

        if i + batch_size < len(texts):
            print(f"  Embedded {i + batch_size}/{len(texts)} texts...")
            time.sleep(0.5)  # Small delay between batches

    return all_embeddings


def _request_embeddings(texts: list[str]) -> list[list[float]]:
    """Make API request for embeddings."""
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    data = {
        "model": EMBEDDING_MODEL,
        "input": texts
    }

    response = httpx.post(
        f"{OPENROUTER_BASE_URL}/embeddings",
        headers=headers,
        json=data,
        timeout=120.0
    )

    if response.status_code != 200:
        raise Exception(f"Embedding API error: {response.status_code} - {response.text}")

    result = response.json()

    # Handle API error in response
    if "error" in result:
        raise Exception(f"Embedding API error: {result['error']}")

    # Validate response structure
    if "data" not in result:
        raise Exception(f"Unexpected API response: {result}")

    # Sort by index to maintain order
    embeddings_data = sorted(result["data"], key=lambda x: x["index"])
    return [item["embedding"] for item in embeddings_data]


def save_embeddings(embeddings: list[list[float]], filepath: Path):
    """Save embeddings as numpy array."""
    arr = np.array(embeddings, dtype=np.float32)
    np.save(filepath, arr)
    print(f"Saved embeddings: {filepath} ({arr.shape})")


def load_embeddings(filepath: Path) -> np.ndarray:
    """Load embeddings from numpy file."""
    return np.load(filepath)


# Test
if __name__ == "__main__":
    test_texts = [
        "Что думаешь про биткоин?",
        "Альткоины сейчас падают сильно."
    ]

    print("Testing embeddings...")
    embeddings = get_embeddings(test_texts)
    print(f"Got {len(embeddings)} embeddings")
    print(f"Dimension: {len(embeddings[0])}")
