"""
Chunker for podcast transcripts.
Parses transcripts and splits them into overlapping chunks for RAG.
"""

import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Utterance:
    """Single utterance from a speaker."""
    timestamp: str
    speaker: str
    text: str

    def __str__(self):
        return f"[{self.timestamp}] {self.speaker}:\n  {self.text}"

    @property
    def char_count(self) -> int:
        return len(self.text)


@dataclass
class Chunk:
    """A chunk of conversation for RAG indexing."""
    id: str
    podcast_id: str
    timestamp_start: str
    timestamp_end: str
    speakers: list[str]
    text: str  # Full text with timestamps for embedding
    speaker_texts: dict[str, list[str]] = field(default_factory=dict)  # Speaker -> their texts

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "podcast_id": self.podcast_id,
            "timestamp_start": self.timestamp_start,
            "timestamp_end": self.timestamp_end,
            "speakers": self.speakers,
            "text": self.text,
            "speaker_texts": self.speaker_texts
        }


def parse_transcript(text: str) -> list[Utterance]:
    """
    Parse transcript text into utterances.

    Format:
    [MM:SS] Speaker:
      Text of the utterance
      can be multiline
    """
    utterances = []

    # Pattern: [timestamp] Speaker:
    # Supports both MM:SS and HH:MM:SS formats
    pattern = r'\[(\d{1,2}:\d{2}(?::\d{2})?)\]\s+([^:]+):'

    # Split by utterance headers
    parts = re.split(pattern, text)

    # parts[0] is before first match (usually empty)
    # Then triplets: timestamp, speaker, text
    i = 1
    while i + 2 <= len(parts):
        timestamp = parts[i]
        speaker = parts[i + 1].strip()

        # Text is everything until next utterance (or end)
        raw_text = parts[i + 2] if i + 2 < len(parts) else ""

        # Clean up text: remove leading/trailing whitespace, normalize spaces
        text_lines = raw_text.strip().split('\n')
        # Remove indent from each line
        text_lines = [line.strip() for line in text_lines if line.strip()]
        clean_text = ' '.join(text_lines)

        if clean_text:  # Only add if there's actual text
            utterances.append(Utterance(
                timestamp=timestamp,
                speaker=speaker,
                text=clean_text
            ))

        i += 3

    return utterances


def estimate_tokens(text: str) -> int:
    """Rough token estimate (avg 4 chars per token for Russian)."""
    return len(text) // 4


def chunk_utterances(
    utterances: list[Utterance],
    podcast_id: str,
    max_tokens: int = 700,
    overlap_tokens: int = 150
) -> list[Chunk]:
    """
    Split utterances into overlapping chunks.

    - Each chunk is max_tokens size
    - Overlap of overlap_tokens with previous chunk
    - Never splits mid-utterance
    """
    if not utterances:
        return []

    chunks = []
    current_utterances: list[Utterance] = []
    current_tokens = 0
    chunk_num = 0

    for utterance in utterances:
        utterance_tokens = estimate_tokens(utterance.text)

        # If adding this utterance exceeds max and we have content, finalize chunk
        if current_tokens + utterance_tokens > max_tokens and current_utterances:
            # Create chunk
            chunk = build_chunk(current_utterances, podcast_id, chunk_num)
            chunks.append(chunk)
            chunk_num += 1

            # Keep last utterances for overlap
            overlap_utterances = get_overlap_utterances(current_utterances, overlap_tokens)
            current_utterances = overlap_utterances
            current_tokens = sum(estimate_tokens(u.text) for u in overlap_utterances)

        current_utterances.append(utterance)
        current_tokens += utterance_tokens

    # Final chunk
    if current_utterances:
        chunk = build_chunk(current_utterances, podcast_id, chunk_num)
        chunks.append(chunk)

    return chunks


def get_overlap_utterances(utterances: list[Utterance], target_tokens: int) -> list[Utterance]:
    """Get last N utterances that fit within target_tokens."""
    result = []
    total_tokens = 0

    for utterance in reversed(utterances):
        tokens = estimate_tokens(utterance.text)
        if total_tokens + tokens > target_tokens and result:
            break
        result.insert(0, utterance)
        total_tokens += tokens

    return result


def build_chunk(utterances: list[Utterance], podcast_id: str, chunk_num: int) -> Chunk:
    """Build a Chunk from list of utterances."""
    # Full text for embedding
    text_parts = [str(u) for u in utterances]
    full_text = "\n\n".join(text_parts)

    # Extract speakers and their texts
    speakers = list(set(u.speaker for u in utterances))
    speaker_texts: dict[str, list[str]] = {}
    for u in utterances:
        if u.speaker not in speaker_texts:
            speaker_texts[u.speaker] = []
        speaker_texts[u.speaker].append(u.text)

    return Chunk(
        id=f"{podcast_id}_{chunk_num:03d}",
        podcast_id=podcast_id,
        timestamp_start=utterances[0].timestamp,
        timestamp_end=utterances[-1].timestamp,
        speakers=speakers,
        text=full_text,
        speaker_texts=speaker_texts
    )


def extract_podcast_id(filepath: Path) -> str:
    """Extract podcast ID from filename."""
    name = filepath.stem
    # Try to extract date from filename
    date_match = re.search(r'(\d{1,2})-yanvarya-(\d{4})', name)
    if date_match:
        day, year = date_match.groups()
        return f"{year}-01-{int(day):02d}"

    # Fallback to filename
    return name[:50]  # Truncate long names


def process_transcript_file(filepath: Path, max_tokens: int = 700, overlap_tokens: int = 150) -> list[Chunk]:
    """Process a single transcript file into chunks."""
    text = filepath.read_text(encoding='utf-8')
    podcast_id = extract_podcast_id(filepath)

    utterances = parse_transcript(text)
    chunks = chunk_utterances(utterances, podcast_id, max_tokens, overlap_tokens)

    return chunks


def process_all_transcripts(
    transcripts_dir: Path,
    max_tokens: int = 700,
    overlap_tokens: int = 150
) -> list[Chunk]:
    """Process all .txt files in directory."""
    all_chunks = []

    for filepath in sorted(transcripts_dir.glob("*.txt")):
        print(f"Processing: {filepath.name}")
        chunks = process_transcript_file(filepath, max_tokens, overlap_tokens)
        all_chunks.extend(chunks)
        print(f"  -> {len(chunks)} chunks")

    return all_chunks


# Test
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from config import PODCASTS_DIR, CHUNK_MAX_TOKENS, CHUNK_OVERLAP_TOKENS

    chunks = process_all_transcripts(PODCASTS_DIR, CHUNK_MAX_TOKENS, CHUNK_OVERLAP_TOKENS)
    print(f"\nTotal chunks: {len(chunks)}")

    if chunks:
        print(f"\nFirst chunk preview:")
        print(f"  ID: {chunks[0].id}")
        print(f"  Speakers: {chunks[0].speakers}")
        print(f"  Time: {chunks[0].timestamp_start} - {chunks[0].timestamp_end}")
        print(f"  Text length: {len(chunks[0].text)} chars")
