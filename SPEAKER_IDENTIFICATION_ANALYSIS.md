# Speaker Identification Implementation Analysis

## Executive Summary

The speaker identification system is implemented as a modular pipeline that runs as **Stage 3 of the transcription process** (lines 305-363 in transcribe.py). It performs voice-based speaker matching using pyannote embeddings and cosine distance similarity.

---

## 1. CURRENT WORKFLOW & TRIGGERING

### 1.1 How Speaker Identification is Triggered

**Location:** Lines 305-363 in `transcribe.py` (function `transcribe_podcast`)

```
Stage 1: Diarization         → outputs diarized segments with speaker labels (SPEAKER_00, SPEAKER_01, etc)
Stage 2: Transcription       → outputs text segments
Stage 3: Speaker Matching    → (OPTIONAL) maps SPEAKER_XX labels to known names
```

**Trigger Conditions:**
- Only executed if `speaker_profiles_path` is provided AND file exists
- Line 307: `if speaker_profiles_path:`
- Line 309: `if profiles_path.exists():`
- Additional check: `profile_manager.has_profiles()` (line 313)

**CLI Integration:**
```bash
# With speaker identification:
python transcribe.py audio.mp3 --token hf_xxxxx  # Uses default speaker_profiles.json

# Explicitly specify profiles:
python transcribe.py audio.mp3 --token hf_xxxxx --speaker-profiles custom_profiles.json

# Without speaker identification:
python transcribe.py audio.mp3 --token hf_xxxxx --speaker-profiles ""
```

---

## 2. SPEAKER IDENTIFICATION WORKFLOW

### 2.1 Complete Processing Pipeline

```
INPUT: Diarized audio segments with SPEAKER_XX labels
  ↓
[A] Profile Loading
  ├─ Load speaker_profiles.json
  ├─ Parse 512-dim centroids for each known speaker
  └─ Time: ~100-200 ms
  ↓
[B] Embedding Extraction (for diarized speakers)
  ├─ Load pyannote/embedding model (first time only)
  ├─ For each SPEAKER_XX label:
  │   ├─ Collect all audio segments for that speaker
  │   ├─ Limit to 180 seconds total duration (max_duration)
  │   ├─ Skip segments < 0.5 seconds
  │   ├─ Extract embedding via model.crop() for each segment
  │   ├─ Average all embeddings → centroid
  │   ├─ L2-normalize for cosine distance
  │   └─ Time per speaker: ~2-10 seconds (depends on segment count)
  ├─ Total time: ~2-30 seconds (varies with speaker count & audio length)
  └─ Output: Dict[label, embedding] for each diarized speaker
  ↓
[C] Profile Matching
  ├─ Compute cosine distances: diarized embeddings → profile centroids
  ├─ Matrix shape: (num_diarized_speakers, num_profiles)
  ├─ For each diarized speaker:
  │   ├─ Find closest profile
  │   ├─ Check if distance < threshold (default 0.65)
  │   ├─ If match: map SPEAKER_XX → speaker_name
  │   ├─ If no match: keep SPEAKER_XX label
  │   └─ Prevent duplicate matches (each profile can match once)
  ├─ Time: ~10-50 ms
  └─ Output: Mapping dict {SPEAKER_XX: speaker_name}
  ↓
[D] Apply Mapping
  ├─ Update all result segments with matched speaker names
  └─ Time: ~1-5 ms
  ↓
OUTPUT: Result segments with identified speaker names
```

### 2.2 Stage-by-Stage Details

#### A. Profile Loading (SpeakerProfileManager)

**File:** `speaker_identification/profiles.py`

**Key Methods:**
- `__init__()` → calls `_load_profiles()`
- `_load_profiles()` → reads JSON, converts centroid arrays from lists

**Data Structure:**
```json
{
  "version": "1.0",
  "embedding_model": "pyannote/embedding",
  "embedding_dimension": 512,
  "samples_dir": "speaker_samples",
  "profiles": {
    "Зенур": {
      "centroid": [0.123, -0.456, ...],  // 512-dim vector
      "samples": ["speaker_samples/Зенур.mp3", ...],
      "created_at": "2025-01-25T12:00:00",
      "updated_at": "2025-01-25T12:00:00"
    },
    ...
  }
}
```

**Performance:**
- File read: ~50 ms
- JSON parse: ~50-100 ms
- Array conversion: ~50 ms
- **Total: ~150-200 ms**

#### B. Embedding Extraction (SpeakerEmbeddingExtractor)

**File:** `speaker_identification/embeddings.py`

**Key Methods:**
- `extract_from_segments(audio_path, segments, max_duration=180.0)` - Main method used in transcribe.py
  - Input: Audio file path + list of time-based segments
  - Returns: Single 512-dim normalized embedding (centroid)

**Algorithm:**
```python
for each segment in segments:
    if total_duration >= 180 seconds:
        break
    if segment_duration < 0.5 seconds:
        continue  # Skip too short segments
    
    embedding = model.crop(audio_file, segment)  # Extract from that time range
    normalize(embedding)
    centroid = average(all_embeddings)
```

**Processing Details:**
- Model: pyannote/embedding (512-dim output)
- Segment limit: 180 seconds per speaker
- Min segment length: 0.5 seconds
- Normalization: L2 norm (for cosine distance)

**Performance:**
- Model load (lazy, first time): ~2-5 seconds
- Per speaker: 2-10 seconds (depends on segment count)
  - Extraction: ~1 second per 10 segments
  - Averaging/normalization: ~10-50 ms
- **Total for N speakers: ~2-5 sec (load) + N × 2-10 sec**

**Code Location:** Lines 325-335 in transcribe.py
```python
extractor = SpeakerEmbeddingExtractor(hf_token, device=str(device))
for label in diarization_labels:
    segments = speaker_segments[label]
    embedding = extractor.extract_from_segments(audio_path, segments)
    if embedding is not None:
        speaker_embeddings_dict[label] = embedding
extractor.unload_model()
```

#### C. Speaker Matching (SpeakerMatcher)

**File:** `speaker_identification/matcher.py`

**Key Methods:**
- `match_speakers(diarization_embeddings, diarization_labels)` → returns mapping dict
- `get_distances(diarization_embeddings, diarization_labels)` → returns distance matrix for debugging

**Algorithm:**
```python
# Compute cosine distances between diarized and profile embeddings
distances = cdist(diarization_embeddings, profile_centroids, metric="cosine")
// Shape: (num_diarized_speakers, num_profiles)

// Greedy matching: process speakers by confidence (lowest distance first)
for each speaker (sorted by min_distance):
    find closest unused profile
    if distance < threshold:
        map speaker → profile_name
        mark profile as used
    else:
        keep original SPEAKER_XX label
```

**Threshold:** Default 0.65 (cosine distance)
- Recommended: 0.3-0.6
- Lower = stricter matching
- Higher = more lenient matching

**Performance:**
- Distance matrix computation: O(num_speakers × num_profiles × 512)
- For 3 speakers × 3 profiles: ~10-50 ms
- Greedy assignment: ~1-5 ms
- **Total: ~10-50 ms**

**Code Location:** Lines 343-352 in transcribe.py
```python
matcher = SpeakerMatcher(profile_manager, threshold=voice_threshold)
speaker_mapping = matcher.match_speakers(diarization_embeddings, valid_labels)

# Show distances for debugging
distances = matcher.get_distances(diarization_embeddings, valid_labels)
for label, dists in distances.items():
    print(f"      {label} {status} [{dists_str}]")
```

---

## 3. CURRENT PROGRESS REPORTING

### 3.1 Console Output Example

**When speaker identification is enabled:**

```
[3/3] Идентификация спикеров по голосу...
      Загружено 3 профилей
      Извлечение эмбеддингов спикеров...
      SPEAKER_00 -> Зенур [Зенур: 0.234, Серега: 0.678, Мария: 0.891]
      SPEAKER_01 -> Серега [Зенур: 0.712, Серега: 0.145, Мария: 0.623]
      SPEAKER_02 -> SPEAKER_02 (не определён) [Зенур: 0.845, Серега: 0.756, Мария: 0.934]
```

### 3.2 Progress Reporting Locations

| Stage | Code Location | Output |
|-------|---------------|--------|
| Start | Line 310 | `[3/3] Идентификация спикеров по голосу...` |
| Profile loading | Line 314 | `Загружено {count} профилей` |
| Embedding extraction | Line 325 | `Извлечение эмбеддингов спикеров...` |
| Matching results | Lines 348-352 | `{label} -> {name} [{distances}]` |
| Completion | Line 365 | `Готово! Итого сегментов: {count}` |

### 3.3 What's Missing from Current Progress Reporting

**No real-time progress for:**
- Loading profile embeddings from disk (instant, but no feedback)
- Extracting embeddings from audio:
  - No per-speaker progress (e.g., "Extracting speaker 1/3")
  - No indication of how many segments are being processed per speaker
  - No time estimate
- Processing speed indicators
- Percentage completion

**Silent operations:**
- Model loading (2-5
