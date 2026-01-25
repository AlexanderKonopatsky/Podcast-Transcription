# Speaker Identification - Quick Reference Summary

## Overview

Speaker identification is **Stage 3** of the transcription process. It matches diarized speakers (SPEAKER_00, SPEAKER_01, etc.) with known speaker profiles using voice embeddings.

**Duration:** ~5-40 seconds (including model load)

---

## Workflow Diagram

```
Diarization Output                  Speaker Profiles
      ↓                                    ↓
[SPEAKER_00: segments]        speaker_profiles.json
[SPEAKER_01: segments]               (contains 3-5 speakers)
[SPEAKER_02: segments]                   ↓
      ↓                                   ↓
      └─────────────────┬─────────────────┘
                        ↓
            ┌───────────────────────────┐
            │  Speaker Identification   │
            ├───────────────────────────┤
            │ A. Load profiles        │ ~200 ms  ✓
            │ B. Extract embeddings   │ ~2-30s   ✗ NO PROGRESS
            │ C. Match speakers       │ ~10-50ms ✓
            │ D. Apply mapping        │ ~1-5ms   ✓
            └───────────────────────────┘
                        ↓
        Result segments with speaker names:
        "Зенур", "Серега", "SPEAKER_02" (if unmatched)
```

---

## Stage Details

### Stage A: Profile Loading (200ms)
- Load JSON file
- Parse 512-dim embeddings
- Validate speaker data
- **Progress:** ✓ Message shown

### Stage B: Embedding Extraction (2-30 seconds) ⚠️ BOTTLENECK
- Load pyannote/embedding model (2-5 sec, first time only)
- For each speaker:
  - Collect all diarization segments
  - Extract embedding from each segment
  - Average and normalize
- **Progress:** ✗ SILENT - no feedback at all

### Stage C: Speaker Matching (10-50ms)
- Compute cosine distances
- Greedy assignment (lowest distance first)
- Apply threshold (default 0.65)
- **Progress:** ✓ Results shown with distances

### Stage D: Apply Mapping (1-5ms)
- Update segment speaker labels
- **Progress:** ✓ Silent but instant

---

## Current Progress Output

```
[3/3] Идентификация спикеров по голосу...
      Загружено 3 профилей
      Извлечение эмбеддингов спикеров...
      SPEAKER_00 -> Зенур [Зенур: 0.234, Серега: 0.678, Мария: 0.891]
      SPEAKER_01 -> Серега [Зенур: 0.712, Серега: 0.145, Мария: 0.623]
      SPEAKER_02 -> SPEAKER_02 (не определён) [Зенур: 0.845, Серега: 0.756, Мария: 0.934]
Готово! Итого сегментов: 5847
```

**Issues:**
- ✗ Silent during longest phase (embedding extraction)
- ✗ No per-speaker progress (don't know if working on speaker 1/3 or 2/3)
- ✗ No time estimates
- ✗ Distance format hard to parse

---

## Data Flow Through Pipeline

```
diarization_segments (output from stage 1):
├─ [{"start": 0.0, "end": 2.5, "speaker": "SPEAKER_00"}, ...]

Reorganize by speaker:
└─ diarization_labels = ["SPEAKER_00", "SPEAKER_01", "SPEAKER_02"]
   speaker_segments = {
     "SPEAKER_00": [5 segments, 2.5 min total],
     "SPEAKER_01": [8 segments, 5.3 min total],
     "SPEAKER_02": [3 segments, 1.2 min total]
   }

Extract embeddings:
└─ speaker_embeddings_dict = {
     "SPEAKER_00": np.array([...512 values...]),  # ← From model
     "SPEAKER_01": np.array([...512 values...]),
     "SPEAKER_02": np.array([...512 values...])
   }

Load profiles:
└─ profile_centroids = {
     "Зенур": np.array([...512 values...]),      # ← From file
     "Серега": np.array([...512 values...]),
     "Мария": np.array([...512 values...])
   }

Compute distances:
└─ distances_matrix = [
     [0.234, 0.678, 0.891],  # SPEAKER_00 distances to each profile
     [0.712, 0.145, 0.623],  # SPEAKER_01 distances
     [0.845, 0.756, 0.934]   # SPEAKER_02 distances
   ]

Create mapping:
└─ speaker_mapping = {
     "SPEAKER_00": "Зенур",        # ✓ distance 0.234 < 0.65
     "SPEAKER_01": "Серега",       # ✓ distance 0.145 < 0.65
     "SPEAKER_02": "SPEAKER_02"    # ✗ best distance 0.756 > 0.65
   }

Update results:
└─ result_segments[i]["speaker"] = speaker_mapping[original_speaker]
```

---

## Performance Breakdown

| Phase | Min | Max | Typical | Depends On |
|-------|-----|-----|---------|------------|
| Profile load | 100ms | 500ms | 200ms | File size |
| Model load | 2s | 5s | 2s | GPU/CPU, first time only |
| Per-speaker extraction | 2s | 10s | 4s | Segment count |
| **Total extraction** (3 speakers) | 6s | 30s | 12s | # speakers, # segments |
| Distance computation | 10ms | 50ms | 20ms | # speakers × # profiles |
| Mapping application | 1ms | 5ms | 2ms | # transcript segments |
| **TOTAL** | 5s | 40s | 14s | Audio length + speaker count |

**Most expensive:** Embedding extraction (90% of time)
**Currently silent:** Embedding extraction

---

## Code Locations

| Component | File | Lines | Purpose |
|-----------|------|-------|---------|
| Main orchestration | transcribe.py | 305-363 | Control flow |
| Profile loading | profiles.py | 31-49 | Load from disk |
| Profile manager | profiles.py | 13-204 | All profile operations |
| Embedding extraction | embeddings.py | 88-154 | Extract from audio |
| Extractor | embeddings.py | 10-162 | Wrapper class |
| Speaker matching | matcher.py | 29-89 | Compute distances & match |
| Matcher | matcher.py | 11-121 | Wrapper class |

---

## Key Metrics for Visualization

### Available Data at Each Stage

**Before extraction (line 316):**
```python
diarization_labels = ["SPEAKER_00", "SPEAKER_01", "SPEAKER_02"]  # ← Total count
speaker_segments = {
    "SPEAKER_00": [seg1, seg2, seg3, ...],  # ← Segment count per speaker
    "SPEAKER_01": [seg1, seg2, ...],
    "SPEAKER_02": [seg1, ...]
}
```

**During extraction:**
```python
for i, label in enumerate(diarization_labels, 1):  # ← Can show [1/3], [2/3], [3/3]
    segments = speaker_segments[label]  # ← Can show segment count
    embedding = extractor.extract_from_segments(audio_path, segments)  # ← Silent
```

**During matching:**
```python
speaker_mapping = matcher.match_speakers(...)  # ← Returns mapping
distances = matcher.get_distances(...)  # ← Returns distance matrix (not used)
```

### Useful Metrics
- **Total speakers:** `len(diarization_labels)`
- **Current speaker (index):** From enumerate loop
- **Segment count per speaker:** `len(speaker_segments[label])`
- **Total duration per speaker:** Sum of (end - start) for all segments
- **Profiles available:** `len(profile_manager.list_speakers())`
- **Match confidence:** `(1 - distance) * 100`

---

## Threshold Configuration

**Default:** `voice_threshold = 0.65` (cosine distance)

**Impact:**
- **Lower threshold (e.g., 0.3):** Stricter matching, more SPEAKER_XX labels
- **Higher threshold (e.g., 0.8):** Lenient matching, more false positives

**Current distances example:**
```
SPEAKER_00:  Зенур: 0.234 (24% distance, 76% match confidence)
SPEAKER_01:  Серега: 0.145 (14% distance, 85% match confidence)
SPEAKER_02:  Зенур: 0.845 (85% distance - TOO FAR, not matched)
```

All are compared to threshold of 0.65:
- 0.234 < 0.65 ✓ Match
- 0.145 < 0.65 ✓ Match
- 0.845 > 0.65 ✗ No match

---

## Suggestions for Progress Improvement

### Quick Wins (5-15 minutes each)

1. **Per-speaker counter**
   ```
   [1/3] SPEAKER_00 (5 segments)...
   [2/3] SPEAKER_01 (8 segments)...
   [3/3] SPEAKER_02 (3 segments)...
   ```

2. **Confidence percentage**
   ```
   SPEAKER_00 -> Зенур (76.6% confidence) [0.234]
   ```

3. **Add tqdm progress bar**
   ```
   [████████░░] 80% (8/10 segments)
   ```

4. **Spinner for model load**
   ```
   ⠋ Loading embedding model... (2.3s)
   ```

### Medium Effort (20-30 minutes)

5. **Time estimates & ETA**
   ```
   [1/3] SPEAKER_00... ETA: 12s
   ```

6. **Stage timing summary**
   ```
   Profile loading:      0.2s
   Model loading:        2.0s
   Embedding extraction: 12.3s
   Matching:            0.03s
   Total:               14.6s
   ```

### Advanced (45+ minutes)

7. **Distance heatmap visualization**
8. **GPU memory monitoring**
9. **Real-time segment tracking**

---

## CLI Options

```bash
# Default (uses speaker_profiles.json if exists)
python transcribe.py audio.mp3 --token hf_xxxxx

# Custom profiles file
python transcribe.py audio.mp3 --token hf_xxxxx --speaker-profiles custom.json

# Adjust matching strictness
python trans
