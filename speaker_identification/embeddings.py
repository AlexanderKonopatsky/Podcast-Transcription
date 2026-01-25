"""
Извлечение speaker embeddings из аудиофайлов.
"""

import torch
import numpy as np
from pathlib import Path


class SpeakerEmbeddingExtractor:
    """Извлечение эмбеддингов спикеров из аудиофайлов."""

    def __init__(self, hf_token: str, device: str = None):
        """
        Args:
            hf_token: HuggingFace токен для доступа к моделям
            device: cuda или cpu (автоопределение если None)
        """
        self.hf_token = hf_token

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self._model = None

    def _load_model(self):
        """Ленивая загрузка модели эмбеддингов."""
        if self._model is None:
            from pyannote.audio import Model, Inference

            model = Model.from_pretrained(
                "pyannote/embedding",
                use_auth_token=self.hf_token
            )
            self._model = Inference(model, window="whole")
            self._model.to(self.device)

        return self._model

    def extract_from_file(self, audio_path: str) -> np.ndarray:
        """
        Извлечь эмбеддинг из аудиофайла с одним спикером.

        Args:
            audio_path: Путь к аудиофайлу (wav, mp3, flac, etc.)

        Returns:
            embedding: np.ndarray shape (512,) - вектор эмбеддинга
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Файл не найден: {audio_path}")

        model = self._load_model()
        embedding = model(str(audio_path))

        # Inference возвращает (1, 512), нам нужен (512,)
        if embedding.ndim == 2:
            embedding = embedding[0]

        return embedding

    def extract_from_files(self, audio_paths: list) -> np.ndarray:
        """
        Извлечь эмбеддинги из нескольких файлов и усреднить.

        Args:
            audio_paths: Список путей к аудиофайлам

        Returns:
            centroid: np.ndarray shape (512,) - усреднённый эмбеддинг
        """
        embeddings = []
        for path in audio_paths:
            emb = self.extract_from_file(path)
            embeddings.append(emb)

        # Усредняем все эмбеддинги
        centroid = np.mean(embeddings, axis=0)

        # Нормализуем для cosine distance
        centroid = centroid / np.linalg.norm(centroid)

        return centroid

    def extract_from_segments(
        self,
        audio_path: str,
        segments: list,
        max_duration: float = 180.0,
        max_segments: int = 100,
        speaker_label: str = None,
        verbose: bool = True,
        segment_timeout: float = 30.0
    ) -> np.ndarray:
        """
        Извлечь эмбеддинг из нескольких сегментов аудиофайла.

        Args:
            audio_path: Путь к аудиофайлу
            segments: Список сегментов [{"start": float, "end": float}, ...]
            max_duration: Максимальная суммарная длительность сегментов (сек, по умолчанию 180)
            max_segments: Максимальное количество сегментов для обработки (по умолчанию 100)
            speaker_label: Метка спикера для логирования (опционально)
            verbose: Выводить прогресс (по умолчанию True)
            segment_timeout: Таймаут на обработку одного сегмента в секундах (по умолчанию 30)

        Returns:
            centroid: np.ndarray shape (512,) - усреднённый эмбеддинг
        """
        from pyannote.core import Segment
        import time
        import signal
        import sys

        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Файл не найден: {audio_path}")

        model = self._load_model()

        embeddings = []
        total_duration = 0.0
        processed_count = 0
        skipped_error = 0

        # Фильтруем сегменты заранее (только достаточно длинные)
        valid_segments = [s for s in segments if (s["end"] - s["start"]) >= 0.5]
        skipped_short = len(segments) - len(valid_segments)

        # Сортируем по длительности (сначала длинные - более информативные для голосового отпечатка)
        valid_segments_sorted = sorted(valid_segments, key=lambda s: s["end"] - s["start"], reverse=True)

        total_segments = len(valid_segments_sorted)
        effective_limit = min(total_segments, max_segments)

        if verbose and speaker_label:
            print(f"          {speaker_label}: {total_segments} сегментов, обработаем до {effective_limit}", flush=True)

        start_time = time.time()
        last_progress_time = start_time

        for i, seg in enumerate(valid_segments_sorted):
            if total_duration >= max_duration:
                if verbose:
                    print(f"          → достигнут лимит длительности {max_duration:.0f} сек", flush=True)
                break

            if processed_count >= max_segments:
                if verbose:
                    print(f"          → достигнут лимит сегментов {max_segments}", flush=True)
                break

            start = seg["start"]
            end = seg["end"]
            duration = end - start

            # Ограничиваем длительность
            if total_duration + duration > max_duration:
                end = start + (max_duration - total_duration)

            # Прогресс каждые 5 секунд или каждые 20 сегментов
            now = time.time()
            if verbose and (now - last_progress_time > 5 or (processed_count > 0 and processed_count % 20 == 0)):
                elapsed = now - start_time
                progress_pct = (i + 1) / effective_limit * 100 if effective_limit else 0
                print(f"          → {processed_count}/{effective_limit} сегментов, "
                      f"{total_duration:.1f}/{max_duration:.0f} сек, "
                      f"прошло {elapsed:.1f} сек ({progress_pct:.0f}%)", flush=True)
                last_progress_time = now

            try:
                segment_start_time = time.time()
                segment = Segment(start, end)
                embedding = model.crop(str(audio_path), segment)

                segment_elapsed = time.time() - segment_start_time

                # Предупреждение о медленных сегментах
                if segment_elapsed > 5.0 and verbose:
                    print(f"          ⚠ Медленный сегмент [{start:.1f}-{end:.1f}]: {segment_elapsed:.1f} сек", flush=True)

                if embedding.ndim == 2:
                    embedding = embedding[0]

                embeddings.append(embedding)
                total_duration += (end - start)
                processed_count += 1

            except Exception as e:
                skipped_error += 1
                if verbose and skipped_error <= 3:
                    print(f"          ⚠ Ошибка сегмента [{start:.1f}-{end:.1f}]: {str(e)[:50]}", flush=True)
                continue

        # Финальная статистика
        if verbose:
            elapsed = time.time() - start_time
            status_parts = [f"{processed_count} обработано"]
            if skipped_short > 0:
                status_parts.append(f"{skipped_short} коротких")
            if skipped_error > 0:
                status_parts.append(f"{skipped_error} ошибок")
            status = ", ".join(status_parts)
            if speaker_label:
                print(f"          {speaker_label}: {status} за {elapsed:.1f} сек", flush=True)
            else:
                print(f"          Готово: {status} за {elapsed:.1f} сек", flush=True)

        if not embeddings:
            return None

        # Усредняем все эмбеддинги
        centroid = np.mean(embeddings, axis=0)

        # Нормализуем для cosine distance
        centroid = centroid / np.linalg.norm(centroid)

        return centroid

    def unload_model(self):
        """Выгрузить модель из памяти."""
        if self._model is not None:
            del self._model
            self._model = None
            torch.cuda.empty_cache()
