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
        max_duration: float = 60.0
    ) -> np.ndarray:
        """
        Извлечь эмбеддинг из нескольких сегментов аудиофайла.

        Args:
            audio_path: Путь к аудиофайлу
            segments: Список сегментов [{"start": float, "end": float}, ...]
            max_duration: Максимальная суммарная длительность сегментов (сек)

        Returns:
            centroid: np.ndarray shape (512,) - усреднённый эмбеддинг
        """
        from pyannote.core import Segment

        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Файл не найден: {audio_path}")

        model = self._load_model()

        embeddings = []
        total_duration = 0.0

        for seg in segments:
            if total_duration >= max_duration:
                break

            start = seg["start"]
            end = seg["end"]
            duration = end - start

            # Пропускаем слишком короткие сегменты
            if duration < 0.5:
                continue

            # Ограничиваем длительность
            if total_duration + duration > max_duration:
                end = start + (max_duration - total_duration)

            try:
                segment = Segment(start, end)
                embedding = model.crop(str(audio_path), segment)

                if embedding.ndim == 2:
                    embedding = embedding[0]

                embeddings.append(embedding)
                total_duration += (end - start)
            except Exception:
                # Пропускаем проблемные сегменты
                continue

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
