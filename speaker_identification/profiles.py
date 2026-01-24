"""
Управление профилями спикеров.
"""

import json
from pathlib import Path
from datetime import datetime
import numpy as np

from .embeddings import SpeakerEmbeddingExtractor


class SpeakerProfileManager:
    """Управление профилями спикеров."""

    DEFAULT_SAMPLES_DIR = "speaker_samples"

    def __init__(
        self,
        profiles_path: str = "speaker_profiles.json",
        samples_dir: str = None
    ):
        """
        Args:
            profiles_path: Путь к файлу с профилями
            samples_dir: Папка с образцами голосов (по умолчанию speaker_samples/)
        """
        self.profiles_path = Path(profiles_path)
        self.samples_dir = Path(samples_dir) if samples_dir else Path(self.DEFAULT_SAMPLES_DIR)
        self.profiles = self._load_profiles()

    def _load_profiles(self) -> dict:
        """Загрузить профили из JSON."""
        if self.profiles_path.exists():
            with open(self.profiles_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                # Конвертируем списки обратно в numpy arrays
                for name, profile in data.get("profiles", {}).items():
                    if "centroid" in profile:
                        profile["centroid"] = np.array(profile["centroid"])
                return data

        return {
            "version": "1.0",
            "embedding_model": "pyannote/embedding",
            "embedding_dimension": 512,
            "samples_dir": str(self.samples_dir),
            "profiles": {}
        }

    def save_profiles(self):
        """Сохранить профили в JSON."""
        # Создаём копию для сериализации
        data = {
            "version": self.profiles.get("version", "1.0"),
            "embedding_model": self.profiles.get("embedding_model", "pyannote/embedding"),
            "embedding_dimension": self.profiles.get("embedding_dimension", 512),
            "samples_dir": str(self.samples_dir),
            "profiles": {}
        }

        for name, profile in self.profiles.get("profiles", {}).items():
            data["profiles"][name] = {
                "centroid": profile["centroid"].tolist() if isinstance(profile["centroid"], np.ndarray) else profile["centroid"],
                "samples": profile.get("samples", []),
                "created_at": profile.get("created_at", datetime.now().isoformat()),
                "updated_at": profile.get("updated_at", datetime.now().isoformat())
            }

        with open(self.profiles_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def add_speaker(
        self,
        name: str,
        extractor: SpeakerEmbeddingExtractor,
        audio_samples: list = None
    ):
        """
        Добавить нового спикера.

        Args:
            name: Имя спикера
            extractor: Экстрактор эмбеддингов
            audio_samples: Список путей к файлам-образцам (если None, ищем в samples_dir)
        """
        if audio_samples is None:
            # Ищем файлы в папке speaker_samples/
            audio_samples = self._find_samples_for_speaker(name)

        if not audio_samples:
            raise ValueError(
                f"Не найдены образцы голоса для '{name}'. "
                f"Добавьте MP3 файлы в папку {self.samples_dir}/ с именем спикера "
                f"(например: {name}.mp3 или {name}_1.mp3, {name}_2.mp3)"
            )

        # Проверяем что файлы существуют
        for sample in audio_samples:
            if not Path(sample).exists():
                raise FileNotFoundError(f"Файл не найден: {sample}")

        print(f"      Извлечение эмбеддингов из {len(audio_samples)} файлов...")
        centroid = extractor.extract_from_files(audio_samples)

        now = datetime.now().isoformat()
        self.profiles["profiles"][name] = {
            "centroid": centroid,
            "samples": [str(s) for s in audio_samples],
            "created_at": now,
            "updated_at": now
        }

        self.save_profiles()
        print(f"      Спикер '{name}' добавлен ({len(audio_samples)} образцов)")

    def _find_samples_for_speaker(self, name: str) -> list:
        """Найти файлы-образцы для спикера в папке samples_dir."""
        if not self.samples_dir.exists():
            return []

        samples = []
        audio_extensions = {'.mp3', '.wav', '.flac', '.ogg', '.m4a'}

        # Ищем файлы вида: name.mp3, name_1.mp3, name_sample.mp3 и т.д.
        name_lower = name.lower()
        for file in self.samples_dir.iterdir():
            if file.is_file() and file.suffix.lower() in audio_extensions:
                file_stem_lower = file.stem.lower()
                # Проверяем что имя файла начинается с имени спикера
                if file_stem_lower == name_lower or file_stem_lower.startswith(name_lower + "_"):
                    samples.append(str(file))

        return sorted(samples)

    def remove_speaker(self, name: str):
        """Удалить профиль спикера."""
        if name not in self.profiles.get("profiles", {}):
            raise KeyError(f"Спикер '{name}' не найден")

        del self.profiles["profiles"][name]
        self.save_profiles()
        print(f"Спикер '{name}' удалён")

    def update_speaker(
        self,
        name: str,
        extractor: SpeakerEmbeddingExtractor,
        audio_samples: list = None
    ):
        """Обновить профиль спикера (пересчитать эмбеддинг)."""
        if name not in self.profiles.get("profiles", {}):
            raise KeyError(f"Спикер '{name}' не найден")

        if audio_samples is None:
            audio_samples = self._find_samples_for_speaker(name)

        if not audio_samples:
            raise ValueError(f"Не найдены образцы голоса для '{name}'")

        print(f"      Пересчёт эмбеддингов из {len(audio_samples)} файлов...")
        centroid = extractor.extract_from_files(audio_samples)

        self.profiles["profiles"][name]["centroid"] = centroid
        self.profiles["profiles"][name]["samples"] = [str(s) for s in audio_samples]
        self.profiles["profiles"][name]["updated_at"] = datetime.now().isoformat()

        self.save_profiles()
        print(f"      Спикер '{name}' обновлён")

    def get_all_centroids(self) -> tuple:
        """
        Получить имена и центроиды всех спикеров.

        Returns:
            names: list[str] - имена спикеров
            centroids: np.ndarray shape (num_speakers, 512)
        """
        profiles = self.profiles.get("profiles", {})

        if not profiles:
            return [], np.array([])

        names = list(profiles.keys())
        centroids = np.array([profiles[name]["centroid"] for name in names])

        return names, centroids

    def list_speakers(self) -> list:
        """Список всех спикеров с метаданными."""
        result = []
        for name, profile in self.profiles.get("profiles", {}).items():
            result.append({
                "name": name,
                "samples_count": len(profile.get("samples", [])),
                "samples": profile.get("samples", []),
                "created_at": profile.get("created_at"),
                "updated_at": profile.get("updated_at")
            })
        return result

    def has_profiles(self) -> bool:
        """Есть ли загруженные профили."""
        return len(self.profiles.get("profiles", {})) > 0
