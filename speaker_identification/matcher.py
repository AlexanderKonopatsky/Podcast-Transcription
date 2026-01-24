"""
Сопоставление спикеров из диаризации с известными профилями.
"""

import numpy as np
from scipy.spatial.distance import cdist

from .profiles import SpeakerProfileManager


class SpeakerMatcher:
    """Сопоставление спикеров с известными профилями."""

    def __init__(
        self,
        profile_manager: SpeakerProfileManager,
        threshold: float = 0.5
    ):
        """
        Args:
            profile_manager: Менеджер профилей спикеров
            threshold: Порог cosine distance для определения спикера (0-1)
                       Меньше = строже, больше = мягче
                       Рекомендуемые значения: 0.3-0.6
        """
        self.profile_manager = profile_manager
        self.threshold = threshold

    def match_speakers(
        self,
        diarization_embeddings: np.ndarray,
        diarization_labels: list
    ) -> dict:
        """
        Сопоставить спикеров диаризации с профилями.

        Args:
            diarization_embeddings: np.ndarray shape (num_speakers, 512)
                                   Центроиды спикеров из диаризации
            diarization_labels: list[str] - метки ["SPEAKER_00", "SPEAKER_01", ...]

        Returns:
            mapping: dict - {"SPEAKER_00": "Иван", "SPEAKER_01": "SPEAKER_01", ...}
                    Если спикер не определён, возвращается его исходная метка
        """
        profile_names, profile_centroids = self.profile_manager.get_all_centroids()

        # Если нет профилей, возвращаем исходные метки
        if len(profile_centroids) == 0:
            return {label: label for label in diarization_labels}

        # Если нет эмбеддингов диаризации, возвращаем исходные метки
        if diarization_embeddings is None or len(diarization_embeddings) == 0:
            return {label: label for label in diarization_labels}

        # Матрица cosine distances: (num_diarization_speakers, num_profile_speakers)
        distances = cdist(diarization_embeddings, profile_centroids, metric="cosine")

        mapping = {}
        used_profiles = set()

        # Сортируем спикеров по минимальному расстоянию до профилей
        # Это позволяет сначала назначить наиболее уверенные совпадения
        speaker_min_distances = [(i, distances[i].min()) for i in range(len(diarization_labels))]
        speaker_min_distances.sort(key=lambda x: x[1])

        for speaker_idx, min_dist in speaker_min_distances:
            label = diarization_labels[speaker_idx]

            # Находим ближайший неиспользованный профиль
            best_profile_idx = None
            best_distance = float('inf')

            for profile_idx in range(len(profile_names)):
                if profile_names[profile_idx] not in used_profiles:
                    dist = distances[speaker_idx, profile_idx]
                    if dist < best_distance:
                        best_distance = dist
                        best_profile_idx = profile_idx

            # Проверяем порог
            if best_profile_idx is not None and best_distance < self.threshold:
                mapping[label] = profile_names[best_profile_idx]
                used_profiles.add(profile_names[best_profile_idx])
            else:
                # Спикер не определён, оставляем исходную метку
                mapping[label] = label

        return mapping

    def get_distances(
        self,
        diarization_embeddings: np.ndarray,
        diarization_labels: list
    ) -> dict:
        """
        Получить расстояния от каждого спикера до всех профилей.
        Полезно для отладки и настройки порога.

        Returns:
            distances: dict - {
                "SPEAKER_00": {"Иван": 0.23, "Мария": 0.67},
                "SPEAKER_01": {"Иван": 0.71, "Мария": 0.18}
            }
        """
        profile_names, profile_centroids = self.profile_manager.get_all_centroids()

        if len(profile_centroids) == 0 or diarization_embeddings is None:
            return {}

        distances_matrix = cdist(diarization_embeddings, profile_centroids, metric="cosine")

        result = {}
        for i, label in enumerate(diarization_labels):
            result[label] = {
                profile_names[j]: round(distances_matrix[i, j], 4)
                for j in range(len(profile_names))
            }

        return result
