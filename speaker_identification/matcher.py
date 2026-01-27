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
        threshold: float = 0.5,
        enable_fallback: bool = True
    ):
        """
        Args:
            profile_manager: Менеджер профилей спикеров
            threshold: Порог cosine distance для определения спикера (0-1)
                       Меньше = строже, больше = мягче
                       Рекомендуемые значения: 0.3-0.6
            enable_fallback: Включить fallback assignment для неопознанных спикеров.
                            Если True и есть неопознанные спикеры + неиспользованные профили,
                            назначить их по минимальному расстоянию (БЕЗ учета threshold)
        """
        self.profile_manager = profile_manager
        self.threshold = threshold
        self.enable_fallback = enable_fallback

    def match_speakers(
        self,
        diarization_embeddings: np.ndarray,
        diarization_labels: list
    ) -> dict:
        """
        Сопоставить спикеров диаризации с профилями.

        Phase 1: Жадное сопоставление с порогом threshold.
        Phase 2 (Fallback): Если enable_fallback=True и есть неопознанные спикеры,
                            назначает их на оставшиеся профили БЕЗ порога.

        Args:
            diarization_embeddings: np.ndarray shape (num_speakers, 512)
                                   Центроиды спикеров из диаризации
            diarization_labels: list[str] - метки ["SPEAKER_00", "SPEAKER_01", ...]

        Returns:
            mapping: dict - {"SPEAKER_00": "Иван", "SPEAKER_01": "SPEAKER_01", ...}
                    Phase 1: Спикер идентифицирован если distance < threshold
                    Phase 2 (Fallback): Неопознанные спикеры назначаются на оставшиеся
                                       профили по минимальному расстоянию (БЕЗ порога)
                    Если спикер не может быть определён, возвращается его исходная метка
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

        # Phase 2: Fallback Assignment
        if self.enable_fallback:
            # Находим неназначенные спикеры (те, что остались SPEAKER_XX)
            unmatched_speakers = [label for label in diarization_labels
                                  if mapping[label] == label]

            # Находим неиспользованные профили
            unused_profiles = set(profile_names) - used_profiles

            # Если есть и те и другие, выполняем fallback
            if unmatched_speakers and unused_profiles:
                # Логируем начало fallback
                print(f"      [Fallback] Обнаружено {len(unmatched_speakers)} неопознанных спикеров "
                      f"и {len(unused_profiles)} неиспользованных профилей")

                # Сортируем неназначенных спикеров по минимальному расстоянию
                # до неиспользованных профилей (сначала наиболее похожие)
                unmatched_with_min_dist = []
                for speaker_label in unmatched_speakers:
                    speaker_idx = diarization_labels.index(speaker_label)
                    min_dist_to_unused = float('inf')

                    for profile_name in unused_profiles:
                        profile_idx = profile_names.index(profile_name)
                        dist = distances[speaker_idx, profile_idx]
                        min_dist_to_unused = min(min_dist_to_unused, dist)

                    unmatched_with_min_dist.append((speaker_label, min_dist_to_unused))

                # Сортируем по минимальному расстоянию (жадный подход)
                unmatched_with_min_dist.sort(key=lambda x: x[1])

                # Назначаем fallback
                for speaker_label, _ in unmatched_with_min_dist:
                    if not unused_profiles:
                        break

                    speaker_idx = diarization_labels.index(speaker_label)

                    # Находим ближайший неиспользованный профиль
                    best_profile = None
                    best_distance = float('inf')

                    for profile_name in unused_profiles:
                        profile_idx = profile_names.index(profile_name)
                        dist = distances[speaker_idx, profile_idx]

                        if dist < best_distance:
                            best_distance = dist
                            best_profile = profile_name

                    # Назначаем
                    if best_profile is not None:
                        mapping[speaker_label] = best_profile
                        unused_profiles.remove(best_profile)

                        # Логируем fallback назначение
                        print(f"      [Fallback] {speaker_label} -> {best_profile} "
                              f"(distance={best_distance:.3f}, превышает threshold={self.threshold:.3f})")

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
