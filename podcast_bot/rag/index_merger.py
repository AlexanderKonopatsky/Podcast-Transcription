"""
Слияние инкрементальных изменений с существующим индексом.
"""

import json
import numpy as np
from pathlib import Path
from typing import Optional


def load_existing_data(data_dir: Path) -> tuple[list[dict], np.ndarray]:
    """
    Загрузка существующих chunks и embeddings.

    Args:
        data_dir: Директория с данными

    Returns:
        Кортеж из (chunks, embeddings)
        Если файлы не существуют, возвращает пустые структуры
    """
    chunks_path = data_dir / "chunks.json"
    embeddings_path = data_dir / "embeddings.npy"

    # Загрузка chunks
    if chunks_path.exists():
        with open(chunks_path, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
    else:
        chunks = []

    # Загрузка embeddings
    if embeddings_path.exists():
        embeddings = np.load(embeddings_path)
    else:
        embeddings = np.array([], dtype=np.float32).reshape(0, 384)  # 384 - размерность

    return chunks, embeddings


def filter_old_chunks(
    chunks: list[dict],
    embeddings: np.ndarray,
    exclude_ids: set[str]
) -> tuple[list[dict], np.ndarray]:
    """
    Удаляет чанки указанных podcast_id из индекса.

    Args:
        chunks: Список чанков
        embeddings: Массив эмбеддингов
        exclude_ids: Множество podcast_id для удаления

    Returns:
        Отфильтрованные (chunks, embeddings)
    """
    if not exclude_ids:
        return chunks, embeddings

    filtered_chunks = []
    filtered_indices = []

    for i, chunk in enumerate(chunks):
        if chunk['podcast_id'] not in exclude_ids:
            filtered_chunks.append(chunk)
            filtered_indices.append(i)

    # Фильтрация embeddings по индексам
    if len(filtered_indices) > 0:
        filtered_embeddings = embeddings[filtered_indices]
    else:
        # Пустой массив с правильной размерностью
        filtered_embeddings = np.array([], dtype=np.float32).reshape(0, embeddings.shape[1])

    return filtered_chunks, filtered_embeddings


def merge_index_data(
    new_chunks: list,
    new_embeddings: list[list[float]],
    modified_ids: set[str],
    deleted_ids: list[str],
    data_dir: Path
) -> int:
    """
    Объединяет новые данные со старыми.

    Алгоритм:
    1. Загрузить существующие chunks и embeddings
    2. Отфильтровать чанки удаленных/измененных подкастов
    3. Добавить новые чанки и эмбеддинги
    4. Атомарно сохранить результат

    Args:
        new_chunks: Новые чанки (объекты Chunk)
        new_embeddings: Новые эмбеддинги (list[list[float]])
        modified_ids: Множество podcast_id модифицированных файлов
        deleted_ids: Список удаленных podcast_id
        data_dir: Директория для сохранения данных

    Returns:
        Общее количество чанков в индексе
    """
    # Шаг 1: Загрузить существующие данные
    old_chunks, old_embeddings = load_existing_data(data_dir)

    # Шаг 2: Отфильтровать удаленные и модифицированные
    exclude_ids = set(deleted_ids) | modified_ids

    filtered_chunks, filtered_embeddings = filter_old_chunks(
        old_chunks,
        old_embeddings,
        exclude_ids
    )

    print(f"  Old chunks: {len(old_chunks)}")
    if exclude_ids:
        removed_count = len(old_chunks) - len(filtered_chunks)
        print(f"  Removed chunks for modified/deleted files: {removed_count}")
        print(f"  Remaining chunks: {len(filtered_chunks)}")

    # Шаг 3: Добавить новые чанки и эмбеддинги
    new_chunks_dicts = [chunk.to_dict() for chunk in new_chunks]
    final_chunks = filtered_chunks + new_chunks_dicts

    # Конвертировать новые эмбеддинги в numpy array
    if new_embeddings:
        new_embeddings_array = np.array(new_embeddings, dtype=np.float32)
        # Объединить с отфильтрованными
        if filtered_embeddings.shape[0] > 0:
            final_embeddings = np.vstack([filtered_embeddings, new_embeddings_array])
        else:
            final_embeddings = new_embeddings_array
    else:
        final_embeddings = filtered_embeddings

    print(f"  New chunks added: {len(new_chunks)}")
    print(f"  Total chunks: {len(final_chunks)}")

    # Валидация синхронизации
    if len(final_chunks) != final_embeddings.shape[0]:
        raise ValueError(
            f"Data mismatch: {len(final_chunks)} chunks vs {final_embeddings.shape[0]} embeddings"
        )

    # Шаг 4: Атомарно сохранить
    atomic_save_index(final_chunks, final_embeddings, data_dir)

    return len(final_chunks)


def atomic_save_index(
    chunks: list[dict],
    embeddings: np.ndarray,
    data_dir: Path
):
    """
    Атомарное сохранение индекса через временные файлы.

    Args:
        chunks: Список чанков
        embeddings: Массив эмбеддингов
        data_dir: Директория для сохранения
    """
    data_dir.mkdir(parents=True, exist_ok=True)

    chunks_path = data_dir / "chunks.json"
    embeddings_path = data_dir / "embeddings.npy"

    chunks_temp = data_dir / "chunks_temp.json"
    embeddings_temp = data_dir / "embeddings_temp"  # np.save добавит .npy

    try:
        # Сохранить во временные файлы
        with open(chunks_temp, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)

        np.save(embeddings_temp, embeddings)  # Создаст embeddings_temp.npy

        # Атомарное переименование
        chunks_temp.replace(chunks_path)
        (data_dir / "embeddings_temp.npy").replace(embeddings_path)

        print(f"  Saved chunks: {chunks_path}")
        print(f"  Saved embeddings: {embeddings_path} ({embeddings.shape})")

    except Exception as e:
        # Удалить временные файлы при ошибке
        if chunks_temp.exists():
            chunks_temp.unlink()
        temp_npy = data_dir / "embeddings_temp.npy"
        if temp_npy.exists():
            temp_npy.unlink()
        raise e


def verify_index_integrity(data_dir: Path) -> dict:
    """
    Проверка целостности индекса.

    Args:
        data_dir: Директория с данными

    Returns:
        Словарь с результатами проверки:
        {
            'valid': bool,
            'errors': list[str],
            'warnings': list[str],
            'stats': {
                'chunks': int,
                'embeddings': int,
                'files': int
            }
        }
    """
    from .metadata import load_metadata

    errors = []
    warnings = []
    stats = {}

    try:
        # Загрузить данные
        chunks, embeddings = load_existing_data(data_dir)
        metadata = load_metadata(data_dir / "index_metadata.json")

        stats['chunks'] = len(chunks)
        stats['embeddings'] = embeddings.shape[0]

        # Проверка 1: Размеры chunks и embeddings совпадают
        if len(chunks) != embeddings.shape[0]:
            errors.append(
                f"Mismatch: {len(chunks)} chunks vs {embeddings.shape[0]} embeddings"
            )

        # Проверка 2: Метаданные существуют и корректны
        if metadata is None:
            warnings.append("Metadata file not found")
            stats['files'] = 0
        else:
            stats['files'] = len(metadata.indexed_files)

            # Проверка соответствия количества чанков
            expected_count = sum(
                f.chunk_count for f in metadata.indexed_files.values()
            )
            if expected_count != len(chunks):
                errors.append(
                    f"Metadata mismatch: {expected_count} expected chunks vs {len(chunks)} actual"
                )

            # Проверка параметров индексирования
            from config import CHUNK_MAX_TOKENS, CHUNK_OVERLAP_TOKENS
            current_params = {
                'chunk_max_tokens': CHUNK_MAX_TOKENS,
                'chunk_overlap_tokens': CHUNK_OVERLAP_TOKENS
            }
            if metadata.indexing_params != current_params:
                warnings.append(
                    "Indexing parameters changed - recommend --full-reindex"
                )

        # Проверка 3: Нет дублирующихся chunk_ids
        chunk_ids = [chunk['id'] for chunk in chunks]
        if len(chunk_ids) != len(set(chunk_ids)):
            errors.append("Duplicate chunk IDs found")

    except Exception as e:
        errors.append(f"Integrity check failed: {e}")

    return {
        'valid': len(errors) == 0,
        'errors': errors,
        'warnings': warnings,
        'stats': stats
    }


def print_verification_result(result: dict):
    """
    Выводит результаты проверки целостности.

    Args:
        result: Результат verify_index_integrity()
    """
    if result['valid']:
        print("[OK] Index integrity OK")
        stats = result['stats']
        print(f"  Files: {stats.get('files', 0)}")
        print(f"  Chunks: {stats.get('chunks', 0)}")
        print(f"  Embeddings: {stats.get('embeddings', 0)}")

        if result['warnings']:
            print("\nWarnings:")
            for warning in result['warnings']:
                print(f"  [!] {warning}")
    else:
        print("[ERROR] Integrity check failed:")
        for error in result['errors']:
            print(f"  - {error}")

        if result['warnings']:
            print("\nWarnings:")
            for warning in result['warnings']:
                print(f"  [!] {warning}")
