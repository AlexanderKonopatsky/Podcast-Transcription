"""
Отслеживание изменений файлов транскриптов.
"""

from enum import Enum
from pathlib import Path
from typing import NamedTuple
from .metadata import IndexMetadata
from .chunker import extract_podcast_id


class FileStatus(Enum):
    """Статус файла относительно индекса."""
    NEW = "new"
    MODIFIED = "modified"
    UNCHANGED = "unchanged"


class FileChange(NamedTuple):
    """Информация об изменении файла."""
    filepath: Path
    podcast_id: str
    status: FileStatus
    old_chunk_count: int = 0


def get_file_status(
    filepath: Path,
    podcast_id: str,
    metadata: IndexMetadata
) -> FileStatus:
    """
    Определяет статус файла относительно индекса.

    Алгоритм:
    1. Если файл не в индексе -> NEW
    2. Если размер или время изменения отличается -> MODIFIED
    3. Иначе -> UNCHANGED

    Args:
        filepath: Путь к файлу транскрипции
        podcast_id: ID подкаста
        metadata: Метаданные индекса

    Returns:
        Статус файла
    """
    # Проверка 1: Файл не в индексе
    if podcast_id not in metadata.indexed_files:
        return FileStatus.NEW

    file_info = metadata.indexed_files[podcast_id]
    stat = filepath.stat()

    # Проверка 2: Быстрая проверка размера и времени модификации
    if stat.st_size != file_info.size or stat.st_mtime != file_info.mtime:
        return FileStatus.MODIFIED

    # Проверка 3: Без изменений
    return FileStatus.UNCHANGED


def scan_transcript_changes(
    transcripts_dir: Path,
    metadata: IndexMetadata
) -> tuple[list[FileChange], list[str]]:
    """
    Сканирует директорию транскриптов и определяет изменения.

    Args:
        transcripts_dir: Директория с .txt файлами транскриптов
        metadata: Метаданные индекса

    Returns:
        Кортеж из:
        - Список измененных файлов (новые + модифицированные)
        - Список удаленных podcast_id
    """
    changed_files: list[FileChange] = []

    # Сканируем все .txt файлы в директории
    for filepath in sorted(transcripts_dir.glob("*.txt")):
        podcast_id = extract_podcast_id(filepath)
        status = get_file_status(filepath, podcast_id, metadata)

        # Добавляем только новые и модифицированные файлы
        if status in (FileStatus.NEW, FileStatus.MODIFIED):
            old_chunk_count = 0
            if status == FileStatus.MODIFIED:
                old_chunk_count = metadata.indexed_files[podcast_id].chunk_count

            changed_files.append(FileChange(
                filepath=filepath,
                podcast_id=podcast_id,
                status=status,
                old_chunk_count=old_chunk_count
            ))

    # Детекция удаленных файлов
    deleted_ids = detect_deleted_files(transcripts_dir, metadata)

    return changed_files, deleted_ids


def detect_deleted_files(
    transcripts_dir: Path,
    metadata: IndexMetadata
) -> list[str]:
    """
    Находит файлы которые есть в индексе, но отсутствуют в директории.

    Args:
        transcripts_dir: Директория с транскриптами
        metadata: Метаданные индекса

    Returns:
        Список podcast_id удаленных файлов
    """
    # Получаем множество существующих podcast_id
    existing_ids = set()
    for filepath in transcripts_dir.glob("*.txt"):
        podcast_id = extract_podcast_id(filepath)
        existing_ids.add(podcast_id)

    # Находим ID которые есть в индексе, но нет в директории
    indexed_ids = set(metadata.indexed_files.keys())
    deleted_ids = list(indexed_ids - existing_ids)

    return deleted_ids


def print_change_summary(changed_files: list[FileChange], deleted_ids: list[str]):
    """
    Выводит красивую сводку об изменениях.

    Args:
        changed_files: Список измененных файлов
        deleted_ids: Список удаленных файлов
    """
    if not changed_files and not deleted_ids:
        print("[OK] Index up to date. No changes detected.")
        return

    print("\nChanges detected:")

    # Подсчет по типам
    new_count = sum(1 for f in changed_files if f.status == FileStatus.NEW)
    modified_count = sum(1 for f in changed_files if f.status == FileStatus.MODIFIED)

    print(f"  New files: {new_count}")
    print(f"  Modified files: {modified_count}")

    if deleted_ids:
        print(f"  Deleted files: {len(deleted_ids)}")

    print()
