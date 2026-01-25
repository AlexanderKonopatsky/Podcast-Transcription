"""
Управление метаданными индекса для инкрементального индексирования.
"""

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass
class FileMetadata:
    """Метаданные одного индексированного файла."""
    filename: str
    size: int
    mtime: float
    chunk_ids: list[str]
    chunk_count: int
    indexed_at: str

    def to_dict(self) -> dict:
        """Преобразование в словарь для JSON."""
        return asdict(self)

    @staticmethod
    def from_dict(data: dict) -> 'FileMetadata':
        """Создание из словаря."""
        return FileMetadata(**data)


@dataclass
class IndexMetadata:
    """Глобальные метаданные индекса."""
    version: str
    last_updated: str
    indexing_params: dict[str, int]
    indexed_files: dict[str, FileMetadata]
    total_chunks: int
    total_files: int

    def to_dict(self) -> dict:
        """Преобразование в словарь для JSON."""
        return {
            'version': self.version,
            'last_updated': self.last_updated,
            'indexing_params': self.indexing_params,
            'indexed_files': {
                podcast_id: file_meta.to_dict()
                for podcast_id, file_meta in self.indexed_files.items()
            },
            'total_chunks': self.total_chunks,
            'total_files': self.total_files
        }

    @staticmethod
    def from_dict(data: dict) -> 'IndexMetadata':
        """Создание из словаря."""
        indexed_files = {
            podcast_id: FileMetadata.from_dict(file_data)
            for podcast_id, file_data in data['indexed_files'].items()
        }
        return IndexMetadata(
            version=data['version'],
            last_updated=data['last_updated'],
            indexing_params=data['indexing_params'],
            indexed_files=indexed_files,
            total_chunks=data['total_chunks'],
            total_files=data['total_files']
        )


def load_metadata(path: Path) -> Optional[IndexMetadata]:
    """
    Загрузка метаданных из файла.

    Args:
        path: Путь к файлу метаданных

    Returns:
        IndexMetadata или None если файл не существует
    """
    if not path.exists():
        return None

    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return IndexMetadata.from_dict(data)
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        print(f"[WARNING] Failed to load metadata from {path}: {e}")
        return None


def save_metadata(metadata: IndexMetadata, path: Path):
    """
    Атомарное сохранение метаданных в файл.

    Использует временный файл для атомарности операции.

    Args:
        metadata: Метаданные для сохранения
        path: Путь к файлу
    """
    # Создать директорию если не существует
    path.parent.mkdir(parents=True, exist_ok=True)

    # Записать во временный файл
    temp_path = path.with_suffix('.tmp')

    try:
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(metadata.to_dict(), f, ensure_ascii=False, indent=2)

        # Атомарное переименование
        temp_path.replace(path)
    except Exception as e:
        # Удалить временный файл в случае ошибки
        if temp_path.exists():
            temp_path.unlink()
        raise e


def create_file_metadata(filepath: Path, chunks: list) -> FileMetadata:
    """
    Создание метаданных для файла на основе его чанков.

    Args:
        filepath: Путь к файлу транскрипции
        chunks: Список чанков созданных из этого файла

    Returns:
        FileMetadata с информацией о файле
    """
    stat = filepath.stat()
    chunk_ids = [chunk.id for chunk in chunks]

    return FileMetadata(
        filename=filepath.name,
        size=stat.st_size,
        mtime=stat.st_mtime,
        chunk_ids=chunk_ids,
        chunk_count=len(chunks),
        indexed_at=datetime.now().isoformat()
    )


def create_initial_metadata(
    chunks_data: list[dict],
    transcripts_dir: Path,
    indexing_params: dict[str, int]
) -> IndexMetadata:
    """
    Создание начальных метаданных из существующего индекса.

    Используется при первом запуске для миграции существующих данных.

    Args:
        chunks_data: Список чанков из chunks.json
        transcripts_dir: Директория с транскрипциями
        indexing_params: Параметры индексирования (chunk_max_tokens, etc.)

    Returns:
        IndexMetadata с информацией обо всех файлах
    """
    # Импорт здесь чтобы избежать циклических зависимостей
    from .chunker import extract_podcast_id

    # Группируем чанки по podcast_id
    files_info: dict[str, list[dict]] = {}
    for chunk in chunks_data:
        podcast_id = chunk['podcast_id']
        if podcast_id not in files_info:
            files_info[podcast_id] = []
        files_info[podcast_id].append(chunk)

    # Создаем FileMetadata для каждого файла
    indexed_files: dict[str, FileMetadata] = {}

    for filepath in sorted(transcripts_dir.glob("*.txt")):
        podcast_id = extract_podcast_id(filepath)

        if podcast_id in files_info:
            file_chunks = files_info[podcast_id]
            stat = filepath.stat()

            indexed_files[podcast_id] = FileMetadata(
                filename=filepath.name,
                size=stat.st_size,
                mtime=stat.st_mtime,
                chunk_ids=[chunk['id'] for chunk in file_chunks],
                chunk_count=len(file_chunks),
                indexed_at=datetime.now().isoformat()
            )

    return IndexMetadata(
        version="1.0",
        last_updated=datetime.now().isoformat(),
        indexing_params=indexing_params,
        indexed_files=indexed_files,
        total_chunks=len(chunks_data),
        total_files=len(indexed_files)
    )


def create_metadata_from_chunks(
    chunks: list,
    transcripts_dir: Path,
    indexing_params: dict[str, int]
) -> IndexMetadata:
    """
    Создание метаданных из списка объектов Chunk.

    Используется при полной переиндексации.

    Args:
        chunks: Список объектов Chunk
        transcripts_dir: Директория с транскрипциями
        indexing_params: Параметры индексирования

    Returns:
        IndexMetadata с информацией обо всех файлах
    """
    # Импорт здесь чтобы избежать циклических зависимостей
    from .chunker import extract_podcast_id

    # Группируем чанки по podcast_id
    files_chunks: dict[str, list] = {}
    for chunk in chunks:
        podcast_id = chunk.podcast_id
        if podcast_id not in files_chunks:
            files_chunks[podcast_id] = []
        files_chunks[podcast_id].append(chunk)

    # Создаем FileMetadata для каждого файла
    indexed_files: dict[str, FileMetadata] = {}

    for filepath in sorted(transcripts_dir.glob("*.txt")):
        podcast_id = extract_podcast_id(filepath)

        if podcast_id in files_chunks:
            file_chunks = files_chunks[podcast_id]
            indexed_files[podcast_id] = create_file_metadata(filepath, file_chunks)

    return IndexMetadata(
        version="1.0",
        last_updated=datetime.now().isoformat(),
        indexing_params=indexing_params,
        indexed_files=indexed_files,
        total_chunks=len(chunks),
        total_files=len(indexed_files)
    )


def update_metadata(
    metadata: IndexMetadata,
    changed_files: list,
    deleted_ids: list[str],
    indexing_params: dict[str, int]
) -> IndexMetadata:
    """
    Обновление метаданных после инкрементальной индексации.

    Args:
        metadata: Текущие метаданные
        changed_files: Список объектов FileChange с информацией об изменениях
        deleted_ids: Список podcast_id удаленных файлов
        indexing_params: Параметры индексирования

    Returns:
        Обновленные метаданные
    """
    # Удалить метаданные для удаленных файлов
    for podcast_id in deleted_ids:
        if podcast_id in metadata.indexed_files:
            del metadata.indexed_files[podcast_id]

    # Обновить метаданные для измененных файлов
    for change in changed_files:
        from .chunker import extract_podcast_id
        podcast_id = extract_podcast_id(change.filepath)

        # Создать метаданные для этого файла из его чанков
        # Предполагается что чанки уже были обработаны в главной функции
        stat = change.filepath.stat()

        # Найти чанки этого файла в change (они должны быть переданы отдельно)
        # Это будет обновлено в основной функции индексирования
        # Здесь мы просто обновляем базовую информацию

        # Временно - эта функция будет вызвана из главного скрипта с нужными данными
        pass

    # Обновить глобальные параметры
    metadata.last_updated = datetime.now().isoformat()
    metadata.indexing_params = indexing_params
    metadata.total_files = len(metadata.indexed_files)

    # total_chunks будет обновлен в главной функции

    return metadata
