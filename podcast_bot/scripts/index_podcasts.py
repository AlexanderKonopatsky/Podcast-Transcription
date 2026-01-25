"""
Script to index all podcasts for RAG.
Creates embeddings and FAISS index.
Supports incremental indexing.
"""

import argparse
import json
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import PODCASTS_DIR, DATA_DIR, CHUNK_MAX_TOKENS, CHUNK_OVERLAP_TOKENS
from rag.chunker import process_all_transcripts, process_transcript_file, extract_podcast_id
from rag.embeddings import get_embeddings, save_embeddings
from rag.metadata import (
    load_metadata, save_metadata, create_initial_metadata,
    create_metadata_from_chunks, create_file_metadata
)
from rag.file_tracker import (
    scan_transcript_changes, print_change_summary, FileStatus
)
from rag.index_merger import (
    merge_index_data, verify_index_integrity, print_verification_result
)


def load_or_create_metadata():
    """
    Загружает метаданные или создает их из существующих данных.

    Используется для миграции при первом запуске.
    """
    metadata_path = DATA_DIR / "index_metadata.json"
    metadata = load_metadata(metadata_path)

    if metadata is None:
        # Проверить наличие существующего индекса
        chunks_path = DATA_DIR / "chunks.json"

        if chunks_path.exists():
            print("Initializing metadata from existing index...")

            # Загрузить существующие chunks
            with open(chunks_path, 'r', encoding='utf-8') as f:
                chunks_data = json.load(f)

            # Создать метаданные
            indexing_params = {
                'chunk_max_tokens': CHUNK_MAX_TOKENS,
                'chunk_overlap_tokens': CHUNK_OVERLAP_TOKENS
            }

            metadata = create_initial_metadata(
                chunks_data, PODCASTS_DIR, indexing_params
            )

            # Сохранить метаданные
            save_metadata(metadata, metadata_path)

            print(f"  Created metadata for {metadata.total_files} files")
            print(f"  Total chunks: {metadata.total_chunks}")
        else:
            # Создать пустые метаданные для первого запуска
            from rag.metadata import IndexMetadata
            from datetime import datetime

            metadata = IndexMetadata(
                version="1.0",
                last_updated=datetime.now().isoformat(),
                indexing_params={
                    'chunk_max_tokens': CHUNK_MAX_TOKENS,
                    'chunk_overlap_tokens': CHUNK_OVERLAP_TOKENS
                },
                indexed_files={},
                total_chunks=0,
                total_files=0
            )

    return metadata


def incremental_index():
    """Инкрементальная индексация - обрабатывает только новые/измененные файлы."""
    print("=" * 50)
    print("Podcast Indexer (Incremental Mode)")
    print("=" * 50)

    # Шаг 1: Загрузить или создать метаданные
    metadata = load_or_create_metadata()

    # Шаг 2: Сканировать изменения
    changed_files, deleted_ids = scan_transcript_changes(PODCASTS_DIR, metadata)

    # Шаг 3: Если нет изменений - выход
    if not changed_files and not deleted_ids:
        print("\n[OK] Index up to date. No changes detected.")
        return

    # Шаг 4: Показать что будет обработано
    print_change_summary(changed_files, deleted_ids)

    # Шаг 5: Обработать измененные файлы
    print(f"[1/3] Processing changed files...")
    new_chunks = []
    modified_ids = set()

    for change in changed_files:
        chunks = process_transcript_file(
            change.filepath,
            max_tokens=CHUNK_MAX_TOKENS,
            overlap_tokens=CHUNK_OVERLAP_TOKENS
        )
        new_chunks.extend(chunks)

        if change.status == FileStatus.MODIFIED:
            modified_ids.add(change.podcast_id)

        status_marker = "[NEW]" if change.status == FileStatus.NEW else "[MOD]"
        print(f"  {status_marker} {change.filepath.name} -> {len(chunks)} chunks")

    # Шаг 6: Создать эмбеддинги только для новых чанков
    if new_chunks:
        print(f"\n[2/3] Creating embeddings for {len(new_chunks)} new chunks...")
        texts = [chunk.text for chunk in new_chunks]
        new_embeddings = get_embeddings(texts)
        print(f"  Created {len(new_embeddings)} embeddings (dimension: {len(new_embeddings[0])})")
    else:
        new_embeddings = []

    # Шаг 7: Объединить с существующими данными
    print(f"\n[3/3] Merging with existing index...")
    total_chunks = merge_index_data(
        new_chunks=new_chunks,
        new_embeddings=new_embeddings,
        modified_ids=modified_ids,
        deleted_ids=deleted_ids,
        data_dir=DATA_DIR
    )

    # Шаг 8: Обновить метаданные
    # Обновляем метаданные для каждого измененного файла
    for change in changed_files:
        podcast_id = change.podcast_id
        # Найти чанки этого файла
        file_chunks = [c for c in new_chunks if c.podcast_id == podcast_id]

        # Создать/обновить метаданные файла
        file_metadata = create_file_metadata(change.filepath, file_chunks)
        metadata.indexed_files[podcast_id] = file_metadata

    # Удалить метаданные удаленных файлов
    for podcast_id in deleted_ids:
        if podcast_id in metadata.indexed_files:
            del metadata.indexed_files[podcast_id]

    # Обновить глобальные параметры
    from datetime import datetime
    metadata.last_updated = datetime.now().isoformat()
    metadata.total_chunks = total_chunks
    metadata.total_files = len(metadata.indexed_files)

    # Сохранить метаданные
    save_metadata(metadata, DATA_DIR / "index_metadata.json")
    print(f"  Updated metadata: {DATA_DIR / 'index_metadata.json'}")

    # Итоговая статистика
    print("\n" + "=" * 50)
    print("[SUCCESS] Incremental indexing complete!")
    print(f"  Total chunks in index: {total_chunks}")
    print(f"  Total files indexed: {metadata.total_files}")
    print("=" * 50)


def full_reindex():
    """Полная переиндексация - обрабатывает все файлы заново."""
    print("=" * 50)
    print("Podcast Indexer (Full Reindex)")
    print("=" * 50)
    print("\n[WARNING] This will reprocess ALL files and recreate the entire index.\n")

    # Шаг 1: Parse and chunk transcripts
    print(f"[1/3] Processing transcripts from: {PODCASTS_DIR}")
    chunks = process_all_transcripts(
        PODCASTS_DIR,
        max_tokens=CHUNK_MAX_TOKENS,
        overlap_tokens=CHUNK_OVERLAP_TOKENS
    )

    if not chunks:
        print("ERROR: No chunks created. Check if transcripts exist.")
        return

    print(f"\nTotal chunks: {len(chunks)}")

    # Шаг 2: Create embeddings
    print(f"\n[2/3] Creating embeddings...")
    texts = [chunk.text for chunk in chunks]
    embeddings = get_embeddings(texts)
    print(f"Created {len(embeddings)} embeddings (dimension: {len(embeddings[0])})")

    # Шаг 3: Save everything
    print(f"\n[3/3] Saving to {DATA_DIR}...")

    # Save embeddings as numpy array
    embeddings_path = DATA_DIR / "embeddings.npy"
    save_embeddings(embeddings, embeddings_path)

    # Save chunks metadata as JSON
    chunks_path = DATA_DIR / "chunks.json"
    chunks_data = [chunk.to_dict() for chunk in chunks]
    with open(chunks_path, 'w', encoding='utf-8') as f:
        json.dump(chunks_data, f, ensure_ascii=False, indent=2)
    print(f"Saved chunks: {chunks_path}")

    # Create and save metadata
    indexing_params = {
        'chunk_max_tokens': CHUNK_MAX_TOKENS,
        'chunk_overlap_tokens': CHUNK_OVERLAP_TOKENS
    }
    metadata = create_metadata_from_chunks(chunks, PODCASTS_DIR, indexing_params)
    save_metadata(metadata, DATA_DIR / "index_metadata.json")
    print(f"Saved metadata: {DATA_DIR / 'index_metadata.json'}")

    # Summary
    print("\n" + "=" * 50)
    print("Indexing complete!")
    print(f"  - Chunks: {len(chunks)}")
    print(f"  - Files: {metadata.total_files}")
    print(f"  - Embeddings: {embeddings_path}")
    print("=" * 50)


def verify():
    """Проверка целостности индекса."""
    print("=" * 50)
    print("Index Integrity Verification")
    print("=" * 50)
    print()

    result = verify_index_integrity(DATA_DIR)
    print_verification_result(result)

    print()


def main():
    parser = argparse.ArgumentParser(
        description='Index podcasts for RAG with incremental support'
    )

    # Режимы работы
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        '--full-reindex',
        action='store_true',
        help='Force full reindexing: reprocess all files and rebuild index'
    )
    mode_group.add_argument(
        '--verify',
        action='store_true',
        help='Verify index integrity and show statistics'
    )

    args = parser.parse_args()

    # Выбор режима
    if args.verify:
        verify()
    elif args.full_reindex:
        full_reindex()
    else:
        incremental_index()  # По умолчанию


if __name__ == "__main__":
    main()
