"""
Система логирования взаимодействий пользователей с ботом.
"""

from pathlib import Path
from datetime import datetime, timezone
from typing import Optional
import json
import logging


class InteractionLogger:
    """Логирование взаимодействий с ботом."""

    def __init__(self, logs_dir: Path, enabled: bool = True):
        """
        Args:
            logs_dir: Корневая папка для логов
            enabled: Включить/выключить логирование
        """
        self.logs_dir = logs_dir
        self.enabled = enabled
        self.logger = logging.getLogger("interaction_logger")

        if enabled:
            logs_dir.mkdir(exist_ok=True, parents=True)

    def log_interaction(
        self,
        user_id: int,
        username: Optional[str],
        speaker: str,
        query: str,
        response: str,
        rag_results: list[dict],
        llm_metadata: dict,
        processing_time_ms: float,
        error: Optional[str] = None
    ) -> None:
        """Логирует одно взаимодействие."""

        if not self.enabled:
            return

        try:
            log_entry = self._build_log_entry(
                user_id=user_id,
                username=username,
                speaker=speaker,
                query=query,
                response=response,
                rag_results=rag_results,
                llm_metadata=llm_metadata,
                processing_time_ms=processing_time_ms,
                error=error
            )

            self._write_log(user_id, log_entry)

        except Exception as e:
            # Логирование НЕ должно ломать основной поток
            self.logger.error(f"Failed to log interaction: {e}", exc_info=True)

    def _build_log_entry(self, **kwargs) -> dict:
        """Строит структуру лога."""
        now = datetime.now(timezone.utc)

        # Обработка RAG results - топ-10 chunks
        chunks_summary = []
        for result in kwargs['rag_results'][:10]:
            chunk = result['chunk']
            chunks_summary.append({
                "chunk_id": chunk['id'],
                "podcast_id": chunk['podcast_id'],
                "timestamp_range": f"{chunk['timestamp_start']}-{chunk['timestamp_end']}",
                "score": round(result['score'], 4),
                "speakers_in_chunk": chunk['speakers'],
                "text": chunk['text']  # Полный текст chunk
            })

        return {
            "timestamp": now.isoformat(),
            "user_id": kwargs['user_id'],
            "username": kwargs.get('username'),
            "speaker": kwargs['speaker'],
            "interaction": {
                "query": kwargs['query'],
                "response": kwargs['response'],
                "response_length": len(kwargs['response'])
            },
            "rag_context": {
                "top_k": len(kwargs['rag_results']),
                "results_count": len(kwargs['rag_results']),
                "chunks": chunks_summary
            },
            "llm_metadata": kwargs['llm_metadata'],
            "processing_time_ms": round(kwargs['processing_time_ms'], 2),
            "error": kwargs.get('error')
        }

    def _write_log(self, user_id: int, log_entry: dict) -> None:
        """Записывает лог в JSONL файл."""
        now = datetime.now(timezone.utc)

        # Путь: logs/YYYY-MM/DD/user_{user_id}.jsonl
        month_dir = self.logs_dir / now.strftime("%Y-%m")
        day_dir = month_dir / now.strftime("%d")
        day_dir.mkdir(exist_ok=True, parents=True)

        log_file = day_dir / f"user_{user_id}.jsonl"

        # Append в JSONL формате
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')


# Singleton instance
_logger_instance: Optional[InteractionLogger] = None


def get_interaction_logger(logs_dir: Path = None, enabled: bool = True) -> InteractionLogger:
    """Получить singleton instance логгера."""
    global _logger_instance
    if _logger_instance is None:
        if logs_dir is None:
            from config import LOGS_DIR, LOG_ENABLED
            logs_dir = LOGS_DIR
            enabled = LOG_ENABLED
        _logger_instance = InteractionLogger(logs_dir, enabled)
    return _logger_instance
