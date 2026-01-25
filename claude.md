# Claude Project Guide

## Обзор проекта

Это приложение для автоматической транскрибации подкастов с определением спикеров (диаризацией) и идентификацией по голосу. Проект состоит из двух основных частей:
1. **Транскрибация с диаризацией** - основной функционал
2. **Podcast Bot** - RAG-бот для работы с транскрибированными подкастами

## Структура проекта

```
.
├── transcribe.py              # Основной скрипт транскрибации
├── speaker_identification/     # Модуль идентификации спикеров по голосу
│   ├── embeddings.py          # Работа с эмбеддингами и профилями спикеров
│   └── __init__.py
├── podcasts/                  # Входные аудиофайлы (MP3, WAV, etc.)
├── output/                    # Результаты транскрибации (TXT, JSON, SRT)
├── speaker_samples/           # Образцы голосов для идентификации
├── speaker_profiles.json      # Профили спикеров (эмбеддинги)
├── podcast_bot/               # RAG-бот для работы с транскрипциями
│   ├── main.py               # Основной скрипт бота
│   ├── bot_main.py           # Альтернативная точка входа
│   ├── config.py             # Конфигурация
│   ├── bot/                  # Модуль бота
│   ├── llm/                  # LLM интеграция
│   ├── rag/                  # RAG система
│   └── scripts/              # Вспомогательные скрипты
└── requirements_backup.txt    # Зависимости проекта
```

## Ключевые компоненты

### 1. Транскрибация ([transcribe.py](transcribe.py))

**Основной скрипт** для обработки аудио. Использует:
- **pyannote/speaker-diarization-3.1** - определение кто говорит
- **faster-whisper (large-v3)** - распознавание речи
- Параллельная обработка диаризации и транскрибации

**Основные режимы:**
- `--batch` - пакетная обработка всех файлов из podcasts/
- Одиночная обработка - `transcribe.py <файл>`
- `--add-speaker` - создание профилей спикеров
- `--list-speakers` - просмотр профилей

### 2. Speaker Identification ([speaker_identification/embeddings.py](speaker_identification/embeddings.py))

**Модуль идентификации спикеров** по голосу:
- Извлечение speaker embeddings (512-мерные векторы)
- Сохранение профилей в speaker_profiles.json
- Сопоставление через cosine distance (порог по умолчанию: 0.5)
- Модель: `pyannote/embedding`

**Workflow:**
1. Пользователь добавляет образцы в speaker_samples/
2. Создается профиль с эмбеддингами
3. При транскрибации SPEAKER_XX заменяются на имена

### 3. Podcast Bot ([podcast_bot/](podcast_bot/))

**RAG-бот** для взаимодействия с транскрибированными подкастами. Позволяет задавать вопросы по содержимому подкастов.

⚠️ **ВАЖНО:** Папка содержит приватные данные и конфиги - не коммитить в публичный репозиторий!

## Технический стек

### ML/AI модели
- **pyannote.audio** - диаризация и embeddings
- **faster-whisper** - ASR (Automatic Speech Recognition)
- **PyTorch** - backend для моделей
- **CUDA** - GPU ускорение (опционально)

### Аудио обработка
- **soundfile** - чтение аудио файлов
- **ffprobe** - определение длительности

### Форматы
- Вход: MP3, WAV, OGG, M4A, FLAC, OPUS, WEBM
- Выход: TXT (читаемый), JSON (полные данные), SRT (субтитры)

## Важные архитектурные особенности

### Патчи для совместимости

#### PyTorch 2.6+ compatibility ([transcribe.py](transcribe.py))
```python
# PyTorch 2.6+ использует weights_only=True по умолчанию
# pyannote требует weights_only=False
torch.load = patched_version(weights_only=False)
```

#### torchaudio nightly patches
Некоторые функции удалены в nightly:
- `torchaudio.info()` → заменено на soundfile
- `torchaudio.load()` → заменено на soundfile + GPU transfer
- `torchaudio.AudioMetaData` → создан dataclass-заглушка

### GPU оптимизации

**Последовательная обработка** для экономии VRAM:
1. Диаризация (освобождает память)
2. `torch.cuda.empty_cache()`
3. Транскрибация

**Batch размеры:**
- Embeddings: 32 (для эффективного GPU использования)
- Whisper: auto (зависит от модели)

### Поток данных

```
Аудио → [Диаризация (WHO)] → Временные сегменты спикеров
      ↘                     ↗
        [Транскрибация (WHAT)] → Текст с временными метками
                                ↓
                          Объединение → TXT/JSON/SRT
```

## Рекомендации по разработке

### При работе с transcribe.py
- Всегда тестировать на коротких файлах сначала
- Проверять VRAM usage при изменении batch размеров
- Учитывать, что диаризация и транскрибация идут параллельно

### При работе с speaker_identification
- Образцы голоса должны быть 15-30+ секунд
- Один голос в файле (не диалог)
- 2-3 образца для лучшей точности
- Threshold 0.5 - хороший баланс (можно настраивать --voice-threshold)

### При работе с podcast_bot
- ⚠️ НЕ коммитить конфиги с приватными данными
- Проверять .gitignore перед коммитом
- Учитывать, что бот работает с транскрипциями из output/

## HuggingFace токен

**Обязательно** для работы pyannote моделей:
1. Создать на https://huggingface.co/settings/tokens
2. Принять условия:
   - https://huggingface.co/pyannote/speaker-diarization-3.1
   - https://huggingface.co/pyannote/segmentation-3.0
   - https://huggingface.co/pyannote/embedding

Передавать через `--token YOUR_HF_TOKEN`

## Известные ограничения

1. **Количество спикеров** - нужно указывать заранее (`--speakers N`)
2. **VRAM** - large-v3 требует ~4-6 GB
3. **Качество диаризации** - зависит от качества аудио и различимости голосов
4. **Speaker ID** - требует предварительные образцы

## Git статус (snapshot)

```
Modified:
- speaker_identification/embeddings.py
- transcribe.py

Untracked:
- podcast_bot/ (содержит приватные данные)
- Различные .md файлы (документация)
```

## Команды быстрого старта

```bash
# Batch обработка
python transcribe.py --batch --token YOUR_HF_TOKEN

# Одиночный файл
python transcribe.py podcasts/file.mp3 --token YOUR_HF_TOKEN

# Добавить спикера
python transcribe.py --add-speaker "Имя" --token YOUR_HF_TOKEN

# Список спикеров
python transcribe.py --list-speakers --token YOUR_HF_TOKEN
```

## Язык проекта

Код и комментарии в основном на **русском языке**, документация смешанная (русский + английский).
