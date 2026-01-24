# Транскрибация подкастов с диаризацией

Автоматическая транскрибация подкастов с определением спикеров (диаризацией) и идентификацией по голосу.

## Структура проекта

```
.
├── podcasts/              # Папка для ваших подкастов (входные файлы)
├── output/                # Папка с результатами транскрибации
├── speaker_samples/       # Образцы голосов спикеров (MP3 файлы)
├── speaker_profiles.json  # Профили спикеров (создаётся автоматически)
├── transcribe.py          # Основной скрипт
├── speaker_identification/  # Модуль идентификации спикеров
└── requirements_backup.txt  # Зависимости
```

## Быстрый старт

### 1. Подготовка

Поместите ваши аудиофайлы в папку `podcasts/`

Поддерживаемые форматы: `.mp3`, `.wav`, `.ogg`, `.m4a`, `.flac`, `.opus`, `.webm`

### 2. Batch обработка всех подкастов

```bash
python transcribe.py --batch --token YOUR_HF_TOKEN
```

Это обработает все файлы из папки `podcasts/` и сохранит результаты в `output/`

### 3. Обработка одного файла

```bash
python transcribe.py podcasts/podcast.mp3 --token YOUR_HF_TOKEN
```

## Настройки

### Количество спикеров

```bash
python transcribe.py --batch --token YOUR_HF_TOKEN --speakers 2
```

### Размер модели

```bash
# Быстрее, но менее точно
python transcribe.py --batch --token YOUR_HF_TOKEN --model medium

# Точнее, но медленнее (по умолчанию)
python transcribe.py --batch --token YOUR_HF_TOKEN --model large-v3
```

Доступные модели: `tiny`, `base`, `small`, `medium`, `large-v3`

### Язык

```bash
python transcribe.py --batch --token YOUR_HF_TOKEN --language en
```

### Другие папки

```bash
python transcribe.py --batch --token YOUR_HF_TOKEN \
  --input my_podcasts \
  --output-dir my_results
```

## Форматы вывода

В batch режиме создаются 3 файла для каждого подкаста:

- `output/podcast_name.txt` - читаемый текст с временными метками
- `output/podcast_name.json` - полные данные в JSON
- `output/podcast_name.srt` - субтитры SRT формата

## Идентификация спикеров по голосу

Вместо SPEAKER_00, SPEAKER_01 можно показывать реальные имена спикеров!

### 1. Добавьте образцы голосов

Поместите MP3 файлы с образцами голосов в папку `speaker_samples/`:

```
speaker_samples/
├── Зенур.mp3           # или Зенур_1.mp3, Зенур_2.mp3
├── Серега.mp3
└── Иван_Петрович.mp3
```

**Требования к образцам:**
- Минимум 15-30 секунд речи
- Один голос в файле (не разговор)
- 2-3 образца для лучшей точности

### 2. Создайте профили спикеров

```bash
python transcribe.py --add-speaker "Зенур" --token YOUR_HF_TOKEN
python transcribe.py --add-speaker "Серега" --token YOUR_HF_TOKEN
```

### 3. Посмотрите список спикеров

```bash
python transcribe.py --list-speakers --token YOUR_HF_TOKEN
```

### 4. Транскрибируйте с определением имён

```bash
# Один файл
python transcribe.py podcast.mp3 --token YOUR_HF_TOKEN

# Batch обработка
python transcribe.py --batch --token YOUR_HF_TOKEN
```

### Результат

**До:**
```
[00:00] SPEAKER_00:
  Всем привет, сегодня у нас 7 марта...
```

**После:**
```
[00:00] Серега:
  Всем привет, сегодня у нас 7 марта...
```

### Дополнительные команды

```bash
# Удалить спикера
python transcribe.py --remove-speaker "Иван" --token YOUR_HF_TOKEN

# Обновить все профили (если добавили новые образцы)
python transcribe.py --update-speakers --token YOUR_HF_TOKEN

# Настроить порог определения (0-1, по умолчанию 0.5)
python transcribe.py podcast.mp3 --token YOUR_HF_TOKEN --voice-threshold 0.4
```

## HuggingFace токен

1. Создайте токен на https://huggingface.co/settings/tokens
2. Примите условия использования моделей:
   - https://huggingface.co/pyannote/speaker-diarization-3.1
   - https://huggingface.co/pyannote/segmentation-3.0
   - https://huggingface.co/pyannote/embedding

## Примеры вывода

### TXT формат
```
[00:15] SPEAKER_00:
  Добро пожаловать в наш подкаст!

[00:20] SPEAKER_01:
  Спасибо, что пригласили.
```

### JSON формат
Содержит полную информацию: сегменты, временные метки, слова, спикеры

### SRT формат
Стандартные субтитры для видео/аудио плееров
