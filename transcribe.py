"""
Транскрибация подкастов с диаризацией спикеров.
Использует faster-whisper для транскрибации и pyannote для диаризации.
"""

import torch
import torch.serialization
import json
import numpy as np
from pathlib import Path
from datetime import timedelta

# Fix для PyTorch 2.6+ (weights_only=True по умолчанию)
# pyannote модели используют старый формат чекпоинтов
_original_torch_load = torch.serialization.load
def _patched_torch_load(f, *args, **kwargs):
    kwargs["weights_only"] = False
    return _original_torch_load(f, *args, **kwargs)
torch.serialization.load = _patched_torch_load
torch.load = _patched_torch_load

# Fix для torchaudio nightly: многие функции были удалены
# Создаем заглушки для совместимости с pyannote
import torchaudio
import soundfile as sf

if not hasattr(torchaudio, 'AudioMetaData'):
    from dataclasses import dataclass
    @dataclass
    class AudioMetaData:
        sample_rate: int
        num_frames: int
        num_channels: int
        bits_per_sample: int
        encoding: str
    torchaudio.AudioMetaData = AudioMetaData

if not hasattr(torchaudio, 'list_audio_backends'):
    torchaudio.list_audio_backends = lambda: ["soundfile"]

if not hasattr(torchaudio, 'info'):
    def _torchaudio_info(filepath, backend=None):
        info = sf.info(filepath)
        return AudioMetaData(
            sample_rate=info.samplerate,
            num_frames=info.frames,
            num_channels=info.channels,
            bits_per_sample=16,  # soundfile default
            encoding=info.subtype or "PCM_16"
        )
    torchaudio.info = _torchaudio_info

# Патч для torchaudio.load() - используем soundfile и загружаем на GPU
_original_torchaudio_load = torchaudio.load
def _patched_torchaudio_load(filepath, frame_offset=0, num_frames=-1, backend=None, **kwargs):
    """Загрузка аудио через soundfile, сразу на GPU если доступен."""
    data, samplerate = sf.read(filepath, dtype='float32', always_2d=True)
    # soundfile возвращает (samples, channels), torch ожидает (channels, samples)
    waveform = torch.from_numpy(data.T)

    # Сразу перемещаем на GPU для ускорения последующих операций (resampling)
    if torch.cuda.is_available():
        waveform = waveform.cuda()

    # Применяем offset и num_frames если указаны
    if frame_offset > 0:
        waveform = waveform[:, frame_offset:]
    if num_frames > 0:
        waveform = waveform[:, :num_frames]

    return waveform, samplerate
torchaudio.load = _patched_torchaudio_load

from faster_whisper import WhisperModel
from pyannote.audio import Pipeline

# Кэш Whisper модели для batch режима: кортеж (model_size, whisper_model)
# НЕ присваивать None - это вызывает деструктор ctranslate2 который падает с segfault
# Модель переиспользуется между файлами если размер совпадает
_whisper_model_cache = None

# Модуль идентификации спикеров по голосу
from speaker_identification import SpeakerEmbeddingExtractor, SpeakerProfileManager, SpeakerMatcher


def get_audio_duration(audio_path: str) -> float:
    """Получить длительность аудио в секундах."""
    import subprocess
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", audio_path],
        capture_output=True, text=True
    )
    try:
        return float(result.stdout.strip())
    except ValueError:
        return 0.0


def transcribe_podcast(
    audio_path: str,
    hf_token: str,
    num_speakers: int = 3,
    language: str = "ru",
    model_size: str = "large-v3",
    speaker_profiles_path: str = None,
    voice_threshold: float = 0.65,
) -> dict:
    """
    Транскрибация подкаста с диаризацией спикеров.

    Args:
        audio_path: Путь к аудиофайлу
        hf_token: HuggingFace токен для pyannote моделей
        num_speakers: Количество спикеров в подкасте
        language: Язык аудио (ru, en, etc.)
        model_size: Размер модели Whisper (tiny, base, small, medium, large-v3)
        speaker_profiles_path: Путь к файлу профилей спикеров (опционально)
        voice_threshold: Порог cosine distance для определения спикера (0-1, по умолчанию: 0.65)
    """
    # С PyTorch nightly (cu128) RTX 5080 поддерживается
    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_available else "cpu")

    print(f"PyTorch CUDA: {cuda_available}")
    if cuda_available:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"Диаризация: {'GPU (pyannote)' if cuda_available else 'CPU (pyannote)'}")
    print(f"Транскрибация: {'GPU (faster-whisper)' if cuda_available else 'CPU (faster-whisper)'}")

    # 1. Получаем информацию об аудио
    print(f"\nАудио: {audio_path}")
    duration = get_audio_duration(audio_path)
    if duration > 0:
        print(f"Длительность: {timedelta(seconds=int(duration))}")

    # 2. Диаризация (определение спикеров)
    print(f"\n[1/2] Диаризация (определение кто когда говорит)...")
    print(f"      Загрузка модели pyannote...")

    diarization_pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token
    )

    # Увеличиваем batch_size для эмбеддингов (по умолчанию 1)
    diarization_pipeline.embedding_batch_size = 32

    # Перемещаем на GPU если доступен
    if cuda_available:
        diarization_pipeline = diarization_pipeline.to(device)

    # Предзагружаем аудио в память для избежания многократного чтения с диска
    print(f"      Загрузка аудио в память...")
    waveform, sample_rate = torchaudio.load(audio_path)
    if cuda_available:
        waveform = waveform.cuda()

    # Передаем waveform вместо пути - pyannote не будет читать с диска
    audio_input = {"waveform": waveform, "sample_rate": sample_rate}

    print(f"      Анализ спикеров (ожидаем {num_speakers} спикеров)...")

    # Hook для отслеживания времени каждого этапа
    import time
    stage_times = {}
    last_stage_time = [time.time()]
    last_stage_name = [None]

    def timing_hook(step_name, step_artifact, file=None, **kwargs):
        now = time.time()
        if last_stage_name[0] is not None:
            elapsed = now - last_stage_time[0]
            stage_times[last_stage_name[0]] = elapsed
            print(f"        [{last_stage_name[0]}] {elapsed:.1f} сек")
        last_stage_time[0] = now
        last_stage_name[0] = step_name

    diarization, speaker_embeddings = diarization_pipeline(
        audio_input,
        num_speakers=num_speakers,
        hook=timing_hook,
        return_embeddings=True
    )

    # Финальный этап
    if last_stage_name[0] is not None:
        elapsed = time.time() - last_stage_time[0]
        stage_times[last_stage_name[0]] = elapsed
        print(f"        [{last_stage_name[0]}] {elapsed:.1f} сек")

    # Собираем сегменты диаризации
    diarization_segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        diarization_segments.append({
            "start": turn.start,
            "end": turn.end,
            "speaker": speaker
        })

    print(f"      Найдено {len(diarization_segments)} сегментов речи", flush=True)

    # Освобождаем память GPU - критично для длинных аудио!
    # Используем None вместо del чтобы избежать segfault в нативных библиотеках
    diarization_pipeline = None
    waveform = None
    audio_input = None
    if cuda_available:
        torch.cuda.empty_cache()
        print(f"      VRAM после очистки: {torch.cuda.memory_allocated() / 1024**3:.2f} GB", flush=True)

    # 3. Транскрибация - на GPU через ctranslate2 (работает с RTX 5080)
    print(f"\n[2/2] Транскрибация (распознавание речи)...", flush=True)

    # faster-whisper использует ctranslate2 который поддерживает новые GPU
    whisper_device = "cuda" if cuda_available else "cpu"
    compute_type = "float16" if whisper_device == "cuda" else "int8"

    # Управление глобальной моделью для batch режима
    # Кэш хранит кортеж (model_size, whisper_model) для переиспользования
    global _whisper_model_cache

    # Проверяем есть ли уже загруженная модель нужного размера
    if _whisper_model_cache is not None:
        cached_size, cached_model = _whisper_model_cache
        if cached_size == model_size:
            # Переиспользуем существующую модель
            print(f"      Используется загруженная модель Whisper ({model_size})", flush=True)
            whisper_model = cached_model
        else:
            # Размер другой - загружаем новую модель
            # НЕ удаляем старую модель - это вызывает segfault в ctranslate2
            print(f"      ВНИМАНИЕ: загрузка модели {model_size} (предыдущая {cached_size} остаётся в памяти)", flush=True)
            print(f"      Загрузка модели Whisper ({model_size})...", flush=True)
            whisper_model = WhisperModel(model_size, device=whisper_device, compute_type=compute_type)
            _whisper_model_cache = (model_size, whisper_model)
    else:
        # Первая загрузка модели
        print(f"      Загрузка модели Whisper ({model_size})...", flush=True)
        whisper_model = WhisperModel(model_size, device=whisper_device, compute_type=compute_type)
        _whisper_model_cache = (model_size, whisper_model)

    print(f"      Распознавание речи...", flush=True)

    # faster-whisper сам загружает аудио через ffmpeg (потоково, без MemoryError)
    segments, info = whisper_model.transcribe(
        str(audio_path),  # передаём путь к файлу, НЕ массив
        language=language,
        beam_size=5,
        word_timestamps=True,
        vad_filter=True,
    )

    print(f"      Язык: {info.language}, вероятность: {info.language_probability:.2f}", flush=True)

    # Собираем транскрипцию с прогрессом (чтобы видеть что скрипт работает)
    transcription_segments = []
    segment_count = 0
    try:
        for segment in segments:
            segment_count += 1
            if segment_count % 100 == 0:
                print(f"      Обработано {segment_count} сегментов...", flush=True)

            transcription_segments.append({
                "start": segment.start,
                "end": segment.end,
                "text": segment.text.strip(),
                "words": [
                    {"start": w.start, "end": w.end, "word": w.word}
                    for w in (segment.words or [])
                ]
            })
    except Exception as e:
        print(f"\n      ОШИБКА при сборе сегментов: {e}", flush=True)
        import traceback
        traceback.print_exc()
        raise

    print(f"      Распознано {len(transcription_segments)} сегментов текста", flush=True)

    # НЕ трогаем whisper_model - любое обращение к нему вызывает segfault в ctranslate2
    # Python сам освободит объект при выходе из функции

    # 4. Объединение результатов (привязка спикеров к тексту)
    print(f"\nОбъединение результатов...", flush=True)

    # Предварительно сортируем сегменты диаризации для бинарного поиска
    diarization_segments.sort(key=lambda x: x["start"])

    result_segments = []
    for i, trans_seg in enumerate(transcription_segments):
        # Находим спикера для этого сегмента
        trans_mid = (trans_seg["start"] + trans_seg["end"]) / 2
        speaker = "Unknown"

        # Бинарный поиск вместо линейного (O(log n) вместо O(n))
        left, right = 0, len(diarization_segments) - 1
        while left <= right:
            mid = (left + right) // 2
            diar_seg = diarization_segments[mid]
            if diar_seg["start"] <= trans_mid <= diar_seg["end"]:
                speaker = diar_seg["speaker"]
                break
            elif trans_mid < diar_seg["start"]:
                right = mid - 1
            else:
                left = mid + 1

        result_segments.append({
            "start": trans_seg["start"],
            "end": trans_seg["end"],
            "text": trans_seg["text"],
            "speaker": speaker,
            "words": trans_seg["words"]
        })

        # Прогресс каждые 1000 сегментов
        if (i + 1) % 1000 == 0:
            print(f"      Обработано {i + 1}/{len(transcription_segments)} сегментов...", flush=True)

    # 5. Идентификация спикеров по профилям (опционально)
    speaker_mapping = None
    if speaker_profiles_path:
        profiles_path = Path(speaker_profiles_path)
        if profiles_path.exists():
            print(f"\n[3/3] Идентификация спикеров по голосу...")
            profile_manager = SpeakerProfileManager(speaker_profiles_path)

            if profile_manager.has_profiles():
                print(f"      Загружено {len(profile_manager.list_speakers())} профилей")

                # Получаем метки спикеров из диаризации
                diarization_labels = list(set(seg["speaker"] for seg in diarization_segments))

                # Группируем сегменты по спикерам
                speaker_segments = {label: [] for label in diarization_labels}
                for seg in diarization_segments:
                    speaker_segments[seg["speaker"]].append(seg)

                # Извлекаем эмбеддинги для каждого спикера из аудио
                import time as _time
                print(f"      Извлечение эмбеддингов спикеров ({len(diarization_labels)} спикеров)...", flush=True)

                # Показываем статистику по сегментам каждого спикера
                print(f"      Статистика сегментов:", flush=True)
                for label in diarization_labels:
                    segs = speaker_segments[label]
                    total_dur = sum(s["end"] - s["start"] for s in segs)
                    print(f"        {label}: {len(segs)} сегментов, {total_dur:.1f} сек речи", flush=True)

                extractor = SpeakerEmbeddingExtractor(hf_token, device=str(device))
                extraction_start = _time.time()

                speaker_embeddings_dict = {}
                for idx, label in enumerate(diarization_labels, 1):
                    print(f"      [{idx}/{len(diarization_labels)}] Обработка {label}...", flush=True)
                    segments = speaker_segments[label]
                    embedding = extractor.extract_from_segments(
                        audio_path,
                        segments,
                        speaker_label=label,
                        verbose=True
                    )
                    if embedding is not None:
                        speaker_embeddings_dict[label] = embedding
                    else:
                        print(f"          ⚠ Не удалось извлечь эмбеддинг для {label}", flush=True)

                extraction_elapsed = _time.time() - extraction_start
                print(f"      Извлечение завершено за {extraction_elapsed:.1f} сек", flush=True)
                extractor.unload_model()

                # Формируем массив эмбеддингов в том же порядке что и labels
                valid_labels = [l for l in diarization_labels if l in speaker_embeddings_dict]
                if valid_labels:
                    diarization_embeddings = np.array([speaker_embeddings_dict[l] for l in valid_labels])

                    # Сопоставляем с профилями
                    matcher = SpeakerMatcher(profile_manager, threshold=voice_threshold)
                    speaker_mapping = matcher.match_speakers(diarization_embeddings, valid_labels)

                    # Показываем расстояния для отладки
                    distances = matcher.get_distances(diarization_embeddings, valid_labels)
                    for label, dists in distances.items():
                        dists_str = ", ".join([f"{name}: {d:.3f}" for name, d in dists.items()])
                        matched = speaker_mapping.get(label, label)
                        status = f"-> {matched}" if matched != label else "(не определён)"
                        print(f"      {label} {status} [{dists_str}]")

                    # Применяем маппинг к результатам
                    for segment in result_segments:
                        original_speaker = segment["speaker"]
                        segment["speaker"] = speaker_mapping.get(original_speaker, original_speaker)
                else:
                    print(f"      Не удалось извлечь эмбеддинги спикеров")
            else:
                print(f"      Профили спикеров пусты")
        else:
            print(f"\n      Файл профилей не найден: {speaker_profiles_path}")

    print(f"\nГотово! Итого сегментов: {len(result_segments)}", flush=True)

    return {
        "segments": result_segments,
        "language": language,
        "duration": duration,
        "num_speakers": num_speakers,
        "speaker_mapping": speaker_mapping
    }


def format_transcript(result: dict, include_timestamps: bool = True) -> str:
    """
    Форматирование результата в читаемый текст.

    Args:
        result: Результат транскрибации с диаризацией
        include_timestamps: Включать временные метки
    """
    lines = []
    current_speaker = None

    for segment in result.get("segments", []):
        speaker = segment.get("speaker", "Unknown")
        text = segment.get("text", "").strip()
        start = segment.get("start", 0)

        if not text:
            continue

        if speaker != current_speaker:
            current_speaker = speaker
            if include_timestamps:
                mins, secs = divmod(int(start), 60)
                hours, mins = divmod(mins, 60)
                if hours > 0:
                    timestamp = f"[{hours:02d}:{mins:02d}:{secs:02d}]"
                else:
                    timestamp = f"[{mins:02d}:{secs:02d}]"
                lines.append(f"\n{timestamp} {speaker}:")
            else:
                lines.append(f"\n{speaker}:")

        lines.append(f"  {text}")

    return "\n".join(lines).strip()


def save_results(result: dict, output_path: str, format: str = "txt"):
    """
    Сохранение результатов в файл.

    Args:
        result: Результат транскрибации
        output_path: Путь для сохранения
        format: Формат (txt, json, srt)
    """
    output_path = Path(output_path)

    try:
        if format == "json":
            output_path = output_path.with_suffix(".json")
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

        elif format == "srt":
            output_path = output_path.with_suffix(".srt")
            lines = []
            for i, segment in enumerate(result.get("segments", []), 1):
                start = segment.get("start", 0)
                end = segment.get("end", 0)
                speaker = segment.get("speaker", "")
                text = segment.get("text", "").strip()

                start_time = f"{int(start//3600):02d}:{int((start%3600)//60):02d}:{int(start%60):02d},{int((start%1)*1000):03d}"
                end_time = f"{int(end//3600):02d}:{int((end%3600)//60):02d}:{int(end%60):02d},{int((end%1)*1000):03d}"

                lines.append(str(i))
                lines.append(f"{start_time} --> {end_time}")
                if speaker:
                    lines.append(f"[{speaker}] {text}")
                else:
                    lines.append(text)
                lines.append("")

            with open(output_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))

        else:  # txt
            output_path = output_path.with_suffix(".txt")
            transcript = format_transcript(result)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(transcript)

        print(f"Сохранено: {output_path}", flush=True)
        return output_path

    except Exception as e:
        print(f"ОШИБКА при сохранении {output_path}: {e}", flush=True)
        import traceback
        traceback.print_exc()
        raise


def process_batch(input_dir: str, output_dir: str, hf_token: str, **kwargs):
    """
    Обработка всех аудиофайлов из папки.
    Пропускает файлы для которых уже существует .txt файл.
    Сохраняет результаты в ту же папку где находится MP3.

    Args:
        input_dir: Папка с подкастами
        output_dir: (DEPRECATED) Игнорируется - результаты сохраняются рядом с MP3
        hf_token: HuggingFace токен
        **kwargs: Параметры для transcribe_podcast
    """
    input_path = Path(input_dir)

    # Предупреждение о deprecated параметре
    if output_dir != "output":  # не стандартное значение
        print(f"⚠ ВНИМАНИЕ: --output-dir игнорируется в batch режиме")
        print(f"  Результаты сохраняются в ту же папку что и MP3 файлы\n")

    # Поддерживаемые форматы
    audio_extensions = {'.mp3', '.wav', '.ogg', '.m4a', '.flac', '.opus', '.webm'}

    # Находим все аудиофайлы
    all_audio_files = [
        f for f in input_path.iterdir()
        if f.is_file() and f.suffix.lower() in audio_extensions
    ]

    if not all_audio_files:
        print(f"В папке {input_dir} не найдено аудиофайлов")
        print(f"Поддерживаемые форматы: {', '.join(audio_extensions)}")
        return 0

    # Фильтруем файлы: пропускаем те, для которых есть TXT
    files_to_process = []
    skipped_files = []

    for audio_file in all_audio_files:
        txt_file = audio_file.with_suffix('.txt')
        if txt_file.exists():
            skipped_files.append(audio_file)
        else:
            files_to_process.append(audio_file)

    # Статистика
    print("=" * 80)
    print("BATCH ОБРАБОТКА")
    print("=" * 80)
    print(f"Папка: {input_dir}")
    print(f"Всего аудиофайлов: {len(all_audio_files)}")
    print(f"  - Новых (будут обработаны): {len(files_to_process)}")
    print(f"  - Пропущено (уже есть .txt): {len(skipped_files)}")

    if skipped_files:
        print(f"\nПропущенные файлы (уже обработаны):")
        for f in skipped_files:
            print(f"  ✓ {f.name}")

    if not files_to_process:
        print("\n" + "=" * 80)
        print("Нет новых файлов для обработки!")
        print("Совет: удалите .txt файл если хотите переобработать")
        print("=" * 80)
        return 0

    print(f"\nФайлы для обработки:")
    for f in files_to_process:
        print(f"  → {f.name}")
    print()

    # Обрабатываем только новые файлы
    success_count = 0
    error_count = 0

    for i, audio_file in enumerate(files_to_process, 1):
        print("=" * 80)
        print(f"[{i}/{len(files_to_process)}] Обработка: {audio_file.name}")
        print("=" * 80)

        try:
            # Транскрибация
            result = transcribe_podcast(str(audio_file), hf_token, **kwargs)

            # Формируем базовый путь в ТОЙ ЖЕ папке где MP3
            output_base = audio_file.with_suffix('')

            # Сохраняем результаты в 3 форматах
            save_results(result, str(output_base), "txt")
            save_results(result, str(output_base), "json")
            save_results(result, str(output_base), "srt")

            success_count += 1
            print(f"\n✓ Завершено: {audio_file.name}\n")

        except Exception as e:
            error_count += 1
            print(f"\n✗ Ошибка при обработке {audio_file.name}: {e}\n")
            import traceback
            traceback.print_exc()
            continue

    # Финальная статистика
    print("=" * 80)
    print("ОБРАБОТКА ЗАВЕРШЕНА")
    print("=" * 80)
    print(f"Результаты:")
    print(f"  ✓ Успешно обработано: {success_count}")
    if error_count > 0:
        print(f"  ✗ Ошибок: {error_count}")
    if skipped_files:
        print(f"  ⊘ Пропущено (уже обработаны): {len(skipped_files)}")
    print(f"\nВсе результаты сохранены в папке: {input_dir}")
    print("=" * 80)

    # НЕ очищаем _whisper_model_cache - это вызывает segfault в ctranslate2
    # Память освободится автоматически при завершении Python процесса

    return 0


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Транскрибация подкастов с диаризацией спикеров",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  # Обработка одного файла
  python transcribe.py audio.mp3 --token hf_xxxxx

  # Batch обработка (пропускает уже обработанные файлы)
  python transcribe.py --batch --token hf_xxxxx

  # Результаты сохраняются рядом с MP3:
  # podcasts/audio.mp3 -> podcasts/audio.txt, audio.json, audio.srt

  # Для переобработки файла - удалите .txt:
  del podcasts/audio.txt
  python transcribe.py --batch --token hf_xxxxx

  # С определением имён спикеров по голосу:
  # 1. Добавьте MP3 файлы в папку speaker_samples/ (например: Зенур.mp3, Серега_1.mp3)
  # 2. Создайте профили:
  python transcribe.py --add-speaker "Зенур" --token hf_xxxxx
  python transcribe.py --add-speaker "Серега" --token hf_xxxxx

  # 3. Посмотреть список спикеров:
  python transcribe.py --list-speakers --token hf_xxxxx

  # 4. Транскрибация с определением имён:
  python transcribe.py audio.mp3 --token hf_xxxxx
  python transcribe.py --batch --token hf_xxxxx

Перед использованием:
  1. Создайте токен на https://huggingface.co/settings/tokens
  2. Примите условия моделей:
     - https://huggingface.co/pyannote/speaker-diarization-3.1
     - https://huggingface.co/pyannote/segmentation-3.0
     - https://huggingface.co/pyannote/embedding
        """,
    )

    parser.add_argument("audio", nargs="?", help="Путь к аудиофайлу (не используется в batch режиме)")
    parser.add_argument("--token", required=True, help="HuggingFace токен")
    parser.add_argument("--batch", action="store_true", help="Режим batch обработки всех файлов из папки")
    parser.add_argument("--input", default="podcasts", help="Папка с подкастами (для batch режима, по умолчанию: podcasts)")
    parser.add_argument("--output-dir", default="output", help="(DEPRECATED в batch режиме) Результаты сохраняются в ту же папку что и MP3 файлы. Для одиночных файлов используйте --output")
    parser.add_argument("--speakers", type=int, default=3, help="Количество спикеров (по умолчанию: 3)")
    parser.add_argument("--language", default="ru", help="Язык аудио (по умолчанию: ru)")
    parser.add_argument(
        "--model",
        default="large-v3",
        choices=["tiny", "base", "small", "medium", "large-v3"],
        help="Размер модели Whisper (по умолчанию: large-v3)",
    )
    parser.add_argument("--output", help="Путь для сохранения одного файла (без расширения)")
    parser.add_argument(
        "--format",
        default="txt",
        choices=["txt", "json", "srt"],
        help="Формат вывода для одного файла (по умолчанию: txt)",
    )

    # Идентификация спикеров
    speaker_group = parser.add_argument_group("Идентификация спикеров по голосу")
    speaker_group.add_argument(
        "--speaker-profiles",
        default="speaker_profiles.json",
        help="Путь к файлу профилей спикеров (по умолчанию: speaker_profiles.json)"
    )
    speaker_group.add_argument(
        "--voice-threshold",
        type=float,
        default=0.65,
        help="Порог cosine distance для определения спикера (0-1, по умолчанию: 0.65)"
    )

    # Управление профилями
    profile_group = parser.add_argument_group("Управление профилями спикеров")
    profile_group.add_argument(
        "--add-speaker",
        metavar="NAME",
        help="Добавить спикера (образцы голоса ищутся в speaker_samples/)"
    )
    profile_group.add_argument(
        "--remove-speaker",
        metavar="NAME",
        help="Удалить спикера из профилей"
    )
    profile_group.add_argument(
        "--list-speakers",
        action="store_true",
        help="Показать список спикеров в профилях"
    )
    profile_group.add_argument(
        "--update-speakers",
        action="store_true",
        help="Обновить профили всех спикеров из speaker_samples/"
    )

    args = parser.parse_args()

    # Команды управления профилями спикеров
    if args.list_speakers:
        profiles_path = Path(args.speaker_profiles)
        if not profiles_path.exists():
            print("Файл профилей не найден. Добавьте спикеров через --add-speaker")
            return 0

        profile_manager = SpeakerProfileManager(args.speaker_profiles)
        speakers = profile_manager.list_speakers()

        if not speakers:
            print("Профили спикеров пусты. Добавьте спикеров через --add-speaker")
            return 0

        print(f"\nПрофили спикеров ({len(speakers)}):")
        print("-" * 60)
        for sp in speakers:
            print(f"  {sp['name']}")
            print(f"    Образцов: {sp['samples_count']}")
            for sample in sp['samples']:
                print(f"      - {sample}")
            print(f"    Создан: {sp['created_at']}")
        print("-" * 60)
        return 0

    if args.add_speaker:
        print(f"\nДобавление спикера: {args.add_speaker}")
        profile_manager = SpeakerProfileManager(args.speaker_profiles)
        extractor = SpeakerEmbeddingExtractor(args.token)

        try:
            profile_manager.add_speaker(args.add_speaker, extractor)
            extractor.unload_model()
            print(f"\nСпикер '{args.add_speaker}' успешно добавлен!")
            print(f"Профили сохранены в: {args.speaker_profiles}")
        except ValueError as e:
            print(f"\nОшибка: {e}")
            print(f"\nДобавьте MP3 файлы в папку speaker_samples/")
            print(f"Примеры имён файлов:")
            print(f"  - {args.add_speaker}.mp3")
            print(f"  - {args.add_speaker}_1.mp3, {args.add_speaker}_2.mp3")
            return 1
        except Exception as e:
            print(f"\nОшибка: {e}")
            return 1
        return 0

    if args.remove_speaker:
        profile_manager = SpeakerProfileManager(args.speaker_profiles)
        try:
            profile_manager.remove_speaker(args.remove_speaker)
            print(f"Спикер '{args.remove_speaker}' удалён")
        except KeyError as e:
            print(f"Ошибка: {e}")
            return 1
        return 0

    if args.update_speakers:
        print("\nОбновление профилей всех спикеров...")
        profile_manager = SpeakerProfileManager(args.speaker_profiles)
        speakers = profile_manager.list_speakers()

        if not speakers:
            print("Нет спикеров для обновления")
            return 0

        extractor = SpeakerEmbeddingExtractor(args.token)
        for sp in speakers:
            print(f"\n  Обновление: {sp['name']}")
            try:
                profile_manager.update_speaker(sp['name'], extractor)
            except Exception as e:
                print(f"    Ошибка: {e}")

        extractor.unload_model()
        print("\nОбновление завершено!")
        return 0

    # Batch режим
    if args.batch:
        # Проверяем папки
        input_path = Path(args.input)
        output_path = Path(args.output_dir)

        if not input_path.exists():
            print(f"Ошибка: папка не найдена: {input_path}")
            return 1

        # Создаем выходную папку если нужно
        output_path.mkdir(exist_ok=True)

        return process_batch(
            str(input_path),
            str(output_path),
            args.token,
            num_speakers=args.speakers,
            language=args.language,
            model_size=args.model,
            speaker_profiles_path=args.speaker_profiles,
            voice_threshold=args.voice_threshold,
        )

    # Обычный режим - один файл
    if not args.audio:
        parser.error("audio argument is required when not using --batch mode")

    # Проверяем файл
    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(f"Ошибка: файл не найден: {audio_path}")
        return 1

    # Транскрибация
    result = transcribe_podcast(
        str(audio_path),
        args.token,
        num_speakers=args.speakers,
        language=args.language,
        model_size=args.model,
        speaker_profiles_path=args.speaker_profiles,
        voice_threshold=args.voice_threshold,
    )

    # Вывод результата
    transcript = format_transcript(result)
    print("\n" + "=" * 60)
    print("ТРАНСКРИПЦИЯ:")
    print("=" * 60)
    print(transcript)

    # Сохранение
    if args.output:
        save_results(result, args.output, args.format)
    else:
        # Сохраняем рядом с аудио
        output_path = audio_path.with_suffix("")
        save_results(result, str(output_path), args.format)

    return 0


if __name__ == "__main__":
    exit(main())
