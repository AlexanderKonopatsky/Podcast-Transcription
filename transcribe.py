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
    voice_threshold: float = 0.5,
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
        voice_threshold: Порог cosine distance для определения спикера (0-1)
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

    print(f"      Найдено {len(diarization_segments)} сегментов речи")

    # Освобождаем память
    del diarization_pipeline
    torch.cuda.empty_cache()

    # 3. Транскрибация - на GPU через ctranslate2 (работает с RTX 5080)
    print(f"\n[2/2] Транскрибация (распознавание речи)...")
    print(f"      Загрузка модели Whisper ({model_size})...")

    # faster-whisper использует ctranslate2 который поддерживает новые GPU
    whisper_device = "cuda" if cuda_available else "cpu"
    compute_type = "float16" if whisper_device == "cuda" else "int8"
    whisper_model = WhisperModel(model_size, device=whisper_device, compute_type=compute_type)

    print(f"      Распознавание речи...")
    segments, info = whisper_model.transcribe(
        audio_path,
        language=language,
        beam_size=5,
        word_timestamps=True,
        vad_filter=True,
    )

    # Собираем транскрипцию
    transcription_segments = []
    for segment in segments:
        transcription_segments.append({
            "start": segment.start,
            "end": segment.end,
            "text": segment.text.strip(),
            "words": [
                {"start": w.start, "end": w.end, "word": w.word}
                for w in (segment.words or [])
            ]
        })

    print(f"      Распознано {len(transcription_segments)} сегментов текста")

    del whisper_model
    torch.cuda.empty_cache()

    # 4. Объединение результатов (привязка спикеров к тексту)
    print(f"\nОбъединение результатов...")

    result_segments = []
    for trans_seg in transcription_segments:
        # Находим спикера для этого сегмента
        trans_mid = (trans_seg["start"] + trans_seg["end"]) / 2
        speaker = "Unknown"

        for diar_seg in diarization_segments:
            if diar_seg["start"] <= trans_mid <= diar_seg["end"]:
                speaker = diar_seg["speaker"]
                break

        result_segments.append({
            "start": trans_seg["start"],
            "end": trans_seg["end"],
            "text": trans_seg["text"],
            "speaker": speaker,
            "words": trans_seg["words"]
        })

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
                print(f"      Извлечение эмбеддингов спикеров...")
                extractor = SpeakerEmbeddingExtractor(hf_token, device=str(device))

                speaker_embeddings_dict = {}
                for label in diarization_labels:
                    segments = speaker_segments[label]
                    embedding = extractor.extract_from_segments(audio_path, segments)
                    if embedding is not None:
                        speaker_embeddings_dict[label] = embedding

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

    print("\nГотово!")

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

    print(f"Сохранено: {output_path}")
    return output_path


def process_batch(input_dir: str, output_dir: str, hf_token: str, **kwargs):
    """
    Обработка всех аудиофайлов из папки.

    Args:
        input_dir: Папка с подкастами
        output_dir: Папка для результатов
        hf_token: HuggingFace токен
        **kwargs: Параметры для transcribe_podcast
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # Поддерживаемые форматы
    audio_extensions = {'.mp3', '.wav', '.ogg', '.m4a', '.flac', '.opus', '.webm'}

    # Находим все аудиофайлы
    audio_files = [
        f for f in input_path.iterdir()
        if f.is_file() and f.suffix.lower() in audio_extensions
    ]

    if not audio_files:
        print(f"В папке {input_dir} не найдено аудиофайлов")
        print(f"Поддерживаемые форматы: {', '.join(audio_extensions)}")
        return 0

    print(f"Найдено {len(audio_files)} аудиофайлов для обработки\n")

    # Обрабатываем каждый файл
    for i, audio_file in enumerate(audio_files, 1):
        print("=" * 80)
        print(f"[{i}/{len(audio_files)}] Обработка: {audio_file.name}")
        print("=" * 80)

        try:
            # Транскрибация
            result = transcribe_podcast(str(audio_file), hf_token, **kwargs)

            # Формируем имя выходного файла
            output_name = audio_file.stem

            # Сохраняем результаты
            output_base = output_path / output_name

            # Сохраняем в нескольких форматах
            save_results(result, str(output_base), "txt")
            save_results(result, str(output_base), "json")
            save_results(result, str(output_base), "srt")

            print(f"\n✓ Завершено: {audio_file.name}\n")

        except Exception as e:
            print(f"\n✗ Ошибка при обработке {audio_file.name}: {e}\n")
            continue

    print("=" * 80)
    print(f"Обработка завершена! Результаты в папке: {output_dir}")
    print("=" * 80)

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

  # Batch обработка (все файлы из папки podcasts)
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
    parser.add_argument("--output-dir", default="output", help="Папка для результатов (для batch режима, по умолчанию: output)")
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
        default=0.5,
        help="Порог cosine distance для определения спикера (0-1, по умолчанию: 0.5)"
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
