# Папка Speaker Samples

Поместите образцы голосов спикеров в эту папку для идентификации по голосу.

## Требования к образцам

- Минимум 15-30 секунд чистой речи
- Один голос в файле (без разговора)
- 2-3 образца на спикера для лучшей точности
- Поддерживаемый формат: `.mp3`

## Пример структуры

```
speaker_samples/
├── Зенур.mp3
├── Серега.mp3
└── Иван_Петрович.mp3
```

## Использование

Создание профилей спикеров:

```bash
python transcribe.py --add-speaker "Зенур" --token YOUR_HF_TOKEN
python transcribe.py --add-speaker "Серега" --token YOUR_HF_TOKEN
```

Просмотр созданных профилей:

```bash
python transcribe.py --list-speakers --token YOUR_HF_TOKEN
```
