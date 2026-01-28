"""
Visualizer for sentiment trends.
"""

import io
import json
from typing import List, Dict
from datetime import datetime
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def create_sentiment_chart(sentiments: List[Dict], entity: str) -> io.BytesIO:
    """
    Create a sentiment trend chart.

    Args:
        sentiments: List of sentiment results
        entity: Entity name for title

    Returns:
        BytesIO buffer with PNG image
    """
    if not sentiments:
        raise ValueError("No sentiments to visualize")

    # Load known speakers from speaker_profiles.json
    try:
        profiles_path = Path(__file__).parent.parent.parent / "speaker_profiles.json"
        if profiles_path.exists():
            with open(profiles_path, 'r', encoding='utf-8') as f:
                profiles = json.load(f)
                known_speakers = set(profiles.get('profiles', {}).keys())
        else:
            known_speakers = None  # If no profiles file, show all
    except Exception as e:
        print(f"Не удалось загрузить speaker_profiles.json: {e}")
        known_speakers = None

    # Filter sentiments - only known speakers
    if known_speakers:
        filtered_sentiments = [
            s for s in sentiments
            if s['speaker'] in known_speakers
        ]

        if not filtered_sentiments:
            # Fallback: if no known speakers found, show all
            print(f"Внимание: Нет известных спикеров в результатах. Показываем всех.")
            filtered_sentiments = sentiments
        else:
            removed_count = len(sentiments) - len(filtered_sentiments)
            if removed_count > 0:
                print(f"Отфильтровано {removed_count} неопознанных спикеров (Unknown, SPEAKER_XX)")
    else:
        filtered_sentiments = sentiments

    # Group by speaker
    by_speaker = {}
    for s in filtered_sentiments:
        speaker = s['speaker']
        if speaker not in by_speaker:
            by_speaker[speaker] = []
        by_speaker[speaker].append(s)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot each speaker
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E']
    for idx, (speaker, speaker_sentiments) in enumerate(by_speaker.items()):
        # Sort by date
        speaker_sentiments.sort(key=lambda x: x['date'])

        # Extract dates and scores
        dates = []
        scores = []
        for s in speaker_sentiments:
            try:
                date_str = s['date']

                # Try ISO format first (YYYY-MM-DD)
                try:
                    date_obj = datetime.fromisoformat(date_str)
                except:
                    # Try other common formats
                    for fmt in ['%Y-%m-%d', '%d.%m.%Y', '%d/%m/%Y', '%Y/%m/%d']:
                        try:
                            date_obj = datetime.strptime(date_str, fmt)
                            break
                        except:
                            continue
                    else:
                        # If all formats fail, skip this entry
                        print(f"Не удалось распарсить дату: {date_str}")
                        continue

                dates.append(date_obj)
                scores.append(s['score'])
            except (ValueError, TypeError, AttributeError) as e:
                print(f"Ошибка обработки даты {s.get('date', 'unknown')}: {e}")
                continue

        if dates:
            color = colors[idx % len(colors)]
            ax.plot(dates, scores, marker='o', label=speaker,
                   linewidth=2, markersize=8, color=color)

    # Styling
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.set_xlabel('Дата', fontsize=12)
    ax.set_ylabel('Sentiment Score', fontsize=12)
    ax.set_title(f'Изменение мнений о "{entity}"', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle=':')

    # Format x-axis - показывать больше дат
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m.%Y'))
    # Показывать каждые 2 недели (14 дней) для лучшей читаемости
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=14))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Y-axis limits
    ax.set_ylim(-1.1, 1.1)

    # Add sentiment zones (background colors)
    ax.axhspan(-1.1, -0.6, alpha=0.05, color='red', label='Очень негативно')
    ax.axhspan(-0.6, -0.2, alpha=0.03, color='red')
    ax.axhspan(-0.2, 0.2, alpha=0.02, color='gray')
    ax.axhspan(0.2, 0.6, alpha=0.03, color='green')
    ax.axhspan(0.6, 1.1, alpha=0.05, color='green', label='Очень позитивно')

    plt.tight_layout()

    # Save to buffer
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
    buffer.seek(0)
    plt.close(fig)

    return buffer
