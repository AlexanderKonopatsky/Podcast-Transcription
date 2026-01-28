"""
TXT exporter for sentiment analysis results.
"""

from typing import List, Dict
from datetime import datetime
import io


def export_sentiment_to_txt(sentiments: List[Dict], entity: str) -> io.BytesIO:
    """
    Export sentiment analysis to detailed TXT report.

    Args:
        sentiments: List of sentiment results with combined_text
        entity: Entity being analyzed

    Returns:
        BytesIO buffer with UTF-8 encoded text
    """
    if not sentiments:
        raise ValueError("No sentiments to export")

    lines = []

    # Header
    lines.append("=" * 80)
    lines.append(f"SENTIMENT ANALYSIS REPORT: {entity}")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Total Podcasts Analyzed: {len(sentiments)}")
    lines.append("=" * 80)
    lines.append("")

    # Individual entries
    for idx, s in enumerate(sentiments, 1):
        lines.append(f"[{idx}] PODCAST: {s['date']} | SPEAKER: {s['speaker']}")
        lines.append("-" * 60)
        lines.append(f"Entity: {s['entity']}")
        lines.append(f"Sentiment: {s['sentiment']}")
        lines.append(f"Score: {s['score']:+.2f}")
        lines.append(f"Reasoning: {s.get('reasoning', 'N/A')}")
        lines.append("")
        lines.append(f'Key Quote: "{s.get("key_quote", "N/A")}"')
        lines.append("")

        # Full context
        if s.get('combined_text'):
            lines.append("-" * 60)
            lines.append("Full Context (Combined Text from Podcast):")
            lines.append("")
            text = s['combined_text']
            lines.append(text[:2000])  # Limit to 2000 chars per entry
            if len(text) > 2000:
                lines.append("")
                lines.append("[... text truncated ...]")

        lines.append("")
        lines.append("=" * 80)
        lines.append("")

    # Summary by speaker
    lines.append("SUMMARY BY SPEAKER:")
    lines.append("=" * 80)
    lines.append("")

    by_speaker = {}
    for s in sentiments:
        speaker = s['speaker']
        if speaker not in by_speaker:
            by_speaker[speaker] = []
        by_speaker[speaker].append(s)

    for speaker, speaker_sentiments in by_speaker.items():
        scores = [s['score'] for s in speaker_sentiments]
        avg_score = sum(scores) / len(scores)

        if len(speaker_sentiments) >= 2:
            trend_change = speaker_sentiments[-1]['score'] - speaker_sentiments[0]['score']
            if trend_change > 0.2:
                trend = f"Становится более позитивным ({trend_change:+.2f} change)"
            elif trend_change < -0.2:
                trend = f"Становится более негативным ({trend_change:+.2f} change)"
            else:
                trend = f"Стабильное мнение ({trend_change:+.2f} change)"
        else:
            trend = "Недостаточно данных для тренда"

        lines.append(f"{speaker}:")
        lines.append(f"  - Average Score: {avg_score:+.2f}")
        lines.append(f"  - Trend: {trend}")
        lines.append(f"  - Appearances: {len(speaker_sentiments)} podcasts")
        lines.append("")

    lines.append("=" * 80)
    lines.append("END OF REPORT")
    lines.append("=" * 80)

    # Create buffer
    content = "\n".join(lines)
    buffer = io.BytesIO(content.encode('utf-8'))
    buffer.seek(0)

    return buffer
