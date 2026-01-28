"""
Sentiment analyzer for extracting opinions from podcast transcripts.
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime, date
import re
import asyncio

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.search import search
from llm.client import chat, achat
from config import SENTIMENT_MAX_TEXT_LENGTH


class SentimentAnalyzer:
    """Analyze sentiment and track opinion changes over time."""

    def __init__(self):
        """Initialize sentiment analyzer."""
        pass

    def analyze_entity(
        self,
        entity: str,
        speaker: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        top_k: int = 50
    ) -> List[Dict]:
        """
        Analyze sentiment about an entity across podcasts.

        Args:
            entity: Topic to analyze (e.g., "биткоин", "eigen layer")
            speaker: Optional speaker filter
            start_date: Optional start date filter (YYYY-MM-DD)
            end_date: Optional end date filter (YYYY-MM-DD)
            top_k: Number of chunks to analyze

        Returns:
            List of sentiment results with dates, scores, quotes
        """
        # Build search query
        search_query = f"{entity}"

        # Set time filter
        time_filter = "historical"  # Get all history
        if start_date and end_date:
            time_filter = {"start": start_date, "end": end_date}

        # Search for mentions
        print(f"🔍 Поиск упоминаний '{entity}'...")
        try:
            results = search(
                search_query,
                speaker=speaker,
                top_k=top_k,
                time_filter=time_filter
            )
        except Exception as e:
            print(f"Ошибка поиска: {e}")
            return []

        if not results:
            print(f"Упоминания '{entity}' не найдены")
            return []

        print(f"Найдено {len(results)} упоминаний")

        # Group by podcast/date
        grouped = self._group_by_podcast(results)
        print(f"В {len(grouped)} подкастах")

        # Analyze each group
        sentiments = []
        for podcast_info, result_list in grouped.items():
            podcast_date, podcast_speaker = podcast_info

            # Combine text from chunks
            combined_text = "\n\n".join([
                result['chunk'].get('text', '')
                for result in result_list
            ])

            # Analyze sentiment via LLM
            sentiment = self._analyze_sentiment_llm(
                entity=entity,
                text=combined_text,
                speaker=podcast_speaker,
                date=podcast_date
            )

            if sentiment:
                sentiments.append(sentiment)

        # Sort by date
        sentiments.sort(key=lambda x: x['date'])

        return sentiments

    async def analyze_entity_async(
        self,
        entity: str,
        speaker: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        top_k: int = 500,
        max_concurrent: int = 10
    ) -> List[Dict]:
        """
        Async version: Analyze sentiment with parallel LLM requests.

        Args:
            entity: Topic to analyze (e.g., "биткоин", "eigen layer")
            speaker: Optional speaker filter
            start_date: Optional start date filter (YYYY-MM-DD)
            end_date: Optional end date filter (YYYY-MM-DD)
            top_k: Number of chunks to analyze
            max_concurrent: Max concurrent LLM requests (rate limiting)

        Returns:
            List of sentiment results with dates, scores, quotes, and combined_text
        """
        # Build search query
        search_query = f"{entity}"

        # Set time filter
        time_filter = "historical"  # Get all history
        if start_date and end_date:
            time_filter = {"start": start_date, "end": end_date}

        # Search for mentions
        print(f"🔍 Поиск упоминаний '{entity}'...")
        try:
            results = search(
                search_query,
                speaker=speaker,
                top_k=top_k,
                time_filter=time_filter
            )
        except Exception as e:
            print(f"Ошибка поиска: {e}")
            return []

        if not results:
            print(f"Упоминания '{entity}' не найдены")
            return []

        print(f"Найдено {len(results)} упоминаний")

        # Group by podcast/date
        grouped = self._group_by_podcast(results)
        print(f"В {len(grouped)} подкастах")

        # Create semaphore for rate limiting
        semaphore = asyncio.Semaphore(max_concurrent)

        # Prepare tasks for parallel execution
        tasks = []
        for podcast_info, result_list in grouped.items():
            podcast_date, podcast_speaker = podcast_info

            # Combine text from chunks
            combined_text = "\n\n".join([
                result['chunk'].get('text', '')
                for result in result_list
            ])

            # Create async task
            task = self._analyze_sentiment_llm_async(
                entity=entity,
                text=combined_text,
                speaker=podcast_speaker,
                date=podcast_date,
                semaphore=semaphore
            )
            tasks.append((task, combined_text, podcast_info))

        # Execute all in parallel with rate limiting
        print(f"⚡ Анализирую {len(tasks)} подкастов (макс. {max_concurrent} параллельно)...")

        results = await asyncio.gather(*[t[0] for t in tasks], return_exceptions=True)

        # Process results
        sentiments = []
        success_count = 0
        error_count = 0

        for (task, combined_text, podcast_info), result in zip(tasks, results):
            if isinstance(result, Exception):
                error_count += 1
                print(f"[ОШИБКА] {podcast_info[0]} - {podcast_info[1]}: {type(result).__name__}: {result}")
                continue

            if result:
                # Add original text for TXT export
                result['combined_text'] = combined_text
                sentiments.append(result)
                success_count += 1

        print(f"✅ Анализ завершен: {success_count} успешно, {error_count} ошибок")

        if error_count > 0 and error_count == len(tasks):
            raise Exception("Все запросы завершились ошибкой. Проверьте подключение к OpenRouter.")

        # Sort by date
        sentiments.sort(key=lambda x: x['date'])

        return sentiments

    def _group_by_podcast(self, results: List[Dict]) -> Dict[tuple, List[Dict]]:
        """Group search results by podcast (date + speaker)."""
        grouped = {}

        for result in results:
            # Result structure: {"chunk": {...}, "score": ..., "podcast_date": ...}
            chunk = result.get('chunk', {})
            podcast_date = str(result.get('podcast_date', 'unknown'))

            # Get speakers from chunk
            speakers = chunk.get('speakers', ['Unknown'])

            # Group by date and each speaker
            for speaker in speakers:
                key = (podcast_date, speaker)
                if key not in grouped:
                    grouped[key] = []
                grouped[key].append(result)

        return grouped

    def _analyze_sentiment_llm(
        self,
        entity: str,
        text: str,
        speaker: str,
        date: str
    ) -> Optional[Dict]:
        """
        Analyze sentiment using LLM.

        Returns:
            Dict with sentiment data or None if failed
        """
        system_prompt = """Ты эксперт по анализу мнений в текстах.
Твоя задача - определить отношение спикера к теме на основе цитат из подкаста."""

        user_prompt = f"""Проанализируй отношение спикера к теме на основе цитат из подкаста.

Тема: {entity}
Спикер: {speaker}
Дата подкаста: {date}

Цитаты из подкаста:
{text[:SENTIMENT_MAX_TEXT_LENGTH]}

Определи:
1. **Sentiment** (одно из): "бычий" (bullish), "медвежий" (bearish), "нейтральный" (neutral)
2. **Score** (число от -1 до +1):
   - -1.0 до -0.6: очень негативно
   - -0.6 до -0.2: негативно
   - -0.2 до +0.2: нейтрально
   - +0.2 до +0.6: позитивно
   - +0.6 до +1.0: очень позитивно
3. **Reasoning**: краткое объяснение (1-2 предложения)
4. **Key_quote**: самая показательная цитата (1-2 предложения)

ВАЖНО: Ответь ТОЛЬКО валидным JSON, без дополнительного текста:
{{"sentiment": "нейтральный", "score": 0.0, "reasoning": "...", "key_quote": "..."}}"""

        try:
            response = chat(system_prompt, user_prompt, model="google/gemini-2.5-flash-lite")

            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                response = json_match.group(0)

            # Parse JSON
            sentiment_data = json.loads(response)

            # Add metadata
            sentiment_data['date'] = date
            sentiment_data['speaker'] = speaker
            sentiment_data['entity'] = entity

            return sentiment_data

        except json.JSONDecodeError as e:
            print(f"Ошибка парсинга JSON для {date}: {e}")
            print(f"Response: {response[:200]}")
            return None
        except Exception as e:
            print(f"Ошибка анализа для {date}: {e}")
            return None

    async def _analyze_sentiment_llm_async(
        self,
        entity: str,
        text: str,
        speaker: str,
        date: str,
        semaphore: asyncio.Semaphore
    ) -> Optional[Dict]:
        """
        Async version: Analyze sentiment using LLM with rate limiting.

        Args:
            entity: Topic being analyzed
            text: Combined text from podcast chunks
            speaker: Speaker name
            date: Podcast date
            semaphore: Asyncio semaphore for rate limiting

        Returns:
            Dict with sentiment data or None if failed
        """
        async with semaphore:  # Acquire slot (max N concurrent)
            system_prompt = """Ты эксперт по анализу мнений в текстах.
Твоя задача - определить отношение спикера к теме на основе цитат из подкаста."""

            user_prompt = f"""Проанализируй отношение спикера к теме на основе цитат из подкаста.

Тема: {entity}
Спикер: {speaker}
Дата подкаста: {date}

Цитаты из подкаста:
{text[:SENTIMENT_MAX_TEXT_LENGTH]}

Определи:
1. **Sentiment** (одно из): "бычий" (bullish), "медвежий" (bearish), "нейтральный" (neutral)
2. **Score** (число от -1 до +1):
   - -1.0 до -0.6: очень негативно
   - -0.6 до -0.2: негативно
   - -0.2 до +0.2: нейтрально
   - +0.2 до +0.6: позитивно
   - +0.6 до +1.0: очень позитивно
3. **Reasoning**: краткое объяснение (1-2 предложения)
4. **Key_quote**: самая показательная цитата (1-2 предложения)

ВАЖНО: Ответь ТОЛЬКО валидным JSON, без дополнительного текста:
{{"sentiment": "нейтральный", "score": 0.0, "reasoning": "...", "key_quote": "..."}}"""

            try:
                response = await achat(
                    system_prompt,
                    user_prompt,
                    model="google/gemini-2.5-flash-lite",
                    timeout=90.0
                )

                # Extract JSON from response
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    response = json_match.group(0)

                # Parse JSON
                sentiment_data = json.loads(response)

                # Add metadata
                sentiment_data['date'] = date
                sentiment_data['speaker'] = speaker
                sentiment_data['entity'] = entity

                return sentiment_data

            except json.JSONDecodeError as e:
                print(f"Ошибка парсинга JSON для {date}: {e}")
                return None
            except Exception as e:
                print(f"Ошибка анализа для {date}: {e}")
                return None

    def summarize_trend(self, sentiments: List[Dict]) -> str:
        """
        Generate a text summary of sentiment trend in HTML format.

        Args:
            sentiments: List of sentiment results

        Returns:
            Human-readable summary in HTML
        """
        if not sentiments:
            return "Нет данных для анализа."

        entity = sentiments[0]['entity']

        # Group by speaker
        by_speaker = {}
        for s in sentiments:
            speaker = s['speaker']
            if speaker not in by_speaker:
                by_speaker[speaker] = []
            by_speaker[speaker].append(s)

        summary_lines = [f"📊 <b>Анализ мнений о '{self._escape_html(entity)}':</b>\n"]

        for speaker, speaker_sentiments in by_speaker.items():
            summary_lines.append(f"\n<b>{self._escape_html(speaker)}:</b>")

            # Show ALL entries (no limit)
            for s in speaker_sentiments:
                score = s['score']
                emoji = self._score_to_emoji(score)
                summary_lines.append(
                    f"  {s['date']}: {emoji} {s['sentiment']} ({score:+.2f})"
                )
                if s.get('key_quote'):
                    quote = s['key_quote'][:80]
                    summary_lines.append(f"    💬 <i>{self._escape_html(quote)}</i>")

            # Calculate trend
            if len(speaker_sentiments) >= 2:
                first_score = speaker_sentiments[0]['score']
                last_score = speaker_sentiments[-1]['score']
                trend_change = last_score - first_score

                if trend_change > 0.2:
                    trend = "📈 Тренд: становится более позитивным"
                elif trend_change < -0.2:
                    trend = "📉 Тренд: становится более негативным"
                else:
                    trend = "📊 Тренд: стабильное мнение"

                summary_lines.append(f"  {trend} (Δ{trend_change:+.2f})")

        # Return full text (no truncation)
        return "\n".join(summary_lines)

    def _escape_html(self, text: str) -> str:
        """Escape HTML special characters."""
        if not text:
            return ""
        return (text
                .replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace('"', "&quot;"))

    def _score_to_emoji(self, score: float) -> str:
        """Convert sentiment score to emoji."""
        if score > 0.6:
            return "🚀"
        elif score > 0.2:
            return "📈"
        elif score > -0.2:
            return "➡️"
        elif score > -0.6:
            return "📉"
        else:
            return "🐻"
