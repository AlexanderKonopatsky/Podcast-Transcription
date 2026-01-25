"""
Console interface for testing the podcast persona bot.
"""

import sys
from pathlib import Path

# Add current dir to path
sys.path.insert(0, str(Path(__file__).parent))

from config import DATA_DIR
from rag.search import search, format_search_results
from llm.prompts import build_system_prompt, build_user_prompt, get_available_speakers
from llm.client import chat


def check_index_exists():
    """Check if index has been created."""
    embeddings_path = DATA_DIR / "embeddings.npy"
    chunks_path = DATA_DIR / "chunks.json"

    if not embeddings_path.exists() or not chunks_path.exists():
        print("ERROR: Index not found!")
        print("Run this first: python scripts/index_podcasts.py")
        return False
    return True


def select_speaker() -> str:
    """Let user select a speaker."""
    speakers = get_available_speakers()

    print("\nДоступные персонажи:")
    for i, speaker in enumerate(speakers, 1):
        print(f"  {i}. {speaker}")

    while True:
        try:
            choice = input("\nВыберите номер персонажа: ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(speakers):
                return speakers[idx]
            print("Неверный номер, попробуйте снова.")
        except ValueError:
            print("Введите число.")


def chat_loop(speaker: str):
    """Main chat loop with selected speaker."""
    print(f"\n{'=' * 50}")
    print(f"Вы общаетесь с: {speaker}")
    print("Команды: /switch - сменить персонажа, /quit - выход")
    print("=" * 50)

    system_prompt = build_system_prompt(speaker)
    history = []

    while True:
        try:
            user_input = input(f"\nВы: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nДо свидания!")
            break

        if not user_input:
            continue

        if user_input.lower() == "/quit":
            print("\nДо свидания!")
            break

        if user_input.lower() == "/switch":
            return "switch"

        # Search for relevant context
        print("  (поиск контекста...)")
        try:
            results = search(user_input, speaker=speaker, top_k=4)
            rag_context = format_search_results(results, speaker=speaker)
        except Exception as e:
            print(f"  (ошибка поиска: {e})")
            rag_context = ""

        # Build prompt
        user_prompt = build_user_prompt(rag_context, user_input, history)

        # Get response
        print("  (генерация ответа...)")
        try:
            response = chat(system_prompt, user_prompt)
        except Exception as e:
            print(f"\nОшибка API: {e}")
            continue

        # Update history
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": response})

        # Keep history limited
        if len(history) > 10:
            history = history[-10:]

        print(f"\n{speaker}: {response}")

    return "quit"


def main():
    print("=" * 50)
    print("  Podcast Persona Bot - Console Test")
    print("=" * 50)

    # Check index
    if not check_index_exists():
        return

    while True:
        speaker = select_speaker()
        result = chat_loop(speaker)

        if result == "quit":
            break
        # "switch" continues the loop


if __name__ == "__main__":
    main()
