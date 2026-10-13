"""Language-model access: the Groq client, system prompt and streaming."""

from datetime import datetime
from typing import Iterator

from groq import Groq

import config

client = Groq(api_key=config.GROQ_API_KEY or "no-key-set")


def build_system_prompt(style: str, rag_context: str = "") -> str:
    """Combine the chosen style, house rules, the real date and any retrieved context.

    The current date is injected on every request so the model never has to guess it.
    """
    now = datetime.now().strftime("%A, %d %B %Y, %I:%M %p")
    parts = [
        config.ASSISTANT_STYLES[style],
        config.BASE_INSTRUCTIONS,
        f"The current date and time is {now}. Treat this as authoritative.",
    ]
    if rag_context:
        parts.append(rag_context)
    return "\n\n".join(parts)


def stream_reply(model: str, messages: list) -> Iterator[str]:
    """Yield the reply in pieces as Groq produces it."""
    try:
        print("DEBUG - Model:", repr(model))

        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True
        )

        for chunk in stream:
            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    except Exception as e:
        print("GROQ ERROR TYPE:", type(e).__name__)
        print("GROQ ERROR DETAILS:", str(e))
        raise