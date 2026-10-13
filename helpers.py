"""Small helpers for working with chat messages."""


def split_content(content):
    """Return (text, image_data_url) for a message.

    A message's content is either a plain string or a list of OpenAI-style blocks
    (used when an image is attached). This hides that difference from the UI.
    """
    if isinstance(content, str):
        return content, None
    text, image_url = "", None
    for block in content:
        if block.get("type") == "text":
            text = block.get("text", "")
        elif block.get("type") == "image_url":
            image_url = block.get("image_url", {}).get("url")
    return text, image_url


def export_transcript(messages: list) -> str:
    """Render the conversation as plain text for download."""
    lines = []
    for message in messages:
        speaker = "You" if message["role"] == "user" else "Assistant"
        text, image_url = split_content(message["content"])
        if image_url:
            text = f"{text} [image attached]".strip()
        lines.append(f"[{message.get('time', '')}] {speaker}: {text}")
    return "\n\n".join(lines)