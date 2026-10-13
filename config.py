"""Central configuration for OmniChat.

Everything tunable lives here so the rest of the code contains no magic values.
"""

import os

from dotenv import load_dotenv

load_dotenv()

# --- Application -------------------------------------------------------------
APP_NAME = "OmniChat"

# --- Groq --------------------------------------------------------------------
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

CHAT_MODEL = "openai/gpt-oss-120b"
VISION_MODEL = "qwen/qwen3.8-27b"  # used automatically when an image is present
STT_MODEL = "whisper-large-v3-turbo"                        # speech to text
TTS_MODEL = "canopylabs/orpheus-v1-english"                 # text to speech
TTS_VOICE = "troy"

# --- Conversation limits -----------------------------------------------------
MAX_CONTEXT_MESSAGES = 20            # past messages sent to the model on each request
MAX_IMAGE_BYTES = 19 * 1024 * 1024   # Groq's request size limit for images
MAX_TTS_CHARS = 900                  # longest reply that is read aloud

# --- Retrieval (RAG) ---------------------------------------------------------
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RAG_CHUNK_SIZE = 150      # words per passage (the embedding model reads about 256 tokens)
RAG_CHUNK_OVERLAP = 30    # words shared between neighbouring passages
RAG_TOP_K = 4             # passages sent to the model per question
RAG_MIN_SCORE = 0.25      # minimum cosine similarity for a passage to count as relevant
DOCUMENT_TYPES = ["pdf", "txt", "md"]
IMAGE_TYPES = ["png", "jpg", "jpeg", "webp"]

# --- Response behaviour ------------------------------------------------------
BASE_INSTRUCTIONS = (
    "Write clear, well-structured answers. Lead with the answer, then give supporting detail. "
    "Use Markdown (short headings, lists, tables, code blocks) only when it improves readability. "
    "Do not use emojis. Do not open with filler such as 'Great question' and do not close with "
    "generic offers of further help. If a request is ambiguous, state your assumption briefly or "
    "ask one clarifying question. If you are unsure, say so. "
    "Reply in the language the user writes in. "
    "You can analyze images the user attaches, and you may be given excerpts from documents the "
    "user uploaded."
)

ASSISTANT_STYLES = {
    "Professional": (
        "You are OmniChat, a precise and knowledgeable assistant. "
        "Write in a clear, neutral and courteous tone."
    ),
    "Technical": (
        "You are OmniChat acting as a senior software engineer. Give correct, idiomatic code "
        "and explain the reasoning and trade-offs briefly."
    ),
    "Concise": (
        "You are OmniChat. Answer as briefly as accuracy allows, usually in one to three "
        "sentences, unless the user asks for more."
    ),
    "Conversational": (
        "You are OmniChat. Use a warm, natural tone while staying accurate and useful."
    ),
}
DEFAULT_STYLE = "Professional"

SUGGESTED_PROMPTS = [
    "Summarize the uploaded documents",
    "Explain how retrieval-augmented generation works",
    "Review a piece of code I paste in",
]