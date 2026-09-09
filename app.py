import os
import io
import base64
import tempfile
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv
from PIL import Image
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import groq
from groq import Groq

load_dotenv()



GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

MODEL = "openai/gpt-oss-120b"                                 # everyday text chat
VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"    # used automatically for images
STT_MODEL = "whisper-large-v3-turbo"                          # speech -> text
TTS_MODEL = "canopylabs/orpheus-v1-english"                   # text -> speech
TTS_VOICE = "troy"

MAX_CONTEXT_MESSAGES = 20     # how many past turns get sent to the model each request
MAX_IMAGE_BYTES = 19 * 1024 * 1024

RAG_CHUNK_SIZE = 220          # words per document chunk
RAG_CHUNK_OVERLAP = 40
RAG_TOP_K = 3
RAG_MIN_SCORE = 0.05

PERSONAS = {
    "Helpful Assistant": "You are a helpful, clear, and concise AI assistant.",
    "Friendly Buddy": "You are a warm, casual, friendly companion. Chat like a close friend and keep it light.",
    "Coding Expert": "You are an expert software engineer. Give precise, well-explained, correct code.",
    "Sarcastic Wit": "You are witty and a little sarcastic, but still genuinely helpful underneath the humor.",
}

client = Groq(api_key=GROQ_API_KEY or "no-key-set")

st.set_page_config(page_title="AI Chat Bot", page_icon="🤖", layout="centered")


# ------------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------------
def extract_text_and_image(content):
    """content is either a plain string, or a list of OpenAI-style content
    blocks (for messages that include an image). Always returns
    (text, data_url_or_None) so the rest of the app doesn't need to care
    which shape it's dealing with."""
    if isinstance(content, str):
        return content, None
    text, data_url = "", None
    for block in content:
        if block.get("type") == "text":
            text = block.get("text", "")
        elif block.get("type") == "image_url":
            data_url = block.get("image_url", {}).get("url")
    return text, data_url


def prepare_image_data_url(img_bytes, mime):
    """Shrinks big phone-camera photos before they're sent to the API —
    keeps requests fast and safely under Groq's size limit."""
    try:
        img = Image.open(io.BytesIO(img_bytes))
        img.thumbnail((1600, 1600))
        is_png = mime == "image/png" and img.mode in ("RGBA", "LA", "P")
        buf = io.BytesIO()
        if is_png:
            img.save(buf, format="PNG")
            out_mime = "image/png"
        else:
            if img.mode not in ("RGB", "L"):
                img = img.convert("RGB")
            img.save(buf, format="JPEG", quality=88)
            out_mime = "image/jpeg"
        return f"data:{out_mime};base64,{base64.b64encode(buf.getvalue()).decode()}"
    except Exception:
        return f"data:{mime};base64,{base64.b64encode(img_bytes).decode()}"


def system_prompt(persona_name):
    # Always inject the REAL current date/time so the model never has to
    # guess it (fixes the classic "wrong date" chatbot bug).
    today_str = datetime.now().strftime("%A, %d %B %Y, %I:%M %p")
    base = PERSONAS[persona_name]
    return (
        f"{base}\n\n"
        f"Important: the real current date and time is {today_str}. Always trust this "
        f"over anything you might otherwise assume, and never say your knowledge is "
        f"limited to an earlier date when asked what today's date is. You can analyze "
        f"any image the user attaches, and you may be given relevant excerpts from a "
        f"document the user uploaded — use them when they help answer the question."
    )


def stream_reply(model, api_messages):
    """A generator that yields text pieces as they arrive from Groq —
    feeds directly into st.write_stream() for a live typing effect."""
    stream = client.chat.completions.create(model=model, messages=api_messages, stream=True)
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


def transcribe_audio(audio_bytes):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    try:
        tmp.write(audio_bytes)
        tmp.close()
        with open(tmp.name, "rb") as f:
            result = client.audio.transcriptions.create(
                file=f, model=STT_MODEL, response_format="text"
            )
    finally:
        os.unlink(tmp.name)
    return result if isinstance(result, str) else getattr(result, "text", str(result))


def synthesize_speech(text):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp.close()
    try:
        response = client.audio.speech.create(
            model=TTS_MODEL, voice=TTS_VOICE, input=text, response_format="wav"
        )
        response.write_to_file(tmp.name)
        with open(tmp.name, "rb") as f:
            data = f.read()
    finally:
        os.unlink(tmp.name)
    return data


def chunk_text(text, size=RAG_CHUNK_SIZE, overlap=RAG_CHUNK_OVERLAP):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + size
        piece = " ".join(words[start:end]).strip()
        if piece:
            chunks.append(piece)
        if end >= len(words):
            break
        start = end - overlap
    return chunks


def build_rag_index(uploaded_pdf):
    reader = PdfReader(uploaded_pdf)
    full_text = ""
    for page in reader.pages:
        full_text += (page.extract_text() or "") + "\n"
    chunks = chunk_text(full_text)
    if not chunks:
        raise ValueError("No extractable text found — this PDF might be a scanned image.")
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(chunks)
    return chunks, vectorizer, matrix


def retrieve_context(query, vectorizer, matrix, chunks, top_k=RAG_TOP_K, min_score=RAG_MIN_SCORE):
    q_vec = vectorizer.transform([query])
    scores = cosine_similarity(q_vec, matrix)[0]
    ranked = scores.argsort()[::-1][:top_k]
    return [chunks[i] for i in ranked if scores[i] >= min_score]


def export_chat_text(messages):
    lines = []
    for m in messages:
        who = "You" if m["role"] == "user" else "Bot"
        text, data_url = extract_text_and_image(m["content"])
        line = text or ""
        if data_url:
            line = (line + " " if line else "") + "[image attached]"
        lines.append(f"[{m.get('time', '')}] {who}: {line}")
    return "\n\n".join(lines)


# ------------------------------------------------------------------
# SESSION STATE
# ------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "persona" not in st.session_state:
    st.session_state.persona = "Helpful Assistant"
if "speak_replies" not in st.session_state:
    st.session_state.speak_replies = False
if "last_audio_hash" not in st.session_state:
    st.session_state.last_audio_hash = None


# ------------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")

    st.session_state.persona = st.selectbox(
        "Persona", list(PERSONAS.keys()),
        index=list(PERSONAS.keys()).index(st.session_state.persona),
    )
    st.session_state.speak_replies = st.toggle(
        "🔊 Speak bot replies", value=st.session_state.speak_replies
    )

    st.divider()
    st.subheader("📄 Document Q&A")
    pdf_file = st.file_uploader("Upload a PDF to ask questions about", type=["pdf"])
    if pdf_file is not None and pdf_file.name != st.session_state.get("rag_filename"):
        with st.spinner("Reading and indexing document..."):
            try:
                chunks, vectorizer, matrix = build_rag_index(pdf_file)
                st.session_state.rag_chunks = chunks
                st.session_state.rag_vectorizer = vectorizer
                st.session_state.rag_matrix = matrix
                st.session_state.rag_filename = pdf_file.name
                st.toast(f"Indexed {len(chunks)} chunks from {pdf_file.name}")
            except Exception as e:
                st.error(f"Couldn't read PDF: {e}")

    if st.session_state.get("rag_filename"):
        st.caption(f"📎 Active: {st.session_state.rag_filename} ({len(st.session_state.rag_chunks)} chunks)")
        if st.button("❌ Remove document"):
            for k in ("rag_chunks", "rag_vectorizer", "rag_matrix", "rag_filename"):
                st.session_state.pop(k, None)
            st.rerun()

    st.divider()
    st.subheader("🎙️ Voice Message")
    audio_value = st.audio_input("Record instead of typing")

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🗑 New Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.last_audio_hash = None
            st.rerun()
    with c2:
        st.download_button(
            "💾 Export", data=export_chat_text(st.session_state.messages),
            file_name="chat_export.txt", use_container_width=True,
            disabled=not st.session_state.messages,
        )

    if not GROQ_API_KEY:
        st.error("No GROQ_API_KEY found. Add it to a .env file next to this app.")


# ------------------------------------------------------------------
# MAIN CHAT AREA
# ------------------------------------------------------------------
st.title("🤖 AI Chat Bot")
st.caption(f"Powered by Groq · {MODEL} · vision, voice & document Q&A enabled")

for m in st.session_state.messages:
    avatar = "🧑" if m["role"] == "user" else "🤖"
    with st.chat_message(m["role"], avatar=avatar):
        text, data_url = extract_text_and_image(m["content"])
        if data_url:
            st.image(data_url)
        if text:
            st.markdown(text)
        if m["role"] == "assistant" and text:
            with st.expander("📋 Copy text"):
                st.code(text, language=None)

# --- handle a freshly recorded voice message (only once per new recording) ---
voice_prompt_text = None
if audio_value is not None:
    audio_bytes = audio_value.getvalue()
    a_hash = hash(audio_bytes)
    if a_hash != st.session_state.last_audio_hash:
        st.session_state.last_audio_hash = a_hash
        with st.spinner("Transcribing your voice message..."):
            try:
                voice_prompt_text = transcribe_audio(audio_bytes)
            except Exception as e:
                st.error(f"Couldn't transcribe audio: {e}")

# --- the chat box itself (supports typing AND attaching an image) ---
chat_value = st.chat_input(
    "Type a message, or attach an image...",
    accept_file=True,
    file_type=["png", "jpg", "jpeg", "webp"],
)

user_text = None
image_data_url = None

if chat_value:
    user_text = chat_value.text.strip() if chat_value.text else ""
    if chat_value.files:
        up_img = chat_value.files[0]
        img_bytes = up_img.getvalue()
        if len(img_bytes) > MAX_IMAGE_BYTES:
            st.error("Image is too large (max ~19MB). Please attach a smaller one.")
        else:
            image_data_url = prepare_image_data_url(img_bytes, up_img.type or "image/jpeg")
elif voice_prompt_text:
    user_text = voice_prompt_text

if user_text or image_data_url:
    now = datetime.now().strftime("%I:%M %p")

    if image_data_url:
        content = [
            {"type": "text", "text": user_text if user_text else "What's in this image?"},
            {"type": "image_url", "image_url": {"url": image_data_url}},
        ]
    else:
        content = user_text

    st.session_state.messages.append({"role": "user", "content": content, "time": now})
    with st.chat_message("user", avatar="🧑"):
        if image_data_url:
            st.image(image_data_url)
        if user_text:
            st.markdown(user_text)

    # Only send the last MAX_CONTEXT_MESSAGES turns to the API — keeps
    # long conversations from exceeding the model's context window.
    trimmed = st.session_state.messages[-MAX_CONTEXT_MESSAGES:]
    uses_image = any(not isinstance(m["content"], str) for m in trimmed)
    model_to_use = VISION_MODEL if uses_image else MODEL

    rag_note = ""
    if st.session_state.get("rag_vectorizer") is not None and user_text:
        relevant = retrieve_context(
            user_text, st.session_state.rag_vectorizer,
            st.session_state.rag_matrix, st.session_state.rag_chunks,
        )
        if relevant:
            joined = "\n\n---\n\n".join(relevant)
            rag_note = (
                f"\n\nRelevant excerpts from the uploaded document "
                f"'{st.session_state.rag_filename}':\n\n{joined}\n\n"
                f"Use these excerpts if they help answer the question. If they don't "
                f"contain the answer, say so and answer from your own knowledge."
            )

    api_messages = [{"role": "system", "content": system_prompt(st.session_state.persona) + rag_note}]
    api_messages += [{"role": m["role"], "content": m["content"]} for m in trimmed]

    try:
        with st.chat_message("assistant", avatar="🤖"):
            full_reply = st.write_stream(stream_reply(model_to_use, api_messages))
            if full_reply:
                with st.expander("📋 Copy text"):
                    st.code(full_reply, language=None)
    except groq.AuthenticationError:
        st.error("Invalid or missing API key. Check GROQ_API_KEY in your .env file.")
    except groq.RateLimitError:
        st.error("Rate limit reached — too many requests, or your free quota ran out. Wait a bit and try again.")
    except groq.APIConnectionError:
        st.error("Couldn't reach Groq's servers. Check your internet connection.")
    except groq.APIStatusError as e:
        st.error(f"Groq API error (status {e.status_code}). Please try again shortly.")
    except Exception as e:
        st.error(f"Unexpected error: {e}")
    else:
        reply_time = datetime.now().strftime("%I:%M %p")
        st.session_state.messages.append({"role": "assistant", "content": full_reply, "time": reply_time})

        if st.session_state.speak_replies and full_reply:
            with st.spinner("Generating voice..."):
                try:
                    audio_out = synthesize_speech(full_reply[:900])  # keep long replies snappy to speak
                    st.audio(audio_out, format="audio/wav", autoplay=True)
                except Exception as e:
                    st.warning(f"Voice reply failed: {e}")