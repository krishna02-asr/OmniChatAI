"""OmniChat: a document-grounded AI assistant.

This file contains only the Streamlit interface and the orchestration of one chat turn.
Model access, retrieval, audio and image handling live in the `core` package.
"""

import hashlib
from datetime import datetime
import sys, asyncio
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
import groq
import streamlit as st

import config
import audio
import rag
import vision
from llm import build_system_prompt, stream_reply
from helpers import export_transcript, split_content
import styles

st.set_page_config(
    page_title=config.APP_NAME,
    layout="centered",
    initial_sidebar_state="expanded",
)
styles.inject_css()


# ------------------------------------------------------------------
# State
# ------------------------------------------------------------------
def init_state() -> None:
    defaults = {
        "messages": [],
        "style": config.DEFAULT_STYLE,
        "speak_replies": False,
        "last_audio_hash": None,
        "rag_docs": {},          # filename -> DocumentIndex
        "pending_prompt": None,  # set when a suggested prompt is clicked
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


# ------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------
def sync_documents(files) -> None:
    """Keep the index in step with the uploader: drop removed files, embed new ones."""
    current = {f.name for f in files or []}
    for name in list(st.session_state.rag_docs):
        if name not in current:
            del st.session_state.rag_docs[name]

    for uploaded in files or []:
        if uploaded.name not in st.session_state.rag_docs:
            with st.spinner(f"Indexing {uploaded.name}"):
                try:
                    st.session_state.rag_docs[uploaded.name] = rag.build_index(uploaded)
                except Exception as exc:
                    st.error(f"{uploaded.name} could not be read. {exc}")

    for name, index in st.session_state.rag_docs.items():
        styles.sidebar_note(f"{name} ({len(index.chunks)} passages)")


def render_sidebar():
    with st.sidebar:
        st.markdown('<div class="side-title">OmniChat</div>', unsafe_allow_html=True)
        styles.sidebar_note("Document-grounded assistant")

        styles.sidebar_section("Assistant")
        st.selectbox("Response style", list(config.ASSISTANT_STYLES), key="style")
        st.toggle("Read replies aloud", key="speak_replies")

        styles.sidebar_section("Documents")
        files = st.file_uploader(
            "Upload PDF or text files",
            type=config.DOCUMENT_TYPES,
            accept_multiple_files=True,
        )
        sync_documents(files)

        styles.sidebar_section("Voice message")
        audio_value = st.audio_input("Record a message", label_visibility="collapsed")

        styles.sidebar_section("Conversation")
        left, right = st.columns(2)
        with left:
            if st.button("New chat"):
                st.session_state.messages = []
                st.session_state.last_audio_hash = None
                st.rerun()
        with right:
            st.download_button(
                "Export",
                data=export_transcript(st.session_state.messages),
                file_name="conversation.txt",
                disabled=not st.session_state.messages,
            )

        if not config.GROQ_API_KEY:
            st.error("GROQ_API_KEY is not set. Add it to a .env file next to app.py and restart.")
    return audio_value


# ------------------------------------------------------------------
# Input
# ------------------------------------------------------------------
def consume_voice(audio_value):
    """Transcribe a new recording once; ignore it on later reruns."""
    if audio_value is None:
        return None
    data = audio_value.getvalue()
    digest = hashlib.md5(data).hexdigest()
    if digest == st.session_state.last_audio_hash:
        return None
    st.session_state.last_audio_hash = digest
    with st.spinner("Transcribing voice message"):
        try:
            return audio.transcribe(data)
        except Exception as exc:
            st.error(f"The voice message could not be transcribed. {exc}")
    return None


def read_input(voice_text):
    """Return (text, image_data_url) from the chat box, a suggested prompt or a voice message."""
    submitted = st.chat_input(
        "Ask a question or attach an image",
        accept_file=True,
        file_type=config.IMAGE_TYPES,
    )
    text, image_url = None, None
    if submitted:
        text = (submitted.text or "").strip()
        if submitted.files:
            image = submitted.files[0]
            raw = image.getvalue()
            if len(raw) > config.MAX_IMAGE_BYTES:
                st.error("That image is larger than 19 MB. Attach a smaller one.")
            else:
                image_url = vision.to_data_url(raw, image.type or "image/jpeg")
    elif st.session_state.pending_prompt:
        text = st.session_state.pending_prompt
        st.session_state.pending_prompt = None
    elif voice_text:
        text = voice_text
    return text, image_url


# ------------------------------------------------------------------
# Rendering
# ------------------------------------------------------------------
def render_sources(sources) -> None:
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})"):
        for source in sources:
            st.text(source)


def render_message(message) -> None:
    with st.chat_message(message["role"]):
        text, image_url = split_content(message["content"])
        if image_url:
            st.image(image_url, width=320)
        if text:
            st.markdown(text)
        render_sources(message.get("sources"))


def render_suggestions() -> None:
    for prompt in config.SUGGESTED_PROMPTS:
        if st.button(prompt, key=f"suggest-{prompt}"):
            st.session_state.pending_prompt = prompt
            st.rerun()


# ------------------------------------------------------------------
# One chat turn
# ------------------------------------------------------------------
def handle_turn(user_text, image_url) -> None:
    if image_url:
        content = [
            {"type": "text", "text": user_text or "Describe this image."},
            {"type": "image_url", "image_url": {"url": image_url}},
        ]
    else:
        content = user_text

    user_message = {
        "role": "user",
        "content": content,
        "time": datetime.now().strftime("%I:%M %p"),
    }
    st.session_state.messages.append(user_message)
    render_message(user_message)

    # Send only recent turns so long conversations stay inside the context window.
    recent = st.session_state.messages[-config.MAX_CONTEXT_MESSAGES:]
    has_image = any(not isinstance(m["content"], str) for m in recent)
    model = config.VISION_MODEL if has_image else config.CHAT_MODEL

    # Retrieval: find passages from the uploaded documents that match the question.
    hits = []
    if st.session_state.rag_docs and user_text:
        earlier_questions = [
            split_content(m["content"])[0]
            for m in st.session_state.messages[:-1]
            if m["role"] == "user"
        ]
        query = rag.build_search_query(earlier_questions, user_text)
        hits = rag.retrieve(query, st.session_state.rag_docs)

    system = build_system_prompt(st.session_state.style, rag.format_context(hits))
    api_messages = [{"role": "system", "content": system}]
    api_messages += [{"role": m["role"], "content": m["content"]} for m in recent]
    sources = [f"{chunk.label} (similarity {score:.2f})" for chunk, score in hits]

    try:
        with st.chat_message("assistant"):
            reply = st.write_stream(stream_reply(model, api_messages))
            render_sources(sources)
    except groq.AuthenticationError:
        st.error("The Groq API key is missing or invalid. Set GROQ_API_KEY in .env and restart.")
    except groq.RateLimitError:
        st.error("Groq's rate limit was reached. Wait a moment, then send the message again.")
    except groq.APIConnectionError:
        st.error("Groq could not be reached. Check your internet connection.")
    except groq.APIStatusError as exc:
        st.error(f"Groq returned an error (status {exc.status_code}). Try again shortly.")
    except Exception as exc:
        st.error(f"The request failed. {exc}")
    else:
        if not reply:
            st.warning("The model returned an empty response. Send the message again.")
            return
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": reply,
                "time": datetime.now().strftime("%I:%M %p"),
                "sources": sources,
            }
        )
        if st.session_state.speak_replies:
            with st.spinner("Generating audio"):
                try:
                    spoken = audio.synthesize(reply[: config.MAX_TTS_CHARS])
                    st.audio(spoken, format="audio/wav", autoplay=True)
                except Exception as exc:
                    st.warning(f"Audio playback is unavailable. {exc}")


# ------------------------------------------------------------------
# Page
# ------------------------------------------------------------------
def main() -> None:
    init_state()
    audio_value = render_sidebar()
    voice_text = consume_voice(audio_value)

    styles.render_header(config.CHAT_MODEL.split("/")[-1], len(st.session_state.rag_docs))

    # Read input first so the empty state never flashes above a new conversation.
    user_text, image_url = read_input(voice_text)
    has_turn = bool(user_text or image_url)

    for message in st.session_state.messages:
        render_message(message)

    if not st.session_state.messages and not has_turn:
        styles.render_empty_state()
        render_suggestions()

    if has_turn:
        handle_turn(user_text, image_url)


main()