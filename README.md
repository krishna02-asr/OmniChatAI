# 🤖 AI Chat Bot

A browser-based AI chatbot built with **Streamlit** and **Groq**, supporting text, images, voice, and document Q&A in one interface.

# Features

- 💬 **Streaming chat** — replies appear word-by-word, like ChatGPT
- 🖼️ **Vision** — attach a photo in the chat box and ask questions about it (auto-switches to a vision model)
- 🎙️ **Voice input** — record a message; it's transcribed with Groq's Whisper model
- 🔊 **Voice output** — optionally have replies read aloud (Groq TTS)
- 📄 **Document Q&A (RAG)** — upload a PDF and ask questions about its contents; relevant excerpts are retrieved with **TF-IDF + cosine similarity** and injected into the prompt
- 🎭 **Personas** — switch the bot's personality on the fly
- 🧠 **Context-aware** — the real current date/time is always passed to the model, and long conversations are automatically trimmed to stay within the model's context window
- 🔐 **Secure config** — the API key is loaded from a local `.env` file, never hardcoded
- 🛡️ **Graceful errors** — specific messages for invalid keys, rate limits, and network issues instead of a crash
- 💾 **Export** — download the full conversation as a `.txt` file

## Architecture

```
app.py                 Streamlit UI + chat/session logic
.streamlit/config.toml Custom app theme
.env                   Local secrets (not committed — see .gitignore)
requirements.txt       Python dependencies
```

Chat history lives in Streamlit's per-session state rather than a shared file on disk, so if this is deployed publicly, different visitors never see each other's conversations.

Document Q&A avoids a full vector-database/embedding-model dependency: PDF text is chunked, vectorized with scikit-learn's `TfidfVectorizer`, and the most relevant chunks are retrieved per question with cosine similarity — a lightweight retrieval-augmented-generation (RAG) approach.

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file in this folder:

```
GROQ_API_KEY=your_key_here
```

Get a free key at [console.groq.com/keys](https://console.groq.com/keys).

Run it:

```bash
streamlit run app.py
```

## Deploying (optional)

Push this repo to GitHub, then deploy for free at [share.streamlit.io](https://share.streamlit.io) to get a live link — add your `GROQ_API_KEY` under the app's "Secrets" settings instead of uploading `.env`.

## Tech Stack

Python · Streamlit · Groq API (Llama 4 Scout, Whisper, Orpheus TTS) · scikit-learn · pypdf · Pillow