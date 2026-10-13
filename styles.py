"""Visual layer: global CSS and the few custom layout components.

Design tokens
    ink      #12202B   text
    petrol   #0E6B72   accent (focus, primary actions, links)
    mist     #F3F5F6   sidebar surface
    rule     #D9DEE2   hairlines
    tint     #E6F0F1   user message background
Type: IBM Plex Sans for the interface, IBM Plex Mono for code.
"""

from html import escape

import streamlit as st

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

:root {
    --ink: #12202B;
    --muted: #5C6B76;
    --petrol: #0E6B72;
    --mist: #F3F5F6;
    --rule: #D9DEE2;
    --tint: #E6F0F1;
}

.stApp { font-family: 'IBM Plex Sans', system-ui, -apple-system, 'Segoe UI', sans-serif; color: var(--ink); }
button, input, textarea { font-family: inherit; }
code, pre, [data-testid="stCode"] * { font-family: 'IBM Plex Mono', ui-monospace, Consolas, monospace; }

/* Remove default Streamlit chrome (toolbar is left alone so the sidebar arrow stays reachable) */
[data-testid="stDecoration"], footer { display: none; }
header[data-testid="stHeader"] { background: transparent; }

/* Reading column */
.block-container { max-width: 800px; padding-top: 2rem; padding-bottom: 7rem; }

/* Header */
.app-header {
    display: flex; justify-content: space-between; align-items: baseline;
    padding-bottom: 0.9rem; margin-bottom: 1.6rem; border-bottom: 1px solid var(--rule);
}
.app-title { font-size: 1.35rem; font-weight: 600; letter-spacing: -0.01em; }
.app-meta { font-size: 0.85rem; color: var(--muted); text-align: right; }
.app-meta strong { color: var(--ink); font-weight: 500; }

/* Messages: assistant replies sit flat on the page, user messages are tinted */
[data-testid="stChatMessage"] {
    background: transparent; padding: 0.9rem 0; gap: 0.9rem;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"], [data-testid="chatAvatarIcon-user"]) {
    background: var(--tint); border-radius: 10px; padding: 0.9rem 1.1rem;
}
[data-testid="stChatMessage"] p { line-height: 1.65; }

/* Empty state */
.empty-state { padding: 2.2rem 0 1rem; max-width: 560px; }
.empty-state h2 { font-size: 1.5rem; font-weight: 600; letter-spacing: -0.01em; margin: 0 0 0.6rem; }
.empty-state p { color: var(--muted); line-height: 1.6; margin: 0 0 1.4rem; }

/* Sidebar */
section[data-testid="stSidebar"] { background: var(--mist); border-right: 1px solid var(--rule); }
.side-title { font-size: 1.05rem; font-weight: 600; margin-bottom: 0.1rem; }
.side-section {
    font-size: 0.9rem; font-weight: 600; margin: 1.5rem 0 0.5rem;
    padding-top: 1rem; border-top: 1px solid var(--rule);
}
.side-note { font-size: 0.82rem; color: var(--muted); line-height: 1.5; margin: 0.15rem 0; }

/* Controls */
.stButton > button, .stDownloadButton > button {
    border-radius: 6px; border: 1px solid var(--rule); background: #fff;
    font-weight: 500; color: var(--ink); width: 100%;
}
.stButton > button:hover, .stDownloadButton > button:hover { border-color: var(--petrol); color: var(--petrol); }
[data-testid="stChatInput"] { border-radius: 10px; }
button:focus-visible, textarea:focus-visible, input:focus-visible { outline: 2px solid var(--petrol); outline-offset: 2px; }

/* Source citations */
[data-testid="stExpander"] { border: 1px solid var(--rule); border-radius: 8px; background: #fff; }
[data-testid="stExpander"] summary { font-size: 0.88rem; color: var(--muted); }
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def render_header(model: str, document_count: int) -> None:
    """Top bar: product name on the left, what the assistant is answering from on the right."""
    if document_count:
        noun = "document" if document_count == 1 else "documents"
        status = f"Answering from <strong>{document_count} {noun}</strong>"
    else:
        status = "No documents loaded, answering from general knowledge"
    st.markdown(
        f'<div class="app-header"><div class="app-title">OmniChat</div>'
        f'<div class="app-meta">{status}<br>Model: {escape(model)}</div></div>',
        unsafe_allow_html=True,
    )


def render_empty_state() -> None:
    st.markdown(
        '<div class="empty-state"><h2>Ask about your documents, or anything else</h2>'
        "<p>Upload PDF or text files in the sidebar to get answers drawn from your own "
        "material, with the source of each answer shown. You can also attach an image or "
        "record a voice message.</p></div>",
        unsafe_allow_html=True,
    )


def sidebar_section(title: str) -> None:
    st.markdown(f'<div class="side-section">{escape(title)}</div>', unsafe_allow_html=True)


def sidebar_note(text: str) -> None:
    st.markdown(f'<div class="side-note">{escape(text)}</div>', unsafe_allow_html=True)