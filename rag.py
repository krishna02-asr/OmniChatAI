"""Retrieval-augmented generation: load documents, index them, find relevant passages.

Pipeline
    load file -> split into overlapping passages -> embed each passage
    question  -> embed -> cosine similarity against all passages -> top-k
"""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

import config


@dataclass(frozen=True)
class Chunk:
    """One passage of a document, with enough metadata to cite it."""

    text: str
    source: str
    page: str = ""

    @property
    def label(self) -> str:
        return f"{self.source}, {self.page}" if self.page else self.source


@dataclass
class DocumentIndex:
    """All passages of one document and their embeddings."""

    chunks: list
    embeddings: np.ndarray


@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformer:
    """Load the embedding model once per process."""
    return SentenceTransformer(config.EMBEDDING_MODEL)


def split_text(text: str, size: int = config.RAG_CHUNK_SIZE,
               overlap: int = config.RAG_CHUNK_OVERLAP) -> list:
    """Split text into passages of `size` words that overlap by `overlap` words.

    The overlap keeps a sentence that straddles a boundary intact in at least one passage.
    """
    words = text.split()
    passages, start = [], 0
    while start < len(words):
        end = start + size
        passage = " ".join(words[start:end]).strip()
        if passage:
            passages.append(passage)
        if end >= len(words):
            break
        start = end - overlap
    return passages


def read_document(uploaded_file) -> list:
    """Return a list of (text, page_label) for a PDF, TXT or MD upload."""
    name = uploaded_file.name.lower()
    if name.endswith(".pdf"):
        reader = PdfReader(uploaded_file)
        return [(page.extract_text() or "", f"p.{number}")
                for number, page in enumerate(reader.pages, start=1)]
    if name.endswith((".txt", ".md")):
        raw = uploaded_file.getvalue()
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            text = raw.decode("latin-1")
        return [(text, "")]
    raise ValueError("Unsupported file type. Use PDF, TXT or MD.")


def build_index(uploaded_file) -> DocumentIndex:
    """Split an uploaded file into passages and embed them."""
    chunks = [
        Chunk(text=passage, source=uploaded_file.name, page=page)
        for text, page in read_document(uploaded_file)
        for passage in split_text(text)
    ]
    if not chunks:
        raise ValueError("No readable text found. Scanned PDFs need OCR before they can be used.")
    embeddings = get_embedder().encode(
        [chunk.text for chunk in chunks], normalize_embeddings=True, batch_size=32
    )
    return DocumentIndex(chunks=chunks, embeddings=embeddings)


def build_search_query(history: list, question: str) -> str:
    """Short follow-ups ("and the second one?") borrow the previous question for context."""
    previous = [text for text in history if text]
    if previous and len(question.split()) < 8:
        return f"{previous[-1]} {question}"
    return question


def retrieve(query: str, indexes: dict, top_k: int = config.RAG_TOP_K,
             min_score: float = config.RAG_MIN_SCORE) -> list:
    """Return the (Chunk, score) pairs most similar to the query across all documents."""
    if not indexes:
        return []
    query_vec = get_embedder().encode([query], normalize_embeddings=True)[0]
    chunks, matrices = [], []
    for index in indexes.values():
        chunks.extend(index.chunks)
        matrices.append(index.embeddings)
    # Vectors are normalised, so the dot product equals cosine similarity.
    scores = np.vstack(matrices) @ query_vec
    best = scores.argsort()[::-1][:top_k]
    return [(chunks[i], float(scores[i])) for i in best if scores[i] >= min_score]


def format_context(hits: list) -> str:
    """Turn retrieved passages into the instruction block appended to the system prompt."""
    if not hits:
        return ""
    excerpts = "\n\n".join(f"[{chunk.label}]\n{chunk.text}" for chunk, _ in hits)
    return (
        "Excerpts from the user's uploaded documents:\n\n"
        f"{excerpts}\n\n"
        "Answer from these excerpts where they apply and cite the source in parentheses, "
        "for example (report.pdf, p.3). If they do not contain the answer, say that the "
        "uploaded documents do not cover it before answering from general knowledge."
    )