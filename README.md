# OmniChat

OmniChat is a chatbot built with Streamlit and the Groq API. You can upload PDF or text files and ask questions about them. It answers from those files and tells you which file and page it used. It can also read images and take voice messages.

## Running it

You need Python 3.10 or newer and a Groq API key.

```
pip install -r requirements.txt
```

Create a file called `.env` next to `app.py` and put your key in it:

```
GROQ_API_KEY=your_key_here
```

Then start the app:

```
streamlit run app.py
```

The first run is slow because it downloads a small embedding model (about 90 MB). After that it starts normally.

## How it answers from documents

Groq doesn't offer an embeddings API, so the document search runs locally with `sentence-transformers`.

1. Each uploaded file is read and cut into passages of about 150 words. Neighbouring passages share 30 words so a sentence on the edge isn't cut in half.
2. Every passage is turned into a vector with the `all-MiniLM-L6-v2` model.
3. When you ask something, the question gets the same treatment and is compared to every passage with cosine similarity.
4. The best four passages (above a minimum score) are added to the system prompt, and the model is told to answer from them and name the source. If nothing relevant is found, it says the documents don't cover the question before falling back on general knowledge.

Short follow-up questions like "and the second one?" are joined with the previous question before searching, otherwise they find nothing useful.

## Files

```
app.py        the Streamlit page and the logic for one chat turn
config.py     model names, limits and retrieval settings
llm.py        Groq client, system prompt, streaming
rag.py        reading files, chunking, embeddings, search
audio.py      speech to text and text to speech
vision.py     resizing images before they are sent
helpers.py    small message helpers, chat export
styles.py     CSS for the page
```

Settings like chunk size, number of passages and the similarity cutoff are at the top of `config.py`.

## Known limits

- The document index lives in memory, so refreshing the page clears it. Saving it to a vector store like ChromaDB would fix that.
- Scanned PDFs have no text layer, so they need OCR before they can be used.
- Search looks at the question only. There is no reranking step, so a vague question can pull in weak passages.