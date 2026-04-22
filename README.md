# 🎵 Lyrics RAG - Music Topic & Sentiment Q&A

A RAG system that answers multiple-choice questions about song topics and sentiment,
using LangChain + ChromaDB + Ollama.

## Project Structure

```
rag_lyrics/
├── data/
│   ├── songs/              ← Raw lyrics as .json files
│   ├── chroma_db/          ← ChromaDB vector store (auto-generated)
│   ├── eval_dataset.json   ← Your MC questions
│   ├── baseline_results.json
│   └── rag_results.json
├── 01_collect_lyrics.py    ← Fetch lyrics via Genius API
├── 01b_baseline_evaluate.py← LLM without RAG (Part 1)
├── 02_build_vectorstore.py ← Embed lyrics into ChromaDB
├── 03_rag_evaluate.py      ← RAG pipeline + evaluation (Part 2)
└── README.md
```

## Setup

```bash
# 1. Install dependencies
pip install lyricsgenius langchain langchain-community chromadb \
            sentence-transformers ollama

# 2. Pull an Ollama model
ollama pull mistral

# 3. Get a free Genius API token
# → https://genius.com/api-clients (create an account, make an API client)
```

## Run Order

```bash
# Step 1 — Collect lyrics
python 01_collect_lyrics.py

# Step 2 — Build vector store
python 02_build_vectorstore.py

# Step 3a — Baseline (no RAG) for Part 1 of assignment
python 01b_baseline_evaluate.py

# Step 3b — Full RAG evaluation for Part 2 of assignment
python 03_rag_evaluate.py
```

## Evaluation Dataset Format

Edit `data/eval_dataset.json` with your questions:

```json
[
  {
    "id": 1,
    "song": "Bohemian Rhapsody",
    "artist": "Queen",
    "question": "What is the main topic of this song?",
    "options": {
      "A": "A romantic love story",
      "B": "An internal struggle with identity and fate",
      "C": "A political protest",
      "D": "A celebration of friendship"
    },
    "answer": "B"
  }
]
```

## Question Types to Include

- **Topic**: "What is the main theme of this song?"
- **Sentiment**: "What is the overall emotional tone?"
- **Subject**: "Who or what is this song directed at?"
- **Aspect**: "How does the artist feel about [specific topic]?"

## Notes

- Ollama must be running locally: `ollama serve`
- Lyrics are stored as JSON with title, artist, and full lyrics
- ChromaDB is local and persistent — no cloud needed
- Use `all-MiniLM-L6-v2` embeddings (free, runs locally, no API key)
