"""
STEP 2 - Build the ChromaDB vector store from saved lyrics
"""

import os
import json
import glob
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# ── Config ───────────────────────────────────────────────────────────────────
SONGS_DIR = "data/songs"
CHROMA_DIR = "data/chroma_db"

# ── Load JSON song files into LangChain Documents ────────────────────────────
documents = []
for filepath in glob.glob(os.path.join(SONGS_DIR, "*.json")):
    with open(filepath, "r", encoding="utf-8") as f:
        song = json.load(f)

    # Each song becomes a Document with metadata
    doc = Document(
        page_content=song["lyrics"],
        metadata={
            "title": song["title"],
            "artist": song["artist"],
            "source": filepath,
        },
    )
    documents.append(doc)

print(f"Loaded {len(documents)} songs.")

# ── Split lyrics into chunks ──────────────────────────────────────────────────
# Songs are short, so chunks can be generous (e.g. verse-sized)
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", " "],
)
chunks = splitter.split_documents(documents)
print(f"Split into {len(chunks)} chunks.")

# ── Embed and store in ChromaDB ───────────────────────────────────────────────
# Uses a free local embedding model (no API key needed)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=CHROMA_DIR,
)
vectorstore.persist()
print(f"Vector store saved to '{CHROMA_DIR}/'")
print("Done! You can now run the RAG pipeline.")
