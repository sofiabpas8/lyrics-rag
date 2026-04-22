"""
STEP 3 - RAG pipeline + evaluation on multiple-choice questions
Install: pip install langchain langchain-community chromadb sentence-transformers ollama
Make sure Ollama is running: `ollama serve` and `ollama pull mistral`
"""

import json
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# ── Config ────────────────────────────────────────────────────────────────────
CHROMA_DIR = "data/chroma_db"
DATASET_PATH = "data/eval_dataset.json"
OLLAMA_MODEL = "mistral"  # or "llama3", "gemma", etc.

# ── Evaluation dataset format (save this as data/eval_dataset.json) ───────────
EXAMPLE_DATASET = [
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
    },
    {
        "id": 2,
        "song": "Alright",
        "artist": "Kendrick Lamar",
        "question": "What is the overall sentiment of this song?",
        "options": {
            "A": "Pessimistic and hopeless",
            "B": "Angry and violent",
            "C": "Hopeful despite adversity",
            "D": "Nostalgic and melancholic"
        },
        "answer": "C"
    },
    # Add more questions here...
]

# Save example dataset if not exists
import os
os.makedirs("data", exist_ok=True)
if not os.path.exists(DATASET_PATH):
    with open(DATASET_PATH, "w") as f:
        json.dump(EXAMPLE_DATASET, f, indent=2)
    print(f"Example dataset saved to {DATASET_PATH}")

# ── Load vector store ─────────────────────────────────────────────────────────
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=embeddings,
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# ── Load LLM (Ollama running locally) ────────────────────────────────────────
llm = Ollama(model=OLLAMA_MODEL)

# ── Prompt template ───────────────────────────────────────────────────────────
PROMPT_TEMPLATE = """
You are a music expert. Use the following song lyrics to answer the question.
Answer ONLY with the letter of the correct option (A, B, C, or D). Nothing else.

Song lyrics context:
{context}

Question: {question}
Options:
A) {option_a}
B) {option_b}
C) {option_c}
D) {option_d}

Answer (just the letter):"""

# ── Evaluation loop ───────────────────────────────────────────────────────────
with open(DATASET_PATH, "r") as f:
    dataset = json.load(f)

results = []
correct = 0

for item in dataset:
    # Build the query combining song info and question
    query = f"{item['song']} by {item['artist']}: {item['question']}"

    # Retrieve relevant chunks from vector store
    docs = retriever.get_relevant_documents(query)
    context = "\n\n".join([d.page_content for d in docs])

    # Format the prompt
    prompt = PROMPT_TEMPLATE.format(
        context=context,
        question=item["question"],
        option_a=item["options"]["A"],
        option_b=item["options"]["B"],
        option_c=item["options"]["C"],
        option_d=item["options"]["D"],
    )

    # Get LLM answer
    raw_answer = llm(prompt).strip().upper()
    predicted = raw_answer[0] if raw_answer and raw_answer[0] in "ABCD" else "?"

    is_correct = predicted == item["answer"]
    if is_correct:
        correct += 1

    result = {
        "id": item["id"],
        "song": item["song"],
        "question": item["question"],
        "expected": item["answer"],
        "predicted": predicted,
        "correct": is_correct,
        "retrieved_sources": [d.metadata.get("title") for d in docs],
    }
    results.append(result)

    status = "✓" if is_correct else "✗"
    print(f"[{status}] Q{item['id']} ({item['song']}): expected={item['answer']}, got={predicted}")

# ── Print accuracy ─────────────────────────────────────────────────────────────
accuracy = correct / len(dataset) * 100
print(f"\n{'='*40}")
print(f"Accuracy: {correct}/{len(dataset)} = {accuracy:.1f}%")
print(f"{'='*40}")

# ── Save results ──────────────────────────────────────────────────────────────
with open("data/rag_results.json", "w") as f:
    json.dump({"accuracy": accuracy, "results": results}, f, indent=2)
print("Results saved to data/rag_results.json")
