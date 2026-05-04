"""
STEP 3 - RAG pipeline + evaluation on multiple-choice questions
"""

import json
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM

# ── Config ────────────────────────────────────────────────────────────────────
CHROMA_DIR = "data/chroma_db"
DATASET_PATH = "data/eval_dataset.json"
OLLAMA_MODEL = "phi"  

import os
os.makedirs("data", exist_ok=True)

# ── Load vector store ─────────────────────────────────────────────────────────
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=embeddings,
)

# ── Load LLM (Ollama running locally) ────────────────────────────────────────
llm = OllamaLLM(model=OLLAMA_MODEL)

# ── Prompt template ───────────────────────────────────────────────────────────
PROMPT_TEMPLATE = """
You are a music expert. Use the following song lyrics to answer the question.
You MUST answer with ONLY a single letter: A, B, C, or D.
Do not explain. Do not write anything else. Just one letter.

Song lyrics context:
{context}

Question: {question}
Options:
A) {option_a}
B) {option_b}
C) {option_c}
D) {option_d}

Your answer (one letter only):"""

# ── Evaluation loop ───────────────────────────────────────────────────────────
with open(DATASET_PATH, "r") as f:
    dataset = json.load(f)

results = []
correct = 0

for item in dataset:
    # Build the query combining song info and question
    query = f"{item['song']} by {item['artist']}: {item['question']}"

    # Retrieve relevant chunks from vector store
    retriever = vectorstore.as_retriever(
        search_kwargs={
            "k": 3,
            "filter": {"title": item["song"].lower()}
        }
    )
    docs = retriever.invoke(query)
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
    raw_answer = llm.invoke(prompt).strip().upper()
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
