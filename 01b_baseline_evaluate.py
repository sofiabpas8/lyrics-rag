"""
STEP 1b - Baseline LLM evaluation WITHOUT RAG (Part 1 of assignment)
This runs the LLM directly on the MC questions, no retrieval, for comparison.
"""

import json
from langchain_community.llms import Ollama

# ── Config ────────────────────────────────────────────────────────────────────
DATASET_PATH = "data/eval_dataset.json"
OLLAMA_MODEL = "phi"

PROMPT_TEMPLATE = """
You are a music expert. Answer the following multiple-choice question.
Answer ONLY with the letter of the correct option (A, B, C, or D). Nothing else.

Question: {question} (Song: {song} by {artist})
Options:
A) {option_a}
B) {option_b}
C) {option_c}
D) {option_d}

Answer (just the letter):"""

# ── Load dataset ──────────────────────────────────────────────────────────────
with open(DATASET_PATH, "r") as f:
    dataset = json.load(f)

llm = Ollama(model=OLLAMA_MODEL)

results = []
correct = 0

for item in dataset:
    prompt = PROMPT_TEMPLATE.format(
        question=item["question"],
        song=item["song"],
        artist=item["artist"],
        option_a=item["options"]["A"],
        option_b=item["options"]["B"],
        option_c=item["options"]["C"],
        option_d=item["options"]["D"],
    )

    raw_answer = llm.invoke(prompt).strip().upper()
    predicted = raw_answer[0] if raw_answer and raw_answer[0] in "ABCD" else "?"

    is_correct = predicted == item["answer"]
    if is_correct:
        correct += 1

    results.append({
        "id": item["id"],
        "song": item["song"],
        "expected": item["answer"],
        "predicted": predicted,
        "correct": is_correct,
    })

    status = "✓" if is_correct else "✗"
    print(f"[{status}] Q{item['id']} ({item['song']}): expected={item['answer']}, got={predicted}")

accuracy = correct / len(dataset) * 100
print(f"\n{'='*40}")
print(f"Baseline LLM Accuracy (no RAG): {correct}/{len(dataset)} = {accuracy:.1f}%")
print(f"{'='*40}")

with open("data/baseline_results.json", "w") as f:
    json.dump({"accuracy": accuracy, "results": results}, f, indent=2)
print("Baseline results saved to data/baseline_results.json")
