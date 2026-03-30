import json
import random
from collections import Counter

from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CLAIMS_TRAIN_PATH = "claims_train.jsonl"
CORPUS_PATH = "corpus.jsonl"
OUTPUT_DIR = "./scifact_sbert_claim_abstract_cpu"

BATCH_SIZE = 2
EPOCHS = 2
LEARNING_RATE = 2e-5
WARMUP_RATIO = 0.1
MAX_LEN = 256
SEED = 42


def set_seed(seed=42):
    random.seed(seed)

def load_corpus(corpus_path):
    corpus = {}

    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            doc_id = obj["doc_id"]
            abstract = obj.get("abstract", [])
            if isinstance(abstract, list):
                abstract_text = " ".join(abstract).strip()
            else:
                abstract_text = str(abstract).strip()
            abstract_text = truncate(abstract_text)
            corpus[doc_id] = abstract_text

    return corpus


def extract_label_from_evidence_list(evidence_list):
    if not evidence_list:
        return None
    first_item = evidence_list[0]
    if isinstance(first_item, dict):
        return first_item.get("label")
    return None


def load_claim_abstract_examples(claims_path, corpus):
    examples = []
    raw_labels = Counter()

    with open(claims_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            claim_text = obj.get("claim", "").strip()
            if not claim_text:
                continue
            cited_doc_ids = obj.get("cited_doc_ids", [])
            evidence = obj.get("evidence", {})
            for doc_id in cited_doc_ids:
                if doc_id not in corpus:
                    continue
                doc_evidence = evidence.get(str(doc_id), evidence.get(doc_id, []))
                label = extract_label_from_evidence_list(doc_evidence)
                if label is None:
                    continue
                label = label.upper().strip()
                raw_labels[label] += 1
                if label == "SUPPORT":
                    score = 1.0
                elif label in {"CONTRADICT", "REFUTE", "REFUTES", "NOT_ENOUGH_INFO"}:
                    score = 0.0
                else:
                    continue
                abstract_text = corpus[doc_id]
                examples.append(
                    InputExample(
                        texts=[claim_text, abstract_text],
                        label=score
                    )
                )
    return examples, raw_labels


def main():
    set_seed(SEED)
    corpus = load_corpus(CORPUS_PATH)
    train_examples, raw_labels = load_claim_abstract_examples(CLAIMS_TRAIN_PATH, corpus)
    if len(train_examples) == 0:
        raise ValueError("train_examples пуст. Проверьте структуру claims_train и corpus.")
    score_counter = Counter(ex.label for ex in train_examples)
    model = SentenceTransformer(MODEL_NAME, device="cpu")

    train_dataloader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=BATCH_SIZE
    )

    train_loss = losses.CosineSimilarityLoss(model)

    warmup_steps = max(1, int(len(train_dataloader) * EPOCHS * WARMUP_RATIO))
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=EPOCHS,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": LEARNING_RATE},
        output_path=OUTPUT_DIR,
        show_progress_bar=True
    )

    print(f"Model saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
