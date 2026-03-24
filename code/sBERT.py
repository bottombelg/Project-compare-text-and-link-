
import re
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-mpnet-base-v2')

WINDOW_SIZE = 2
TOP_K = 3

def preprocess(text):
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def split_into_sentences(text):
    sentences = re.split(r'[.!?]', text)
    return [s.strip() for s in sentences if s.strip()]
  
def build_windows(sentences, window_size=WINDOW_SIZE):
    windows = []
    for i in range(len(sentences)):
        window = " ".join(sentences[i:i + window_size])
        if window:
            windows.append(window)
    return windows


def top_k_mean(scores, k=TOP_K):
    flat = scores.flatten()
    k = min(k, flat.shape[0])
    topk = torch.topk(flat, k).values
    return topk.mean().item()

def compute_similarity(source_text, mention_text):
    source_sentences = split_into_sentences(source_text)
    mention_sentences = split_into_sentences(mention_text)
    if not source_sentences or not mention_sentences:
        return 0.0
    source_windows = build_windows(source_sentences)
    mention_windows = build_windows(mention_sentences)
  
    source_windows = [preprocess(s) for s in source_windows]
    mention_windows = [preprocess(s) for s in mention_windows]

    source_embs = model.encode(
        source_windows,
        convert_to_tensor=True,
        normalize_embeddings=True
    )
  
    mention_embs = model.encode(
        mention_windows,
        convert_to_tensor=True,
        normalize_embeddings=True
    )

    scores = util.cos_sim(mention_embs, source_embs)
    score = top_k_mean(scores, k=TOP_K)
    return score

with open("file1.txt") as f:
  mention = f.read()

with open("file2.txt") as f:
  source = f.read()

print("Similarity:", compute_similarity(source, mention))
