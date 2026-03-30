import re
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util
import json
model = SentenceTransformer('./scifact_sbert_claim_abstract_cpu')

WINDOW_SIZE = 3
TOP_K = 5

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
    
with open("corpus.jsonl") as f :
    b = []
    count = 0
    for line in f :
        count += 1
        a = json.loads(line)['abstract']
        strsum = " ".join(a)
        b.append(strsum)
        if(count == 2) :
            break
with open("claims_test.jsonl") as f :
    test = []
    count = 0
    for line in f :
        count += 1
        test.append(json.loads(line)['claim'])
        if count == 2 :
            break
for elem in b :
    for j in test :
        print(compute_similarity(elem,j))
        print("source :", elem)
        print("mention:", j)
        print()

