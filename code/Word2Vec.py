import re
import numpy as np
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def split_into_sentences(text):
    sentences = re.split(r'[.!?]', text)
    return [s.strip() for s in sentences if s.strip()]

def tokenize(text):
    return text.split()

def train_word2vec(sentences):
    tokenized = [tokenize(s) for s in sentences]

    model = Word2Vec(
        sentences=tokenized,
        vector_size=300,
        window=5,
        min_count=1,
        workers=4
    )
    return model

def sentence_embedding(text, model):
    words = text.split()
    vectors = [model.wv[w] for w in words if w in model.wv]
    if len(vectors) == 0:
        return np.zeros(model.vector_size)
    return np.mean(vectors, axis=0)

def compute_similarity_w2v(source_text, mention_text):
    sentences = split_into_sentences(source_text)
    if len(sentences) == 0:
        return 0.0
    sentences = [preprocess(s) for s in sentences]
    mention_text = preprocess(mention_text)
    all_sentences = sentences + [mention_text]
    model = train_word2vec(all_sentences)
    mention_vec = sentence_embedding(mention_text, model)
    sentence_vecs = [sentence_embedding(s, model) for s in sentences]
    similarities = cosine_similarity([mention_vec], sentence_vecs)[0]

    top_k = min(3, len(similarities))
    top_scores = np.sort(similarities)[-top_k:]

    return float(np.mean(top_scores))

with open("file1.txt") as f:
    mention = f.read()

with open("file2.txt") as f:
    source = f.read()

print(compute_similarity_w2v(source, mention))
