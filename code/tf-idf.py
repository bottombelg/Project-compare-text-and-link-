import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  
    return text

def split_into_sentences(text):
    sentences = re.split(r'[.!?]', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 0]
    return sentences
  
def compute_similarity(source_text, mention_text):
    sentences = split_into_sentences(source_text)
    sentences = [preprocess(s) for s in sentences]
    mention_text = preprocess(mention_text)
    corpus = sentences + [mention_text]
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),     
        max_features=5000
    )
    tfidf_matrix = vectorizer.fit_transform(corpus)
    mention_vec = tfidf_matrix[-1]
    similarities = cosine_similarity(mention_vec, tfidf_matrix[:-1])[0]
    max_score = np.max(similarities)
    return max_score
    
with open("file1.txt") as f :
  mention = f.read()
with open("file2.txt") as f :
  source = f.read()
print(compute_similarity(source,mention))
