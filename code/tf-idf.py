from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

with open("file1.txt") as f :
  source_text = f.read()
with open("file2.txt") as f :
  mention_text = f.read()
  
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform([source_text, mention_text])

similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])

print(similarity[0][0])
