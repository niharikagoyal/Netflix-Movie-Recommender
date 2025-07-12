import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib

# Load movie metadata
movies = pd.read_csv("movies.csv")

# Replace | with space for TF-IDF
movies["genres"] = movies["genres"].str.replace("|", " ")

# TF-IDF on genres
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies["genres"])

# Cosine Similarity
cosine_sim = cosine_similarity(tfidf_matrix)

# Save model
joblib.dump(cosine_sim, "content_model.pkl")
print("✅ Content-based model saved!")
