import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import joblib

ratings = pd.read_csv("ratings.csv")

# Pivot table: users as rows, movies as columns
pivot = ratings.pivot_table(index='userId', columns='movieId', values='rating').fillna(0)

# Cosine similarity between users
user_sim = cosine_similarity(pivot)

# Save pivot and similarity matrix
joblib.dump((pivot, user_sim), "user_cf_model.pkl")

print("✅ User-based collaborative filtering model saved!")
