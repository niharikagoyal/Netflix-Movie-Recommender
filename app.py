import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load data and models
movies = pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv")
cos_sim = joblib.load("content_model.pkl")  # genre-based similarity
pivot, user_sim = joblib.load("user_cf_model.pkl")  # user similarity + rating pivot

# Preprocess genre list
all_genres = sorted(set(g for genre_str in movies['genres'] for g in genre_str.split('|')))

# Streamlit page setup
st.set_page_config(page_title="🎬 Netflix Recommender", layout="wide")
st.title("🍿 Netflix Movie Recommender System ")

# Sidebar
st.sidebar.header("🎯 Recommendation Type")
method = st.sidebar.radio("Choose a method", ["Content-Based", "User-Based Collaborative"])
selected_genre = st.sidebar.selectbox("Filter by Genre (Optional)", ["All"] + all_genres)

# Movie title lookup
movie_indices = pd.Series(movies.index, index=movies['title'])

# Content-based filtering
def recommend_content(title, n=5):
    idx = movie_indices[title]
    sim_scores = list(enumerate(cos_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:n+1]
    movie_ids = [i[0] for i in sim_scores]
    return movies.iloc[movie_ids][['title', 'genres']]

# User-based collaborative filtering
def recommend_user_based(user_id, n=5):
    if user_id not in pivot.index:
        return pd.DataFrame(columns=["title", "genres"])

    user_idx = pivot.index.get_loc(user_id)
    sim_scores = list(enumerate(user_sim[user_idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    top_users = [i[0] for i in sim_scores[1:6]]  # top 5 similar users
    mean_ratings = pivot.iloc[top_users].mean().sort_values(ascending=False)
    unseen_movies = mean_ratings[pivot.loc[user_id] == 0].head(n).index.tolist()
    
    return movies[movies["movieId"].isin(unseen_movies)][["title", "genres"]]

# UI logic
if method == "Content-Based":
    selected_title = st.selectbox("Choose a movie you like 🎥", movies['title'].values)
    if st.button("Show Recommendations"):
        st.subheader(f"🎬 Similar Movies{' in ' + selected_genre if selected_genre != 'All' else ''}:")
        results = recommend_content(selected_title)
        if selected_genre != "All":
            results = results[results['genres'].str.contains(selected_genre)]
        for _, row in results.iterrows():
            st.markdown(f"**{row['title']}** — _{row['genres']}_")

else:
    selected_user = st.selectbox("Select your User ID 👤", pivot.index)
    if st.button("Show Personalized Picks"):
        st.subheader(f"🎯 Movies Recommended for You{' in ' + selected_genre if selected_genre != 'All' else ''}:")
        results = recommend_user_based(selected_user)
        if selected_genre != "All":
            results = results[results['genres'].str.contains(selected_genre)]
        for _, row in results.iterrows():
            st.markdown(f"**{row['title']}** — _{row['genres']}_")
