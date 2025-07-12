## 🎬 Netflix Hybrid Movie Recommender System
A smart movie recommender system that blends Content-Based Filtering and User-Based Collaborative Filtering, allowing users to explore personalized movie recommendations through an interactive Streamlit interface.

✅ No external API needed
✅ Runs fully offline using local CSV files and pre-trained .pkl models
✅ Built without SVD — using user similarity matrix for collaborative filtering

## 🚀 Features
🔍 Two Recommendation Modes

Content-Based Filtering — Suggests movies similar to a selected one using cosine similarity from genre vectors.

User-Based Collaborative Filtering — Personalized recommendations based on user rating patterns.

🎭 Genre Filter — Filter content-based recommendations by selected genres (optional).

💾 Offline Functionality — Works without external API calls using movies.csv, ratings.csv, and .pkl model files.

🌐 Interactive UI — Built with Streamlit for a clean, modern user experience.

## 🧠 Tech Stack
Component	Technology
Programming Language	Python
Data Handling	Pandas, NumPy
ML / Recommendation	Scikit-learn, Cosine Similarity
Model Storage	joblib (.pkl files)
Web Framework	Streamlit

## 📂 Dataset Used
movies.csv — Contains movie titles and genres

ratings.csv — Contains userId, movieId, and rating

(Dataset based on MovieLens structure)

## ⚙️ How to Use
1️⃣ Train Models (Optional)
If you want to regenerate models:
python build_content_model.py
python build_user_cf_model.py
2️⃣ Run the Application
streamlit run app.py

## 🧪 How It Works
Content-Based Recommender:

Uses TF-IDF or CountVectorizer on genres

Computes cosine similarity

Recommends movies similar to the one selected

User-Based Collaborative Filtering:

Builds user-item matrix

Finds similar users based on cosine similarity

Predicts ratings for unseen movies

## 🌱 Future Scope
🖼️ Add TMDB API to display real-time movie posters

🔐 User login system + live rating submission

🌍 Deploy via Streamlit Cloud / Render

👩‍💻 Author
Niharika Goyal
