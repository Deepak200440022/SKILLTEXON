from dataset_summary import merged_df
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# TF-IDF Vectorization (stop_words param improves quality)
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(merged_df["tags"])

# Cosine similarity
similarity_matrix = cosine_similarity(tfidf_matrix)

# Movie title to index mapping (cached)
title_to_index = {title.lower(): idx for idx, title in enumerate(merged_df["title"])}

# Recommendation function
def recommend_similar_movies(title, top_n=10):
    idx = title_to_index.get(title.lower())
    if idx is None:
        raise ValueError(f"Title '{title}' not found.")

    sim_scores = similarity_matrix[idx]
    top_indices = np.argpartition(-sim_scores, range(1, top_n + 1))[1:top_n + 1]
    top_indices = top_indices[np.argsort(-sim_scores[top_indices])]

    return [(i, sim_scores[i]) for i in top_indices]



# Example usage
# print(recommend_similar_movies("The Avengers", top_n=10))
