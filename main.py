import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the movies and ratings datasets
movies = pd.read_csv("movies_dataset/movie.csv")
ratings = pd.read_csv("movies_dataset/rating.csv")

# Merge the datasets
data = pd.merge(ratings, movies, on="movieId")

# Use a subset of the data for testing (first 10,000 movies)
data_subset = data[:10000]
genre_matrix = CountVectorizer(tokenizer=lambda x: x.split('|'), token_pattern=None).fit_transform(data_subset['genres'])

# Compute the similarity matrix
similarity_matrix = cosine_similarity(genre_matrix)

# Recommendation function
def recommend_movies(movie_title, num_recommendations=5):
    try:
        # Get the index of the movie
        movie_idx = data_subset[data_subset['title'] == movie_title].index[0]
        similarity_scores = list(enumerate(similarity_matrix[movie_idx]))
        sorted_movies = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

        # Collect unique recommendations
        recommended_indices = []
        seen_titles = set()
        for i, _ in sorted_movies:
            title = data_subset.iloc[i]['title']
            if title not in seen_titles and title != movie_title:
                recommended_indices.append(i)
                seen_titles.add(title)
            if len(recommended_indices) >= num_recommendations:
                break

        return data_subset.iloc[recommended_indices][['title', 'genres']]
    except IndexError:
        return "Movie not found in dataset!"

# Test the recommendation system
print("\nRecommended Movies for 'Toy Story (1995)':")
print(recommend_movies('Toy Story (1995)'))