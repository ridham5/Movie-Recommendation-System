# Movie Recommendation System

This project is a **content-based movie recommendation system** built in Python. It uses movie genres to recommend movies that are similar to a given title. The system demonstrates fundamental concepts of data preprocessing, feature extraction, and similarity measurement.

The project is built to showcase programming skills in **Python**, **data handling**, and **machine learning techniques** to potential recruiters.

---

# Features
- Recommends movies based on their genres using **content-based filtering**.
- Utilizes **cosine similarity** to compute movie-to-movie relationships.
- Handles duplicate recommendations to ensure unique results.
- Tested with a subset of the [MovieLens Dataset](https://www.kaggle.com/datasets/grouplens/movielens-20m-dataset).
- Modular and easy-to-extend code for future enhancements.

# How to Run
1. Clone this repository to your local machine:
   ```bash
   git clone https://github.com/your-username/movie-recommendation-system.git
2. Navigate to the project directory:
    cd movie-recommendation-system
3. Install the required Python libraries:
    pip install pandas numpy scikit-learn
4. Download the Kaggle dataset:
    Visit the Kaggle MovieLens Dataset.
    Extract the downloaded files into the movies_dataset folder in the project directory.
5. Run the script:
    python main.py

# Example Output
For the movie "Toy Story (1995)", the system recommends:

Rank	Recommended Movie	Genres
1	Adventures of Rocky and Bullwinkle, The (2000)	Adventure, Animation, Children
2	Antz (1998)	Adventure, Animation, Children, Comedy
3	Toy Story 2 (1999)	Adventure, Animation, Children, Comedy
4	Monsters, Inc. (2001)	Adventure, Animation, Children, Fantasy
5	Emperor's New Groove, The (2000)	Adventure, Animation, Children, Comedy

# Dataset
The system uses the Kaggle MovieLens Dataset to train the recommendation model. It processes the movies.csv and ratings.csv files to build the recommendations.

# Skills and Technologies Used
- Python: Data preprocessing, analysis, and modeling.
- Pandas: Efficient handling of large datasets.
- Scikit-Learn: Feature extraction and similarity measurement.
- Cosine Similarity: For calculating the similarity between movies.
- Content-Based Filtering: Personalized recommendations based on genres.

# Future Enhancements
- Expand recommendations to consider user ratings (collaborative filtering).
- Integrate with a web-based user interface for better usability.
- Deploy the system as a REST API for real-world applications.