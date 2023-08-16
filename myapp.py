import pandas as pd
import seaborn as sns
import streamlit as st
import numpy as np
import pickle
import joblib
from matplotlib import pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

st.markdown("<h1 style='text-align: center; color: black;'>MOVIE RECOMMENDER SYSTEM</h1>", unsafe_allow_html=True)
st.text("")
st.text("")
st.markdown("<h6 style='text-align: center;'>Welcome! Please fill in this form before we can continue recommend the next movie for you to watch!</h6>", unsafe_allow_html=True)
st.divider()

#Display sneak peak of data
movies = pd.read_csv('mergedata.csv')
st.dataframe(movies)

#FORM
recommendationform = st.form(key='form_recommend', clear_on_submit=True)
inserted_userid = recommendationform.text_input("Enter your User ID")
button_label = 'RECOMMEND'
submitted = recommendationform.form_submit_button(label=f'{button_label}')

if submitted:
        userid = int(inserted_userid)
        st.write('Here are some movie recommendations for you', userid, '! We hope you like it!')

        # Read in data
        ratings = pd.read_csv('ratings.csv')
        # Take a look at the data
        print(ratings.head())

        # Get the dataset information
        ratings.info()

        # Number of users
        print('The ratings dataset has', ratings['userId'].nunique(), 'unique users')
        # Number of movies
        print('The ratings dataset has', ratings['movieId'].nunique(), 'unique movies')
        # Number of ratings
        print('The ratings dataset has', ratings['rating'].nunique(), 'unique ratings')
        # List of unique ratings
        print('The unique ratings are', sorted(ratings['rating'].unique()))

        # Read in data
        movies = pd.read_csv('movies.csv')
        # Take a look at the data
        print(movies.head())

        # Merge ratings and movies datasets
        df = pd.merge(ratings, movies, on='movieId', how='inner')
        # save to a new .csv file
        output_file_path = 'mergedata.csv'
        df.to_csv(output_file_path, index=False)
        # Take a look at the data
        print(df.head())

        # Aggregate by movie
        agg_ratings = df.groupby('title').agg(mean_rating=('rating', 'mean'),
                                              number_of_ratings=('rating', 'count')).reset_index()
        # Keep the movies with over 100 ratings
        agg_ratings_GT100 = agg_ratings[agg_ratings['number_of_ratings'] > 100]
        agg_ratings_GT100.info()

        # Check popular movies
        print(agg_ratings_GT100.sort_values(by='number_of_ratings', ascending=False).head())

        # Visulization
        sns.jointplot(x='mean_rating', y='number_of_ratings', data=agg_ratings_GT100)
        plt.show()

        # Merge data
        df_GT100 = pd.merge(df, agg_ratings_GT100[['title']], on='title', how='inner')
        df_GT100.info()

        # Number of users
        print('The ratings dataset has', df_GT100['userId'].nunique(), 'unique users')
        # Number of movies
        print('The ratings dataset has', df_GT100['movieId'].nunique(), 'unique movies')
        # Number of ratings
        print('The ratings dataset has', df_GT100['rating'].nunique(), 'unique ratings')
        # List of unique ratings
        print('The unique ratings are', sorted(df_GT100['rating'].unique()))

        # Create user-item matrix
        matrix = df_GT100.pivot_table(index='userId', columns='title', values='rating')
        print(matrix.head())

        # Normalize user-item matrix
        matrix_norm = matrix.subtract(matrix.mean(axis=1), axis='rows')
        print(matrix_norm.head())
        # After normalization, the movies with a rating less than the user’s average rating
        # get a negative value, and the movies with a rating more than the user’s average rating get a positive value.

        # User similarity matrix using Pearson correlation
        user_similarity = matrix_norm.T.corr()
        print(user_similarity.head())

        # User similarity matrix using cosine similarity
        user_similarity_cosine = cosine_similarity(matrix_norm.fillna(0))
        print(user_similarity_cosine)

        # Pick a user ID
        #picked_userid = 1
        picked_userid = userid

        # Remove picked user ID from the candidate list
        user_similarity.drop(index=picked_userid, inplace=True)
        # Take a look at the data
        print(user_similarity.head())

        # -1 means opposite movie preference and 1 means same movie preference
        # Number of similar users
        n = 10
        # User similarity threashold
        user_similarity_threshold = 0.3
        # Get top n similar users
        similar_users = user_similarity[user_similarity[picked_userid] > user_similarity_threshold][
                            picked_userid].sort_values(ascending=False)[:n]
        # Print out top n similar users
        print(f'The similar users for user {picked_userid} are', similar_users)

        # Now, Remove the movies that have been watched by the target user
        # Keep only the movies that similar users have watched
        # Movies that the target user has watched
        picked_userid_watched = matrix_norm[matrix_norm.index == picked_userid].dropna(axis=1, how='all')
        print('\nThe movie that user has similar taste with other', picked_userid_watched)

        # Movies that similar users watched. Remove movies that none of the similar users have watched
        similar_user_movies = matrix_norm[matrix_norm.index.isin(similar_users.index)].dropna(axis=1, how='all')
        print('\nRemove other movie that user not similar taste with other', similar_user_movies)

        # Remove the watched movie from the movie list
        similar_user_movies.drop(picked_userid_watched.columns, axis=1, inplace=True, errors='ignore')
        # Take a look at the data
        print('\nRemove the watched movie from the movie list', similar_user_movies)

        # this code loops through items and users to get the item score,
        # rank the score from high to low and pick the top 10 movies to recommend to user ID 1.
        # A dictionary to store item scores
        item_score = {}
        # Loop through items
        for i in similar_user_movies.columns:
            # Get the ratings for movie i
            movie_rating = similar_user_movies[i]
            # Create a variable to store the score
            total = 0
            # Create a variable to store the number of scores
            count = 0
            # Loop through similar users
            for u in similar_users.index:
                # If the movie has rating
                if pd.isna(movie_rating[u]) == False:
                    # Score is the sum of user similarity score multiply by the movie rating
                    score = similar_users[u] * movie_rating[u]
                    # Add the score to the total score for the movie so far
                    total += score
                    # Add 1 to the count
                    count += 1
            # Get the average score for the item
            item_score[i] = total / count
        # Convert dictionary to pandas dataframe
        item_score = pd.DataFrame(item_score.items(), columns=['movie', 'movie_score'])

        # Sort the movies by score
        ranked_item_score = item_score.sort_values(by='movie_score', ascending=False)
        # Select top m movies
        m = 10
        print(ranked_item_score.head(m))
        st.write(ranked_item_score.head(10))

        # Average rating for the picked user (optional)
        avg_rating = matrix[matrix.index == picked_userid].T.mean()[picked_userid]
        # Print the average movie rating for user 1
        print(f'\nThe average movie rating for user {picked_userid} is {avg_rating:.2f}')

        # Calcuate the predicted rating
        ranked_item_score['predicted_rating'] = ranked_item_score['movie_score'] + avg_rating
        # Take a look at the data
        print(ranked_item_score.head(m))

#st.write("Hope you like it!")