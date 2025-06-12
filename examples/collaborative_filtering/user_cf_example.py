# examples/collaborative_filtering/user_cf_example.py
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
import sys

# --- User-Based Collaborative Filtering: Basic Explanation ---
# User-Based Collaborative Filtering (UBCF) is a recommendation algorithm that suggests items
# to a user based on the preferences of other users with similar tastes.
# The core idea is: "If users A and B have similar preferences for items they both rated,
# then user A will likely prefer items that user B liked but user A has not yet seen."
#
# How it works:
# 1. Build a User-Item Interaction Matrix: This matrix (often sparse) contains users' interactions
#    with items. Values can be explicit ratings (e.g., 1-5 stars) or implicit feedback (e.g., views, clicks).
#    Rows represent users, and columns represent items.
# 2. Calculate User-User Similarity: Determine how similar users are to each other based on their
#    interaction history. A common method is Cosine Similarity, calculated on user vectors
#    (rows in the user-item matrix). Other metrics like Pearson correlation can also be used.
# 3. Generate Recommendations: To recommend items for a target user:
#    a. Identify "Neighboring Users": Find a subset of users who are most similar to the target user
#       (e.g., top N similar users).
#    b. Predict Scores for Unseen Items: For items the target user hasn't interacted with, predict
#       their likely rating. This is often done by taking a weighted average of the ratings given by
#       the neighboring users to that item. The weights are typically the similarity scores of these neighbors
#       to the target user.
#    c. Rank and Recommend: Sort items by their predicted scores and recommend the top K items.
#
# Pros:
# - Simplicity and Intuitiveness: The underlying concept is straightforward and easy to explain.
# - Effective for Diverse Tastes: Can uncover nuanced preferences if there are users with similar,
#   non-obvious taste patterns.
# - Serendipity: Can sometimes recommend items that are not directly similar (content-wise) to what
#   the user has consumed before, by leveraging the tastes of similar users.
# - No Item Feature Engineering: Doesn't require knowledge about item characteristics.
#
# Cons:
# - Data Sparsity: Performance degrades significantly if the user-item matrix is very sparse.
#   If users have rated few items in common, similarity calculations become unreliable.
# - Scalability of Similarity Calculation: Computing user-user similarity is computationally
#   expensive. For U users and I items, it's typically O(U^2 * I) for dense data or
#   O(U^2 * avg_items_rated_per_user) for sparse data when using pairwise cosine similarity.
#   This can be prohibitive for millions of users.
# - New User (Cold-Start): Difficult to provide recommendations for new users with few or no
#   interactions, as their similarity to others cannot be reliably computed.
# - Popularity Bias: May tend to over-recommend items that are popular among many users,
#   reducing personalization for users with niche tastes if not handled.
# - Dynamic User Preferences: User tastes can change over time. The user similarity model
#   might need frequent recalculation.
# ---

# Dynamically add project root to sys.path for module imports
# This allows the script to be run from any directory and still find project modules.
# __file__ is examples/collaborative_filtering/user_cf_example.py
# os.path.dirname(__file__) is examples/collaborative_filtering
# os.path.join(os.path.dirname(__file__), '..', '..') navigates up two levels to the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root) # Insert at the beginning for higher precedence

# 1. Data Loading and Preprocessing
# Time Complexity:
# - Reading CSV: O(N_interactions) where N_interactions is the number of rows in the CSV.
# - Pivot table: O(N_interactions) to iterate through data, then O(U*I) to construct the dense matrix,
#   where U is the number of unique users and I is the number of unique items.
# - fillna(0): O(U*I) in the worst case for a dense matrix.
# Overall: Roughly O(N_interactions + U*I).
def load_and_preprocess_data(filepath='data/dummy_interactions.csv'):
    """
    Loads interaction data from a CSV file and transforms it into a user-item matrix.
    If the specified CSV file is not found, it attempts to generate dummy data.

    Args:
        filepath (str): Path to the CSV data file. Assumed to be relative to project root
                        if it starts with 'data/', otherwise an absolute path.

    Returns:
        pandas.DataFrame: A user-item matrix where rows are user_id, columns are item_id,
                          and values are ratings. Returns None if data loading fails.
    """
    # Construct absolute path to data file, assuming 'data/' paths are relative to project root
    abs_filepath = filepath
    if not os.path.isabs(filepath) and filepath.startswith('data/'):
        abs_filepath = os.path.join(project_root, filepath)

    if not os.path.exists(abs_filepath):
        print(f"Error: Data file not found at {abs_filepath}.")
        # Specifically check if it's the default dummy data path to offer generation
        if abs_filepath.endswith('data/dummy_interactions.csv'):
            print("Attempting to generate dummy data by running 'data/generate_dummy_data.py'...")
            try:
                # Dynamically import and run the data generation script
                from data.generate_dummy_data import generate_dummy_data
                generate_dummy_data()
                print("Dummy data generated successfully.")
                # Verify if the file was created
                if not os.path.exists(abs_filepath):
                    print(f"Error: Dummy data generation completed, but file still not found at {abs_filepath}.")
                    return None
            except ImportError as e_import:
                print(f"ImportError: Failed to import 'generate_dummy_data'. Ensure it's in the 'data' directory. "
                      f"Project root: {project_root}, sys.path: {sys.path}. Error: {e_import}")
                return None
            except Exception as e_general:
                print(f"Error: Failed to generate dummy data. Exception: {e_general}")
                return None
        else:
            # The missing file is not the one we know how to generate
            return None

    # Load data from CSV
    df = pd.read_csv(abs_filepath)
    # Create the user-item matrix:
    # - index='user_id': users become rows
    # - columns='item_id': items become columns
    # - values='rating': ratings fill the matrix cells
    # - fillna(0): missing ratings (where a user hasn't rated an item) are filled with 0.
    #   This is a common practice, but can influence similarity (e.g., implies neutrality or no interaction).
    user_item_matrix = df.pivot_table(index='user_id', columns='item_id', values='rating').fillna(0)
    return user_item_matrix

# 2. User Similarity Calculation
# Time Complexity: O(U^2 * I) where U is the number of users and I is the number of items.
# This is because cosine_similarity calculates similarity between all pairs of U users,
# and each user's preference vector has length I. This step can be a bottleneck for large U.
def calculate_user_similarity(user_item_matrix):
    """
    Calculates cosine similarity between users based on their item ratings.

    Args:
        user_item_matrix (pandas.DataFrame): The user-item rating matrix.

    Returns:
        pandas.DataFrame: A square matrix where rows and columns are user_ids,
                          and values are the cosine similarity scores between users.
    """
    # cosine_similarity expects samples in rows, which is how our user_item_matrix is structured (users as rows).
    user_similarity_matrix = cosine_similarity(user_item_matrix)
    # Convert the resulting NumPy array back to a DataFrame for easier handling,
    # using original user_ids as index and columns.
    user_similarity_df = pd.DataFrame(user_similarity_matrix,
                                      index=user_item_matrix.index,
                                      columns=user_item_matrix.index)
    return user_similarity_df

# 3. User-Based Collaborative Filtering Recommendations
# Time Complexity for a single target user:
# - Get target user's ratings: O(I)
# - Get similarities for target user: O(U)
# - Find top N similar users: O(U log U) if sorting all, or O(U log N_similar) with a heap (nlargest).
# - Iterate through top_n_similar_users (let this be K_sim):
#   - For each similar user, iterate through their items (avg_items_rated_by_sim_user, let this be I_avg_sim):
#     - Dictionary lookups/updates for scores: O(1) on average.
# - Sorting final recommendations: O(M log M) where M is the number of candidate items.
# Overall: Roughly O(U + K_sim * I_avg_sim + M log M).
# If K_sim is small and M is not too large, this can be efficient per user.
def get_user_based_recommendations(target_user_id, user_item_matrix, user_similarity_df, num_recommendations=5, top_n_similar_users=10):
    """
    Generates item recommendations for a target user using User-Based Collaborative Filtering.

    Args:
        target_user_id (int or str): The ID of the user for whom to generate recommendations.
        user_item_matrix (pandas.DataFrame): The user-item rating matrix.
        user_similarity_df (pandas.DataFrame): The user-user similarity matrix.
        num_recommendations (int): The number of items to recommend.
        top_n_similar_users (int): The number of most similar users (neighbors) to consider.

    Returns:
        list: A list of tuples, where each tuple contains (item_id, predicted_score),
              sorted by predicted_score in descending order.
    """
    if target_user_id not in user_item_matrix.index:
        print(f"Error: Target user ID {target_user_id} not found in the user-item matrix.")
        return []

    # Get items already rated by the target user to exclude them from recommendations
    target_user_ratings = user_item_matrix.loc[target_user_id]
    target_rated_items = target_user_ratings[target_user_ratings > 0].index

    # Get similarity scores of all other users to the target user
    # Exclude the target user itself from the similarity list
    similarities_to_target = user_similarity_df.loc[target_user_id].drop(target_user_id, errors='ignore')

    # Select the top N most similar users (neighbors)
    # Users with higher similarity scores are considered more influential.
    similar_users = similarities_to_target.nlargest(top_n_similar_users).index

    if not len(similar_users):
        print(f"No similar users found for user {target_user_id} with the current criteria.")
        return []

    # Store predicted scores for items
    recommendation_scores = {}
    # Store sum of similarity scores for weighted average calculation
    sum_similarity_for_item = {}

    # Iterate through each similar user
    for similar_user_id in similar_users:
        # Get the similarity score between the target user and the current similar user
        similarity_score = user_similarity_df.loc[target_user_id, similar_user_id]

        # Optional: Skip users with non-positive similarity (depends on similarity metric and data)
        if similarity_score <= 0:
            continue

        # Get ratings from the current similar user
        similar_user_ratings = user_item_matrix.loc[similar_user_id]

        # Iterate through items rated by the similar user
        for item_id, rating in similar_user_ratings.items():
            # Recommend only if the item is positively rated by the similar user
            # and not already rated by the target user
            if rating > 0 and item_id not in target_rated_items:
                # Aggregate scores: Add (similarity_score * rating)
                recommendation_scores[item_id] = recommendation_scores.get(item_id, 0) + similarity_score * rating
                # Aggregate similarity scores for the denominator of the weighted average
                sum_similarity_for_item[item_id] = sum_similarity_for_item.get(item_id, 0) + similarity_score

    # Calculate the weighted average score for each candidate item
    # final_score = sum(similarity_score * rating_from_similar_user) / sum(similarity_scores_of_users_who_rated_the_item)
    final_scores = {}
    for item_id, total_weighted_score in recommendation_scores.items():
        if sum_similarity_for_item.get(item_id, 0) > 0: # Avoid division by zero
            final_scores[item_id] = total_weighted_score / sum_similarity_for_item[item_id]
        # Items with sum_similarity_for_item == 0 would mean they were not processed, or only by users with 0 similarity.

    # Sort recommendations by predicted score in descending order
    sorted_recommendations = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)

    return sorted_recommendations[:num_recommendations]

# Main execution block
if __name__ == "__main__":
    print("--- User-Based Collaborative Filtering Example ---")

    # Define the path to the data file (relative to project root)
    data_file_path = 'data/dummy_interactions.csv'
    print(f"\nAttempting to load and preprocess data from: {data_file_path}")
    user_item_matrix = load_and_preprocess_data(filepath=data_file_path)

    if user_item_matrix is not None and not user_item_matrix.empty:
        print("\nUser-Item Matrix (first 5 rows/users):")
        print(user_item_matrix.head())

        print("\nCalculating User Similarity Matrix... (this may take a while for many users)")
        user_similarity_df = calculate_user_similarity(user_item_matrix)
        print("\nUser Similarity Matrix (first 5x5 users):")
        # Ensure there are enough users/items to display 5x5, otherwise adjust
        display_limit = min(5, user_similarity_df.shape[0])
        print(user_similarity_df.iloc[:display_limit, :display_limit])

        # Select a target user for recommendations
        # For robustness, pick the first user. Handle cases where the matrix might be small.
        if not user_item_matrix.empty:
            target_user_id_example = user_item_matrix.index[0]

            # Check if the target user has rated any items for more meaningful recommendations
            if user_item_matrix.loc[target_user_id_example].sum() == 0:
                print(f"\nWarning: Target user ID {target_user_id_example} has no rated items in the matrix. "
                      "Recommendations might be less meaningful or empty.")
                # Attempt to find a user with some ratings for a better demonstration
                for uid in user_item_matrix.index:
                    if user_item_matrix.loc[uid].sum() > 0:
                        target_user_id_example = uid
                        print(f"Switching to user ID {target_user_id_example} for a better demo as they have rated items.")
                        break

            print(f"\nGenerating recommendations for User ID: {target_user_id_example}...")
            # Get recommendations
            # top_n_similar_users: How many neighbors to consider.
            # A larger number might increase coverage but could include less similar users.
            recommendations = get_user_based_recommendations(
                target_user_id_example,
                user_item_matrix,
                user_similarity_df,
                num_recommendations=5, # Number of items to recommend
                top_n_similar_users=10 # Consider top 10 similar users
            )

            if recommendations:
                print(f"\nTop {len(recommendations)} recommendations for User {target_user_id_example}:")
                for item_id, score in recommendations:
                    print(f"- Item ID: {item_id}, Predicted Score: {score:.4f}")
            else:
                print(f"No recommendations could be generated for User {target_user_id_example}. "
                      "This could be due to no similar users found, or similar users "
                      "not rating any new items not already seen by the target user.")
        else:
            print("\nUser-Item matrix is empty. Cannot generate recommendations.")
    else:
        print("\nData loading failed or resulted in an empty matrix. Cannot proceed with the example.")

    print("\n--- User-Based Collaborative Filtering Example Finished ---")
