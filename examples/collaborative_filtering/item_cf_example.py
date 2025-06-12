# examples/collaborative_filtering/item_cf_example.py
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
import sys

# --- Item-Based Collaborative Filtering: Basic Explanation ---
# Item-Based Collaborative Filtering (IBCF) recommends items to a user based on the similarity
# between items the user has previously interacted with and other items.
# The core idea: "If a user liked item X, and item X is similar to item Y, then the user might also like item Y."
#
# How it works:
# 1. Build a User-Item Interaction Matrix: Represents users' interactions (e.g., ratings) with items.
#    Rows are users, columns are items. This is the same matrix as in User-Based CF.
# 2. Calculate Item-Item Similarity: Compute similarity between all pairs of items.
#    This is typically done by creating item vectors (columns from the user-item matrix, possibly after transposing)
#    and calculating cosine similarity between these vectors. An item's vector represents how it's rated by all users.
#    The result is an item-item similarity matrix.
# 3. Generate Recommendations for a User:
#    a. Identify items the target user has positively rated or interacted with.
#    b. For each unrated item (candidate item for recommendation):
#       i. Calculate a predicted score. This is often a weighted sum of the user's ratings for their interacted items.
#       ii. The weight for each rated item is its similarity to the candidate item.
#       iii. Formula for predicted score for item 'j' for user 'u':
#           P(u,j) = sum( S(j,k) * R(u,k) ) / sum( |S(j,k)| )
#           where:
#             - S(j,k) is the similarity between item 'j' (candidate) and item 'k' (rated by user).
#             - R(u,k) is the rating user 'u' gave to item 'k'.
#             - The sum is over all items 'k' rated by user 'u'.
#    c. Rank candidate items by their predicted scores and recommend the top N.
#    d. Filter out items the user has already interacted with.
#
# Pros:
# - Stability: Item similarities often change less frequently than user-user similarities,
#   meaning the item-item similarity matrix can be pre-computed and updated less often.
#   This is beneficial if the item catalog is more static than the user base or user preferences.
# - Scalability for User Base: More scalable when the number of users is much larger than the number of items,
#   as the expensive similarity calculation is on items.
# - Explainability: Recommendations can be easily explained (e.g., "Because you liked Item X, you might like Item Y, which is similar to X").
# - Handles New Users: New users who rate even a few items can get decent recommendations if those items have known similarities.
#
# Cons:
# - Data Sparsity: If the user-item matrix is very sparse, it's hard to find users who have rated the same pairs of items,
#   making item similarity calculations less reliable.
# - Scalability of Similarity Calculation (for items): Computing item-item similarity can be computationally
#   expensive if the number of items (I) is very large. For I items and U users, it's typically O(I^2 * U)
#   for dense data or O(I^2 * avg_users_rated_per_item) for sparse data.
# - Popularity Bias: May tend to recommend popular items, similar to UBCF.
# - Cold-Start for New Items: Cannot recommend new items that have no interaction data, as their similarity
#   to other items cannot be determined.
# - Limited Serendipity: May recommend items too similar to what the user already knows, potentially reducing discovery of novel items.
# ---

# Dynamically add project root to sys.path for module imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 1. Data Loading and Preprocessing
# Time Complexity:
# - Reading CSV: O(N_interactions)
# - Pivot table & fillna: O(N_interactions + U*I)
# Overall: Roughly O(N_interactions + U*I).
def load_and_preprocess_data(filepath='data/dummy_interactions.csv'):
    """
    Loads interaction data and transforms it into a user-item matrix.
    Attempts to generate dummy data if the specified file is not found.
    """
    abs_filepath = filepath
    if not os.path.isabs(filepath) and filepath.startswith('data/'):
        abs_filepath = os.path.join(project_root, filepath)

    if not os.path.exists(abs_filepath):
        print(f"Error: Data file not found at {abs_filepath}.")
        if abs_filepath.endswith('data/dummy_interactions.csv'):
            print("Attempting to generate dummy data...")
            try:
                from data.generate_dummy_data import generate_dummy_data
                generate_dummy_data()
                print("Dummy data generated successfully.")
                if not os.path.exists(abs_filepath):
                    print(f"Error: Dummy data generation ran, but file still not found at {abs_filepath}.")
                    return None
            except ImportError as e_import:
                print(f"ImportError: Failed to import 'generate_dummy_data'. Project root: {project_root}, sys.path: {sys.path}. Error: {e_import}")
                return None
            except Exception as e_general:
                print(f"Error generating dummy data: {e_general}")
                return None
        else:
            return None

    df = pd.read_csv(abs_filepath)
    # Pivot to create user-item matrix (users as rows, items as columns, ratings as values)
    # Fill missing interactions with 0 (neutral or no interaction)
    user_item_matrix = df.pivot_table(index='user_id', columns='item_id', values='rating').fillna(0)
    return user_item_matrix

# 2. Item Similarity Calculation
# Time Complexity: O(I^2 * U), where I is the number of items and U is the number of users.
# user_item_matrix.T transposes the matrix to (I x U).
# cosine_similarity then computes similarities between all pairs of I items,
# where each item's vector has length U.
def calculate_item_similarity(user_item_matrix):
    """
    Calculates cosine similarity between items based on user ratings.
    """
    # Transpose the matrix so items are rows and users are columns (I x U)
    # This allows cosine_similarity to compute similarities between item vectors.
    item_user_matrix = user_item_matrix.T
    item_similarity_matrix = cosine_similarity(item_user_matrix)
    # Convert to DataFrame for easier use, with item_ids as index and columns
    item_similarity_df = pd.DataFrame(item_similarity_matrix,
                                      index=user_item_matrix.columns, # Original item_ids
                                      columns=user_item_matrix.columns) # Original item_ids
    return item_similarity_df

# 3. Item-Based Collaborative Filtering Recommendations
# Time Complexity for a single user:
# - Get user's rated items: O(I)
# - Iterate through unrated items (I_unrated, worst case I):
#   - For each unrated item, iterate through user's rated items (I_rated):
#     - Look up similarity: O(1) (DataFrame lookup)
#     - Perform calculations.
# - Sorting recommendations: O(M log M) where M is number of candidate items (I_unrated).
# Overall: Roughly O(I_unrated * I_rated + M log M).
# If I_rated is small, this is more efficient than iterating all I items for similarity.
def get_item_based_recommendations(user_id, user_item_matrix, item_similarity_df, num_recommendations=5):
    """
    Generates item recommendations for a target user using Item-Based Collaborative Filtering.
    """
    if user_id not in user_item_matrix.index:
        print(f"Error: User ID {user_id} not found in the data.")
        return []

    # Get the ratings provided by the target user
    user_ratings_series = user_item_matrix.loc[user_id]
    # Filter for items the user has positively rated (rating > 0)
    rated_items_with_scores = user_ratings_series[user_ratings_series > 0]
    rated_item_ids = rated_items_with_scores.index

    # Identify items the user has not yet rated
    all_item_ids = user_item_matrix.columns
    unrated_item_ids = all_item_ids.difference(rated_item_ids)

    if not rated_item_ids.tolist(): # Handle case where user has no rated items
        print(f"User {user_id} has no rated items. Cannot generate item-based recommendations.")
        return []

    if not unrated_item_ids.tolist(): # Handle case where user has rated all items
        print(f"User {user_id} has rated all available items. No new recommendations to generate.")
        return []

    # Dictionary to store the predicted scores for unrated items
    recommendation_scores = {}

    # For each item the user has not rated
    for item_to_predict_score_for in unrated_item_ids:
        weighted_sum_of_ratings = 0
        sum_of_similarity_scores = 0

        # For each item the user has already rated
        for previously_rated_item_id in rated_item_ids:
            # Get the similarity between the item we want to predict and the item already rated by the user
            similarity = item_similarity_df.loc[item_to_predict_score_for, previously_rated_item_id]

            # Consider only positive similarities
            if similarity > 0:
                # Get the rating the user gave to the already-rated item
                rating_for_previously_rated_item = rated_items_with_scores.loc[previously_rated_item_id]

                # Add to the weighted sum: similarity * user's rating for the similar item
                weighted_sum_of_ratings += similarity * rating_for_previously_rated_item
                # Add to the sum of similarities (for the denominator of weighted average)
                sum_of_similarity_scores += similarity

        # Calculate the predicted score if there were similar items
        if sum_of_similarity_scores > 0:
            predicted_score = weighted_sum_of_ratings / sum_of_similarity_scores
            recommendation_scores[item_to_predict_score_for] = predicted_score

    # Sort the unrated items by their predicted scores in descending order
    sorted_recommendations = sorted(recommendation_scores.items(), key=lambda x: x[1], reverse=True)

    return sorted_recommendations[:num_recommendations]

# Main execution block
if __name__ == "__main__":
    print("--- Item-Based Collaborative Filtering Example ---")

    data_file_path = 'data/dummy_interactions.csv' # Path relative to project root
    print(f"\nLoading and preprocessing data from: {data_file_path}...")
    user_item_matrix = load_and_preprocess_data(filepath=data_file_path)

    if user_item_matrix is not None and not user_item_matrix.empty:
        print("\nUser-Item Matrix (first 5 rows/users):")
        print(user_item_matrix.head())

        print("\nCalculating Item Similarity Matrix... (this may take a while for many items)")
        item_similarity_df = calculate_item_similarity(user_item_matrix)
        print("\nItem Similarity Matrix (first 5x5 items):")
        display_limit = min(5, item_similarity_df.shape[0])
        print(item_similarity_df.iloc[:display_limit, :display_limit])

        if not user_item_matrix.empty:
            target_user_id = user_item_matrix.index[0] # Example: use the first user

            # Check if the target user has rated any items for more meaningful recommendations
            if user_item_matrix.loc[target_user_id].sum() == 0:
                print(f"\nWarning: Target user ID {target_user_id} has no rated items in the matrix. "
                      "Recommendations might be less meaningful or empty.")
                # Attempt to find a user with some ratings for a better demonstration
                for uid in user_item_matrix.index:
                    if user_item_matrix.loc[uid].sum() > 0:
                        target_user_id = uid
                        print(f"Switching to user ID {target_user_id} for a better demo as they have rated items.")
                        break

            print(f"\nGenerating recommendations for User ID: {target_user_id}...")
            recommendations = get_item_based_recommendations(
                target_user_id,
                user_item_matrix,
                item_similarity_df,
                num_recommendations=5
            )

            if recommendations:
                print(f"\nTop {len(recommendations)} recommendations for User {target_user_id}:")
                for item_id, score in recommendations:
                    print(f"- Item ID: {item_id}, Predicted Score: {score:.4f}")
            else:
                print(f"No recommendations could be generated for User {target_user_id}. "
                      "This could be because the user has rated all items, no rated items, "
                      "or no similar items were found for the unrated items.")
        else:
            print("\nUser-Item matrix is empty. Cannot generate recommendations.")
    else:
        print("\nData loading failed or resulted in an empty matrix. Cannot proceed with the example.")

    print("\n--- Item-Based Collaborative Filtering Example Finished ---")
