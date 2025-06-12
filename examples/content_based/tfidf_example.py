# examples/content_based/tfidf_example.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel # Faster than cosine_similarity for normalized vectors
import os
import sys
import numpy as np

# --- Content-Based Filtering with TF-IDF: Basic Explanation ---
# Content-Based Filtering recommends items to users based on the similarity of item attributes
# (content) to items the user has previously liked or interacted with.
# It focuses on the properties of items rather than on other users' opinions.
#
# How it works:
# 1. Item Representation (Content Extraction): Each item needs to be described by a set of features.
#    For items with textual content (e.g., movie descriptions, article text, product details),
#    TF-IDF (Term Frequency-Inverse Document Frequency) is a common technique to convert text into numerical vectors.
#    - TF (Term Frequency): How often a term appears in a specific item's description.
#      TF(t,d) = (Number of times term t appears in document d) / (Total number of terms in document d)
#    - IDF (Inverse Document Frequency): Importance of a term across the entire collection of item descriptions.
#      Terms that are common across many items (e.g., "the", "a") get a low IDF score, while rare terms get a higher score.
#      IDF(t,D) = log(Total number of documents D / Number of documents containing term t)
#    - TF-IDF Score: The product of TF and IDF (TF * IDF). This score is high for terms that are frequent
#      in a specific item's description but relatively rare across all item descriptions, making them good discriminators.
#    Each item is thus represented as a vector where each dimension corresponds to a term's TF-IDF score.
#
# 2. User Profile Creation (Optional but common for complex systems):
#    A user's profile can be built based on the content of items they have positively interacted with.
#    This could be an aggregated vector (e.g., average or weighted sum) of the TF-IDF vectors of items they liked.
#    In simpler cases (like this example), recommendations are made by comparing new items directly
#    to the content of specific items the user has liked.
#
# 3. Similarity Calculation: The similarity between item vectors is calculated.
#    Cosine similarity is a common choice for TF-IDF vectors because it measures the cosine of the angle
#    between two vectors, effectively capturing orientation rather than magnitude.
#    `linear_kernel` from scikit-learn can compute this efficiently, especially if TF-IDF vectors are L2-normalized
#    (which TfidfVectorizer can do by default or can be done explicitly).
#
# 4. Recommendation Generation:
#    To recommend items for a user:
#    a. Identify items the user has liked (e.g., items with high ratings).
#    b. For each such liked item (or for the user's aggregated profile), find other items in the dataset
#       that are most similar based on their TF-IDF vector similarity.
#    c. Rank these similar items by their similarity score.
#    d. Filter out items the user has already interacted with and present the top N items.
#
# Pros:
# - User Independence: Recommendations for one user do not depend on other users' data, only on their own interactions.
#   This means it doesn't suffer from the "new user cold-start" problem as much as collaborative filtering,
#   as long as the new user interacts with at least one item.
# - Transparency & Explainability: Recommendations can be easily explained by listing the content features
#   that caused the similarity (e.g., "Recommended because it shares genres/keywords like 'sci-fi thriller'
#   with items you've previously enjoyed").
# - Handles New Items (Item Cold-Start): New items can be recommended as soon as their content features are available
#   and vectorized, without needing any user interaction data for that new item.
#
# Cons:
# - Limited Serendipity (Filter Bubble): Tends to recommend items very similar in content to what the user
#   already knows, making it harder to discover items outside their current taste profile.
#   Users might get stuck in an "overspecialization" or "filter bubble."
# - Extensive Feature Engineering: Effectiveness heavily depends on the quality, completeness, and representation
#   of the item features (content descriptions, tags, genres, etc.). Requires domain knowledge and careful preprocessing.
# - User Cold-Start (to some extent): While it can recommend to new users after one interaction, the quality
#   of recommendations improves as more interactions define the user's content preferences.
# ---

# Dynamically add project root to sys.path for module imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root) # Insert at the beginning for priority

# --- Data Loading ---
# Time Complexity: O(N_interactions + N_items) for reading CSVs, where N_interactions is rows in interaction data
# and N_items is rows in metadata.
def load_data(interactions_filepath_rel='data/dummy_interactions.csv', metadata_filepath_rel='data/dummy_item_metadata.csv'):
    """
    Loads interaction and item metadata from specified relative paths from project root.
    Ensures dummy data is generated if the default files are missing.

    Args:
        interactions_filepath_rel (str): Relative path to interactions CSV.
        metadata_filepath_rel (str): Relative path to item metadata CSV.

    Returns:
        tuple: (pandas.DataFrame, pandas.DataFrame) for interactions and items, or (None, None) on failure.
    """
    interactions_filepath_abs = os.path.join(project_root, interactions_filepath_rel)
    metadata_filepath_abs = os.path.join(project_root, metadata_filepath_rel)

    needs_generation = False
    # Check existence of both files before attempting generation
    if not os.path.exists(interactions_filepath_abs) and interactions_filepath_rel == 'data/dummy_interactions.csv':
        print(f"Warning: Interaction data file not found at {interactions_filepath_abs}.")
        needs_generation = True
    if not os.path.exists(metadata_filepath_abs) and metadata_filepath_rel == 'data/dummy_item_metadata.csv':
        print(f"Warning: Item metadata file not found at {metadata_filepath_abs}.")
        needs_generation = True

    if needs_generation:
        print("Attempting to generate dummy data by running 'data/generate_dummy_data.py'...")
        try:
            from data.generate_dummy_data import generate_dummy_data
            generate_dummy_data() # This should create both dummy files
            print("Dummy data generation script executed.")
            # Verify that files exist after generation attempt
            if not os.path.exists(interactions_filepath_abs):
                print(f"Error: Dummy interaction data still not found at {interactions_filepath_abs} after generation.")
                return None, None
            if not os.path.exists(metadata_filepath_abs):
                print(f"Error: Dummy item metadata still not found at {metadata_filepath_abs} after generation.")
                return None, None
            print("Dummy data files should now be available.")
        except ImportError as e_import:
            print(f"ImportError: Failed to import 'generate_dummy_data'. Error: {e_import}")
            return None, None
        except Exception as e_general:
            print(f"Error during dummy data generation: {e_general}")
            return None, None

    try:
        df_interactions = pd.read_csv(interactions_filepath_abs)
        df_items = pd.read_csv(metadata_filepath_abs)
        # Ensure item_id is consistently typed for potential merges or lookups.
        # In dummy data, item_id is int. Convert to string if mixing with string IDs, or ensure consistency.
        if not df_items.empty and 'item_id' in df_items.columns:
             df_items['item_id'] = df_items['item_id'].astype(str) # Standardize to string for this example
        if not df_interactions.empty and 'item_id' in df_interactions.columns:
            df_interactions['item_id'] = df_interactions['item_id'].astype(str)

        return df_interactions, df_items
    except FileNotFoundError as e:
        print(f"Error loading data files: {e}. Please ensure paths are correct or dummy data exists.")
        return None, None
    except Exception as e:
        print(f"An unexpected error occurred during data loading: {e}")
        return None, None

# --- Recommendation Logic ---
# Time Complexity:
# - TF-IDF Vectorization: O(N_items * avg_doc_length + V_size * avg_doc_length) where N_items is number of items,
#   avg_doc_length is average number of terms per item description, V_size is vocabulary size. Can be significant.
# - linear_kernel (Similarity Matrix): O(N_items^2 * N_features) where N_features is the TF-IDF vocabulary size.
#   This is the most computationally intensive part if N_items is large.
# - Per-user recommendation (after matrix computation):
#   - Identifying user's rated items: O(N_user_interactions).
#   - Getting similarity scores for one item: O(N_items).
#   - Sorting these scores: O(N_items * log(N_items)).
#   - Filtering and selecting top N: O(N_items).
# Overall, pre-calculating the full similarity matrix is costly.
def get_content_based_recommendations(user_id, df_interactions, df_items, num_recommendations=5):
    """
    Generates content-based recommendations for a user using TF-IDF on item descriptions.
    This example bases recommendations on the last item the user interacted with.
    """
    if df_items is None or df_interactions is None or df_items.empty or df_interactions.empty:
        print("Error: DataFrames are invalid or empty. Cannot generate recommendations.")
        return []

    if 'description' not in df_items.columns or 'item_id' not in df_items.columns:
        print("Error: Item metadata must contain 'item_id' and 'description' columns.")
        return []

    # Standardize user_id to string for consistent lookups
    user_id = str(user_id)
    df_interactions['user_id'] = df_interactions['user_id'].astype(str)


    # Ensure item descriptions are strings and handle missing values
    df_items['description'] = df_items['description'].fillna('').astype(str)

    # 1. TF-IDF Vectorization
    # TfidfVectorizer converts a collection of raw documents to a matrix of TF-IDF features.
    # - stop_words='english': Removes common English stop words.
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    # fit_transform learns the vocabulary and idf from 'description', then transforms it into a document-term matrix.
    tfidf_matrix = tfidf_vectorizer.fit_transform(df_items['description'])
    # tfidf_matrix is a sparse matrix of shape (num_items, num_features/vocabulary_size)

    # 2. Compute Cosine Similarity Matrix
    # linear_kernel computes the dot product, which is equivalent to cosine similarity for L2-normalized vectors.
    # TfidfVectorizer typically produces L2-normalized vectors (norm='l2' by default).
    # This creates an (num_items x num_items) matrix where each entry (i, j) is the similarity between item i and item j.
    cosine_sim_matrix = linear_kernel(tfidf_matrix, tfidf_matrix)

    # Create a mapping from item_id (string) to DataFrame index for quick lookups in cosine_sim_matrix
    item_id_to_idx = pd.Series(df_items.index, index=df_items['item_id'])

    try:
        # Get items rated/interacted by the target user
        user_interacted_items_series = df_interactions[df_interactions['user_id'] == user_id]['item_id']

        if user_interacted_items_series.empty:
            print(f"User ID {user_id} has no interaction data. Cannot generate content-based recommendations on this basis.")
            return []

        # For simplicity, this example uses the *last* item the user interacted with as the basis for recommendations.
        # A more robust approach might use an aggregate of all liked items or most highly rated items.
        last_interacted_item_id = user_interacted_items_series.iloc[-1] # This is a string due to earlier astype(str)

        if last_interacted_item_id not in item_id_to_idx:
            print(f"Error: The item ID {last_interacted_item_id} (last interacted by user {user_id}) "
                  "is not found in the item metadata's index map. It might be missing from df_items.")
            return []

        # Get the index of this last interacted item in the similarity matrix
        last_interacted_item_idx = item_id_to_idx[last_interacted_item_id]

        # Get pairwise similarity scores of all items with this last interacted item
        # cosine_sim_matrix[last_interacted_item_idx] gives a row of similarities
        item_similarity_scores = list(enumerate(cosine_sim_matrix[last_interacted_item_idx]))

        # Sort the items based on their similarity scores in descending order
        # x[0] is the index of the item, x[1] is its similarity score
        sorted_similar_items = sorted(item_similarity_scores, key=lambda x: x[1], reverse=True)

        recommendations = []
        # Keep track of items already recommended or interacted with by the user to avoid duplicates
        # Convert to set for efficient lookups
        seen_item_ids = set(user_interacted_items_series.values)
        # The item itself (last_interacted_item_id) will have a similarity of 1.0 and should be excluded if not already by being in seen_item_ids.
        # However, the loop structure naturally handles this by checking `if recommended_item_id not in seen_item_ids`.

        print(f"\nGenerating recommendations based on similarity to item ID: {last_interacted_item_id} (Description: '{df_items.loc[item_id_to_idx[last_interacted_item_id], 'description'][:100]}...')")

        for item_matrix_idx, score in sorted_similar_items:
            if len(recommendations) >= num_recommendations:
                break # Stop when enough recommendations are found

            # Map DataFrame index (from enumerate) back to actual item_id
            recommended_item_id = df_items.iloc[item_matrix_idx]['item_id'] # This is a string

            # Add to recommendations if not already seen/interacted with by the user
            if recommended_item_id not in seen_item_ids:
                recommendations.append({'item_id': recommended_item_id, 'similarity_score': score})
                seen_item_ids.add(recommended_item_id) # Add to seen set to prevent re-recommending

        return recommendations

    except KeyError as e:
        print(f"KeyError during recommendation generation for user {user_id}. Possibly an ID mismatch. Error: {e}")
        return []
    except Exception as e:
        print(f"An unexpected error occurred during recommendation generation: {e}")
        return []

# Main execution block
if __name__ == "__main__":
    print("--- Content-Based Filtering (TF-IDF) Example ---")

    print("\nLoading data...")
    df_interactions, df_items = load_data(
        interactions_filepath_rel='data/dummy_interactions.csv',
        metadata_filepath_rel='data/dummy_item_metadata.csv'
    )

    if df_interactions is not None and df_items is not None:
        if not df_interactions.empty and not df_items.empty:
            if df_interactions['user_id'].nunique() > 0:
                # Example: use the first unique user_id from interactions data
                target_user_id = df_interactions['user_id'].unique()[0]
                # Ensure target_user_id is string, as used internally by get_content_based_recommendations
                target_user_id = str(target_user_id)

                print(f"\nAttempting to generate content-based recommendations for User ID: {target_user_id}...")
                recommendations = get_content_based_recommendations(
                    target_user_id,
                    df_interactions,
                    df_items,
                    num_recommendations=5
                )

                if recommendations:
                    print(f"\nTop {len(recommendations)} recommendations for User {target_user_id} (based on TF-IDF item similarity):")
                    recs_df = pd.DataFrame(recommendations)
                    # Merge with df_items to get more details (description, genres) for printing
                    # Ensure item_id in recs_df and df_items are of the same type (string for this example)
                    recs_df['item_id'] = recs_df['item_id'].astype(str)
                    df_items_for_merge = df_items[['item_id', 'description', 'genres']].copy()
                    df_items_for_merge['item_id'] = df_items_for_merge['item_id'].astype(str)

                    recs_df = pd.merge(recs_df, df_items_for_merge, on='item_id', how='left')

                    for index, row in recs_df.iterrows():
                        print(f"- Item ID: {row['item_id']} (Similarity: {row['similarity_score']:.4f}) "
                              f"| Genres: {row.get('genres', 'N/A')} "
                              f"| Description snippet: {row.get('description', 'N/A')[:70]}...")
                else:
                    print(f"No recommendations could be generated for User {target_user_id}.")
            else:
                print("Error: No unique users found in the interaction data.")
        else:
            print("Error: One or both data files (interactions, items) are empty.")
    else:
        print("\nData loading failed. Cannot proceed with the example.")

    print("\n--- Content-Based Filtering (TF-IDF) Example Finished ---")
