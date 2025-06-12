# examples/matrix_factorization/svd_example.py
import pandas as pd
import os
import sys
from surprise import Dataset, Reader, SVD
# train_test_split is not used in this specific example but often useful
# from surprise.model_selection import train_test_split
from collections import defaultdict

# --- Singular Value Decomposition (SVD) for Recommendations: Basic Explanation ---
# Matrix Factorization, particularly variants inspired by SVD, is a popular class of algorithms
# for recommendation systems. The core idea is to decompose the sparse user-item interaction
# matrix (e.g., ratings) into lower-dimensional matrices representing latent factors (hidden features)
# of users and items.
#
# How it works (specifically for Surprise's SVD, often a Funk SVD or SVD++ variant):
# 1. User-Item Interaction Data: Input is typically a list of (user_id, item_id, rating) triples.
# 2. Latent Factor Model:
#    - We assume there are 'k' latent factors that capture underlying properties (e.g., for movies:
#      action level, comedy content, romantic elements; for users: preference for these aspects).
#    - Each user 'u' is associated with a k-dimensional vector p_u (user factors).
#    - Each item 'i' is associated with a k-dimensional vector q_i (item factors).
# 3. Rating Prediction:
#    The predicted rating r_ui for user 'u' and item 'i' is often modeled as:
#    r_ui_predicted = μ + b_u + b_i + p_u^T * q_i
#    Where:
#      - μ (mu): Global average rating across all items.
#      - b_u: User bias (how this user tends to rate compared to the average).
#      - b_i: Item bias (how this item tends to be rated compared to the average).
#      - p_u^T * q_i: The dot product of user and item factor vectors, capturing the interaction
#        between user preferences and item characteristics in the latent space.
#    Note: The SVD algorithm in Surprise is not the "pure" mathematical SVD but rather a model
#    inspired by it, optimized for recommendation tasks (predicting known ratings).
#
# 4. Training (Learning Parameters):
#    The model parameters (μ, b_u, b_i, p_u, q_i for all users and items) are learned by minimizing
#    an objective function. This function usually consists of the sum of squared errors between
#    predicted ratings and actual known ratings, plus regularization terms to prevent overfitting.
#    Optimization is typically performed using Stochastic Gradient Descent (SGD) or Alternating Least Squares (ALS).
#    Surprise's SVD uses SGD by default.
#
# Pros:
# - Handles Sparsity Well: By learning latent factors, SVD can generalize from known ratings to predict
#   unknown ratings, often performing better than neighborhood-based CF on sparse data.
# - Compact Representation: User and item factors provide a lower-dimensional, dense representation.
# - Captures Complex Relationships: Latent factors can implicitly capture complex user preferences
#   and item attributes, even those not explicitly present in metadata.
# - Scalability: While training can be intensive, prediction is relatively fast once factors are learned.
#
# Cons:
# - Interpretability: The learned latent factors are often not directly interpretable in human terms.
# - Cold-Start Problem:
#   - New Users: If a user has no ratings, their user factor vector p_u cannot be learned.
#   - New Items: If an item has no ratings, its item factor vector q_i cannot be learned.
#   Fallbacks (e.g., content-based features, average ratings) are needed for these cases.
# - Training Complexity: Training can be computationally intensive for very large datasets.
#   Complexity is roughly O(N_interactions * N_factors * N_epochs) for SGD.
# ---

# Dynamically add project root to sys.path for module imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root) # Insert at the beginning for priority

# --- Data Loading for Surprise ---
# Time Complexity: O(N_interactions) for reading CSV and loading into Surprise Dataset.
def load_data_for_surprise(base_filepath='data/dummy_interactions.csv'):
    """
    Loads interaction data from a CSV file into a Surprise Dataset object.
    The CSV file must contain columns for user ID, item ID, and rating.
    Handles file path relative to the project root and attempts dummy data generation if needed.

    Args:
        base_filepath (str): Path to the CSV data file, relative to the project root if starts with 'data/'.

    Returns:
        surprise.Dataset: A Dataset object ready for use with Surprise algorithms, or None on failure.
    """
    filepath = base_filepath
    # Construct absolute path if base_filepath is a known relative path pattern
    if not os.path.isabs(base_filepath) and base_filepath.startswith('data/'):
        filepath = os.path.join(project_root, base_filepath)

    if not os.path.exists(filepath):
        print(f"Error: Data file not found at {filepath}.")
        # Specifically check if it's the default dummy data path to offer generation
        if filepath.endswith('data/dummy_interactions.csv'):
            print("Attempting to generate dummy data using 'data/generate_dummy_data.py'...")
            try:
                from data.generate_dummy_data import generate_dummy_data
                generate_dummy_data()
                print("Dummy data generation script executed.")
                if not os.path.exists(filepath): # Re-check after generation attempt
                    print(f"Error: Dummy data file still not found at {filepath} after generation attempt.")
                    return None
                print("Dummy data file should now be available.")
            except ImportError as e_import:
                print(f"ImportError: Failed to import 'generate_dummy_data'. Error: {e_import}")
                return None
            except Exception as e_general:
                print(f"Error during dummy data generation: {e_general}")
                return None
        else:
            # The missing file is not the one we know how to generate
            return None

    # The Reader object is used to parse the file or DataFrame.
    # We need to define the rating_scale (e.g., 1-5 stars).
    reader = Reader(rating_scale=(1, 5)) # Adjust if your rating scale is different

    try:
        df = pd.read_csv(filepath)
        # Ensure required columns exist. Surprise expects columns in order: user, item, rating.
        required_cols = ['user_id', 'item_id', 'rating']
        if not all(col in df.columns for col in required_cols):
            print(f"Error: CSV file must contain the columns: {', '.join(required_cols)}.")
            return None

        # Load the DataFrame into a Surprise Dataset.
        # Only these three columns are used by Surprise.
        data = Dataset.load_from_df(df[['user_id', 'item_id', 'rating']], reader)
        return data
    except Exception as e:
        print(f"Error loading data into Surprise Dataset: {e}")
        return None

# --- Top-N Recommendation Generation ---
def get_top_n_recommendations(predictions, n=10):
    """
    Extracts the top-N recommendations for each user from a list of Surprise predictions.

    Args:
        predictions (list of surprise.Prediction objects): A list of predictions,
            typically generated by an algorithm's `test` method or multiple `predict` calls.
        n (int): The number of recommendations to return for each user.

    Returns:
        defaultdict: A dictionary where keys are user (raw) IDs and values are lists of
                     (raw_item_id, estimated_rating) tuples, sorted by estimated_rating.
    """
    # Map predictions to each user.
    top_n = defaultdict(list)
    for pred in predictions:
        # pred attributes: uid (raw user id), iid (raw item id), r_ui (true rating, often None for prediction), est (estimated rating)
        top_n[pred.uid].append((pred.iid, pred.est))

    # Sort the predictions for each user by estimated rating (descending) and retrieve the top n.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]
    return top_n

# --- Main Execution Block ---
if __name__ == "__main__":
    print("--- SVD (Matrix Factorization) Example using Surprise library ---")

    data_file_path = 'data/dummy_interactions.csv' # Path relative to project root

    # 1. Load Data using Surprise Reader and Dataset
    print(f"\nLoading data from '{data_file_path}' for Surprise...")
    data = load_data_for_surprise(base_filepath=data_file_path)

    if data:
        # 2. Build Full Trainset
        # For this example, we'll train on the whole dataset.
        # In a real scenario, you'd split into train/test sets (e.g., using train_test_split from Surprise).
        trainset = data.build_full_trainset()
        print("Data loaded and full training set built successfully.")

        # 3. Train SVD Model
        # Time Complexity for training: O(N_interactions * N_factors * N_epochs)
        print("\nTraining SVD model...")
        # SVD Parameters:
        # - n_factors: Number of latent factors (k in the explanation). Default is 100.
        # - n_epochs: Number of iterations of the SGD optimization. Default is 20.
        # - biased: If True (default), the model includes biases (μ, b_u, b_i). This is generally recommended.
        # - random_state: For reproducibility.
        algo = SVD(n_factors=50, n_epochs=20, biased=True, random_state=42)
        algo.fit(trainset)
        print("SVD model training completed.")

        # 4. Generate Top-N Recommendations for a Specific User
        # This involves:
        #  a. Identifying all items the user hasn't rated yet.
        #  b. Predicting ratings for these unrated items using the trained model.
        #  c. Selecting the items with the highest predicted ratings.

        target_user_raw_id = None # Will be determined from the loaded data
        try:
            # Load the original CSV again just to get a list of unique user IDs.
            # This ensures we pick a user ID that actually exists in our dataset for demonstration.
            # In a real application, you'd get the target user ID from your application context.
            df_interactions_for_user_selection = pd.read_csv(os.path.join(project_root, data_file_path))
            if not df_interactions_for_user_selection.empty:
                target_user_raw_id = df_interactions_for_user_selection['user_id'].unique()[0]
                # Ensure it's a string if your IDs are strings, or int if they are ints.
                # Surprise handles raw IDs as they are provided (str, int).
            else:
                print("Warning: Interaction data file is empty. Cannot select a target user for recommendations.")
        except FileNotFoundError:
             print(f"Error: Could not find '{data_file_path}' to select a target user. Please ensure dummy data exists.")
        except IndexError:
             print("Error: No user IDs found in the dummy data. Check data generation.")

        # Convert to string if it's not, to match how Surprise might store it internally if loaded from mixed-type raw ids
        # However, Surprise generally preserves original ID types. For dummy data, they are usually integers.
        # Let's assume IDs from CSV are integers for this dummy data.

        if target_user_raw_id is not None:
            print(f"\nGenerating top-N recommendations for User ID (raw): {target_user_raw_id}...")

            # Get items the user has already rated (from the trainset)
            rated_item_raw_ids = set()
            try:
                # Convert raw user ID to inner ID used by Surprise
                target_user_inner_id = trainset.to_inner_uid(target_user_raw_id)
                # Get (inner_item_id, rating) tuples for this user
                user_ratings_inner = trainset.ur[target_user_inner_id]
                # Convert inner item IDs back to raw item IDs
                rated_item_raw_ids = {trainset.to_raw_iid(inner_iid) for (inner_iid, _rating) in user_ratings_inner}
                print(f"User {target_user_raw_id} has rated {len(rated_item_raw_ids)} items. These will be excluded from recommendations.")
            except ValueError:
                # This happens if the target_user_raw_id is not in the trainset (e.g., new user)
                print(f"Warning: User ID {target_user_raw_id} not found in the training set. "
                      "Assuming no items rated, so all items are candidates for recommendation.")

            # Get all unique item IDs present in the training set
            all_items_in_trainset_raw_ids = {trainset.to_raw_iid(inner_iid) for inner_iid in trainset.all_items()}

            # Identify items to predict: all items in trainset MINUS items already rated by the user
            items_to_predict_raw_ids = [iid for iid in all_items_in_trainset_raw_ids if iid not in rated_item_raw_ids]

            if not items_to_predict_raw_ids:
                print(f"User {target_user_raw_id} has already rated all items in the training set, or no new items to recommend.")
            else:
                # Generate predictions for these unrated items
                # Time Complexity for N predictions: O(N * N_factors)
                print(f"Predicting ratings for {len(items_to_predict_raw_ids)} unrated items for user {target_user_raw_id}...")
                user_predictions = []
                for item_raw_id in items_to_predict_raw_ids:
                    # algo.predict() takes raw user and item IDs
                    prediction = algo.predict(uid=target_user_raw_id, iid=item_raw_id)
                    user_predictions.append(prediction)

                # Extract top-N recommendations from the predictions
                # Time Complexity for sorting: O(N_predictions * log(N_predictions))
                top_recommendations = get_top_n_recommendations(user_predictions, n=5)

                if top_recommendations.get(target_user_raw_id):
                    print(f"\nTop 5 recommendations for User {target_user_raw_id} (SVD based):")
                    for item_raw_id, estimated_rating in top_recommendations[target_user_raw_id]:
                        print(f"- Item ID (raw): {item_raw_id}, Predicted Rating: {estimated_rating:.3f}")
                else:
                    print(f"No recommendations could be generated for User {target_user_raw_id}.")
        else:
            if data: # Only print if data loading itself was successful
                print("\nCould not determine a target user for generating recommendations.")
    else:
        print("\nData loading failed. Cannot proceed with the SVD example.")

    print("\n--- SVD (Singular Value Decomposition) example execution complete ---")
