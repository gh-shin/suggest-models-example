# examples/deep_learning/dnn_recommender.py
import pandas as pd
import numpy as np
import os
import sys
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Concatenate, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# --- Deep Learning (DNN) with Embeddings for Recommendations: Basic Explanation ---
# Deep Neural Networks (DNNs), when combined with embedding layers, offer a flexible and powerful
# approach for recommendation systems. They excel at capturing complex, non-linear relationships
# within user-item interaction data and can learn rich, dense vector representations (embeddings)
# for both users and items.
#
# How it typically works (for a rating prediction task using a two-tower model structure):
# 1. Input Data: User IDs, Item IDs, and their corresponding ratings (or implicit feedback).
# 2. Embedding Layers:
#    - User IDs are mapped to dense vectors using a User Embedding layer. This layer learns to represent
#      each user in a lower-dimensional space where similar users (in terms of preference) are closer.
#      Input: User ID (integer), Output: User embedding vector (e.g., 32-dimensional).
#    - Item IDs are similarly mapped using an Item Embedding layer.
#      Input: Item ID (integer), Output: Item embedding vector.
#    These embeddings effectively capture latent features or characteristics from the interaction data.
# 3. Combining Embeddings: The learned user and item embedding vectors are then combined to model
#    their interaction. A common method is:
#    - Concatenation: The two vectors are joined end-to-end, creating a single, longer vector.
#      This combined vector then serves as input to subsequent dense layers.
#    - Dot Product (or Element-wise Product): This explicitly models the interaction, similar to
#      how traditional matrix factorization predicts ratings. This can also be fed into dense layers.
# 4. Deep Neural Network (DNN) Layers: The combined (or interacted) embedding vector is passed
#    through one or more dense (fully connected) layers. These layers apply non-linear transformations
#    (using activation functions like ReLU) to learn higher-level patterns from the concatenated embeddings.
#    Dropout layers are often interspersed to provide regularization and prevent overfitting.
# 5. Output Layer: A final dense layer outputs the predicted value.
#    - For rating prediction (regression): This layer typically has a single neuron and a linear activation
#      function (or sometimes sigmoid if ratings are scaled to [0,1]).
#    - For click-through rate prediction (binary classification): A single neuron with a sigmoid activation.
# 6. Training: The model is trained end-to-end by minimizing a suitable loss function
#    (e.g., Mean Squared Error (MSE) for rating prediction, Binary Cross-Entropy for classification)
#    using an optimizer like Adam or RMSprop.
#
# Pros:
# - Powerful Feature Representation: Automatically learns rich, dense embeddings for users and items,
#   capturing nuanced characteristics without manual feature engineering for IDs.
# - Captures Non-linear Relationships: DNNs can model complex interactions and patterns that simpler
#   linear models (like basic SVD or linear regression) might miss.
# - High Flexibility: The architecture is highly customizable. Easy to incorporate additional features
#   (e.g., user demographics, item metadata by adding more input branches with their own embeddings or
#   direct numerical inputs) and to adjust the depth/width of the network.
# - State-of-the-art Performance: Often achieves high accuracy on many recommendation benchmarks,
#   especially with large datasets.
#
# Cons:
# - Complexity and Training Time: Can be computationally expensive and time-consuming to train,
#   especially with very large datasets, large embedding dimensions, or deep/wide networks.
#   Requires more computational resources (like GPUs) for efficient training.
# - Data Requirements ("Data Hungry"): Typically requires a substantial amount of interaction data
#   to learn effectively and avoid overfitting. Performance might not be optimal on very sparse datasets
#   without proper regularization or architectural choices.
# - Interpretability Challenges: Like many deep learning models, the learned relationships and the
#   meaning of individual dimensions in the embedding vectors can be hard to interpret directly.
# - Cold-Start Problem: Still faces challenges with new users or new items that have no (or very few)
#   interactions in the training data, as meaningful embeddings cannot be learned for them.
#   Hybrid approaches or content-feature integration are often needed to mitigate this.
# ---

# Dynamically add project root to sys.path for module imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root) # Ensure project root is at the start for priority

# --- Data Loading and Preprocessing ---
# Time Complexity:
# - Reading CSV: O(N_interactions)
# - LabelEncoding: Roughly O(N_interactions) for fitting and transforming.
# - train_test_split: O(N_interactions).
# Overall: Dominated by operations proportional to the number of interactions.
def load_and_preprocess_data(base_filepath='data/dummy_interactions.csv', test_size=0.2, random_state=42):
    """
    Loads interaction data from a CSV, preprocesses it for the DNN model
    (encodes user/item IDs), and splits it into training and testing sets.

    Args:
        base_filepath (str): Path to the interaction data CSV, relative to project root.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Seed for random operations for reproducibility.

    Returns:
        tuple: Contains (X_train, X_test, y_train, y_test, num_users, num_items,
                 user_encoder, item_encoder, df_all_interactions) or Nones on failure.
    """
    filepath = os.path.join(project_root, base_filepath)

    if not os.path.exists(filepath):
        print(f"Error: Data file not found at {filepath}.")
        if base_filepath == 'data/dummy_interactions.csv': # Check if it's the default dummy file
            print("Attempting to generate dummy data using 'data/generate_dummy_data.py'...")
            try:
                from data.generate_dummy_data import generate_dummy_data
                generate_dummy_data()
                print("Dummy data generation script executed.")
                if not os.path.exists(filepath): # Re-check after generation
                    print(f"Error: Dummy data file still not found at {filepath} after generation.")
                    return None, None, None, None, None, None, None, None, None
                print("Dummy data file should now be available.")
            except ImportError as e_import:
                print(f"ImportError: Failed to import 'generate_dummy_data'. Error: {e_import}")
                return None, None, None, None, None, None, None, None, None
            except Exception as e_general:
                print(f"Error during dummy data generation: {e_general}")
                return None, None, None, None, None, None, None, None, None
        else:
            # The missing file is not the one we know how to generate
            return None, None, None, None, None, None, None, None, None

    df = pd.read_csv(filepath)
    if df.empty:
        print("Error: Data file is empty.")
        return None, None, None, None, None, None, None, None, None

    # Encode User and Item IDs into 0-indexed integer indices
    # This is necessary because Embedding layers in Keras expect integer inputs.
    user_encoder = LabelEncoder()
    df['user_idx'] = user_encoder.fit_transform(df['user_id'])

    item_encoder = LabelEncoder()
    df['item_idx'] = item_encoder.fit_transform(df['item_id'])

    # Number of unique users and items, crucial for defining Embedding layer input_dim
    num_users = df['user_idx'].nunique()
    num_items = item_encoder.classes_.shape[0] # More robust way to get unique encoded item count

    # Prepare features (X) and target (y)
    # X will be a 2D array where each row is [user_idx, item_idx]
    X = df[['user_idx', 'item_idx']].values
    # y is the rating, ensuring it's float for TensorFlow/Keras model training
    y = df['rating'].values.astype(np.float32)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    print(f"Data loaded: {len(df)} interactions.")
    print(f"Number of unique users: {num_users}, Number of unique items: {num_items}")
    return X_train, X_test, y_train, y_test, num_users, num_items, user_encoder, item_encoder, df


# --- DNN Model Building ---
def build_dnn_model(num_users, num_items, embedding_dim=32, dense_layers=[64, 32], dropout_rate=0.1):
    """
    Builds a Deep Neural Network (DNN) model for recommendation using Keras.
    The model uses separate embedding layers for users and items, concatenates them,
    and passes them through dense layers to predict ratings.

    Args:
        num_users (int): Total number of unique users (for user embedding layer).
        num_items (int): Total number of unique items (for item embedding layer).
        embedding_dim (int): Dimensionality of the embedding vectors.
        dense_layers (list of int): List specifying the number of units in each dense layer.
        dropout_rate (float): Dropout rate for regularization.

    Returns:
        tensorflow.keras.models.Model: The compiled Keras model.
    """
    # User Embedding Pathway
    # Input layer for user index (a single integer)
    user_input = Input(shape=(1,), name='user_input')
    # Embedding layer: Maps each user index to a dense vector of size `embedding_dim`.
    # `input_dim` is the vocabulary size (number of unique users).
    user_embedding = Embedding(input_dim=num_users, output_dim=embedding_dim, name='user_embedding')(user_input)
    # Flatten the embedding output to a 1D vector
    user_vec = Flatten(name='flatten_user_embedding')(user_embedding)
    user_vec = Dropout(dropout_rate, name='dropout_user_vec')(user_vec) # Apply dropout for regularization

    # Item Embedding Pathway
    # Input layer for item index
    item_input = Input(shape=(1,), name='item_input')
    # Embedding layer for items
    item_embedding = Embedding(input_dim=num_items, output_dim=embedding_dim, name='item_embedding')(item_input)
    # Flatten the embedding output
    item_vec = Flatten(name='flatten_item_embedding')(item_embedding)
    item_vec = Dropout(dropout_rate, name='dropout_item_vec')(item_vec) # Apply dropout

    # Combine user and item embedding vectors
    # Concatenate operation joins the two vectors side-by-side.
    concat_embeddings = Concatenate(name='concatenate_embeddings')([user_vec, item_vec])
    concat_dropout = Dropout(dropout_rate, name='dropout_concatenated')(concat_embeddings)

    # Fully Connected (Dense) Layers
    # Start with the concatenated (and possibly dropped-out) embeddings
    current_dense_layer = concat_dropout
    for i, units in enumerate(dense_layers):
        current_dense_layer = Dense(units, activation='relu', name=f'dense_layer_{i+1}')(current_dense_layer)
        current_dense_layer = Dropout(dropout_rate, name=f'dropout_dense_{i+1}')(current_dense_layer)

    # Output Layer
    # Predicts a single value (the rating).
    # 'linear' activation is used for regression tasks (predicting continuous rating values).
    # If ratings were normalized to [0,1], 'sigmoid' could also be an option.
    output_layer = Dense(1, activation='linear', name='rating_output')(current_dense_layer)

    # Create and compile the Keras Model
    model = Model(inputs=[user_input, item_input], outputs=output_layer)
    return model

# --- Recommendation Generation ---
def get_dnn_recommendations(model, user_id_original, user_encoder, item_encoder, df_all_interactions, num_total_items_encoded, num_recommendations=5):
    """
    Generates top-N recommendations for a specific user using the trained DNN model.

    Args:
        model (tensorflow.keras.models.Model): The trained Keras DNN model.
        user_id_original: The original ID of the user for whom to generate recommendations.
        user_encoder (LabelEncoder): The fitted user ID encoder.
        item_encoder (LabelEncoder): The fitted item ID encoder.
        df_all_interactions (pd.DataFrame): DataFrame containing all user-item interactions.
        num_total_items_encoded (int): Total number of unique items known to item_encoder.
        num_recommendations (int): Number of recommendations to return.

    Returns:
        list: A list of dictionaries, each containing 'item_id' (original) and 'predicted_rating'.
    """
    try:
        # Convert the original user ID to its integer index used during training
        user_idx_encoded = user_encoder.transform([user_id_original])[0]
    except ValueError:
        print(f"Error: User ID '{user_id_original}' was not found in the training data (new user). "
              "Cannot generate recommendations for this user with this model.")
        return []

    # Identify items the user has already interacted with (using original item IDs for clarity)
    # These items should be excluded from the recommendations.
    items_rated_by_user_original_ids = set()
    if 'user_id' in df_all_interactions.columns and 'item_id' in df_all_interactions.columns:
        items_rated_by_user_original_ids = set(
            df_all_interactions[df_all_interactions['user_id'] == user_id_original]['item_id'].unique()
        )
    print(f"User {user_id_original} has already interacted with {len(items_rated_by_user_original_ids)} items.")

    # Generate a list of all possible item indices (0 to num_total_items_encoded - 1)
    all_item_indices_encoded = np.arange(num_total_items_encoded)

    # Filter out items the user has already interacted with.
    # We need to predict ratings only for items the user hasn't seen.
    items_to_predict_encoded_indices = []
    for item_idx_encoded in all_item_indices_encoded:
        # Convert encoded item index back to original item ID to check if user has rated it
        original_item_id_for_check = item_encoder.inverse_transform([item_idx_encoded])[0]
        if original_item_id_for_check not in items_rated_by_user_original_ids:
            items_to_predict_encoded_indices.append(item_idx_encoded)

    items_to_predict_encoded_indices = np.array(items_to_predict_encoded_indices, dtype=np.int32)

    if len(items_to_predict_encoded_indices) == 0:
        print(f"User {user_id_original} has already rated all available items, or no new items to recommend.")
        return []

    # Prepare input arrays for the Keras model prediction
    # We need to repeat the user_idx_encoded for each item we want to predict
    user_input_array = np.full(len(items_to_predict_encoded_indices), user_idx_encoded, dtype=np.int32)
    item_input_array = items_to_predict_encoded_indices # Already ensured dtype=np.int32

    # Predict ratings for the unrated items
    # Prediction Time Complexity for N items: O(N * Complexity_of_Forward_Pass)
    print(f"Predicting ratings for {len(items_to_predict_encoded_indices)} unrated items for user {user_id_original}...")
    predicted_ratings = model.predict([user_input_array, item_input_array], verbose=0).flatten()

    # Combine item indices with their predicted ratings
    recommendation_results = list(zip(items_to_predict_encoded_indices, predicted_ratings))
    # Sort by predicted rating in descending order
    recommendation_results.sort(key=lambda x: x[1], reverse=True)

    # Format the top N recommendations
    top_n_recommendations = []
    for item_idx_encoded, score in recommendation_results[:num_recommendations]:
        original_item_id = item_encoder.inverse_transform([item_idx_encoded])[0] # Convert back to original ID
        top_n_recommendations.append({'item_id': original_item_id, 'predicted_rating': score})

    return top_n_recommendations

# --- Main Execution Block ---
# Training Time Complexity: Dominated by model.fit(), roughly:
# O(N_epochs * (N_train_samples / Batch_size) * (Sum of (Input_features * Output_features) for each layer part of backprop))
# Embedding layers contribute significantly: (num_users * emb_dim + num_items * emb_dim) parameters.
if __name__ == "__main__":
    print("--- DNN (Deep Neural Network) Based Recommendation Example (TensorFlow/Keras) ---")

    # 1. Load and Preprocess Data
    print("\nLoading and preprocessing data...")
    X_train, X_test, y_train, y_test, num_users, num_items, user_encoder, item_encoder, df_all_interactions = \
        load_and_preprocess_data(base_filepath='data/dummy_interactions.csv')

    if X_train is not None and df_all_interactions is not None:
        print(f"Training data samples: {X_train.shape[0]}, Test data samples: {X_test.shape[0]}")
        # num_users and num_items are already printed in load_and_preprocess_data

        # 2. Build DNN Model
        print("\nBuilding DNN model...")
        # Hyperparameters for the model
        embedding_dimension = 50       # Dimensionality of embedding vectors
        dnn_dense_layers = [128, 64, 32] # Units in each dense layer
        dnn_dropout_rate = 0.2         # Dropout rate for regularization
        learning_rate = 0.001          # Learning rate for the Adam optimizer

        model = build_dnn_model(
            num_users,
            num_items,
            embedding_dim=embedding_dimension,
            dense_layers=dnn_dense_layers,
            dropout_rate=dnn_dropout_rate
        )
        # Compile the model: configure optimizer, loss function, and metrics
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='mean_squared_error', # Suitable for rating prediction (regression)
            metrics=['mae']            # Mean Absolute Error, another common regression metric
        )
        model.summary() # Print model architecture

        # 3. Train the Model
        print("\nTraining the model... (This may take some time depending on epochs and data size)")
        # Training Hyperparameters
        num_epochs = 10                # Number of times to iterate over the entire training dataset
        training_batch_size = 64       # Number of samples per gradient update

        # Ensure input data to `fit` is of the correct type (integer indices for embeddings)
        # X_train[:, 0] is user_idx, X_train[:, 1] is item_idx
        history = model.fit(
            [X_train[:, 0].astype(np.int32), X_train[:, 1].astype(np.int32)],
            y_train,
            epochs=num_epochs,
            batch_size=training_batch_size,
            validation_data=([X_test[:, 0].astype(np.int32), X_test[:, 1].astype(np.int32)], y_test),
            verbose=1 # Show training progress
        )
        print("Model training completed.")

        # Evaluate the model on the test set
        loss, mae = model.evaluate(
            [X_test[:, 0].astype(np.int32), X_test[:, 1].astype(np.int32)],
            y_test,
            verbose=0
        )
        print(f"\nEvaluation on Test Dataset: Loss = {loss:.4f}, MAE = {mae:.4f}")

        # 4. Generate Recommendations for a Specific User
        if not df_all_interactions.empty and user_encoder.classes_.shape[0] > 0:
            # Select a target user (e.g., the first user from the original dataset) for demonstration
            target_user_original_id = user_encoder.classes_[0] # Get the first original user ID known to encoder

            print(f"\nGenerating recommendations for User ID (original): {target_user_original_id}...")

            # num_items is item_encoder.classes_.shape[0], which is num_total_items_encoded
            recommendations = get_dnn_recommendations(
                model,
                target_user_original_id,
                user_encoder,
                item_encoder,
                df_all_interactions, # Used to find items already rated by the user
                num_items,             # Total number of unique items (for generating candidates)
                num_recommendations=5
            )

            if recommendations:
                print(f"\nTop recommendations for User {target_user_original_id} (DNN-based):")
                for rec in recommendations:
                    print(f"- Item ID (original): {rec['item_id']}, Predicted Rating: {rec['predicted_rating']:.3f}")
            else:
                print(f"No new recommendations could be generated for User {target_user_original_id}.")
        else:
            print("\nCannot generate recommendations as interaction data is empty or no users were encoded.")
    else:
        print("\nData loading and preprocessing failed. Cannot proceed with the DNN example.")

    print("\n--- DNN Based Recommendation Example Finished ---")
