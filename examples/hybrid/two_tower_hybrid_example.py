# examples/hybrid/two_tower_hybrid_example.py
import pandas as pd
import numpy as np
import os
import sys
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Concatenate, GlobalAveragePooling1D, Dot
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# --- Two-Tower Hybrid Recommendation Model: Basic Explanation ---
# The Two-Tower model is a highly effective and scalable architecture for building recommendation systems.
# It's particularly popular for candidate generation in large-scale industrial recommenders (e.g., YouTube, Spotify).
# The core idea is to learn separate neural network representations (embeddings) for users and items
# in two independent "towers." These embeddings are then combined, usually via a dot product, to predict
# user-item interaction likelihood.
#
# How it works:
# 1. User Tower:
#    - Input: Takes various user features as input. This can include:
#        - User ID (categorical feature, learned via an embedding layer).
#        - User demographic data (e.g., age, location; categorical or numerical).
#        - User activity history (e.g., embeddings of previously interacted items, pooled or processed via RNN/Transformer).
#    - Architecture: Typically, categorical features are passed through embedding layers. Numerical features might be
#      normalized. These processed features are then concatenated and fed through one or more dense (fully connected)
#      layers to produce a final, fixed-size user embedding vector.
#    - In this simplified example: User ID -> User Embedding Layer -> User Vector.
#
# 2. Item Tower:
#    - Input: Takes various item features as input. This can include:
#        - Item ID (categorical feature, learned via an embedding layer).
#        - Item metadata (e.g., genres, categories, brand; categorical features often embedded).
#        - Item content (e.g., textual descriptions processed by TF-IDF, word embeddings + CNN/RNN; image features from a CNN).
#    - Architecture: Similar to the user tower, item features are processed (embeddings for categorical,
#      specialized networks for content) and then typically concatenated and passed through dense layers
#      to produce a final, fixed-size item embedding vector.
#    - In this example: Item ID -> Item Embedding; Item Genres (multi-hot) -> Genre Embeddings (pooled) -> Concatenated with Item ID embedding -> Dense Layers -> Item Vector.
#
# 3. Interaction & Prediction (Similarity Learning):
#    - The user embedding (U_vector) from the user tower and the item embedding (I_vector) from the item tower are generated.
#    - The predicted affinity or interaction score for a (user, item) pair is commonly calculated using the
#      dot product: score = dot(U_vector, I_vector).
#    - A high dot product suggests high similarity or preference. Cosine similarity (normalized dot product) is also common.
#    - For training with binary labels (interacted/not interacted), this score is often passed through a sigmoid
#      function to get a probability: P(interaction | user, item) = sigmoid(score).
#
# 4. Training (Learning Embeddings):
#    - The model is trained on observed user-item interactions (positive pairs).
#    - Since implicit feedback data often lacks explicit negative signals, negative samples (items the user
#      did not interact with) are crucial. These are often generated through random sampling or more sophisticated
#      methods (e.g., sampling items popular among other users but not interacted with by the current user).
#    - Loss Function:
#        - For explicit feedback (ratings): Mean Squared Error (if dot product directly predicts rating).
#        - For implicit feedback (clicks, views): Binary Cross-Entropy is commonly used with the sigmoid output.
#          Ranking losses (like Hinge loss, BPR loss, Triplet loss) are also effective for learning relative preferences.
#
# 5. Serving / Recommendation Generation:
#    - Efficiency at scale is a key advantage.
#    - User Tower: For a given user, compute their user embedding vector U_vector.
#    - Item Tower: Precompute item embedding vectors I_vector for all items in the corpus. This can be done offline.
#    - Candidate Retrieval: Use an Approximate Nearest Neighbor (ANN) search system (e.g., FAISS, ScaNN) to efficiently
#      find the top-K item embeddings from the precomputed corpus that have the highest dot product (or cosine similarity)
#      with the user's embedding U_vector. These items form the candidates for recommendation.
#    - This "candidate generation" step is often followed by a more complex "ranking" model that re-ranks the candidates.
#
# Pros:
# - Scalability for Serving: The separation of towers allows item embeddings to be precomputed and indexed,
#   making real-time recommendation retrieval very fast even with millions of items.
# - Flexibility in Feature Engineering: Easy to incorporate a wide variety of user and item features (categorical,
#   numerical, textual, visual) into their respective towers. Each tower can have its own specialized architecture.
# - Effective for Candidate Generation: Highly efficient at narrowing down a massive item corpus to a smaller,
#   relevant set of candidates for further ranking or direct display.
# - Good Generalization: Learns meaningful dense representations that can capture complex relationships.
#
# Cons:
# - Limited Interaction Modeling: The final user-item interaction is often simplified to a dot product (or similar shallow interaction).
#   This might not capture all nuances of user-item affinity as effectively as models that allow for deeper, more
#   explicit cross-feature interactions earlier in the network (e.g., DCN, xDeepFM).
# - Feature Engineering within Towers: While flexible, the performance still relies heavily on the quality and
#   informativeness of the features engineered for each tower.
# - Cold-Start: Like many embedding-based models, it can struggle with new users or items for whom interaction
#   data is sparse or non-existent, as meaningful embeddings cannot be readily learned without sufficient data.
#   Content features in the towers can help mitigate this for new items to some extent.
# ---

# Dynamically add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- Model Hyperparameters ---
USER_EMBEDDING_DIM = 32       # Dimensionality of the user embedding vector.
ITEM_EMBEDDING_DIM = 32       # Dimensionality of the item ID embedding vector.
GENRE_EMBEDDING_DIM = 16      # Dimensionality of the genre embedding vector.
MAX_GENRES_PER_ITEM = 5       # Maximum number of genres to consider per item (for padding genre sequences).
DENSE_UNITS = [64, USER_EMBEDDING_DIM] # Dense layers in item tower. Last layer must match user embedding dim for dot product.
LEARNING_RATE = 0.001         # Learning rate for the Adam optimizer.
EPOCHS = 5                    # Number of training epochs (small for example).
BATCH_SIZE = 64               # Batch size for training.
NEGATIVE_SAMPLES = 4          # Number of negative samples to generate for each positive interaction during training.

# --- Data Loading and Preprocessing ---
# Time Complexity: Dominated by reading CSVs (O(N_interactions + N_items_metadata)) and negative sampling.
# Negative sampling can be O(N_positive_samples * NEGATIVE_SAMPLES * Avg_time_to_find_negative).
def load_and_preprocess_data(
    base_interactions_filepath='data/dummy_interactions.csv',
    base_metadata_filepath='data/dummy_item_metadata.csv'
):
    """
    Loads interaction and item metadata, preprocesses them for the Two-Tower model.
    - Handles dummy data generation if files are missing.
    - Encodes user and item IDs.
    - Tokenizes and pads genre information for items.
    - Generates negative samples for training.
    - Splits data into training and testing sets.
    """
    interactions_filepath = os.path.join(project_root, base_interactions_filepath)
    metadata_filepath = os.path.join(project_root, base_metadata_filepath)

    # Ensure data files exist, attempt to generate dummy data if they don't.
    files_missing = False
    for fp_abs, fp_rel_for_check in [(interactions_filepath, 'data/dummy_interactions.csv'),
                                     (metadata_filepath, 'data/dummy_item_metadata.csv')]:
        if not os.path.exists(fp_abs):
            print(f"Warning: Data file not found at {fp_abs} (expected relative: {fp_rel_for_check}).")
            if fp_abs.endswith(fp_rel_for_check): # Only try to generate if it's the specific dummy file
                 files_missing = True
            else: # A specific non-dummy file is missing, cannot proceed
                print(f"Error: Required file {fp_abs} is missing and is not a default dummy file.")
                return None


    if files_missing:
        print("Attempting to generate dummy data (generate_sequences=False)...")
        try:
            from data.generate_dummy_data import generate_dummy_data
            # Two-tower model doesn't require pre-generated sequences like SASRec by default.
            generate_dummy_data(num_users=200, num_items=100, num_interactions=2000, generate_sequences=False)
            print("Dummy data generation script executed.")
            # Verify files again
            if not os.path.exists(interactions_filepath) or not os.path.exists(metadata_filepath):
                print("Error: One or both dummy data files still not found after generation attempt.")
                return None
            print("Dummy data files should now be available.")
        except ImportError as e_import:
            print(f"ImportError: Failed to import 'generate_dummy_data'. Error: {e_import}")
            return None
        except Exception as e:
            print(f"Error during dummy data generation: {e}")
            return None

    df_interactions = pd.read_csv(interactions_filepath)
    df_items_meta = pd.read_csv(metadata_filepath)

    if df_interactions.empty or df_items_meta.empty:
        print("Error: Interaction or item metadata is empty after loading.")
        return None

    # Standardize ID types (important for consistent encoding and merging)
    df_interactions['user_id'] = df_interactions['user_id'].astype(str)
    df_interactions['item_id'] = df_interactions['item_id'].astype(str)
    df_items_meta['item_id'] = df_items_meta['item_id'].astype(str)

    # Encode User IDs using LabelEncoder (maps original IDs to 0-indexed integers)
    user_encoder = LabelEncoder()
    df_interactions['user_idx'] = user_encoder.fit_transform(df_interactions['user_id'])
    num_users = len(user_encoder.classes_)

    # Encode Item IDs consistently across both DataFrames
    # Collect all unique item IDs from interactions and metadata before fitting LabelEncoder
    all_item_ids_from_interactions = df_interactions['item_id'].unique()
    all_item_ids_from_metadata = df_items_meta['item_id'].unique()
    combined_item_ids = pd.Series(list(set(all_item_ids_from_interactions) | set(all_item_ids_from_metadata))).astype(str)

    item_encoder = LabelEncoder()
    item_encoder.fit(combined_item_ids)

    df_interactions['item_idx'] = item_encoder.transform(df_interactions['item_id'])
    df_items_meta['item_idx'] = item_encoder.transform(df_items_meta['item_id'])
    num_items = len(item_encoder.classes_)

    # Preprocess Genres:
    # 1. Split genre strings into lists. Handle missing genres by filling with an empty string.
    df_items_meta['genres_list'] = df_items_meta['genres'].fillna('').astype(str).apply(lambda x: x.split(';') if x else [])

    # 2. Use Keras Tokenizer to convert genre names into integer sequences.
    #    `oov_token` handles genres seen in test/new data but not in training.
    genre_tokenizer = Tokenizer(oov_token="<unk>")
    genre_tokenizer.fit_on_texts(df_items_meta['genres_list'])

    # 3. Convert lists of genre strings to lists of integer token sequences.
    item_genre_sequences = genre_tokenizer.texts_to_sequences(df_items_meta['genres_list'])

    # 4. Pad genre sequences to a fixed length (MAX_GENRES_PER_ITEM).
    #    `padding='post'`: Add padding tokens at the end.
    #    `truncating='post'`: Truncate from the end if too long.
    #    `value=0`: Use 0 as the padding token. Tokenizer reserves 0 for padding if not specified,
    #               but explicit is good. `word_index` is 1-based.
    item_genre_padded = pad_sequences(item_genre_sequences, maxlen=MAX_GENRES_PER_ITEM, padding='post', truncating='post', value=0)

    # Vocabulary size for genre embedding layer (+1 for the padding token 0, if Tokenizer doesn't account for it implicitly,
    # which it does if value=0 is used in pad_sequences and 0 is not in word_index).
    # `len(word_index)` is number of unique words. `+1` makes space for index 0 if it's reserved for padding/OOV.
    # Keras Tokenizer's word_index is 1-based. So, max_index = len(word_index). We need input_dim = max_index + 1.
    num_genres_vocab = len(genre_tokenizer.word_index) + 1 # +1 because word_index is 1-based, embedding needs 0 for padding.

    # Create a mapping from encoded item_idx to its padded genre sequence for easy lookup.
    item_idx_to_genres_padded_seq = {row['item_idx']: item_genre_padded[i] for i, row in df_items_meta.iterrows()}

    # --- Create Training Data with Negative Sampling ---
    # Positive samples: actual user-item interactions. Label = 1.0
    positive_samples_df = df_interactions[['user_idx', 'item_idx']].copy()
    positive_samples_df['label'] = 1.0 # Using float for labels, compatible with Keras loss functions.

    # Negative samples: user-item pairs that were NOT interacted with. Label = 0.0
    # This is a crucial step for training models with implicit feedback.
    negative_samples_list = []
    all_possible_item_indices_encoded = np.arange(num_items) # All encoded item indices: [0, 1, ..., num_items-1]

    # Create a set of existing interactions for fast lookup during negative sampling
    user_item_interaction_set = set(zip(positive_samples_df['user_idx'], positive_samples_df['item_idx']))

    print(f"Generating {NEGATIVE_SAMPLES} negative samples for each of {len(positive_samples_df)} positive interactions...")
    for _, row in positive_samples_df.iterrows():
        user_idx = row['user_idx']
        num_neg_generated = 0
        attempts = 0 # To prevent infinite loops if a user has rated almost all items
        while num_neg_generated < NEGATIVE_SAMPLES and attempts < num_items * 2: # Safety break
            negative_item_idx = np.random.choice(all_possible_item_indices_encoded)
            if (user_idx, negative_item_idx) not in user_item_interaction_set:
                negative_samples_list.append({'user_idx': user_idx, 'item_idx': negative_item_idx, 'label': 0.0})
                num_neg_generated += 1
            attempts +=1
        if num_neg_generated < NEGATIVE_SAMPLES:
            print(f"Warning: Could only generate {num_neg_generated}/{NEGATIVE_SAMPLES} negative samples for user_idx {user_idx}.")


    df_negative_samples = pd.DataFrame(negative_samples_list)

    # Combine positive and negative samples and shuffle
    training_data_df = pd.concat([positive_samples_df, df_negative_samples]).sample(frac=1, random_state=42).reset_index(drop=True)

    # Split into training and testing sets
    train_df, test_df = train_test_split(training_data_df, test_size=0.2, random_state=42, stratify=training_data_df['label'])

    print(f"Data preprocessing complete. Training samples: {len(train_df)}, Test samples: {len(test_df)}")
    print(f"Unique users: {num_users}, Unique items: {num_items}, Unique genre tokens (incl. padding/OOV): {num_genres_vocab}")

    return (train_df, test_df,
            user_encoder, item_encoder, genre_tokenizer,
            num_users, num_items, num_genres_vocab, # Use num_genres_vocab
            item_idx_to_genres_padded_seq, df_items_meta)

# --- Model Definition ---
def build_two_tower_model(num_users, num_items, num_genres_vocab):
    """
    Builds the Two-Tower Keras model.
    This involves creating a user tower, an item tower, and then combining them
    for training to predict user-item interaction.

    Args:
        num_users (int): Total number of unique users (for user embedding layer).
        num_items (int): Total number of unique items (for item ID embedding layer).
        num_genres_vocab (int): Vocabulary size for genres (for genre embedding layer).

    Returns:
        tuple: (training_model, user_model, item_model)
               - training_model: The model used for training, takes user & item features, outputs interaction score.
               - user_model: The user tower model, takes user ID, outputs user embedding.
               - item_model: The item tower model, takes item ID & genres, outputs item embedding.
    """
    # --- User Tower ---
    # Input: User's encoded integer index.
    user_input_idx = Input(shape=(1,), name='user_input_idx_tower', dtype='int32')
    # Embedding layer: Maps user index to a dense vector of size USER_EMBEDDING_DIM.
    user_embedding_layer = Embedding(input_dim=num_users, output_dim=USER_EMBEDDING_DIM, name='user_embedding')
    user_embedding_vector = user_embedding_layer(user_input_idx)
    # Flatten the embedding output to a 1D vector (batch_size, USER_EMBEDDING_DIM).
    user_vector = Flatten(name='flatten_user_embedding')(user_embedding_vector)
    # Optional: Add more Dense layers here to further transform the user embedding if needed.
    # For simplicity, this example uses the direct embedding as the user's final representation.
    user_model = Model(inputs=user_input_idx, outputs=user_vector, name='user_tower')

    # --- Item Tower ---
    # Input 1: Item's encoded integer index.
    item_id_input_idx = Input(shape=(1,), name='item_id_input_idx_tower', dtype='int32')
    # Input 2: Item's padded sequence of encoded genre tokens.
    item_genre_input_seq = Input(shape=(MAX_GENRES_PER_ITEM,), name='item_genre_input_seq_tower', dtype='int32')

    # Item ID Embedding
    item_id_embedding_layer = Embedding(input_dim=num_items, output_dim=ITEM_EMBEDDING_DIM, name='item_id_embedding')
    item_id_vector = Flatten(name='flatten_item_id_embedding')(item_id_embedding_layer(item_id_input_idx))

    # Genre Embeddings
    # `mask_zero=False` is used because 0 is a valid token from Tokenizer if not explicitly handled otherwise.
    # However, if 0 is strictly for padding from pad_sequences, mask_zero=True could be used if desired,
    # though GlobalAveragePooling1D handles variable length by averaging over non-masked steps.
    genre_embedding_layer = Embedding(input_dim=num_genres_vocab, output_dim=GENRE_EMBEDDING_DIM, name='genre_embedding', mask_zero=False)
    genre_embedding_vectors = genre_embedding_layer(item_genre_input_seq) # Shape: (batch, MAX_GENRES_PER_ITEM, GENRE_EMBEDDING_DIM)
    # Pool the genre embeddings into a single fixed-size vector for the item.
    # GlobalAveragePooling1D averages the embeddings across the MAX_GENRES_PER_ITEM dimension.
    genre_pooled_vector = GlobalAveragePooling1D(name='pool_genre_embeddings')(genre_embedding_vectors)

    # Concatenate the item ID embedding and the pooled genre embedding.
    concatenated_item_features = Concatenate(name='concat_item_features')([item_id_vector, genre_pooled_vector])

    # Pass the concatenated item features through Dense layers to get the final item embedding.
    current_item_dense_layer = concatenated_item_features
    for i, units in enumerate(DENSE_UNITS):
        # The last dense layer should output a vector of USER_EMBEDDING_DIM (or whatever dimension is chosen for dot product).
        # Using 'linear' activation for the final layer of the tower before dot product is common.
        activation = 'relu' if i < len(DENSE_UNITS) - 1 else 'linear'
        current_item_dense_layer = Dense(units, activation=activation, name=f'item_dense_layer_{i+1}')(current_item_dense_layer)
    item_vector = current_item_dense_layer # This is the final item embedding from the item tower.

    # Item Tower Model (can be used for precomputing item embeddings)
    item_model = Model(inputs=[item_id_input_idx, item_genre_input_seq], outputs=item_vector, name='item_tower')

    # --- Combine Towers for Training Model ---
    # Define inputs for the combined training model. These will be fed during `model.fit()`.
    # The names here ('user_idx_input', 'item_idx_input', 'genre_seq_input') must match the keys
    # in the dictionary passed to `model.fit(x=...)`.
    user_idx_training_input = Input(shape=(1,), name='user_idx_input', dtype='int32')
    item_idx_training_input = Input(shape=(1,), name='item_idx_input', dtype='int32')
    genre_seq_training_input = Input(shape=(MAX_GENRES_PER_ITEM,), name='genre_seq_input', dtype='int32')

    # Get embeddings from the respective towers
    user_embedding_for_training = user_model(user_idx_training_input)
    item_embedding_for_training = item_model([item_idx_training_input, genre_seq_training_input])

    # Calculate the dot product between user and item embeddings. This is the similarity score.
    # `axes=1` means dot product along the embedding dimension.
    # `normalize=False` for standard dot product. `normalize=True` would compute cosine similarity.
    interaction_score = Dot(axes=1, normalize=False, name='dot_product_interaction')([user_embedding_for_training, item_embedding_for_training])

    # Output layer: A single neuron with a sigmoid activation to predict the probability of interaction (0 or 1).
    output_probability = Dense(1, activation='sigmoid', name='interaction_probability_output')(interaction_score)

    # The complete model used for training
    training_model = Model(
        inputs={ # Dictionary of inputs, keys match the Input layer names
            'user_idx_input': user_idx_training_input,
            'item_idx_input': item_idx_training_input,
            'genre_seq_input': genre_seq_training_input
        },
        outputs=output_probability,
        name='two_tower_training_model'
    )
    return training_model, user_model, item_model

# --- Recommendation Generation ---
# Time Complexity for serving recommendations for one user:
# - User embedding: O(Complexity_User_Tower_Forward_Pass) (fast)
# - Item embeddings: O(N_items * Complexity_Item_Tower_Forward_Pass) (can be precomputed and stored)
# - Similarity (dot product) + Top-K: O(N_items * Embedding_Dim) if brute-force, or O(log N_items) with ANN.
def get_two_tower_recommendations(user_id_original, user_encoder, item_encoder,
                                  user_model, item_model,
                                  item_idx_to_genres_padded_seq, all_possible_item_indices_encoded,
                                  num_recommendations=5):
    """
    Generates top-N recommendations for a given user using the trained Two-Tower model.
    This involves getting the user's embedding, then comparing it against all item embeddings.

    Args:
        user_id_original: The original ID of the user.
        user_encoder (LabelEncoder): Fitted user ID encoder.
        item_encoder (LabelEncoder): Fitted item ID encoder.
        user_model (Model): Trained user tower Keras model.
        item_model (Model): Trained item tower Keras model.
        item_idx_to_genres_padded_seq (dict): Mapping from encoded item_idx to its padded genre sequence.
        all_possible_item_indices_encoded (np.array): Array of all unique encoded item indices.
        num_recommendations (int): Number of recommendations to return.

    Returns:
        list: List of recommended items, each a dict {'item_id': original_id, 'similarity_score': score}.
    """
    try:
        # Convert original user ID to its encoded integer index
        user_idx_encoded = user_encoder.transform(np.array([user_id_original]))[0] # Ensure input is array-like
    except ValueError:
        print(f"Error: User ID '{user_id_original}' not found in user_encoder. Cannot generate recommendations.")
        return []

    # 1. Get the embedding for the target user using the user_model.
    # Input needs to be a NumPy array.
    user_embedding = user_model.predict(np.array([user_idx_encoded]), verbose=0)

    # 2. Prepare inputs for the item_model to get embeddings for ALL items.
    #    - all_item_ids_input: Array of all encoded item IDs.
    #    - all_item_genres_input: Corresponding array of padded genre sequences for these items.
    all_item_ids_input_for_model = all_possible_item_indices_encoded.reshape(-1, 1) # Reshape for Keras model input

    # Retrieve genre sequences for all items. Handle missing ones with padding.
    default_genre_padding = [0] * MAX_GENRES_PER_ITEM
    all_item_genres_input_for_model = np.array(
        [item_idx_to_genres_padded_seq.get(idx, default_genre_padding) for idx in all_possible_item_indices_encoded]
    )

    # 3. Get embeddings for all items using the item_model.
    #    This can be precomputed and stored in a real system for efficiency.
    print(f"Generating embeddings for all {len(all_possible_item_indices_encoded)} items...")
    all_item_embeddings = item_model.predict(
        [all_item_ids_input_for_model, all_item_genres_input_for_model],
        batch_size=BATCH_SIZE, # Use a reasonable batch size for prediction
        verbose=0
    )
    print("Item embeddings generated.")

    # 4. Calculate similarity (dot product) between the user's embedding and all item embeddings.
    #    user_embedding shape: (1, USER_EMBEDDING_DIM)
    #    all_item_embeddings shape: (num_items, USER_EMBEDDING_DIM)
    #    Resulting similarities shape: (num_items,)
    similarities = np.dot(all_item_embeddings, user_embedding.T).flatten()

    # 5. Get indices of items with highest similarity scores.
    #    `np.argsort` returns indices that would sort in ascending order.
    #    Slicing `[-num_recommendations:]` gets the top N largest.
    #    `[::-1]` reverses them to be in descending order of score.
    top_n_indices_in_all_items_array = np.argsort(similarities)[-num_recommendations:][::-1]

    # 6. Format recommendations: convert encoded item indices back to original item IDs.
    recommendations = []
    for array_idx in top_n_indices_in_all_items_array:
        # `array_idx` is an index into `all_possible_item_indices_encoded` and `similarities`
        item_idx_encoded = all_possible_item_indices_encoded[array_idx]
        original_item_id = item_encoder.inverse_transform([item_idx_encoded])[0] # inverse_transform expects an array
        recommendations.append({
            'item_id': original_item_id,
            'similarity_score': similarities[array_idx]
        })
    return recommendations

# --- Main Execution ---
# Overall Training Time Complexity: O(EPOCHS * (N_train_samples / BATCH_SIZE) * Complexity_of_Forward_Backward_Pass)
# Complexity of Forward/Backward Pass is influenced by embedding lookups and dense layer operations in both towers.
if __name__ == "__main__":
    print("--- Two-Tower Hybrid Recommender Example ---")

    print("\nLoading and preprocessing data...")
    # Load and preprocess data
    processed_data = load_and_preprocess_data()

    if processed_data:
        (train_df, test_df,
         user_encoder, item_encoder, genre_tokenizer,
         num_users, num_items, num_genres_vocab, # Renamed num_genres to num_genres_vocab
         item_idx_to_genres_padded_seq, df_items_meta) = processed_data

        print(f"\nBuilding Two-Tower model (User features: ID; Item features: ID, Genres)...")
        training_model, user_model, item_model = build_two_tower_model(num_users, num_items, num_genres_vocab)

        # Compile the training model
        # Using BinaryCrossentropy as we are predicting interaction probability (0 or 1).
        training_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
            loss='binary_crossentropy',
            metrics=['accuracy'] # Accuracy of predicting the binary label.
        )

        print("\n--- Training Model Summary ---")
        training_model.summary()
        print("\n--- User Tower Summary ---")
        user_model.summary()
        print("\n--- Item Tower Summary ---")
        item_model.summary()

        # Prepare training and testing data inputs for the Keras model.
        # The input must be a dictionary where keys match the `name` attributes of the Input layers.
        train_inputs_dict = {
            'user_idx_input': train_df['user_idx'].values.astype(np.int32),
            'item_idx_input': train_df['item_idx'].values.astype(np.int32),
            'genre_seq_input': np.array([item_idx_to_genres_padded_seq.get(idx, [0]*MAX_GENRES_PER_ITEM) for idx in train_df['item_idx']]).astype(np.int32)
        }
        train_labels_array = train_df['label'].values.astype(np.float32)

        test_inputs_dict = {
            'user_idx_input': test_df['user_idx'].values.astype(np.int32),
            'item_idx_input': test_df['item_idx'].values.astype(np.int32),
            'genre_seq_input': np.array([item_idx_to_genres_padded_seq.get(idx, [0]*MAX_GENRES_PER_ITEM) for idx in test_df['item_idx']]).astype(np.int32)
        }
        test_labels_array = test_df['label'].values.astype(np.float32)

        print("\nTraining the Two-Tower model...")
        history = training_model.fit(
            train_inputs_dict,
            train_labels_array,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_data=(test_inputs_dict, test_labels_array),
            verbose=1 # Show training progress
        )
        print("Model training completed.")

        # Generate recommendations for an example user
        if hasattr(user_encoder, 'classes_') and user_encoder.classes_.size > 0:
            # Select the first user from the encoder's known classes as an example
            target_user_original_id = user_encoder.classes_[0]

            # Get all unique encoded item indices that the item_encoder knows about
            all_item_indices_encoded = item_encoder.transform(item_encoder.classes_)

            print(f"\nGenerating recommendations for User ID (original): {target_user_original_id}...")
            recommendations = get_two_tower_recommendations(
                target_user_original_id, user_encoder, item_encoder,
                user_model, item_model,
                item_idx_to_genres_padded_seq, # Corrected variable name
                all_item_indices_encoded,
                num_recommendations=5
            )

            print("\nRecommended items:")
            if recommendations:
                for rec in recommendations:
                    # For displaying more details, merge with item metadata
                    item_details = df_items_meta[df_items_meta['item_id'] == str(rec['item_id'])] # Ensure ID is string for lookup
                    description = item_details['description'].iloc[0] if not item_details.empty else "N/A"
                    genres_display = item_details['genres'].iloc[0] if not item_details.empty else "N/A"
                    print(f"- Item ID: {rec['item_id']} (Similarity Score: {rec['similarity_score']:.4f}) "
                          f"| Genres: {genres_display} | Description: {description[:60]}...")
            else:
                print("No recommendations could be generated for this user.")
        else:
            print("\nCannot generate recommendations as no users were encoded or available.")
    else:
        print("\nData loading and preprocessing failed. Cannot proceed with the Two-Tower example.")

    print("\n--- Two-Tower Hybrid Recommender Example Finished ---")
