# examples/sequential/transformer_sasrec_example.py
import pandas as pd
import numpy as np
import os
import sys
import tensorflow as tf
from tensorflow.keras.layers import Layer, Embedding, Dense, Dropout, LayerNormalization, MultiHeadAttention, Input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

# --- SASRec (Self-Attentive Sequential Recommendation): Basic Explanation ---
# SASRec (Self-Attentive Sequential Recommendation) is a recommendation model designed to capture
# user preferences based on their historical interaction sequences. It leverages the self-attention
# mechanism, inspired by the Transformer model, to understand which items in a user's past
# sequence are most influential for predicting the next item.
#
# How it works:
# 1. Input Representation:
#    - User's interaction history is represented as a sequence of item IDs (e.g., [item_A, item_B, item_C]).
#    - The task is to predict the next item (e.g., item_D) that the user is likely to interact with.
#
# 2. Item Embeddings:
#    - Each item ID in the input sequence is mapped to a dense vector representation (item embedding).
#    - This embedding captures the latent characteristics of the item.
#    - A special padding token (often 0) is used for sequences shorter than MAX_SEQ_LENGTH.
#
# 3. Positional Embeddings:
#    - Since self-attention itself doesn't inherently understand item order, positional information is crucial.
#    - Positional embeddings are learned vectors, one for each position in the sequence (1 to MAX_SEQ_LENGTH).
#    - These are added to the corresponding item embeddings to give the model a sense of item order.
#
# 4. Self-Attention Block(s): This is the core of SASRec, typically composed of one or more identical blocks.
#    Each block aims to refine the representation of each item in the sequence by considering its context.
#    A block usually contains:
#    a. Causal Multi-Head Self-Attention:
#       - "Self-Attention": Each item in the sequence attends to all other items *before it* (and itself)
#         to calculate a weighted sum of their representations. This means the model learns which
#         past items are most relevant for understanding the current item's role in the sequence.
#       - "Causal": Ensures that when predicting the item at position 't', the model only attends to items
#         at positions 'j <= t'. This prevents data leakage from future items.
#       - "Multi-Head": The attention mechanism is run multiple times in parallel with different learned
#         linear projections (heads). This allows the model to jointly attend to information from
#         different representation subspaces at different positions. The outputs are then concatenated
#         and linearly transformed.
#    b. Point-wise Feed-Forward Network (FFN):
#       - Applied independently to each item's representation after the self-attention step.
#       - Typically consists of two dense layers with a non-linear activation (e.g., ReLU) in between.
#       - This allows for more complex transformations of each item's representation.
#    c. Add & Norm (Residual Connections and Layer Normalization):
#       - Residual connections (adding the input of a sub-layer to its output) help in training deeper models
#         by mitigating vanishing gradient problems.
#       - Layer Normalization stabilizes the learning process.
#
# 5. Prediction Output:
#    - After the input sequence passes through all Transformer blocks, the output representation of the
#      *last item* in the (input) sequence is taken. This vector is considered to summarize the user's
#      current interest based on their history.
#    - This final vector is then passed through a Dense layer with `num_items_for_embedding` units (no activation or softmax,
#      as loss function will handle logits) to produce a score (logit) for every possible next item.
#    - During training, `SparseCategoricalCrossentropy` loss is used to compare these logits against the
#      actual next item in the sequence.
#
# Pros:
# - Captures Sequential Dynamics: Effectively models the order and dependencies within user interaction sequences.
#   Self-attention can identify long-range dependencies.
# - Contextual Understanding: Learns to weigh the importance of different past items dynamically based on the current context
#   when predicting the next item.
# - Parallelizable Training: Computations within the Transformer blocks (especially self-attention) can be highly parallelized,
#   making training efficient on modern hardware like GPUs/TPUs.
#
# Cons:
# - Data Requirements: Performs best with a sufficient amount of user sequence data to learn meaningful patterns.
# - Computational Cost: Self-attention has a complexity of O(sequence_length^2 * embedding_dim), which can be
#   demanding for very long sequences. However, sequence lengths in recommendation are often moderate.
# - Item Cold-Start: Cannot directly recommend new items that were not present in the training data (as they lack embeddings).
#   Requires strategies like retraining or using item content features for new items.
# ---

# Dynamically add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- Model Hyperparameters (kept small for this example for quick execution) ---
MAX_SEQ_LENGTH = 10  # Maximum length of a user's item interaction sequence considered by the model.
EMBEDDING_DIM = 32   # Dimensionality of item and positional embedding vectors.
NUM_HEADS = 2        # Number of attention heads in the MultiHeadAttention layer.
NUM_BLOCKS = 1       # Number of Transformer blocks to stack. SASRec paper suggests 2 for some datasets.
FFN_UNITS = 64       # Number of units in the hidden layer of the Feed-Forward Network.
DROPOUT_RATE = 0.2   # Dropout rate for regularization in various layers.
LEARNING_RATE = 0.001# Learning rate for the Adam optimizer.
EPOCHS = 5           # Number of training epochs (small for example).
BATCH_SIZE = 32      # Number of sequences per training batch.

# --- Data Loading and Preprocessing ---
# Time Complexity:
# - Reading CSV: O(N_interactions_total)
# - LabelEncoding: O(N_total_items_in_sequences)
# - Sequence padding and splitting: O(N_sequences * MAX_SEQ_LENGTH)
def load_sequences(base_filepath='data/dummy_sequences.csv', max_seq_len=MAX_SEQ_LENGTH):
    """
    Loads item interaction sequences, preprocesses them for SASRec, and creates training samples.
    - Reads sequences from a CSV file.
    - Encodes item IDs to 0-indexed integers.
    - Generates (input_sequence, target_item) pairs.
    - Pads input sequences to a fixed length.

    Args:
        base_filepath (str): Path to the sequence data CSV, relative to project root.
        max_seq_len (int): The fixed length to pad/truncate sequences to.

    Returns:
        tuple: (X_padded, y_array, num_items_for_embedding, item_encoder) or (None, None, None, None) on failure.
               - X_padded: Padded input sequences.
               - y_array: Target items for each input sequence.
               - num_items_for_embedding: Vocabulary size for item embedding (num unique items + 1 for padding).
               - item_encoder: Fitted LabelEncoder for item IDs.
    """
    filepath = os.path.join(project_root, base_filepath)
    if not os.path.exists(filepath):
        print(f"Error: Sequence data file not found at {filepath}.")
        print("Attempting to generate dummy sequence data using 'data/generate_dummy_data.py'...")
        try:
            from data.generate_dummy_data import generate_dummy_data
            # Ensure generate_sequences=True is called for dummy data generation
            generate_dummy_data(num_users=200, num_items=100, num_interactions=2000, generate_sequences=True)
            print("Dummy sequence data generation script executed.")
            if not os.path.exists(filepath): # Re-check after generation attempt
                print(f"Error: Dummy sequence file still not found at {filepath} after generation.")
                return None, None, None, None
            print("Dummy sequence data file should now be available.")
        except ImportError as e_import:
            print(f"ImportError: Failed to import 'generate_dummy_data'. Error: {e_import}")
            return None, None, None, None
        except Exception as e_general:
            print(f"Error during dummy data generation: {e_general}")
            return None, None, None, None

    df_sequences = pd.read_csv(filepath)
    if df_sequences.empty or 'item_ids_sequence' not in df_sequences.columns:
        print(f"Error: File {filepath} is empty or missing 'item_ids_sequence' column.")
        return None, None, None, None

    # Convert space-separated item ID strings in CSV to lists of integers
    sequences_orig_ids = df_sequences['item_ids_sequence'].apply(lambda s: [int(i) for i in str(s).split(' ')]).tolist()

    # Create a vocabulary of all unique items present in the sequences
    all_items_flat_list = [item for seq in sequences_orig_ids for item in seq]
    if not all_items_flat_list:
        print("Error: No items found in the sequences.")
        return None, None, None, None

    # Use LabelEncoder to map original item IDs to 0-indexed integers
    item_encoder = LabelEncoder()
    item_encoder.fit(all_items_flat_list)

    # Vocabulary size for embedding layer: number of unique items + 1 for padding token (0)
    # The padding token will have index 0. Actual items will have indices 1 to N.
    num_items_for_embedding = len(item_encoder.classes_) + 1

    # Transform sequences to encoded item indices. Add 1 to all encoded indices
    # so that actual item indices start from 1, reserving 0 exclusively for padding.
    encoded_sequences = [item_encoder.transform(s) + 1 for s in sequences_orig_ids]

    # Create (input, target) pairs for training
    # For a sequence [i1, i2, i3, i4]:
    #   Input: [i1], Target: i2
    #   Input: [i1, i2], Target: i3
    #   Input: [i1, i2, i3], Target: i4
    X_train_seqs, y_train_targets = [], []
    for seq in encoded_sequences:
        for i in range(1, len(seq)): # Start from 1 because we need at least one item as input
            input_sub_sequence = seq[:i]    # Sequence up to item i-1 (exclusive of i)
            target_item_label = seq[i]      # Item at index i is the target
            X_train_seqs.append(input_sub_sequence)
            y_train_targets.append(target_item_label)

    if not X_train_seqs:
        print("Error: Could not generate any training samples (X, y pairs). "
              "This might happen if all sequences have length 1 or less.")
        return None, None, None, None

    # Pad input sequences to `max_seq_len`
    # 'pre' padding: add 0s at the beginning of sequences shorter than max_seq_len.
    # 'pre' truncating: remove elements from the beginning for sequences longer than max_seq_len.
    # `value=0`: use 0 as the padding value.
    X_padded = pad_sequences(X_train_seqs, maxlen=max_seq_len, padding='pre', truncating='pre', value=0)
    y_array = np.array(y_train_targets) # Target items are already 1-based encoded item indices

    print(f"Data loaded: {len(df_sequences)} original sequences, resulting in {len(X_padded)} training samples.")
    return X_padded, y_array, num_items_for_embedding, item_encoder

# --- SASRec Model Components ---
class PositionalEmbedding(Layer):
    """
    Custom Keras layer for adding positional embeddings to item embeddings.
    The position of an item in a sequence is important for sequential models.
    """
    def __init__(self, max_seq_len, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim
        # Embedding layer for positions. Input dimension is max_seq_len + 1
        # because positions are 1-indexed (1 to max_seq_len).
        # The 0th embedding can be considered for a padding position if needed,
        # though typically not explicitly used if items at padding positions are masked.
        self.pos_embeddings = Embedding(input_dim=max_seq_len + 1, output_dim=embed_dim, name=f"{self.name}_pos_emb_lookup")

    def call(self, x_item_embeddings): # x_item_embeddings has shape (batch_size, seq_len, embed_dim)
        # Get the actual sequence length from the input item embeddings tensor
        # (might be shorter than max_seq_len if not padded, or if masking is handled downstream)
        seq_len = tf.shape(x_item_embeddings)[1]

        # Create position indices: [1, 2, ..., seq_len]
        positions = tf.range(start=1, limit=seq_len + 1, delta=1)
        # Look up embeddings for these positions
        embedded_positions = self.pos_embeddings(positions) # Shape: (seq_len, embed_dim)
        # This will be added to item_embeddings, broadcasting over the batch dimension.
        return embedded_positions

    def get_config(self): # For model saving and loading
        config = super().get_config()
        config.update({
            "max_seq_len": self.max_seq_len,
            "embed_dim": self.embed_dim,
        })
        return config

class TransformerBlock(Layer):
    """
    A single Transformer block, consisting of Multi-Head Self-Attention and a Feed-Forward Network.
    Includes residual connections and layer normalization.
    """
    def __init__(self, embed_dim, num_heads, ffn_units, dropout_rate, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ffn_units = ffn_units
        self.dropout_rate = dropout_rate

        if embed_dim % num_heads != 0:
            raise ValueError(
                f"Embedding dimension ({embed_dim}) must be divisible by the number of attention heads ({num_heads})."
            )

        # Multi-Head Self-Attention layer
        # key_dim = embed_dim // num_heads ensures projection heads have compatible dimensions.
        self.mha = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim // num_heads, dropout=dropout_rate, name=f"{self.name}_mha")

        # Point-wise Feed-Forward Network (FFN)
        self.ffn = tf.keras.Sequential([
            Dense(ffn_units, activation="relu", name=f"{self.name}_ffn_dense1"),
            Dropout(dropout_rate, name=f"{self.name}_ffn_dropout"),
            Dense(embed_dim, name=f"{self.name}_ffn_dense2") # Project back to embedding dimension
        ], name=f"{self.name}_ffn")

        # Layer Normalization layers
        self.layernorm1 = LayerNormalization(epsilon=1e-6, name=f"{self.name}_layernorm1")
        self.layernorm2 = LayerNormalization(epsilon=1e-6, name=f"{self.name}_layernorm2")

        # Dropout layer after MHA (before adding residual connection)
        self.dropout_mha_output = Dropout(dropout_rate, name=f"{self.name}_mha_output_dropout")

    def call(self, inputs, training=False, attention_mask=None): # Renamed causal_mask to attention_mask for clarity
        # `inputs` shape: (batch_size, seq_len, embed_dim)
        # `attention_mask` is the causal mask to prevent attending to future positions.
        # It should be shape (batch_size, seq_len, seq_len) where mask[b, i, j] is True if item j
        # should be masked (not attended to) when computing representation for item i in batch b.

        # Multi-Head Self-Attention sub-layer
        # Query, Value, Key are all the same `inputs` for self-attention.
        # The `attention_mask` ensures causality.
        attn_output = self.mha(query=inputs, value=inputs, key=inputs, attention_mask=attention_mask, training=training)
        attn_output_dropped = self.dropout_mha_output(attn_output, training=training)
        # Add residual connection and apply Layer Normalization
        out1 = self.layernorm1(inputs + attn_output_dropped)

        # Feed-Forward Network sub-layer
        ffn_output = self.ffn(out1, training=training)
        # Add residual connection and apply Layer Normalization
        # (Dropout is already part of self.ffn definition if needed after Dense layers there)
        out2 = self.layernorm2(out1 + ffn_output) # Corrected: should be out1 + ffn_output
        return out2

    def get_config(self): # For model saving and loading
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ffn_units": self.ffn_units,
            "dropout_rate": self.dropout_rate,
        })
        return config

# --- SASRec Model Building Function ---
# Time Complexity for one forward pass: Dominated by Transformer blocks.
# Each block: O(MAX_SEQ_LENGTH^2 * EMBEDDING_DIM + MAX_SEQ_LENGTH * EMBEDDING_DIM * FFN_UNITS)
# Total for NUM_BLOCKS: O(NUM_BLOCKS * (MAX_SEQ_LENGTH^2 * EMBEDDING_DIM + MAX_SEQ_LENGTH * EMBEDDING_DIM * FFN_UNITS))
def build_sasrec_model(max_seq_len, num_items_for_embedding, embed_dim, num_blocks, num_heads, ffn_units, dropout_rate):
    """
    Builds the SASRec model architecture using Keras Functional API.
    """
    # Input layer for sequences of item indices (padded with 0s)
    input_seq = Input(shape=(max_seq_len,), name="input_sequence", dtype='int32')

    # Item Embedding Layer
    # `mask_zero=True` tells subsequent layers to ignore padding tokens (0)
    item_embedding_layer = Embedding(
        input_dim=num_items_for_embedding, # Vocabulary size (num unique items + 1 for padding)
        output_dim=embed_dim,
        name="item_embedding",
        mask_zero=True # Important for handling padded sequences
    )
    item_embs = item_embedding_layer(input_seq) # Shape: (batch_size, max_seq_len, embed_dim)

    # Positional Embedding Layer
    pos_embedding_layer = PositionalEmbedding(max_seq_len, embed_dim, name="positional_embedding")
    pos_embs = pos_embedding_layer(item_embs) # Shape: (max_seq_len, embed_dim)

    # Add item embeddings and positional embeddings
    # Positional embeddings are broadcasted across the batch dimension.
    seq_embs = item_embs + pos_embs
    seq_embs = Dropout(dropout_rate, name="input_dropout")(seq_embs)

    # --- Causal Self-Attention Mask ---
    # This mask ensures that an item at position 'i' can only attend to items at positions 'j <= i'.
    # It prevents the model from "seeing" future items during training for a given target.
    # The mask should have True for positions that *should be masked* (i.e., not attended to).
    # Keras MHA layer expects attention_mask shape (batch_size, Tq, Tv) or (batch_size, num_heads, Tq, Tv).
    # For self-attention, Tq (target/query sequence length) = Tv (source/value sequence length) = max_seq_len.

    # 1. Create a boolean matrix of shape (max_seq_len, max_seq_len)
    #    `tf.linalg.band_part(tf.ones((L, L)), -1, 0)` creates a lower triangular matrix (1s on and below diagonal).
    #    `1.0 - lower_triangular_matrix` inverts it, so future positions (upper triangle) are 1s.
    #    `tf.cast(..., dtype=tf.bool)` converts to boolean (True for future positions).
    causal_mask_matrix = tf.cast(
        1.0 - tf.linalg.band_part(tf.ones((max_seq_len, max_seq_len)), -1, 0),
        dtype=tf.bool
    )
    # Add a batch dimension for broadcasting: (1, max_seq_len, max_seq_len)
    # This single mask will be broadcast across all samples in the batch and all attention heads.
    causal_attention_mask_for_mha = causal_mask_matrix[tf.newaxis, :, :]

    # The `mask_zero=True` in the Embedding layer automatically generates a padding mask.
    # Keras's MultiHeadAttention layer is designed to correctly combine an explicit `attention_mask` (like our causal one)
    # with the implicit padding mask propagated from the Embedding layer.

    # Transformer Blocks
    transformer_output = seq_embs
    for i in range(num_blocks):
        transformer_output = TransformerBlock(
            embed_dim, num_heads, ffn_units, dropout_rate, name=f"transformer_block_{i+1}"
        )(transformer_output, attention_mask=causal_attention_mask_for_mha) # Pass the causal mask

    # Use the representation of the *last item* in the input sequence for prediction.
    # Due to 'pre' padding, the last actual item is at index -1 of the sequence.
    # Its representation has captured information from all preceding items via self-attention.
    last_item_representation = transformer_output[:, -1, :] # Shape: (batch_size, embed_dim)

    # Output Layer: Predict scores (logits) for all possible next items.
    # The number of units is `num_items_for_embedding` (vocab size including padding).
    # No activation function here, as `SparseCategoricalCrossentropy(from_logits=True)` expects raw logits.
    output_logits = Dense(num_items_for_embedding, activation=None, name="output_logits")(last_item_representation)

    model = Model(inputs=input_seq, outputs=output_logits)
    return model

# --- Recommendation Generation for a Given Sequence ---
def get_sasrec_recommendations(model, input_sequence_encoded_1based, item_encoder, num_items_for_embedding, num_recommendations=5):
    """
    Generates next-item recommendations for a given 1-based encoded input sequence.

    Args:
        model: The trained SASRec Keras model.
        input_sequence_encoded_1based (list of int): The user's current interaction sequence,
                                                     with item IDs already encoded and 1-based.
        item_encoder (LabelEncoder): Fitted item encoder to map back to original IDs.
        num_items_for_embedding (int): Total number of items in embedding layer (vocab size + 1).
        num_recommendations (int): Number of top recommendations to return.

    Returns:
        list: List of recommended items, each a dict {'item_id': original_id, 'score': predicted_score}.
    """
    if not input_sequence_encoded_1based: # Check if the input list is empty
        print("Input sequence is empty. Cannot generate recommendations.")
        return []

    # Pad the input sequence to MAX_SEQ_LENGTH, using 'pre' padding with value 0.
    padded_input_sequence = pad_sequences(
        [input_sequence_encoded_1based], maxlen=MAX_SEQ_LENGTH, padding='pre', truncating='pre', value=0
    )

    # Get model predictions (logits for all items in the vocabulary)
    # Prediction Time Complexity for one sequence: O(MAX_SEQ_LENGTH^2 * EMBEDDING_DIM + ...) (dominated by Transformer)
    predicted_logits = model.predict(padded_input_sequence, verbose=0)[0] # Get scores for the single input sequence

    # Mask items that should not be recommended:
    # 1. The padding token (index 0)
    predicted_logits[0] = -np.inf

    # 2. Items already present in the input sequence (user has already interacted with them)
    for item_idx_1based_in_input in input_sequence_encoded_1based:
        if 0 < item_idx_1based_in_input < len(predicted_logits): # Check bounds
             predicted_logits[item_idx_1based_in_input] = -np.inf # Mask by setting score to negative infinity

    # Get indices of the top N items with highest scores (logits)
    # `np.argsort` returns indices that would sort the array in ascending order.
    # Slicing `[-num_recommendations:]` gets the top N largest.
    # `[::-1]` reverses them to be in descending order of score.
    top_n_item_indices_1based = np.argsort(predicted_logits)[-num_recommendations:][::-1]

    recommended_items = []
    for item_idx_1based in top_n_item_indices_1based:
        if item_idx_1based == 0: # Should be effectively filtered by -np.inf, but double-check
            continue

        # Convert 1-based index (used in model due to padding at 0) back to 0-based for item_encoder
        item_idx_0based_for_encoder = item_idx_1based - 1

        # Check if the index is valid for the encoder
        if 0 <= item_idx_0based_for_encoder < len(item_encoder.classes_):
            original_item_id = item_encoder.inverse_transform([item_idx_0based_for_encoder])[0]
            recommended_items.append({'item_id': original_item_id, 'score': predicted_logits[item_idx_1based]})
        else:
            print(f"Warning: Skipping recommended item index {item_idx_1based} as it's out of bounds for item_encoder after adjustment.")


    return recommended_items

# --- Main Execution ---
# Training Time Complexity: O(EPOCHS * (N_train_samples / BATCH_SIZE) * Complexity_of_SASRec_Forward_Pass)
if __name__ == "__main__":
    print("--- SASRec (Self-Attentive Sequential Recommendation) Example ---")

    print("\nLoading and preprocessing sequence data...")
    X_padded, y_array, num_items_for_embedding, item_encoder = load_sequences()

    if X_padded is not None and item_encoder is not None:
        print(f"Padded input sequences shape: {X_padded.shape}")
        print(f"Target item array shape: {y_array.shape}")
        print(f"Number of unique items for embedding (incl. padding): {num_items_for_embedding}")

        print("\nBuilding SASRec model...")
        model = build_sasrec_model(
            MAX_SEQ_LENGTH,
            num_items_for_embedding,
            EMBEDDING_DIM,
            NUM_BLOCKS,
            NUM_HEADS,
            FFN_UNITS,
            DROPOUT_RATE
        )

        # Compile the model
        # For predicting the next item (which is a class from N items), use SparseCategoricalCrossentropy.
        # `from_logits=True` because our model outputs raw scores (logits), not probabilities (e.g., from softmax).
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'] # Accuracy of predicting the exact next item.
        )
        model.summary() # Print model architecture

        print("\nTraining the model...")
        history = model.fit(
            X_padded,
            y_array,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=1 # Show training progress
        )
        print("Model training completed.")

        # Example: Generate recommendations for a sample sequence
        if hasattr(item_encoder, 'classes_') and len(item_encoder.classes_) > 0:
            # Try to get an example sequence from the original dummy data for a realistic test
            raw_sequences_df_path = os.path.join(project_root, 'data/dummy_sequences.csv')
            example_input_sequence_orig_ids = []
            if os.path.exists(raw_sequences_df_path):
                raw_sequences_df = pd.read_csv(raw_sequences_df_path)
                if not raw_sequences_df.empty and 'item_ids_sequence' in raw_sequences_df.columns:
                    # Take the first sequence from the dummy data as an example
                    example_raw_sequence_str = raw_sequences_df['item_ids_sequence'].iloc[0]
                    example_input_sequence_orig_ids = [int(i) for i in example_raw_sequence_str.split(' ')]

            if not example_input_sequence_orig_ids: # Fallback if file loading failed or empty
                print("\nWarning: Could not load example sequence from file, using a fallback sequence.")
                # Ensure fallback items exist in item_encoder.classes_ or use encoded values directly
                if len(item_encoder.classes_) >= 3: # Check if encoder knows at least 3 items
                   example_input_sequence_orig_ids = item_encoder.classes_[:min(3, MAX_SEQ_LENGTH-1)].tolist() # Take first 3 known items
                else:
                   example_input_sequence_orig_ids = [] # Cannot form a meaningful sequence

            if example_input_sequence_orig_ids:
                 # Filter out items not known to the encoder (e.g. if dummy data changed)
                known_items_in_sequence_orig_ids = [item for item in example_input_sequence_orig_ids if item in item_encoder.classes_]

                if known_items_in_sequence_orig_ids:
                    # Take a portion of the sequence, e.g., last few items up to MAX_SEQ_LENGTH-1, as input to predict the next
                    # The model expects an input sequence to predict what comes *after* it.
                    input_for_recs_orig_ids = known_items_in_sequence_orig_ids[:MAX_SEQ_LENGTH-1] # Max input len is MAX_SEQ_LENGTH-1 to predict next
                    if input_for_recs_orig_ids: # Ensure it's not empty after filtering and slicing
                        input_for_recs_encoded_1based = item_encoder.transform(input_for_recs_orig_ids) + 1

                        print(f"\nExample: Input sequence for recommendation (Original IDs): {input_for_recs_orig_ids}")
                        print(f"Encoded 1-based input sequence: {input_for_recs_encoded_1based.tolist()}")

                        recommendations = get_sasrec_recommendations(
                            model,
                            input_for_recs_encoded_1based.tolist(),
                            item_encoder,
                            num_items_for_embedding
                        )

                        print("\nRecommended next items:")
                        if recommendations:
                            for rec in recommendations:
                                print(f"- Item ID (original): {rec['item_id']}, Predicted Score (logit): {rec['score']:.3f}")
                        else:
                            print("No recommendations could be generated for the example sequence.")
                    else:
                        print("\nWarning: Example input sequence became empty after filtering for known items or slicing.")
                else:
                    print("\nWarning: None of the items in the example sequence are known to the item encoder.")
            else:
                print("\nCould not prepare a valid example input sequence for recommendation.")
        else:
            print("\nItem encoder not available or has no classes; cannot generate example recommendations.")
    else:
        print("\nData loading failed. Cannot proceed with the SASRec example.")

    print("\n--- SASRec Example Finished ---")
