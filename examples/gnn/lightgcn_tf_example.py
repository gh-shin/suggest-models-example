# examples/gnn/lightgcn_tf_example.py
import pandas as pd
import numpy as np
import scipy.sparse as sp
import os
import sys
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Lambda, Add, Multiply, Dot, Subtract, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder

# --- LightGCN (Light Graph Convolution Network): Basic Explanation ---
# LightGCN is a simplified Graph Neural Network (GNN) model tailored for recommendation tasks.
# It leverages the user-item interaction graph to learn embeddings for users and items.
# The core idea is that user preferences can be refined by signals from users who interacted
# with similar items, and item characteristics can be refined by users who also interacted with them.
#
# How it works:
# 1. Graph Construction:
#    - A bipartite graph is constructed where users and items are nodes.
#    - An edge exists between a user node and an item node if the user has interacted with the item.
#    - This graph is often represented by an adjacency matrix A.
#
# 2. Embedding Initialization (E^(0)):
#    - Both users and items are initialized with trainable embedding vectors (e.g., from a Keras Embedding layer).
#    - These initial embeddings E^(0) represent the 0-th layer embeddings (i.e., before any graph propagation).
#
# 3. Light Graph Convolution (Propagation Rule):
#    - LightGCN iteratively refines embeddings by propagating them through the graph structure.
#    - The embedding of a node (user or item) at layer (k+1) is computed by aggregating the
#      embeddings of its neighbors from layer (k).
#    - The key simplification in LightGCN compared to standard GCNs is that it *removes*
#      feature transformation (multiplying by a weight matrix W^(k)) and non-linear activation
#      functions (like ReLU) from the propagation step.
#    - The propagation rule for node 'u' (can be a user or item) is:
#      E_u^(k+1) = sum_{v in N_u} (1 / sqrt(|N_u| * |N_v|)) * E_v^(k)
#      where N_u is the set of neighbors of node 'u', and |N_u| is its degree.
#      This is equivalent to performing matrix multiplication with a symmetrically normalized adjacency matrix:
#      E^(k+1) = (D^(-0.5) * A * D^(-0.5)) * E^(k)
#      where D is the diagonal degree matrix.
#
# 4. Layer Combination (Final Embedding Generation):
#    - After 'K' layers of graph convolutions, we obtain 'K+1' sets of embeddings for each user/item
#      (E^(0), E^(1), ..., E^(K)). Each E^(k) captures information from k-hop neighbors.
#    - The final embedding for a user/item is typically a weighted sum (often simple average) of the
#      embeddings learned at each layer:
#      E_final = (1 / (K+1)) * sum_{k=0 to K} (E^(k))
#
# 5. Prediction:
#    - The predicted preference score for a user-item pair is usually calculated as the dot product
#      of their final embeddings: score(user, item) = dot(E_final_user, E_final_item).
#
# 6. Training (Optimization):
#    - LightGCN is commonly trained using a pairwise loss function, such as Bayesian Personalized Ranking (BPR) loss.
#    - BPR loss aims to rank observed (positive) items higher than unobserved (negative) items for a given user.
#      It samples triplets (user, positive_item, negative_item) and tries to maximize the margin between
#      score(user, positive_item) and score(user, negative_item).
#    - Only the initial embeddings E^(0) are trainable parameters. The propagation steps are parameter-free.
#
# Pros:
# - Simplicity and Efficiency: By removing weight matrices and non-linearities in propagation, LightGCN is
#   computationally less expensive and has fewer parameters than many other GCNs.
# - Effectiveness: Achieves state-of-the-art performance on many recommendation benchmarks, demonstrating
#   that the simplifications are well-suited for collaborative filtering.
# - Captures Higher-Order Connectivity: Multiple propagation layers allow embeddings to implicitly capture
#   information from more distant neighbors in the graph, modeling complex collaborative signals.
#
# Cons:
# - Over-smoothing: With a large number of layers (K), embeddings of different nodes might become too similar,
#   losing their discriminative power. This is a common issue in GCNs.
# - Graph Dependent: Performance relies on a well-constructed user-item interaction graph. It might not be
#   as effective for very sparse graphs or users/items with few interactions (cold-start).
# - No Feature Transformation during Propagation: While a strength for CF, it means it doesn't learn to transform
#   features during the neighborhood aggregation, relying solely on the initial embeddings and graph structure.
# ---

# Dynamically add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Need Reshape layer
from tensorflow.keras.layers import Reshape

# --- Model Hyperparameters ---
EMBEDDING_DIM = 32          # Dimensionality of user and item embeddings.
NUM_LAYERS = 2              # Number of LightGCN propagation layers (K in the paper).
LEARNING_RATE = 0.001       # Learning rate for the Adam optimizer.
BATCH_SIZE_BPR = 1024       # Batch size for BPR training (number of (user, positive_item, negative_item) triplets).
EPOCHS = 10                 # Number of training epochs (small for example, typically 100s or 1000s).
REG_WEIGHT = 1e-4           # L2 regularization weight for the initial embeddings (E^0).

# --- Data Loading and Preprocessing ---
# Time Complexity:
# - Reading CSV: O(N_interactions)
# - LabelEncoding: O(N_interactions)
# - Sparse matrix creation: O(N_interactions)
# - Adjacency matrix normalization: Depends on sparse matrix operations, generally efficient for sparse graphs.
def load_and_preprocess_data(base_filepath='data/dummy_interactions.csv'):
    """
    Loads interaction data, encodes user/item IDs, and constructs the normalized
    adjacency matrix required for LightGCN.

    Returns:
        dict: A dictionary containing preprocessed data components, or None on failure.
              Keys: 'df_interactions', 'user_encoder', 'item_encoder', 'num_users',
                    'num_items', 'norm_adj_mat'.
    """
    filepath = os.path.join(project_root, base_filepath)
    if not os.path.exists(filepath):
        print(f"Error: Data file not found at {filepath}.")
        try:
            from data.generate_dummy_data import generate_dummy_data
            print("Attempting to generate dummy data...")
            # Ensure enough interactions for a meaningful graph
            generate_dummy_data(num_users=200, num_items=100, num_interactions=2500, generate_sequences=False)
            print("Dummy data generation script executed.")
        except ImportError as e_import:
            print(f"ImportError: Failed to import 'generate_dummy_data'. Error: {e_import}")
            return None
        except Exception as e:
            print(f"Error during dummy data generation: {e}")
            return None
        if not os.path.exists(filepath): # Re-check after attempt
            print(f"Error: Dummy data file still not found at {filepath} after generation attempt.")
            return None
        print("Dummy data file should now be available.")

    df = pd.read_csv(filepath)
    if df.empty:
        print(f"Error: Data file {filepath} is empty.")
        return None

    # Encode user_id and item_id to 0-indexed integers
    user_encoder = LabelEncoder()
    df['user_idx'] = user_encoder.fit_transform(df['user_id'])

    item_encoder = LabelEncoder()
    df['item_idx'] = item_encoder.fit_transform(df['item_id'])

    num_users = df['user_idx'].nunique()
    num_items = df['item_idx'].nunique()

    # Create the user-item interaction matrix R (sparse)
    # R_ui = 1 if user u interacted with item i, 0 otherwise.
    R = sp.csr_matrix((np.ones(len(df)), (df['user_idx'], df['item_idx'])), shape=(num_users, num_items))

    # Construct the full adjacency matrix A_tilde for the bipartite graph:
    # A_tilde = [[0, R], [R.T, 0]]
    # This matrix has dimensions (num_users + num_items) x (num_users + num_items).
    # Top-left block: User-User (0 matrix)
    # Top-right block: User-Item (R matrix)
    # Bottom-left block: Item-User (R.T matrix - transpose of R)
    # Bottom-right block: Item-Item (0 matrix)
    adj_mat_top_block = sp.hstack([sp.csr_matrix((num_users, num_users), dtype=R.dtype), R])
    adj_mat_bottom_block = sp.hstack([R.T, sp.csr_matrix((num_items, num_items), dtype=R.dtype)])
    adj_mat = sp.vstack([adj_mat_top_block, adj_mat_bottom_block]).tocsr() # Ensure CSR format

    # Calculate degree matrix D and its inverse square root D^(-0.5)
    # Degree of a node is the sum of its row in the adjacency matrix.
    row_sum_degrees = np.array(adj_mat.sum(axis=1)).flatten()

    # For D^(-0.5), we take -0.5 power of degrees. Handle division by zero for nodes with no edges.
    # Nodes with degree 0 will have d_inv_sqrt = 0, so they won't aggregate messages.
    d_inv_sqrt = np.power(row_sum_degrees, -0.5, where=row_sum_degrees > 0)
    d_inv_sqrt[np.isinf(d_inv_sqrt) | np.isnan(d_inv_sqrt)] = 0.0 # Replace inf/NaN with 0

    # Create diagonal matrix from d_inv_sqrt
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

    # Symmetrically normalized adjacency matrix: A_norm = D^(-0.5) * A_tilde * D^(-0.5)
    # This is the matrix used in the LightGCN propagation rule.
    norm_adj_mat = d_mat_inv_sqrt.dot(adj_mat).dot(d_mat_inv_sqrt).tocsr()

    print(f"Data loaded and preprocessed: {num_users} users, {num_items} items.")
    print(f"Normalized adjacency matrix shape: {norm_adj_mat.shape}")

    return {
        'df_interactions': df,
        'user_encoder': user_encoder,
        'item_encoder': item_encoder,
        'num_users': num_users,
        'num_items': num_items,
        'norm_adj_mat': norm_adj_mat
    }

# --- BPR Triplet Generation ---
# Time Complexity: O(N_positive_interactions * Avg_attempts_for_negative_sample)
# In worst case, can be O(N_positive_interactions * N_items) if finding negative samples is very hard.
def get_bpr_triplets(df_interactions, num_items, user_col='user_idx', item_col='item_idx'):
    """
    Generates (user, positive_item, negative_item) triplets for BPR loss.
    For each user and a positive item they interacted with, a negative item
    (one they haven't interacted with) is randomly sampled.
    """
    # Create a dictionary mapping each user to the set of items they've interacted with.
    user_item_interaction_sets = df_interactions.groupby(user_col)[item_col].apply(set).to_dict()
    all_item_indices = np.arange(num_items) # Array of all possible item indices [0, ..., num_items-1]

    users, positive_items, negative_items = [], [], []
    print(f"Generating BPR triplets for {df_interactions.shape[0]} positive interactions...")
    for user_idx, interacted_items_set in user_item_interaction_sets.items():
        for positive_item_idx in interacted_items_set:
            # Simple random negative sampling: pick a random item until one is found
            # that the user has not interacted with.
            negative_item_idx = np.random.choice(all_item_indices)
            # Ensure the negatively sampled item is not in the user's interacted items.
            # Add a counter to prevent potential infinite loops for users who interacted with almost all items.
            max_attempts = num_items * 2
            current_attempts = 0
            while negative_item_idx in interacted_items_set and current_attempts < max_attempts:
                negative_item_idx = np.random.choice(all_item_indices)
                current_attempts += 1

            if current_attempts >= max_attempts and negative_item_idx in interacted_items_set:
                # This user might have interacted with all or nearly all items. Skip adding this triplet.
                # Or, could sample from a global list of non-interacted items for this user, but that's more complex.
                # print(f"Warning: Could not find a valid negative sample for user {user_idx}, item {positive_item_idx} after {max_attempts} attempts.")
                continue


            users.append(user_idx)
            positive_items.append(positive_item_idx)
            negative_items.append(negative_item_idx)

    print(f"Generated {len(users)} BPR triplets.")
    return np.array(users, dtype=np.int32), np.array(positive_items, dtype=np.int32), np.array(negative_items, dtype=np.int32)


class LightGCNLayer(tf.keras.layers.Layer):
    """
    A single LightGCN propagation layer. It performs a sparse matrix multiplication
    of the normalized adjacency matrix with the current embeddings.
    E^(k+1) = A_norm * E^(k)
    """
    def __init__(self, norm_adj_mat_sp, **kwargs):
        super(LightGCNLayer, self).__init__(**kwargs)
        # Convert the SciPy sparse matrix to a TensorFlow SparseTensor.
        # This is done once at layer initialization for efficiency.
        coo = norm_adj_mat_sp.tocoo()
        indices = np.mat([coo.row, coo.col]).transpose()
        self.norm_adj_mat_tf = tf.SparseTensor(indices, coo.data.astype(np.float32), coo.shape)
        # self.is_built = False # Not strictly necessary here as TF handles build status

    def call(self, current_embeddings_concat):
        """
        Performs the graph propagation.
        Args:
            current_embeddings_concat (tf.Tensor): Concatenated user and item embeddings from the previous layer,
                                                 shape ((num_users + num_items), embedding_dim).
        Returns:
            tf.Tensor: Propagated embeddings, same shape as input.
        """
        # Sparse matrix multiplication: A_norm * E_current
        # E_current contains embeddings for both users and items, stacked.
        return tf.sparse.sparse_dense_matmul(self.norm_adj_mat_tf, current_embeddings_concat)

    def get_config(self):
        config = super().get_config()
        # Note: self.norm_adj_mat_tf (SparseTensor) is not directly serializable by Keras.
        # For saving/loading a model with this layer, the adjacency matrix needs to be
        # passed again to the constructor or handled via a custom save/load mechanism.
        # For this example, we are not focusing on model serialization.
        return config

# Define LightGCN as a Keras Model subclass
class LightGCNModel(tf.keras.Model):
    """
    LightGCN model implemented as a Keras subclassed Model.
    It handles embedding initialization, multiple propagation layers,
    and final embedding combination for prediction.
    """
    def __init__(self, num_users, num_items, norm_adj_mat_sp, # Pass the scipy sparse matrix
                 embedding_dim=EMBEDDING_DIM, num_layers=NUM_LAYERS, reg_weight=REG_WEIGHT, **kwargs):
        super(LightGCNModel, self).__init__(**kwargs)

        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers # K: Number of propagation layers

        # Initial (0-th layer) embeddings for users and items. These are the only trainable parameters.
        # L2 regularization is applied as suggested in the LightGCN paper.
        self.e0_user_embedding_layer = Embedding(
            input_dim=num_users, output_dim=embedding_dim,
            name='user_E0_embedding',
            embeddings_initializer='glorot_uniform', # Default, but can be specified
            embeddings_regularizer=tf.keras.regularizers.l2(reg_weight)
        )
        self.e0_item_embedding_layer = Embedding(
            input_dim=num_items, output_dim=embedding_dim,
            name='item_E0_embedding',
            embeddings_initializer='glorot_uniform',
            embeddings_regularizer=tf.keras.regularizers.l2(reg_weight)
        )

        # List of LightGCN propagation layers
        self.gcn_propagation_layers = [
            LightGCNLayer(norm_adj_mat_sp, name=f'lightgcn_propagation_layer_{k+1}')
            for k in range(num_layers)
        ]

        # Helper Lambda layers for TensorFlow operations within the Keras model structure
        self.concat_initial_embeddings = Concatenate(axis=0, name='concat_E0_user_item_embeddings')
        self.stack_layer_embeddings = Lambda(lambda x: tf.stack(x, axis=0), name='stack_all_layer_embeddings')
        self.mean_pool_final_embeddings = Lambda(lambda x: tf.reduce_mean(x, axis=0), name='mean_pool_layer_embeddings')

        # Lambdas to split the final (num_users + num_items, emb_dim) tensor back into user and item embeddings
        self.split_final_user_embeddings = Lambda(
            lambda x: tf.slice(x, [0, 0], [self.num_users, -1]), # Slice for user embeddings
            name='final_user_embeddings_split'
        )
        self.split_final_item_embeddings = Lambda(
            lambda x: tf.slice(x, [self.num_users, 0], [self.num_items, -1]), # Slice for item embeddings
            name='final_item_embeddings_split'
        )

        # Lambdas for BPR loss calculation during training
        # Gathers specific user/item embeddings based on input indices
        self.gather_embeddings = Lambda(
            lambda params_ids_tuple: tf.gather(params_ids_tuple[0], params_ids_tuple[1]),
            name='gather_embeddings_for_bpr'
        )
        # Reshapes gathered embeddings from (batch_size, 1, emb_dim) to (batch_size, emb_dim) if needed,
        # though gather on 1D indices from a 2D table should already give (batch_size, emb_dim).
        # Using Reshape for explicit control.
        self.reshape_gathered_embeddings = Reshape((self.embedding_dim,), name='reshape_gathered_bpr_embeddings')

        self.dot_product_pos = Dot(axes=1, name='bpr_positive_score_dot')
        self.dot_product_neg = Dot(axes=1, name='bpr_negative_score_dot')
        self.bpr_difference = Subtract(name="bpr_score_difference")

    def _propagate_embeddings(self):
        """
        Performs LightGCN propagation for all layers and combines them.
        This method computes the final user and item embeddings based on E_final = mean(E_0, E_1, ..., E_K).
        It's called by `call` (for training) and `get_final_embeddings_for_recommendation` (for inference).
        """
        # Get initial (E^0) embeddings for all users and items
        # tf.range creates sequences [0, 1, ..., num_users-1] and [0, 1, ..., num_items-1]
        all_E0_user_embeddings = self.e0_user_embedding_layer(tf.range(self.num_users, dtype=tf.int32))
        all_E0_item_embeddings = self.e0_item_embedding_layer(tf.range(self.num_items, dtype=tf.int32))

        # Concatenate user and item E^0 embeddings: E_concat^0 = [E_user^0; E_item^0]
        # Shape: (num_users + num_items, embedding_dim)
        current_propagated_embeddings = self.concat_initial_embeddings([all_E0_user_embeddings, all_E0_item_embeddings])

        # Store embeddings from all layers (E^0, E^1, ..., E^K)
        all_layer_wise_embeddings = [current_propagated_embeddings] # Start with E^0

        # Perform K layers of graph propagation
        for gcn_layer in self.gcn_propagation_layers:
            current_propagated_embeddings = gcn_layer(current_propagated_embeddings) # E^(k+1) = A_norm * E^(k)
            all_layer_wise_embeddings.append(current_propagated_embeddings)

        # Combine embeddings from all layers using mean pooling (as per LightGCN paper)
        # 1. Stack embeddings from all layers: shape (K+1, num_users + num_items, embedding_dim)
        if len(all_layer_wise_embeddings) > 1: # If propagation happened (NUM_LAYERS > 0)
            stacked_all_layer_embeddings = self.stack_layer_embeddings(all_layer_wise_embeddings)
            # 2. Mean pool across layers (axis=0): shape (num_users + num_items, embedding_dim)
            final_combined_embeddings = self.mean_pool_final_embeddings(stacked_all_layer_embeddings)
        else: # Only E^0 exists if NUM_LAYERS = 0
            final_combined_embeddings = all_layer_wise_embeddings[0]

        # Split the combined tensor back into final user and item embeddings
        final_user_embeddings_table = self.split_final_user_embeddings(final_combined_embeddings)
        final_item_embeddings_table = self.split_final_item_embeddings(final_combined_embeddings)

        return final_user_embeddings_table, final_item_embeddings_table

    def call(self, inputs):
        """
        Forward pass for training with BPR loss.
        Args:
            inputs (list of tf.Tensor): A list containing three tensors:
                - user_input_indices: Batch of user indices.
                - positive_item_input_indices: Batch of positive item indices for corresponding users.
                - negative_item_input_indices: Batch of negative item indices for corresponding users.
        Returns:
            tf.Tensor: The difference between positive scores and negative scores (y_pred_diff for BPR loss).
        """
        user_input_indices, positive_item_input_indices, negative_item_input_indices = inputs

        # Get the final (propagated and combined) embeddings for all users and items
        final_user_embeddings_table, final_item_embeddings_table = self._propagate_embeddings()

        # Gather embeddings for the specific users, positive items, and negative items in the batch
        # `tf.gather` selects rows from the embedding tables based on input indices.
        user_batch_embeddings_gathered = self.gather_embeddings([final_user_embeddings_table, user_input_indices])
        pos_item_batch_embeddings_gathered = self.gather_embeddings([final_item_embeddings_table, positive_item_input_indices])
        neg_item_batch_embeddings_gathered = self.gather_embeddings([final_item_embeddings_table, negative_item_input_indices])

        # Reshape embeddings if `gather` added an extra dimension of size 1 (common with tf.gather on 2D params and 1D indices)
        # This ensures they are (batch_size, embedding_dim) for dot product.
        user_batch_embeddings = self.reshape_gathered_embeddings(user_batch_embeddings_gathered)
        pos_item_batch_embeddings = self.reshape_gathered_embeddings(pos_item_batch_embeddings_gathered)
        neg_item_batch_embeddings = self.reshape_gathered_embeddings(neg_item_batch_embeddings_gathered)

        # Calculate dot product for (user, positive_item) pairs
        positive_scores = self.dot_product_pos([user_batch_embeddings, pos_item_batch_embeddings])
        # Calculate dot product for (user, negative_item) pairs
        negative_scores = self.dot_product_neg([user_batch_embeddings, neg_item_batch_embeddings])

        # BPR difference: positive_score - negative_score
        # This difference is then fed into a log_sigmoid for the BPR loss.
        return self.bpr_difference([positive_scores, negative_scores])

    def get_final_embeddings_for_recommendation(self):
        """
        Computes and returns the final (propagated and combined) embeddings for all users and items.
        This method is intended for use during inference/recommendation generation.
        """
        return self._propagate_embeddings()

# --- BPR Loss Function ---
def bpr_loss(_, y_pred_difference):
    """
    Bayesian Personalized Ranking (BPR) loss.
    Args:
        _ (tf.Tensor): True labels (ignored in BPR, usually ones).
        y_pred_difference (tf.Tensor): The difference between positive and negative item scores
                                     (output of the LightGCNModel's call method).
    Returns:
        tf.Tensor: The BPR loss value.
    """
    # BPR loss aims to maximize: log(sigmoid(score_positive - score_negative))
    # This is equivalent to minimizing: -log(sigmoid(score_positive - score_negative))
    return -tf.reduce_mean(tf.math.log_sigmoid(y_pred_difference))

# --- Recommendation Generation ---
# Time Complexity for one user:
# - Embedding propagation (if not precomputed): O(N_edges_in_graph * EMBEDDING_DIM * NUM_LAYERS)
# - Dot products for all items: O(N_items * EMBEDDING_DIM)
# - Sorting: O(N_items * log N_items)
def get_lightgcn_recommendations(user_id_original, user_encoder, item_encoder,
                                 trained_lightgcn_model, # Pass the trained model instance
                                 num_recommendations=5):
    """
    Generates top-N recommendations for a given user using the trained LightGCN model.
    """
    try:
        # Convert original user ID to its encoded integer index
        user_idx_encoded = user_encoder.transform([user_id_original])[0]
    except ValueError:
        print(f"Error: User ID '{user_id_original}' not found in user_encoder. Cannot generate recommendations.")
        return []

    # Get final propagated embeddings for all users and items from the trained model
    final_all_user_embeddings, final_all_item_embeddings = trained_lightgcn_model.get_final_embeddings_for_recommendation()

    # Get the embedding for the target user
    target_user_embedding_vector = tf.expand_dims(final_all_user_embeddings[user_idx_encoded], axis=0) # Shape: (1, embedding_dim)

    # Calculate scores for all items for this user by dot product: UserEmbedding * AllItemEmbeddings^T
    # `tf.matmul` with transpose_b=True achieves this efficiently.
    all_item_scores_for_user = tf.matmul(target_user_embedding_vector, final_all_item_embeddings, transpose_b=True)
    all_item_scores_for_user_np = tf.squeeze(all_item_scores_for_user).numpy() # Squeeze to 1D array

    # Get indices of items with highest scores
    # `np.argsort` returns indices that would sort in ascending order.
    # Slicing `[-num_recommendations:]` gets the top N largest.
    # `[::-1]` reverses them to be in descending order of score.
    top_n_item_indices_encoded = np.argsort(all_item_scores_for_user_np)[-num_recommendations:][::-1]

    recommendations = []
    for item_idx_encoded in top_n_item_indices_encoded:
        original_item_id = item_encoder.inverse_transform([item_idx_encoded])[0] # inverse_transform expects an array
        recommendations.append({
            'item_id': original_item_id,
            'score': all_item_scores_for_user_np[item_idx_encoded]
        })
    return recommendations

# --- Main Execution ---
# Overall Training Time Complexity: O(EPOCHS * (N_BPR_Triplets / BATCH_SIZE_BPR) * EMBEDDING_DIM + EPOCHS * N_edges_in_graph * EMBEDDING_DIM * NUM_LAYERS)
# The first term is for processing BPR triplets (dot products, subtractions).
# The second term is for the graph propagation part within _propagate_embeddings, called once per batch effectively (though shared across samples).
if __name__ == "__main__":
    print("--- LightGCN (TensorFlow/Keras) Example ---")

    print("\nLoading and preprocessing data...")
    data_dict = load_and_preprocess_data()

    if data_dict:
        df_interactions = data_dict['df_interactions']
        user_encoder = data_dict['user_encoder']
        item_encoder = data_dict['item_encoder']
        num_users = data_dict['num_users']
        num_items = data_dict['num_items']
        norm_adj_mat_sp = data_dict['norm_adj_mat'] # This is a SciPy sparse matrix

        print(f"Unique users: {num_users}, Unique items: {num_items}")
        if norm_adj_mat_sp is not None:
            print(f"Normalized adjacency matrix shape: {norm_adj_mat_sp.shape}")

        print("\nBuilding LightGCN model...")
        # Instantiate the LightGCNModel, passing the SciPy sparse matrix
        lightgcn_model_instance = LightGCNModel(
            num_users, num_items, norm_adj_mat_sp, # Pass the SciPy sparse matrix here
            embedding_dim=EMBEDDING_DIM, num_layers=NUM_LAYERS, reg_weight=REG_WEIGHT
        )

        # Compile the model with BPR loss and Adam optimizer
        lightgcn_model_instance.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss=bpr_loss)

        # Generate BPR training triplets (user_idx, positive_item_idx, negative_item_idx)
        print("\nGenerating BPR triplets for training...")
        train_users_bpr, train_pos_items_bpr, train_neg_items_bpr = get_bpr_triplets(
            df_interactions, num_items, user_col='user_idx', item_col='item_idx'
        )

        # Dummy y_true for BPR loss (it's ignored by the loss function, but Keras fit expects it)
        # Its values don't matter, only its shape for batching.
        dummy_y_true_for_bpr = np.ones_like(train_users_bpr, dtype=np.float32)

        print(f"\nStarting LightGCN model training for {EPOCHS} epochs...")
        # Inputs for the model.call method are a list or tuple
        history = lightgcn_model_instance.fit(
            [train_users_bpr, train_pos_items_bpr, train_neg_items_bpr], # List of inputs
            dummy_y_true_for_bpr,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE_BPR,
            verbose=1 # Show training progress
        )
        print("Model training completed.")

        print("\n--- LightGCN Model Summary ---")
        # Summary will show layers, including the custom LightGCNLayer instances.
        # The model needs to be built first, which happens upon first call (e.g. during fit).
        lightgcn_model_instance.summary()

        # Generate recommendations for an example user
        if hasattr(user_encoder, 'classes_') and user_encoder.classes_.size > 0:
            # Select the first user from the encoder's known classes as an example
            target_user_original_id = user_encoder.classes_[0]

            print(f"\nGenerating recommendations for User ID (original): {target_user_original_id}...")
            recommendations = get_lightgcn_recommendations(
                target_user_original_id, user_encoder, item_encoder,
                lightgcn_model_instance, # Pass the trained model instance
                num_recommendations=5
            )

            print("\nRecommended items:")
            if recommendations:
                for rec in recommendations:
                    print(f"- Item ID (original): {rec['item_id']}, Predicted Score: {rec['score']:.4f}")
            else:
                print("No recommendations could be generated for this user.")
        else:
            print("\nCannot generate recommendations as no users were encoded or available.")
    else:
        print("\nData loading and preprocessing failed. Cannot proceed with LightGCN example.")

    print("\n--- LightGCN (TensorFlow/Keras) Example Finished ---")
