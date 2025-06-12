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
# LightGCN is a simplified Graph Neural Network model specifically designed for recommendation.
# It aims to capture collaborative filtering signals by modeling user-item interactions as a bipartite graph
# and learning user/item embeddings through graph convolutions.
#
# How it works:
# 1. Graph Construction: User-item interactions are represented as a bipartite graph. Users and items are nodes.
#    An edge exists if a user has interacted with an item.
# 2. Embedding Initialization (E_0): Users and items are initialized with trainable embedding vectors (e.g., from an Embedding layer).
# 3. Light Graph Convolution (Propagation Rule):
#    - For each user/item, its embedding at layer 'k+1' is computed by aggregating the embeddings of its neighbors
#      from layer 'k'.
#    - Crucially, LightGCN *only* uses neighborhood aggregation. It does *not* use feature transformation (weight matrices)
#      or non-linear activation functions during propagation, which are common in other GCNs. This simplification
#      reduces model complexity and is found to be effective for recommendations.
#    - The aggregation is typically a sum or mean of normalized neighbor embeddings. The normalization involves
#      degree matrices (D^-0.5 * A * D^-0.5, where A is the adjacency matrix).
# 4. Layer Combination:
#    - After 'K' layers of graph convolutions, we obtain 'K+1' sets of embeddings for each user/item (E_0, E_1, ..., E_K).
#    - The final embedding for a user/item is a weighted sum (or average, or concatenation) of the embeddings
#      learned at each layer. A common choice is to average them: E_final = (1/(K+1)) * sum(E_k for k in 0..K).
# 5. Prediction:
#    - The predicted score for a user-item pair is typically the dot product of their final embeddings: score(u,i) = E_final_user^T * E_final_item.
# 6. Training:
#    - Often trained with a pairwise loss function like Bayesian Personalized Ranking (BPR) loss, which aims to rank
#      observed items higher than unobserved items for a given user.
#
# Pros:
# - Simplicity and Efficiency: Fewer parameters and simpler propagation than many other GCNs.
# - Effectiveness: Achieves strong performance on many recommendation benchmarks.
# - Captures Higher-Order Connectivity: Multiple propagation layers allow embeddings to capture information
#   from more distant neighbors (e.g., items liked by users similar to users who liked items similar to the target item).
#
# Cons:
# - Over-smoothing: With too many layers, embeddings of different nodes might become too similar, losing discriminative power.
# - Requires careful graph construction and normalization.
# ---

# Dynamically add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Need Reshape layer
from tensorflow.keras.layers import Reshape

# --- Model Hyperparameters ---
EMBEDDING_DIM = 32
NUM_LAYERS = 2  # Number of LightGCN layers (K in the paper)
LEARNING_RATE = 0.001 # Common default for Adam
BATCH_SIZE_BPR = 1024 # Batch size for BPR training (number of (u,i,j) triplets)
EPOCHS = 10 # Small for example, typically 100s or 1000s
REG_WEIGHT = 1e-4 # L2 regularization weight for embeddings (common in LightGCN)


def load_and_preprocess_data(base_filepath='data/dummy_interactions.csv'):
    """Loads interaction data and prepares it for LightGCN."""
    filepath = os.path.join(project_root, base_filepath)
    if not os.path.exists(filepath):
        print(f"오류: {filepath} 파일을 찾을 수 없습니다.")
        try:
            from data.generate_dummy_data import generate_dummy_data
            print("더미 데이터 생성 시도 중...")
            generate_dummy_data(num_users=200, num_items=100, num_interactions=2500) # Ensure enough data
            print("더미 데이터 생성 완료.")
        except Exception as e:
            print(f"더미 데이터 생성 실패: {e}")
            return None
        if not os.path.exists(filepath):
            print(f"파일 재생성 실패: {filepath}")
            return None

    df = pd.read_csv(filepath)
    if df.empty:
        print(f"데이터 파일 {filepath}이 비어있습니다.")
        return None

    user_encoder = LabelEncoder()
    df['user_idx'] = user_encoder.fit_transform(df['user_id'])

    item_encoder = LabelEncoder()
    df['item_idx'] = item_encoder.fit_transform(df['item_id'])

    num_users = df['user_idx'].nunique()
    num_items = df['item_idx'].nunique()

    R = sp.csr_matrix((np.ones(len(df)), (df['user_idx'], df['item_idx'])), shape=(num_users, num_items))

    # Adjacency matrix A_tilde for LightGCN (R in top-right, R.T in bottom-left)
    # A = [[0, R], [R.T, 0]]
    adj_mat_top = sp.hstack([sp.csr_matrix((num_users, num_users), dtype=R.dtype), R])
    adj_mat_bottom = sp.hstack([R.T, sp.csr_matrix((num_items, num_items), dtype=R.dtype)])
    adj_mat = sp.vstack([adj_mat_top, adj_mat_bottom]).tocsr() # Ensure CSR for consistent operations

    # Normalized adjacency matrix D^-0.5 * A * D^-0.5
    row_sum = np.array(adj_mat.sum(axis=1)).flatten()
    # Replace 0s in row_sum with 1s to avoid division by zero, then compute d_inv_sqrt
    # This means nodes with no edges will have d_inv_sqrt = 1, effectively not changing their embeddings via neighbors
    # A better approach might be to handle them or ensure all nodes have at least a self-loop if appropriate.
    # For LightGCN, D_ii = sum of outgoing edges. If sum is 0, D^-0.5 is undefined.
    # LightGCN paper uses D_ii = |N_i|.
    # We should keep 0 for nodes with no edges if d_inv_sqrt is for D^-0.5.
    # If row_sum is 0, power(-0.5) is inf.
    d_inv_sqrt = np.power(row_sum, -0.5, where=row_sum>0) # Compute only where row_sum > 0
    d_inv_sqrt[np.isinf(d_inv_sqrt) | np.isnan(d_inv_sqrt)] = 0.0 # Set inf/nan to 0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

    norm_adj_mat = d_mat_inv_sqrt.dot(adj_mat).dot(d_mat_inv_sqrt).tocsr()

    return {
        'df_interactions': df,
        'user_encoder': user_encoder,
        'item_encoder': item_encoder,
        'num_users': num_users,
        'num_items': num_items,
        'norm_adj_mat': norm_adj_mat
    }

def get_bpr_triplets(df_interactions, num_items, user_col='user_idx', item_col='item_idx'):
    """Generates (user, positive_item, negative_item) triplets for BPR loss."""
    user_item_set = df_interactions.groupby(user_col)[item_col].apply(set).to_dict()
    all_item_indices = np.arange(num_items)

    users, pos_items, neg_items = [], [], []
    for user_idx, interacted_items in user_item_set.items():
        for pos_item_idx in interacted_items:
            # Simple random negative sampling
            neg_item_idx = np.random.choice(all_item_indices)
            while neg_item_idx in interacted_items:
                neg_item_idx = np.random.choice(all_item_indices)

            users.append(user_idx)
            pos_items.append(pos_item_idx)
            neg_items.append(neg_item_idx)

    return np.array(users, dtype=np.int32), np.array(pos_items, dtype=np.int32), np.array(neg_items, dtype=np.int32)


class LightGCNLayer(tf.keras.layers.Layer):
    def __init__(self, norm_adj_mat_sp, **kwargs):
        super(LightGCNLayer, self).__init__(**kwargs)
        coo = norm_adj_mat_sp.tocoo()
        indices = np.mat([coo.row, coo.col]).transpose()
        self.norm_adj_mat_tf = tf.SparseTensor(indices, coo.data.astype(np.float32), coo.shape)
        self.is_built = False # To handle potential re-build issues if any

    def call(self, embeddings):
        return tf.sparse.sparse_dense_matmul(self.norm_adj_mat_tf, embeddings)

    def get_config(self):
        config = super().get_config()
        # self.norm_adj_mat_tf cannot be serialized directly.
        # For saving/loading, the matrix needs to be handled separately.
        return config

# Define LightGCN as a Keras Model subclass
class LightGCNModel(tf.keras.Model):
    def __init__(self, num_users, num_items, norm_adj_mat_sp,
                 embedding_dim=EMBEDDING_DIM, num_layers=NUM_LAYERS, reg_weight=REG_WEIGHT, **kwargs):
        super(LightGCNModel, self).__init__(**kwargs)

        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        # self.norm_adj_mat_sp = norm_adj_mat_sp # Not storing, passed to GCN layers

        self.e0_user_embedding_layer = Embedding(
            num_users, embedding_dim, name='user_E0_embedding',
            embeddings_regularizer=tf.keras.regularizers.l2(reg_weight)
        )
        self.e0_item_embedding_layer = Embedding(
            num_items, embedding_dim, name='item_E0_embedding',
            embeddings_regularizer=tf.keras.regularizers.l2(reg_weight)
        )

        self.gcn_layers = [LightGCNLayer(norm_adj_mat_sp, name=f'lightgcn_prop_layer_{k+1}') for k in range(num_layers)]

        self.concat_E0 = Concatenate(axis=0, name='concat_E0_embeddings_model')
        self.stack_layers_lambda = Lambda(lambda x: tf.stack(x, axis=0), name='stack_layer_embeddings_model')
        self.mean_pool_lambda = Lambda(lambda x: tf.reduce_mean(x, axis=0), name='mean_pooled_embeddings_model')

        self.split_user_lambda = Lambda(
            lambda x: tf.split(x, [self.num_users, self.num_items], axis=0)[0],
            output_shape=(self.num_users, self.embedding_dim), name='final_user_split_model'
        )
        self.split_item_lambda = Lambda(
            lambda x: tf.split(x, [self.num_users, self.num_items], axis=0)[1],
            output_shape=(self.num_items, self.embedding_dim), name='final_item_split_model'
        )

        gather_output_shape = (1, self.embedding_dim)
        self.gather_lambda = Lambda(
            lambda params_ids_tuple: tf.gather(params_ids_tuple[0], params_ids_tuple[1]),
            output_shape=gather_output_shape, name='gather_bpr_model'
        )
        # Using Reshape instead of Lambda(tf.squeeze) for more explicit shape handling by Keras
        self.reshape_for_squeeze = Reshape((self.embedding_dim,), name='reshape_for_squeeze_bpr_model')

        self.pos_dot_layer = Dot(axes=1, name='pos_score_dot_model')
        self.neg_dot_layer = Dot(axes=1, name='neg_score_dot_model')
        self.subtract_layer = Subtract(name="bpr_diff_model")

    def _propagate_embeddings(self):
        all_E0_user_embs = self.e0_user_embedding_layer(tf.range(self.num_users, dtype=tf.int32))
        all_E0_item_embs = self.e0_item_embedding_layer(tf.range(self.num_items, dtype=tf.int32))

        current_layer_embeddings = self.concat_E0([all_E0_user_embs, all_E0_item_embs])
        all_layer_embeddings_list = [current_layer_embeddings]

        for gcn_layer_instance in self.gcn_layers:
            current_layer_embeddings = gcn_layer_instance(current_layer_embeddings)
            all_layer_embeddings_list.append(current_layer_embeddings)

        if len(all_layer_embeddings_list) > 1:
            stacked_embeddings = self.stack_layers_lambda(all_layer_embeddings_list)
            final_embeddings_tensor = self.mean_pool_lambda(stacked_embeddings)
        else:
            final_embeddings_tensor = all_layer_embeddings_list[0]

        final_user_embs_table = self.split_user_lambda(final_embeddings_tensor)
        final_item_embs_table = self.split_item_lambda(final_embeddings_tensor)
        return final_user_embs_table, final_item_embs_table

    def call(self, inputs): # inputs is a list: [user_input, pos_item_input, neg_item_input]
        user_input, pos_item_input, neg_item_input = inputs

        final_user_embs_table, final_item_embs_table = self._propagate_embeddings()

        user_emb_g = self.gather_lambda([final_user_embs_table, user_input])
        pos_item_emb_g = self.gather_lambda([final_item_embs_table, pos_item_input])
        neg_item_emb_g = self.gather_lambda([final_item_embs_table, neg_item_input])

        user_emb = self.reshape_for_squeeze(user_emb_g)
        pos_item_emb = self.reshape_for_squeeze(pos_item_emb_g)
        neg_item_emb = self.reshape_for_squeeze(neg_item_emb_g)

        pos_score = self.pos_dot_layer([user_emb, pos_item_emb])
        neg_score = self.neg_dot_layer([user_emb, neg_item_emb])

        return self.subtract_layer([pos_score, neg_score])

    # For recommendation, we need the final full embedding tables
    def get_final_embeddings_for_recommendation(self):
        return self._propagate_embeddings()


def bpr_loss(_, y_pred_diff): # y_true is ignored
    return -tf.reduce_mean(tf.math.log_sigmoid(y_pred_diff))


def get_lightgcn_recommendations(user_id_original, user_encoder, item_encoder,
                                 trained_lightgcn_model, # Pass the trained model instance
                                 # num_users, num_items are known by the model for splitting internally
                                 num_recommendations=5):
    try:
        user_idx = user_encoder.transform([user_id_original])[0]
    except ValueError:
        print(f"User ID {user_id_original} not found in encoder.")
        return []

    # Get final propagated embeddings from the trained model
    final_user_embs_rec, final_item_embs_rec = trained_lightgcn_model.get_final_embeddings_for_recommendation()

    target_user_embedding = tf.expand_dims(final_user_embs_rec[user_idx], axis=0)
    all_item_scores = tf.matmul(target_user_embedding, final_item_embs_rec, transpose_b=True)
    all_item_scores = tf.squeeze(all_item_scores).numpy()

    top_n_item_indices = np.argsort(all_item_scores)[-num_recommendations:][::-1]

    recommendations = []
    for item_idx in top_n_item_indices:
        original_item_id = item_encoder.inverse_transform([item_idx])[0]
        recommendations.append({'item_id': original_item_id, 'score': all_item_scores[item_idx]})
    return recommendations

# --- Main Execution ---
if __name__ == "__main__":
    print("--- LightGCN (TensorFlow/Keras) 예제 시작 ---")

    print("\n데이터 로드 및 전처리 중...")
    data_dict = load_and_preprocess_data()

    if data_dict:
        df_interactions = data_dict['df_interactions']
        user_encoder, item_encoder = data_dict['user_encoder'], data_dict['item_encoder']
        num_users, num_items = data_dict['num_users'], data_dict['num_items']
        norm_adj_mat_sp = data_dict['norm_adj_mat']

        print(f"고유 사용자: {num_users}, 고유 아이템: {num_items}")
        print(f"정규화된 인접 행렬 형태: {norm_adj_mat_sp.shape if norm_adj_mat_sp is not None else 'N/A'}")

        print("\nLightGCN 모델 구축 중...")
        # Instantiate the LightGCNModel
        model = LightGCNModel(
            num_users, num_items, norm_adj_mat_sp,
            embedding_dim=EMBEDDING_DIM, num_layers=NUM_LAYERS, reg_weight=REG_WEIGHT
        )

        model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss=bpr_loss)

        # Call model on dummy inputs to build it for summary, or define input_shape in build
        # For functional models, summary works after definition. For subclassed, after first call or explicit build.
        # Let's build it by calling with dummy shapes (not ideal, but for summary)
        # Or simply proceed to fit, summary will be available after first batch implicitly.
        # For now, we'll see summary after fit or can be printed if needed.

        print("\nBPR 트리플릿 생성 및 모델 학습 중...")
        train_users, train_pos_items, train_neg_items = get_bpr_triplets(
            df_interactions, num_items, user_col='user_idx', item_col='item_idx'
        )

        dummy_y_true = np.ones_like(train_users, dtype=np.float32)

        # Inputs for the model.call method are a list or tuple
        history = model.fit(
            [train_users, train_pos_items, train_neg_items],
            dummy_y_true,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE_BPR,
            verbose=1
        )
        print("모델 학습 완료.")
        model.summary() # Print summary after training (model is built)

        if hasattr(user_encoder, 'classes_') and user_encoder.classes_.size > 0:
            target_user_original_id = user_encoder.classes_[0]

            print(f"\n사용자 ID {target_user_original_id}에 대한 추천 생성 중...")
            recommendations = get_lightgcn_recommendations(
                target_user_original_id, user_encoder, item_encoder,
                model, # Pass the trained model instance
                # num_users and num_items are not directly needed by the function call
                # as the model's get_final_embeddings_for_recommendation method handles it.
                num_recommendations=5
            )

            print("\n추천된 아이템:")
            if recommendations:
                for rec in recommendations:
                    print(f"- 아이템 {rec['item_id']}: 예측 점수 {rec['score']:.4f}")
            else:
                print("추천할 아이템이 없습니다.")
        else:
            print("\n추천을 생성할 사용자가 없습니다.")
    else:
        print("\n데이터 로드 및 전처리에 실패하여 예제를 실행할 수 없습니다.")

    print("\n--- LightGCN (TensorFlow/Keras) 예제 실행 완료 ---")
