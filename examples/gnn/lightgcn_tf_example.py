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

# --- LightGCN (Light Graph Convolution Network): 기본 설명 ---
# LightGCN은 추천 작업에 맞춰진 단순화된 그래프 신경망(GNN) 모델입니다.
# 사용자-아이템 상호작용 그래프를 활용하여 사용자와 아이템에 대한 임베딩을 학습합니다.
# 핵심 아이디어는 사용자 선호도가 유사한 아이템과 상호작용한 사용자로부터의 신호로 정제될 수 있고,
# 아이템 특성은 해당 아이템과 상호작용한 사용자에 의해 정제될 수 있다는 것입니다.
#
# 작동 방식:
# 1. 그래프 구성:
#    - 사용자와 아이템이 노드인 이분 그래프가 구성됩니다.
#    - 사용자가 아이템과 상호작용한 경우 사용자 노드와 아이템 노드 사이에 엣지가 존재합니다.
#    - 이 그래프는 종종 인접 행렬 A로 표현됩니다.
#
# 2. 임베딩 초기화 (E^(0)):
#    - 사용자와 아이템 모두 학습 가능한 임베딩 벡터로 초기화됩니다 (예: Keras Embedding 레이어에서).
#    - 이러한 초기 임베딩 E^(0)은 0번째 레이어 임베딩을 나타냅니다 (즉, 그래프 전파 전).
#
# 3. Light Graph Convolution (전파 규칙):
#    - LightGCN은 그래프 구조를 통해 임베딩을 전파하여 반복적으로 정제합니다.
#    - 레이어 (k+1)에서 노드(사용자 또는 아이템)의 임베딩은 레이어 (k)의 이웃 임베딩을 집계하여 계산됩니다.
#    - 표준 GCN과 비교하여 LightGCN의 주요 단순화는 전파 단계에서 특징 변환(가중치 행렬 W^(k) 곱하기)과
#      비선형 활성화 함수(예: ReLU)를 제거한다는 것입니다.
#    - 노드 'u'(사용자 또는 아이템일 수 있음)에 대한 전파 규칙은 다음과 같습니다:
#      E_u^(k+1) = sum_{v in N_u} (1 / sqrt(|N_u| * |N_v|)) * E_v^(k)
#      여기서 N_u는 노드 'u'의 이웃 집합이고 |N_u|는 해당 차수입니다.
#      이는 대칭적으로 정규화된 인접 행렬과의 행렬 곱셈을 수행하는 것과 동일합니다:
#      E^(k+1) = (D^(-0.5) * A * D^(-0.5)) * E^(k)
#      여기서 D는 대각 차수 행렬입니다.
#
# 4. 레이어 조합 (최종 임베딩 생성):
#    - 'K'개의 그래프 컨볼루션 레이어 후 각 사용자/아이템에 대해 'K+1'개의 임베딩 집합을 얻습니다
#      (E^(0), E^(1), ..., E^(K)). 각 E^(k)는 k-홉 이웃으로부터 정보를 포착합니다.
#    - 사용자/아이템에 대한 최종 임베딩은 일반적으로 각 레이어에서 학습된 임베딩의 가중 합(종종 단순 평균)입니다:
#      E_final = (1 / (K+1)) * sum_{k=0 to K} (E^(k))
#
# 5. 예측:
#    - 사용자-아이템 쌍에 대한 예측된 선호도 점수는 일반적으로 최종 임베딩의 내적으로 계산됩니다:
#      score(user, item) = dot(E_final_user, E_final_item).
#
# 6. 학습 (최적화):
#    - LightGCN은 일반적으로 베이즈 개인화 순위(BPR) 손실과 같은 쌍별 손실 함수를 사용하여 학습됩니다.
#    - BPR 손실은 주어진 사용자에 대해 관찰된 (긍정적인) 아이템을 관찰되지 않은 (부정적인) 아이템보다 높게 순위를 매기는 것을 목표로 합니다.
#      (사용자, 긍정적_아이템, 부정적_아이템) 삼중항을 샘플링하고 score(사용자, 긍정적_아이템)와 score(사용자, 부정적_아이템) 간의
#      마진을 최대화하려고 시도합니다.
#    - 초기 임베딩 E^(0)만 학습 가능한 매개변수입니다. 전파 단계는 매개변수가 없습니다.
#
# Pros (장점):
# - 단순성 및 효율성: 전파에서 가중치 행렬과 비선형성을 제거함으로써 LightGCN은
#   다른 많은 GCN보다 계산 비용이 적게 들고 매개변수가 적습니다.
# - 효과성: 많은 추천 벤치마크에서 최첨단 성능을 달성하여 단순화가 협업 필터링에 적합함을 보여줍니다.
# - 고차 연결성 포착: 여러 전파 레이어를 통해 임베딩이 그래프에서 더 멀리 떨어진 이웃의 정보를 암묵적으로 포착하여
#   복잡한 협업 신호를 모델링할 수 있습니다.
#
# Cons (단점):
# - 과도한 평활화 (Over-smoothing): 많은 수의 레이어(K)를 사용하면 다른 노드의 임베딩이 너무 유사해져
#   판별력을 잃을 수 있습니다. 이는 GCN의 일반적인 문제입니다.
# - 그래프 의존성: 성능은 잘 구성된 사용자-아이템 상호작용 그래프에 의존합니다. 매우 희소한 그래프나
#   상호작용이 거의 없는 사용자/아이템(cold-start)에는 효과적이지 않을 수 있습니다.
# - 전파 중 특징 변환 없음: CF의 강점이지만, 이웃 집계 중에 특징을 변환하는 방법을 학습하지 않고
#   초기 임베딩과 그래프 구조에만 의존한다는 의미입니다.
# ---

# 프로젝트 루트를 sys.path에 동적으로 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Reshape 레이어 필요
from tensorflow.keras.layers import Reshape

# --- 모델 하이퍼파라미터 ---
EMBEDDING_DIM = 32          # 사용자 및 아이템 임베딩의 차원.
NUM_LAYERS = 2              # LightGCN 전파 레이어 수 (논문의 K).
LEARNING_RATE = 0.001       # Adam 옵티마이저의 학습률.
BATCH_SIZE_BPR = 1024       # BPR 학습을 위한 배치 크기 ((사용자, 긍정적_아이템, 부정적_아이템) 삼중항 수).
EPOCHS = 10                 # 학습 에포크 수 (예제를 위해 작게 설정, 일반적으로 100s 또는 1000s).
REG_WEIGHT = 1e-4           # 초기 임베딩(E^0)에 대한 L2 정규화 가중치.

# --- 데이터 로딩 및 전처리 ---
# Time Complexity:
# - CSV 읽기: O(N_interactions)
# - LabelEncoding: O(N_interactions)
# - 희소 행렬 생성: O(N_interactions)
# - 인접 행렬 정규화: 희소 행렬 연산에 따라 다르며, 일반적으로 희소 그래프에 대해 효율적임.
def load_and_preprocess_data(base_filepath='data/dummy_interactions.csv'):
    """
    한국어: 상호작용 데이터를 로드하고, 사용자/아이템 ID를 인코딩하며, LightGCN에 필요한
    정규화된 인접 행렬을 구성합니다.

    Loads interaction data, encodes user/item IDs, and constructs the normalized
    adjacency matrix required for LightGCN.

    Returns:
        dict: A dictionary containing preprocessed data components, or None on failure.
              Keys: 'df_interactions', 'user_encoder', 'item_encoder', 'num_users',
                    'num_items', 'norm_adj_mat'.
    """
    filepath = os.path.join(project_root, base_filepath)
    if not os.path.exists(filepath):
        print(f"Error: Data file not found at {filepath}.") # 오류: {filepath}에서 데이터 파일을 찾을 수 없습니다.
        try:
            from data.generate_dummy_data import generate_dummy_data
            print("Attempting to generate dummy data...") # 더미 데이터 생성 시도 중...
            # 의미 있는 그래프를 위해 충분한 상호작용 보장
            generate_dummy_data(num_users=200, num_items=100, num_interactions=2500, generate_sequences=False)
            print("Dummy data generation script executed.") # 더미 데이터 생성 스크립트 실행됨.
        except ImportError as e_import:
            print(f"ImportError: Failed to import 'generate_dummy_data'. Error: {e_import}") # ImportError: 'generate_dummy_data'를 임포트하지 못했습니다. 오류: {e_import}
            return None
        except Exception as e:
            print(f"Error during dummy data generation: {e}") # 더미 데이터 생성 중 오류: {e}
            return None
        if not os.path.exists(filepath): # 시도 후 다시 확인
            print(f"Error: Dummy data file still not found at {filepath} after generation attempt.") # 오류: 생성 시도 후에도 {filepath}에서 더미 데이터 파일을 찾을 수 없습니다.
            return None
        print("Dummy data file should now be available.") # 이제 더미 데이터 파일을 사용할 수 있습니다.

    df = pd.read_csv(filepath)
    if df.empty:
        print(f"Error: Data file {filepath} is empty.") # 오류: 데이터 파일 {filepath}이 비어 있습니다.
        return None

    # user_id 및 item_id를 0부터 시작하는 정수로 인코딩
    user_encoder = LabelEncoder()
    df['user_idx'] = user_encoder.fit_transform(df['user_id'])

    item_encoder = LabelEncoder()
    df['item_idx'] = item_encoder.fit_transform(df['item_id'])

    num_users = df['user_idx'].nunique()
    num_items = df['item_idx'].nunique()

    # 사용자-아이템 상호작용 행렬 R (희소) 생성
    # 사용자가 아이템 i와 상호작용한 경우 R_ui = 1, 그렇지 않으면 0.
    R = sp.csr_matrix((np.ones(len(df)), (df['user_idx'], df['item_idx'])), shape=(num_users, num_items))

    # 이분 그래프에 대한 전체 인접 행렬 A_tilde 구성:
    # A_tilde = [[0, R], [R.T, 0]]
    # 이 행렬의 차원은 (num_users + num_items) x (num_users + num_items)입니다.
    # 왼쪽 상단 블록: 사용자-사용자 (0 행렬)
    # 오른쪽 상단 블록: 사용자-아이템 (R 행렬)
    # 왼쪽 하단 블록: 아이템-사용자 (R.T 행렬 - R의 전치)
    # 오른쪽 하단 블록: 아이템-아이템 (0 행렬)
    adj_mat_top_block = sp.hstack([sp.csr_matrix((num_users, num_users), dtype=R.dtype), R])
    adj_mat_bottom_block = sp.hstack([R.T, sp.csr_matrix((num_items, num_items), dtype=R.dtype)])
    adj_mat = sp.vstack([adj_mat_top_block, adj_mat_bottom_block]).tocsr() # CSR 형식 확인

    # 차수 행렬 D와 그 역제곱근 D^(-0.5) 계산
    # 노드의 차수는 인접 행렬에서 해당 행의 합입니다.
    row_sum_degrees = np.array(adj_mat.sum(axis=1)).flatten()

    # D^(-0.5)의 경우 차수의 -0.5 거듭제곱을 취합니다. 엣지가 없는 노드에 대한 0으로 나누기 처리.
    # 차수가 0인 노드는 d_inv_sqrt = 0이 되어 메시지를 집계하지 않습니다.
    d_inv_sqrt = np.power(row_sum_degrees, -0.5, where=row_sum_degrees > 0)
    d_inv_sqrt[np.isinf(d_inv_sqrt) | np.isnan(d_inv_sqrt)] = 0.0 # inf/NaN을 0으로 대체

    # d_inv_sqrt로부터 대각 행렬 생성
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

    # 대칭적으로 정규화된 인접 행렬: A_norm = D^(-0.5) * A_tilde * D^(-0.5)
    # 이것이 LightGCN 전파 규칙에 사용되는 행렬입니다.
    norm_adj_mat = d_mat_inv_sqrt.dot(adj_mat).dot(d_mat_inv_sqrt).tocsr()

    print(f"Data loaded and preprocessed: {num_users} users, {num_items} items.") # 데이터 로드 및 전처리됨: 사용자 {num_users}명, 아이템 {num_items}개.
    print(f"Normalized adjacency matrix shape: {norm_adj_mat.shape}") # 정규화된 인접 행렬 모양: {norm_adj_mat.shape}

    return {
        'df_interactions': df,
        'user_encoder': user_encoder,
        'item_encoder': item_encoder,
        'num_users': num_users,
        'num_items': num_items,
        'norm_adj_mat': norm_adj_mat
    }

# --- BPR 삼중항 생성 ---
# Time Complexity: O(N_positive_interactions * Avg_attempts_for_negative_sample)
# 최악의 경우, 부정적 샘플을 찾기가 매우 어려우면 O(N_positive_interactions * N_items)가 될 수 있음.
def get_bpr_triplets(df_interactions, num_items, user_col='user_idx', item_col='item_idx'):
    """
    한국어: BPR 손실을 위한 (사용자, 긍정적_아이템, 부정적_아이템) 삼중항을 생성합니다.
    각 사용자와 해당 사용자가 상호작용한 긍정적 아이템에 대해,
    (상호작용하지 않은) 부정적 아이템이 무작위로 샘플링됩니다.

    Generates (user, positive_item, negative_item) triplets for BPR loss.
    For each user and a positive item they interacted with, a negative item
    (one they haven't interacted with) is randomly sampled.
    """
    # 각 사용자를 해당 사용자가 상호작용한 아이템 집합에 매핑하는 딕셔너리 생성.
    user_item_interaction_sets = df_interactions.groupby(user_col)[item_col].apply(set).to_dict()
    all_item_indices = np.arange(num_items) # 가능한 모든 아이템 인덱스 배열 [0, ..., num_items-1]

    users, positive_items, negative_items = [], [], []
    print(f"Generating BPR triplets for {df_interactions.shape[0]} positive interactions...") # {df_interactions.shape[0]}개의 긍정적 상호작용에 대한 BPR 삼중항 생성 중...
    for user_idx, interacted_items_set in user_item_interaction_sets.items():
        for positive_item_idx in interacted_items_set:
            # 단순 무작위 부정적 샘플링: 사용자가 상호작용하지 않은 아이템을 찾을 때까지
            # 무작위 아이템을 선택합니다.
            negative_item_idx = np.random.choice(all_item_indices)
            # 부정적으로 샘플링된 아이템이 사용자의 상호작용 아이템에 없는지 확인합니다.
            # 거의 모든 아이템과 상호작용한 사용자에 대한 잠재적인 무한 루프를 방지하기 위해 카운터 추가.
            max_attempts = num_items * 2
            current_attempts = 0
            while negative_item_idx in interacted_items_set and current_attempts < max_attempts:
                negative_item_idx = np.random.choice(all_item_indices)
                current_attempts += 1

            if current_attempts >= max_attempts and negative_item_idx in interacted_items_set:
                # 이 사용자는 모든 또는 거의 모든 아이템과 상호작용했을 수 있습니다. 이 삼중항 추가를 건너<0xEB><0x9B><0x84>니다.
                # 또는 이 사용자에 대한 상호작용하지 않은 아이템의 전역 목록에서 샘플링할 수 있지만 이는 더 복잡합니다.
                # print(f"Warning: Could not find a valid negative sample for user {user_idx}, item {positive_item_idx} after {max_attempts} attempts.") # 경고: 사용자 {user_idx}, 아이템 {positive_item_idx}에 대해 {max_attempts}번 시도 후 유효한 부정적 샘플을 찾을 수 없습니다.
                continue


            users.append(user_idx)
            positive_items.append(positive_item_idx)
            negative_items.append(negative_item_idx)

    print(f"Generated {len(users)} BPR triplets.") # {len(users)}개의 BPR 삼중항 생성됨.
    return np.array(users, dtype=np.int32), np.array(positive_items, dtype=np.int32), np.array(negative_items, dtype=np.int32)


class LightGCNLayer(tf.keras.layers.Layer):
    """
    A single LightGCN propagation layer. It performs a sparse matrix multiplication
    of the normalized adjacency matrix with the current embeddings.
    E^(k+1) = A_norm * E^(k)
    """ # 단일 LightGCN 전파 레이어입니다. 정규화된 인접 행렬과 현재 임베딩의 희소 행렬 곱셈을 수행합니다.
    def __init__(self, norm_adj_mat_sp, **kwargs):
        super(LightGCNLayer, self).__init__(**kwargs)
        # SciPy 희소 행렬을 TensorFlow 희소 텐서로 변환합니다.
        # 효율성을 위해 레이어 초기화 시 한 번 수행됩니다.
        coo = norm_adj_mat_sp.tocoo()
        indices = np.mat([coo.row, coo.col]).transpose()
        self.norm_adj_mat_tf = tf.SparseTensor(indices, coo.data.astype(np.float32), coo.shape)
        # self.is_built = False # TF가 빌드 상태를 처리하므로 여기서 엄격하게 필요하지 않음

    def call(self, current_embeddings_concat):
        """
        Performs the graph propagation. # 그래프 전파를 수행합니다.
        Args:
            current_embeddings_concat (tf.Tensor): Concatenated user and item embeddings from the previous layer,
                                                 shape ((num_users + num_items), embedding_dim).
        Returns:
            tf.Tensor: Propagated embeddings, same shape as input.
        """
        # 희소 행렬 곱셈: A_norm * E_current
        # E_current는 사용자와 아이템 모두에 대한 임베딩을 포함하며 쌓여 있습니다.
        return tf.sparse.sparse_dense_matmul(self.norm_adj_mat_tf, current_embeddings_concat)

    def get_config(self):
        config = super().get_config()
        # 참고: self.norm_adj_mat_tf (SparseTensor)는 Keras에서 직접 직렬화할 수 없습니다.
        # 이 레이어가 있는 모델을 저장/로드하려면 인접 행렬을 생성자에 다시 전달하거나
        # 사용자 정의 저장/로드 메커니즘을 통해 처리해야 합니다.
        # 이 예제에서는 모델 직렬화에 중점을 두지 않습니다.
        return config

# Keras 모델 서브클래스로 LightGCN 정의
class LightGCNModel(tf.keras.Model):
    """
    LightGCN model implemented as a Keras subclassed Model. # Keras 서브클래스 모델로 구현된 LightGCN 모델입니다.
    It handles embedding initialization, multiple propagation layers,
    and final embedding combination for prediction.
    """ # 임베딩 초기화, 다중 전파 레이어 및 예측을 위한 최종 임베딩 조합을 처리합니다.
    def __init__(self, num_users, num_items, norm_adj_mat_sp, # SciPy 희소 행렬 전달
                 embedding_dim=EMBEDDING_DIM, num_layers=NUM_LAYERS, reg_weight=REG_WEIGHT, **kwargs):
        super(LightGCNModel, self).__init__(**kwargs)

        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers # K: 전파 레이어 수

        # 사용자와 아이템을 위한 초기(0번째 레이어) 임베딩. 이것이 유일하게 학습 가능한 매개변수입니다.
        # LightGCN 논문에서 제안된 대로 L2 정규화가 적용됩니다.
        self.e0_user_embedding_layer = Embedding(
            input_dim=num_users, output_dim=embedding_dim,
            name='user_E0_embedding',
            embeddings_initializer='glorot_uniform', # 기본값이지만 지정 가능
            embeddings_regularizer=tf.keras.regularizers.l2(reg_weight)
        )
        self.e0_item_embedding_layer = Embedding(
            input_dim=num_items, output_dim=embedding_dim,
            name='item_E0_embedding',
            embeddings_initializer='glorot_uniform',
            embeddings_regularizer=tf.keras.regularizers.l2(reg_weight)
        )

        # LightGCN 전파 레이어 목록
        self.gcn_propagation_layers = [
            LightGCNLayer(norm_adj_mat_sp, name=f'lightgcn_propagation_layer_{k+1}')
            for k in range(num_layers)
        ]

        # Keras 모델 구조 내 TensorFlow 연산을 위한 헬퍼 Lambda 레이어
        self.concat_initial_embeddings = Concatenate(axis=0, name='concat_E0_user_item_embeddings')
        self.stack_layer_embeddings = Lambda(lambda x: tf.stack(x, axis=0), name='stack_all_layer_embeddings')
        self.mean_pool_final_embeddings = Lambda(lambda x: tf.reduce_mean(x, axis=0), name='mean_pool_layer_embeddings')

        # 최종 (num_users + num_items, emb_dim) 텐서를 사용자 및 아이템 임베딩으로 다시 분할하는 Lambda
        self.split_final_user_embeddings = Lambda(
            lambda x: tf.slice(x, [0, 0], [self.num_users, -1]), # 사용자 임베딩 슬라이스
            name='final_user_embeddings_split'
        )
        self.split_final_item_embeddings = Lambda(
            lambda x: tf.slice(x, [self.num_users, 0], [self.num_items, -1]), # 아이템 임베딩 슬라이스
            name='final_item_embeddings_split'
        )

        # 학습 중 BPR 손실 계산을 위한 Lambda
        # 입력 인덱스를 기반으로 특정 사용자/아이템 임베딩 수집
        self.gather_embeddings = Lambda(
            lambda params_ids_tuple: tf.gather(params_ids_tuple[0], params_ids_tuple[1]),
            name='gather_embeddings_for_bpr'
        )
        # 수집된 임베딩을 (batch_size, 1, emb_dim)에서 (batch_size, emb_dim)으로 재구성 (필요한 경우),
        # 2D 테이블의 1D 인덱스에 대한 gather는 이미 (batch_size, emb_dim)을 제공해야 함.
        # 명시적 제어를 위해 Reshape 사용.
        self.reshape_gathered_embeddings = Reshape((self.embedding_dim,), name='reshape_gathered_bpr_embeddings')

        self.dot_product_pos = Dot(axes=1, name='bpr_positive_score_dot')
        self.dot_product_neg = Dot(axes=1, name='bpr_negative_score_dot')
        self.bpr_difference = Subtract(name="bpr_score_difference")

    def _propagate_embeddings(self):
        """
        Performs LightGCN propagation for all layers and combines them. # 모든 레이어에 대해 LightGCN 전파를 수행하고 결합합니다.
        This method computes the final user and item embeddings based on E_final = mean(E_0, E_1, ..., E_K).
        It's called by `call` (for training) and `get_final_embeddings_for_recommendation` (for inference).
        """ # 이 메소드는 E_final = mean(E_0, E_1, ..., E_K)를 기반으로 최종 사용자 및 아이템 임베딩을 계산합니다. 학습을 위해 `call`에 의해 호출되고 추론을 위해 `get_final_embeddings_for_recommendation`에 의해 호출됩니다.
        # 모든 사용자와 아이템에 대한 초기 (E^0) 임베딩 가져오기
        # tf.range는 [0, 1, ..., num_users-1] 및 [0, 1, ..., num_items-1] 시퀀스를 생성합니다.
        all_E0_user_embeddings = self.e0_user_embedding_layer(tf.range(self.num_users, dtype=tf.int32))
        all_E0_item_embeddings = self.e0_item_embedding_layer(tf.range(self.num_items, dtype=tf.int32))

        # 사용자 및 아이템 E^0 임베딩 연결: E_concat^0 = [E_user^0; E_item^0]
        # 모양: (num_users + num_items, embedding_dim)
        current_propagated_embeddings = self.concat_initial_embeddings([all_E0_user_embeddings, all_E0_item_embeddings])

        # 모든 레이어의 임베딩 저장 (E^0, E^1, ..., E^K)
        all_layer_wise_embeddings = [current_propagated_embeddings] # E^0으로 시작

        # K개 레이어의 그래프 전파 수행
        for gcn_layer in self.gcn_propagation_layers:
            current_propagated_embeddings = gcn_layer(current_propagated_embeddings) # E^(k+1) = A_norm * E^(k)
            all_layer_wise_embeddings.append(current_propagated_embeddings)

        # 평균 풀링을 사용하여 모든 레이어의 임베딩 결합 (LightGCN 논문에 따름)
        # 1. 모든 레이어의 임베딩 스택: 모양 (K+1, num_users + num_items, embedding_dim)
        if len(all_layer_wise_embeddings) > 1: # 전파가 발생한 경우 (NUM_LAYERS > 0)
            stacked_all_layer_embeddings = self.stack_layer_embeddings(all_layer_wise_embeddings)
            # 2. 레이어 간 평균 풀링 (axis=0): 모양 (num_users + num_items, embedding_dim)
            final_combined_embeddings = self.mean_pool_final_embeddings(stacked_all_layer_embeddings)
        else: # NUM_LAYERS = 0인 경우 E^0만 존재
            final_combined_embeddings = all_layer_wise_embeddings[0]

        # 결합된 텐서를 최종 사용자 및 아이템 임베딩으로 다시 분할
        final_user_embeddings_table = self.split_final_user_embeddings(final_combined_embeddings)
        final_item_embeddings_table = self.split_final_item_embeddings(final_combined_embeddings)

        return final_user_embeddings_table, final_item_embeddings_table

    def call(self, inputs):
        """
        Forward pass for training with BPR loss. # BPR 손실로 학습하기 위한 순방향 패스입니다.
        Args:
            inputs (list of tf.Tensor): A list containing three tensors:
                - user_input_indices: Batch of user indices.
                - positive_item_input_indices: Batch of positive item indices for corresponding users.
                - negative_item_input_indices: Batch of negative item indices for corresponding users.
        Returns:
            tf.Tensor: The difference between positive scores and negative scores (y_pred_diff for BPR loss).
        """ # (BPR 손실에 대한 y_pred_diff)
        user_input_indices, positive_item_input_indices, negative_item_input_indices = inputs

        # 모든 사용자와 아이템에 대한 최종 (전파되고 결합된) 임베딩 가져오기
        final_user_embeddings_table, final_item_embeddings_table = self._propagate_embeddings()

        # 배치의 특정 사용자, 긍정적 아이템 및 부정적 아이템에 대한 임베딩 수집
        # `tf.gather`는 입력 인덱스를 기반으로 임베딩 테이블에서 행을 선택합니다.
        user_batch_embeddings_gathered = self.gather_embeddings([final_user_embeddings_table, user_input_indices])
        pos_item_batch_embeddings_gathered = self.gather_embeddings([final_item_embeddings_table, positive_item_input_indices])
        neg_item_batch_embeddings_gathered = self.gather_embeddings([final_item_embeddings_table, negative_item_input_indices])

        # `gather`가 크기 1의 추가 차원을 추가한 경우 임베딩 재구성 (2D 매개변수 및 1D 인덱스에 대한 tf.gather에서 일반적)
        # 내적을 위해 (batch_size, embedding_dim)인지 확인합니다.
        user_batch_embeddings = self.reshape_gathered_embeddings(user_batch_embeddings_gathered)
        pos_item_batch_embeddings = self.reshape_gathered_embeddings(pos_item_batch_embeddings_gathered)
        neg_item_batch_embeddings = self.reshape_gathered_embeddings(neg_item_batch_embeddings_gathered)

        # (사용자, 긍정적_아이템) 쌍에 대한 내적 계산
        positive_scores = self.dot_product_pos([user_batch_embeddings, pos_item_batch_embeddings])
        # (사용자, 부정적_아이템) 쌍에 대한 내적 계산
        negative_scores = self.dot_product_neg([user_batch_embeddings, neg_item_batch_embeddings])

        # BPR 차이: positive_score - negative_score
        # 이 차이는 BPR 손실을 위해 log_sigmoid에 입력됩니다.
        return self.bpr_difference([positive_scores, negative_scores])

    def get_final_embeddings_for_recommendation(self):
        """
        Computes and returns the final (propagated and combined) embeddings for all users and items. # 모든 사용자와 아이템에 대한 최종 (전파되고 결합된) 임베딩을 계산하고 반환합니다.
        This method is intended for use during inference/recommendation generation.
        """ # 이 메소드는 추론/추천 생성 중에 사용하기 위한 것입니다.
        return self._propagate_embeddings()

# --- BPR 손실 함수 ---
def bpr_loss(_, y_pred_difference):
    """
    Bayesian Personalized Ranking (BPR) loss. # 베이즈 개인화 순위 (BPR) 손실입니다.
    Args:
        _ (tf.Tensor): True labels (ignored in BPR, usually ones).
        y_pred_difference (tf.Tensor): The difference between positive and negative item scores
                                     (output of the LightGCNModel's call method).
    Returns:
        tf.Tensor: The BPR loss value.
    """ # BPR 손실 값입니다.
    # BPR 손실은 다음을 최대화하는 것을 목표로 합니다: log(sigmoid(score_positive - score_negative))
    # 이는 다음을 최소화하는 것과 같습니다: -log(sigmoid(score_positive - score_negative))
    return -tf.reduce_mean(tf.math.log_sigmoid(y_pred_difference))

# --- 추천 생성 ---
# 사용자 한 명에 대한 Time Complexity:
# - 임베딩 전파 (미리 계산되지 않은 경우): O(N_edges_in_graph * EMBEDDING_DIM * NUM_LAYERS)
# - 모든 아이템에 대한 내적: O(N_items * EMBEDDING_DIM)
# - 정렬: O(N_items * log N_items)
def get_lightgcn_recommendations(user_id_original, user_encoder, item_encoder,
                                 trained_lightgcn_model, # 학습된 모델 인스턴스 전달
                                 num_recommendations=5):
    """
    Generates top-N recommendations for a given user using the trained LightGCN model. # 학습된 LightGCN 모델을 사용하여 주어진 사용자에 대한 상위 N개 추천을 생성합니다.
    """
    try:
        # 원본 사용자 ID를 인코딩된 정수 인덱스로 변환
        user_idx_encoded = user_encoder.transform([user_id_original])[0]
    except ValueError:
        print(f"Error: User ID '{user_id_original}' not found in user_encoder. Cannot generate recommendations.") # 오류: 사용자 ID '{user_id_original}'를 user_encoder에서 찾을 수 없습니다. 추천을 생성할 수 없습니다.
        return []

    # 학습된 모델에서 모든 사용자와 아이템에 대한 최종 전파된 임베딩 가져오기
    final_all_user_embeddings, final_all_item_embeddings = trained_lightgcn_model.get_final_embeddings_for_recommendation()

    # 대상 사용자에 대한 임베딩 가져오기
    target_user_embedding_vector = tf.expand_dims(final_all_user_embeddings[user_idx_encoded], axis=0) # 모양: (1, embedding_dim)

    # 이 사용자에 대한 모든 아이템의 점수를 내적으로 계산: UserEmbedding * AllItemEmbeddings^T
    # `tf.matmul` (transpose_b=True 사용)은 이를 효율적으로 달성합니다.
    all_item_scores_for_user = tf.matmul(target_user_embedding_vector, final_all_item_embeddings, transpose_b=True)
    all_item_scores_for_user_np = tf.squeeze(all_item_scores_for_user).numpy() # 1D 배열로 압축

    # 가장 높은 점수를 가진 아이템의 인덱스 가져오기
    # `np.argsort`는 오름차순으로 정렬하는 인덱스를 반환합니다.
    # `[-num_recommendations:]` 슬라이싱은 상위 N개 최대값을 가져옵니다.
    # `[::-1]`은 점수 내림차순으로 정렬하기 위해 역순으로 만듭니다.
    top_n_item_indices_encoded = np.argsort(all_item_scores_for_user_np)[-num_recommendations:][::-1]

    recommendations = []
    for item_idx_encoded in top_n_item_indices_encoded:
        original_item_id = item_encoder.inverse_transform([item_idx_encoded])[0] # inverse_transform은 배열을 예상함
        recommendations.append({
            'item_id': original_item_id,
            'score': all_item_scores_for_user_np[item_idx_encoded]
        })
    return recommendations

# --- 메인 실행 ---
# 전체 학습 Time Complexity: O(EPOCHS * (N_BPR_Triplets / BATCH_SIZE_BPR) * EMBEDDING_DIM + EPOCHS * N_edges_in_graph * EMBEDDING_DIM * NUM_LAYERS)
# 첫 번째 항은 BPR 삼중항 처리(내적, 뺄셈)를 위한 것입니다.
# 두 번째 항은 _propagate_embeddings 내의 그래프 전파 부분을 위한 것으로, 배치당 한 번 효과적으로 호출됩니다(샘플 간 공유되지만).
if __name__ == "__main__":
    print("--- LightGCN (TensorFlow/Keras) Example ---") # --- LightGCN (TensorFlow/Keras) 예제 ---

    print("\nLoading and preprocessing data...") # 데이터 로드 및 전처리 중...
    data_dict = load_and_preprocess_data()

    if data_dict:
        df_interactions = data_dict['df_interactions']
        user_encoder = data_dict['user_encoder']
        item_encoder = data_dict['item_encoder']
        num_users = data_dict['num_users']
        num_items = data_dict['num_items']
        norm_adj_mat_sp = data_dict['norm_adj_mat'] # SciPy 희소 행렬입니다.

        print(f"Unique users: {num_users}, Unique items: {num_items}") # 고유 사용자: {num_users}, 고유 아이템: {num_items}
        if norm_adj_mat_sp is not None:
            print(f"Normalized adjacency matrix shape: {norm_adj_mat_sp.shape}") # 정규화된 인접 행렬 모양: {norm_adj_mat_sp.shape}

        print("\nBuilding LightGCN model...") # LightGCN 모델 구축 중...
        # LightGCNModel 인스턴스화, SciPy 희소 행렬 전달
        lightgcn_model_instance = LightGCNModel(
            num_users, num_items, norm_adj_mat_sp, # 여기에 SciPy 희소 행렬 전달
            embedding_dim=EMBEDDING_DIM, num_layers=NUM_LAYERS, reg_weight=REG_WEIGHT
        )

        # BPR 손실 및 Adam 옵티마이저로 모델 컴파일
        lightgcn_model_instance.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss=bpr_loss)

        # BPR 학습 삼중항 생성 (user_idx, positive_item_idx, negative_item_idx)
        print("\nGenerating BPR triplets for training...") # 학습을 위한 BPR 삼중항 생성 중...
        train_users_bpr, train_pos_items_bpr, train_neg_items_bpr = get_bpr_triplets(
            df_interactions, num_items, user_col='user_idx', item_col='item_idx'
        )

        # BPR 손실을 위한 더미 y_true (손실 함수에서 무시되지만 Keras fit에서 예상함)
        # 값은 중요하지 않으며 배치 처리를 위한 모양만 중요합니다.
        dummy_y_true_for_bpr = np.ones_like(train_users_bpr, dtype=np.float32)

        print(f"\nStarting LightGCN model training for {EPOCHS} epochs...") # {EPOCHS} 에포크 동안 LightGCN 모델 학습 시작 중...
        # model.call 메소드의 입력은 리스트 또는 튜플입니다.
        history = lightgcn_model_instance.fit(
            [train_users_bpr, train_pos_items_bpr, train_neg_items_bpr], # 입력 목록
            dummy_y_true_for_bpr,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE_BPR,
            verbose=1 # 학습 진행 상황 표시
        )
        print("Model training completed.") # 모델 학습 완료.

        print("\n--- LightGCN Model Summary ---") # --- LightGCN 모델 요약 ---
        # 요약에는 사용자 정의 LightGCNLayer 인스턴스를 포함한 레이어가 표시됩니다.
        # 모델은 먼저 빌드되어야 하며, 이는 첫 번째 호출 시(예: fit 중) 발생합니다.
        lightgcn_model_instance.summary()

        # 예제 사용자에 대한 추천 생성
        if hasattr(user_encoder, 'classes_') and user_encoder.classes_.size > 0:
            # 인코더의 알려진 클래스에서 첫 번째 사용자를 예제로 선택
            target_user_original_id = user_encoder.classes_[0]

            print(f"\nGenerating recommendations for User ID (original): {target_user_original_id}...") # 사용자 ID (원본): {target_user_original_id}에 대한 추천 생성 중...
            recommendations = get_lightgcn_recommendations(
                target_user_original_id, user_encoder, item_encoder,
                lightgcn_model_instance, # 학습된 모델 인스턴스 전달
                num_recommendations=5
            )

            print("\nRecommended items:") # 추천 아이템:
            if recommendations:
                for rec in recommendations:
                    print(f"- Item ID (original): {rec['item_id']}, Predicted Score: {rec['score']:.4f}")
            else:
                print("No recommendations could be generated for this user.") # 이 사용자에 대한 추천을 생성할 수 없습니다.
        else:
            print("\nCannot generate recommendations as no users were encoded or available.") # 인코딩되었거나 사용 가능한 사용자가 없어 추천을 생성할 수 없습니다.
    else:
        print("\nData loading and preprocessing failed. Cannot proceed with LightGCN example.") # 데이터 로드 및 전처리에 실패했습니다. LightGCN 예제를 진행할 수 없습니다.

    print("\n--- LightGCN (TensorFlow/Keras) Example Finished ---") # --- LightGCN (TensorFlow/Keras) 예제 완료 ---
