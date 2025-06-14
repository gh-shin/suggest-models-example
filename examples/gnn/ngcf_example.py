# examples/gnn/ngcf_example.py
import pandas as pd
import numpy as np
import os
import sys
import tensorflow as tf
from tensorflow.keras.layers import Layer, Embedding, Dense, LeakyReLU, Concatenate, Multiply, Add
from tensorflow.keras.models import Model
from sklearn.preprocessing import LabelEncoder
# 실제 구현에서는 희소 행렬 연산과 그래프 데이터 처리 방법이 필요합니다:
# import scipy.sparse as sp

# --- NGCF (Neural Graph Collaborative Filtering): 상세 설명 ---
# NGCF (Neural Graph Collaborative Filtering)는 사용자-아이템 상호작용 그래프 구조를 명시적으로 활용하여
# 사용자와 아이템에 대한 임베딩을 학습하는 추천 모델입니다. 그래프에서 고차 연결성을 포착하는 것을 목표로 하며,
# 이는 직접적인 이웃을 넘어선 관계(예: 유사한 아이템을 좋아한 사용자와 유사한 사용자)를 고려한다는 의미입니다.
#
# 참고 자료:
# - 원본 논문: Wang, X., He, X., Wang, M., Feng, F., & Chua, T. S. (2019).
#   Neural graph collaborative filtering. In Proceedings of the 42nd international acm sigir conference
#   on Research and development in Information Retrieval (pp. 165-174).
#   링크: https://dl.acm.org/doi/10.1145/3331184.3331267
# - 예제 오픈소스 구현 (예: 원저자 또는 평판 좋은 출처):
#   (참고: 실제 사용을 위해서는 RecBole, Microsoft Recommenders 등과 같은 확립된 라이브러리를 참조하십시오.)
#
# 작동 방식:
# 1. 그래프 구성:
#    - 상호작용 데이터로부터 사용자-아이템 이분 그래프가 구성됩니다. 사용자와 아이템은 노드입니다.
#    - 사용자 'u'가 아이템 'i'와 상호작용한 경우 엣지 (u, i)가 존재합니다.
#
# 2. 임베딩 레이어 (초기 임베딩 E^(0)):
#    - 사용자 및 아이템 ID는 먼저 초기 밀집 임베딩 벡터(E_u^(0), E_i^(0))에 매핑됩니다.
#    - 이는 일반적으로 학습 중에 학습되며 기본 표현으로 사용됩니다.
#
# 3. 임베딩 전파 레이어 (고차 연결성 포착):
#    - NGCF는 그래프 전체에 임베딩을 전파하기 위해 여러 레이어를 사용합니다. 각 레이어는 그래프의 직접적인 이웃으로부터
#      정보를 집계하여 노드의 임베딩을 정제합니다.
#    - 메시지 구성: 사용자 'u'(또는 아이템 'i')에 대해 이웃 'v'(아이템 또는 사용자)로부터의 메시지는
#      일반적으로 다음에 기반하여 구성됩니다:
#        - 이웃의 임베딩 (이전 레이어의 E_v^(k-1)).
#        - 자가 임베딩 (사용자 'u'의 E_u^(k-1)).
#        - 상호작용 항, 종종 자가 임베딩과 이웃 임베딩의 요소별 곱 (Hadamard 곱): E_u^(k-1) * E_v^(k-1).
#          이 항은 선호도 또는 상호작용 강도를 모델링합니다.
#      메시지 m_{u<-v}는 다음과 같이 공식화될 수 있습니다:
#        m_{u<-v} = (1/sqrt(|N_u||N_v|)) * (W1 * E_v^(k-1) + W2 * (E_u^(k-1) * E_v^(k-1)))
#        여기서 N_u, N_v는 이웃 집합이고, W1, W2는 변환을 위한 학습 가능한 가중치 행렬입니다.
#        (1/sqrt(|N_u||N_v|)) 항은 그래프 라플라시안 정규화 계수입니다.
#    - 메시지 집계: 레이어 'k'에서 사용자 'u'에 대한 임베딩 E_u^(k)는 모든 이웃 'v'로부터의 메시지를 집계하고
#      이전 레이어의 자체 표현과 결합하여 형성됩니다.
#      일반적인 집계는 합계이며, 비선형 활성화(예: LeakyReLU)가 뒤따릅니다:
#        E_u^(k) = LeakyReLU( sum_{v in N_u} (m_{u<-v}) + m_{u<-u} )
#        여기서 m_{u<-u} = W1 * E_u^(k-1) (자가 메시지, 때로는 단순화되거나 다름).
#        원본 NGCF 공식은 다음과 같습니다:
#        E_u^(k) = LeakyReLU(W1*E_u^(k-1) + sum_{i in N_u} (1/sqrt(|N_u||N_i|)) * (W1*E_i^(k-1) + W2*(E_u^(k-1) * E_i^(k-1))) )
#        이 프로세스는 여러 레이어(예: 1~3개 레이어)에 대해 반복됩니다.
#
# 4. 예측 레이어:
#    - L개의 전파 레이어 후 각 사용자/아이템에 대해 여러 임베딩이 얻어집니다: (E^(0), E^(1), ..., E^(L)).
#    - 다른 레이어의 이러한 임베딩은 다른 차수의 연결성을 포착합니다.
#    - 최종 사용자 임베딩(U_final)과 아이템 임베딩(I_final)은 일반적으로 모든 레이어의 임베딩을 연결하여 형성됩니다:
#        U_final = Concat(E_u^(0), E_u^(1), ..., E_u^(L))
#        I_final = Concat(E_i^(0), E_i^(1), ..., E_i^(L))
#    - 사용자와 아이템 간의 예측된 상호작용 점수는 최종 연결된 임베딩의 내적을 취하여 계산됩니다:
#      score(u,i) = dot(U_final, I_final).
#
# 5. 학습:
#    - LightGCN과 유사하게 NGCF는 종종 BPR 손실과 같은 쌍별 손실로 학습되어
#      모델이 관찰된 아이템을 사용자에 대해 관찰되지 않은 (부정적인) 아이템보다 높게 순위를 매기도록 권장합니다.
#
# Pros (장점):
# - 명시적인 고차 정보: 전파 레이어는 그래프에서 다양한 길이의 경로를 따라 정보 흐름을 명시적으로 모델링하여
#   고차 협업 신호를 포착합니다.
# - 학습 가능한 전파: LightGCN의 매개변수 없는 전파와 달리 NGCF의 전파 레이어에는 학습 가능한 가중치 행렬(W1, W2)이
#   포함되어 있어 모델이 이웃 정보를 결합하고 변환하는 방법을 학습할 수 있습니다.
#
# Cons (단점):
# - 복잡성 및 계산 비용: 전파 레이어의 추가 가중치 행렬 및 요소별 곱으로 인해 LightGCN보다 복잡합니다.
#   이로 인해 학습 시간이 길어지고 매개변수가 많아질 수 있습니다.
# - 과적합 가능성: 복잡성이 증가하면 특히 희소 데이터셋에서 과적합되기 쉬울 수 있습니다.
# - 과도한 평활화 (Over-smoothing): 다른 GCN과 마찬가지로 많은 레이어를 사용하면 노드 임베딩이 너무 유사해질 수 있습니다.
# - LightGCN과의 비교: 추천을 위한 GCN 모델의 단순화(전파에서 W와 비선형성 제거)인 LightGCN은
#   실제로 NGCF보다 성능이 우수한 경우가 많으며, 이는 NGCF의 추가적인 복잡성이 항상 유익하지 않고
#   때로는 해로울 수 있음을 시사합니다.
#
# 일반적인 사용 사례:
# - 사용자-아이템 상호작용에서 복잡한 다중 홉 관계를 포착하는 것이 바람직한 추천 시나리오.
# - 그래프 구조를 통해 상호작용 유형 또는 강도를 명시적으로 모델링하는 것이 유익한 데이터셋.
# ---

# 프로젝트 루트를 sys.path에 동적으로 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- 모델 하이퍼파라미터 (개념적) ---
EMBEDDING_DIM = 64       # 초기 사용자/아이템 임베딩의 차원.
PROPAGATION_LAYERS = [64, 64, 64] # 각 NGCF 전파 레이어의 출력 차원.
NODE_DROPOUT_RATE = 0.1  # 전파 중 노드 임베딩에 대한 Dropout 비율 (사용되는 경우).
MESS_DROPOUT_RATE = 0.1  # 집계 중 메시지에 대한 Dropout 비율.
LEARNING_RATE = 0.001
BPR_BATCH_SIZE = 2048
EPOCHS = 5 # 예제를 위해 작게 설정
REG_WEIGHT_EMB = 1e-5 # 초기 임베딩에 대한 L2 정규화
REG_WEIGHT_LAYERS = 1e-5 # NGCF 레이어의 W1, W2에 대한 L2 정규화

# --- 플레이스홀더 데이터 로딩 및 전처리 ---
def load_and_preprocess_ngcf_data(base_filepath='data/dummy_interactions.csv'):
    """
    한국어: NGCF를 위한 상호작용 데이터 로딩, ID 인코딩 및 정규화된 인접 행렬(또는 기타 그래프 표현) 생성을 위한 플레이스홀더입니다.
    이는 LightGCN의 전처리와 유사하지만 특정 NGCF 구현 세부 정보에 따라 약간의 변형이 필요할 수 있습니다.

    Placeholder for loading interaction data, encoding IDs, and creating the
    normalized adjacency matrix (or other graph representations) for NGCF.
    This would be similar to LightGCN's preprocessing but might need slight
    variations based on specific NGCF implementation details.
    """
    print(f"Attempting to load data from: {base_filepath} (relative to project root)") # 데이터 로드 시도 중: {base_filepath} (프로젝트 루트 기준)
    filepath = os.path.join(project_root, base_filepath)
    if not os.path.exists(filepath):
        print(f"Error: Data file not found at {filepath}.") # 오류: {filepath}에서 데이터 파일을 찾을 수 없습니다.
        try:
            from data.generate_dummy_data import generate_dummy_data
            print("Attempting to generate dummy data...") # 더미 데이터 생성 시도 중...
            generate_dummy_data(num_users=50, num_items=30, num_interactions=500, generate_sequences=False)
            print("Dummy data generation script executed.") # 더미 데이터 생성 스크립트 실행됨.
        except Exception as e:
            print(f"Error during dummy data generation: {e}") # 더미 데이터 생성 중 오류: {e}
            return None
        if not os.path.exists(filepath):
            print(f"Error: Dummy data file still not found at {filepath} after generation.") # 오류: 생성 후에도 {filepath}에서 더미 데이터 파일을 찾을 수 없습니다.
            return None

    df = pd.read_csv(filepath)
    if df.empty:
        print("Error: Data file is empty.") # 오류: 데이터 파일이 비어 있습니다.
        return None

    user_encoder = LabelEncoder()
    df['user_idx'] = user_encoder.fit_transform(df['user_id'])
    item_encoder = LabelEncoder()
    df['item_idx'] = item_encoder.fit_transform(df['item_id'])

    num_users = df['user_idx'].nunique()
    num_items = df['item_idx'].nunique()

    print(f"Data loaded: {num_users} users, {num_items} items, {len(df)} interactions.") # 데이터 로드됨: 사용자 {num_users}명, 아이템 {num_items}개, 상호작용 {len(df)}개.

    # 인접 행렬 (A) 및 정규화된 버전 (A_norm 또는 유사) 플레이스홀더
    # NGCF의 경우 일반적으로 희소 인접 행렬이 필요합니다.
    # 정규화 (1/sqrt(|N_u||N_v|))는 종종 NGCFLayer 내에서 적용됩니다.
    # adj_matrix_sp = sp.csr_matrix(...) # df_interactions로부터 구성
    # 이 플레이스홀더의 경우 행렬에 대해 None을 반환합니다.
    print("Placeholder: Adjacency matrix construction would happen here.") # 플레이스홀더: 인접 행렬 구성이 여기서 수행됩니다.

    return {
        'df_interactions': df,
        'user_encoder': user_encoder,
        'item_encoder': item_encoder,
        'num_users': num_users,
        'num_items': num_items,
        'adj_matrix_sp': None # 실제 희소 인접 행렬에 대한 플레이스홀더
    }

def generate_bpr_triplets_placeholder(df_interactions, num_items):
    """Placeholder for BPR triplet generation.""" # BPR 삼중항 생성을 위한 플레이스홀더입니다.
    print("Placeholder: BPR triplet generation would occur here.") # 플레이스홀더: BPR 삼중항 생성이 여기서 발생합니다.
    # 예시: 올바른 유형이지만 작은 크기의 더미 np.array 반환
    dummy_users = np.array([0], dtype=np.int32)
    dummy_pos_items = np.array([0], dtype=np.int32)
    dummy_neg_items = np.array([1], dtype=np.int32)
    if num_items <=1: # neg_item이 pos_item과 다른지 확인
        print("Warning: Not enough items to generate distinct positive/negative BPR triplets.") # 경고: 고유한 긍정적/부정적 BPR 삼중항을 생성하기에 아이템이 충분하지 않습니다.
        if num_items == 1: # 아이템이 하나만 존재하는 경우
            dummy_neg_items = np.array([0], dtype=np.int32)

    return dummy_users, dummy_pos_items, dummy_neg_items

def bpr_loss_placeholder(_, y_pred_diff):
    """Placeholder for BPR loss function.""" # BPR 손실 함수를 위한 플레이스홀더입니다.
    print("Placeholder: BPR loss calculation would occur here (using y_pred_diff).") # 플레이스홀더: BPR 손실 계산이 여기서 발생합니다 (y_pred_diff 사용).
    return -tf.reduce_mean(tf.math.log_sigmoid(y_pred_diff))


# --- NGCF 모델 구성 요소 (플레이스홀더) ---
class NGCFLayer(Layer):
    """
    A single NGCF Embedding Propagation Layer (conceptual outline). # 단일 NGCF 임베딩 전파 레이어 (개념적 윤곽).
    Implements the message construction and aggregation steps.
    E_u^(k) = LeakyReLU( (W1*E_u^(k-1) + sum_{i in N_u} (1/sqrt(|N_u||N_i|)) * (W1*E_i^(k-1) + W2*(E_u^(k-1) * E_i^(k-1)))) )
    Simplified: E_u^(k) = LeakyReLU( Message_from_self_and_neighbors )
    """ # 메시지 구성 및 집계 단계를 구현합니다.
    def __init__(self, in_dim, out_dim, node_dropout=0.0, mess_dropout=0.0, reg_weight=1e-5, **kwargs):
        super(NGCFLayer, self).__init__(**kwargs)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.node_dropout_rate = node_dropout # 전파 전 노드 임베딩에 대한 Dropout
        self.mess_dropout_rate = mess_dropout # 메시지에 대한 Dropout

        # 이 레이어의 학습 가능한 가중치 행렬 W1 및 W2
        self.W1 = Dense(out_dim, use_bias=True, kernel_regularizer=tf.keras.regularizers.l2(reg_weight), name="W1_transform")
        self.W2 = Dense(out_dim, use_bias=True, kernel_regularizer=tf.keras.regularizers.l2(reg_weight), name="W2_interaction_transform")

        self.activation = LeakyReLU()
        # 사용되는 경우 노드 드롭아웃 및 메시지 드롭아웃 레이어가 여기에 정의됩니다.
        # 플레이스홀더의 단순성을 위해 call()에서 직접 적용하는 것을 스케치할 수 있습니다.

        print(f"Placeholder NGCFLayer initialized: in_dim={in_dim}, out_dim={out_dim}") # 플레이스홀더 NGCFLayer 초기화됨: in_dim={in_dim}, out_dim={out_dim}

    def call(self, current_user_embeddings, current_item_embeddings, adj_matrix_norm_sp):
        """
        Conceptual outline of the call method. # 호출 메소드의 개념적 윤곽입니다.
        `adj_matrix_norm_sp` would be the symmetrically normalized adjacency matrix (or similar for NGCF message passing).
        """ # `adj_matrix_norm_sp`는 대칭적으로 정규화된 인접 행렬(또는 NGCF 메시지 전달에 유사한 것)입니다.
        print(f"  Placeholder NGCFLayer.call(): Propagating embeddings...") #   플레이스홀더 NGCFLayer.call(): 임베딩 전파 중...

        # 1. 전파를 위해 사용자 및 아이템 임베딩 결합
        # E_concat = tf.concat([current_user_embeddings, current_item_embeddings], axis=0)

        # 2. 자가 메시지 (W1로 변환된 임베딩)
        # self_messages = self.W1(E_concat) # 예: W1 * E_u^(k-1)

        # 3. 이웃으로부터의 메시지 (플레이스홀더를 위해 단순화됨)
        # 이것이 핵심 NGCF 메시지 전달이 발생하는 곳입니다.
        # 각 노드 u에 대해 이웃 v로부터 집계:
        #   message_neighbor_part = W1(E_v)
        #   message_interaction_part = W2(E_u * E_v) # 요소별 곱
        #   aggregated_messages = sum over neighbors ( (message_neighbor_part + message_interaction_part) * laplacian_norm_factor )
        # 이는 일반적으로 (정규화된) 인접 행렬과의 희소 행렬 곱셈을 포함합니다.
        # 예제 개념적 스케치 (실제 희소 연산 없이는 실행 불가):
        # ego_embeddings = E_concat
        # side_embeddings = E_concat # 자가 루프 및 이웃용
        #
        # transformed_ego_embeddings = self.W1(ego_embeddings) # W1 * E_u
        # transformed_side_embeddings = self.W1(side_embeddings) # W1 * E_i
        #
        # # W1 * E_i (또는 E_v)의 전파
        # propagated_neighbor_embeddings = tf.sparse.sparse_dense_matmul(adj_matrix_norm_sp, transformed_side_embeddings)
        #
        # # 상호작용 항: E_u * E_i (요소별)
        # # 이는 브로드캐스팅 또는 그래프 상호작용을 위한 특정 연산의 신중한 처리가 필요합니다.
        # # 단순화를 위해 sum_neighbors(W2 * (E_u * E_i))를 계산하는 방법이 있다고 가정합니다.
        # # 이는 E_u를 확장하고, 이웃 E_i와 곱한 다음, 변환하고 합산하는 것을 포함할 수 있습니다.
        # # 이 부분은 TF 희소 연산으로 올바르게 구현하기 가장 복잡합니다.
        # interaction_messages_summed = tf.zeros_like(ego_embeddings) # 합산된 상호작용 메시지에 대한 플레이스홀더

        # 이 플레이스홀더의 경우 변환된 임베딩을 그대로 전달합니다.
        # 실제 구현에서는 훨씬 더 복잡합니다.
        next_user_embeddings = self.activation(self.W1(current_user_embeddings)) # 단순화된 플레이스홀더
        next_item_embeddings = self.activation(self.W1(current_item_embeddings)) # 단순화된 플레이스홀더

        print(f"  Placeholder NGCFLayer.call(): Embeddings processed (conceptually).") #   플레이스홀더 NGCFLayer.call(): 임베딩 처리됨 (개념적으로).
        return next_user_embeddings, next_item_embeddings

    def get_config(self):
        config = super().get_config()
        config.update({
            "in_dim": self.in_dim,
            "out_dim": self.out_dim,
            "node_dropout": self.node_dropout_rate,
            "mess_dropout": self.mess_dropout_rate
        })
        return config

class NGCFModel(Model):
    """
    NGCF Model (conceptual outline). # NGCF 모델 (개념적 윤곽).
    Combines initial embeddings with multiple NGCF propagation layers.
    Final embeddings are concatenations from all layers.
    """ # 초기 임베딩과 여러 NGCF 전파 레이어를 결합합니다. 최종 임베딩은 모든 레이어의 연결입니다.
    def __init__(self, num_users, num_items, adj_matrix_sp, # 그래프용 플레이스홀더
                 embedding_dim=EMBEDDING_DIM, propagation_layer_dims=PROPAGATION_LAYERS,
                 node_dropout=NODE_DROPOUT_RATE, mess_dropout=MESS_DROPOUT_RATE, reg_emb=REG_WEIGHT_EMB, reg_layers=REG_WEIGHT_LAYERS, **kwargs):
        super(NGCFModel, self).__init__(**kwargs)
        self.num_users = num_users
        self.num_items = num_items
        self.adj_matrix_sp = adj_matrix_sp # 레이어에서 사용하기 위해 저장 (개념적으로)

        # 초기 (E^0) 임베딩
        self.user_embedding_E0 = Embedding(num_users, embedding_dim, name="user_E0_embedding",
                                           embeddings_regularizer=tf.keras.regularizers.l2(reg_emb))
        self.item_embedding_E0 = Embedding(num_items, embedding_dim, name="item_E0_embedding",
                                           embeddings_regularizer=tf.keras.regularizers.l2(reg_emb))

        # NGCF 전파 레이어
        self.ngcf_layers = []
        current_dim = embedding_dim
        for layer_idx, layer_out_dim in enumerate(propagation_layer_dims):
            self.ngcf_layers.append(
                NGCFLayer(current_dim, layer_out_dim, node_dropout, mess_dropout, reg_layers, name=f"ngcf_layer_{layer_idx+1}")
            )
            current_dim = layer_out_dim # 다음 레이어의 입력 차원은 현재 레이어의 출력 차원

        self.concat_embeddings = Concatenate(axis=-1, name="concat_final_embeddings")
        print(f"Placeholder NGCFModel initialized with {len(self.ngcf_layers)} propagation layers.") # 플레이스홀더 NGCFModel이 {len(self.ngcf_layers)}개의 전파 레이어로 초기화되었습니다.
        print(f"Initial embedding dim: {embedding_dim}, Propagation layer output dims: {propagation_layer_dims}") # 초기 임베딩 차원: {embedding_dim}, 전파 레이어 출력 차원: {propagation_layer_dims}

    def get_all_layer_embeddings(self):
        """Helper to get embeddings from all layers (E0 up to EL).""" # 모든 레이어(E0부터 EL까지)의 임베딩을 가져오는 헬퍼입니다.
        all_user_layer_embeddings = [self.user_embedding_E0.embeddings] # 사용자를 위한 E0
        all_item_layer_embeddings = [self.item_embedding_E0.embeddings] # 아이템을 위한 E0

        current_user_embs = self.user_embedding_E0.embeddings
        current_item_embs = self.item_embedding_E0.embeddings

        for ngcf_layer in self.ngcf_layers:
            # 현재 임베딩과 (개념적) 인접 행렬 전달
            next_user_embs, next_item_embs = ngcf_layer(current_user_embs, current_item_embs, self.adj_matrix_sp)
            all_user_layer_embeddings.append(next_user_embs)
            all_item_layer_embeddings.append(next_item_embs)
            current_user_embs, current_item_embs = next_user_embs, next_item_embs

        # 최종 표현을 위해 모든 레이어의 임베딩 연결
        final_user_embeddings = self.concat_embeddings(all_user_layer_embeddings)
        final_item_embeddings = self.concat_embeddings(all_item_layer_embeddings)
        return final_user_embeddings, final_item_embeddings

    def call(self, inputs, training=False):
        """
        Forward pass for BPR training (conceptual outline). # BPR 학습을 위한 순방향 패스 (개념적 윤곽).
        inputs: [user_indices, positive_item_indices, negative_item_indices]
        """
        user_indices, pos_item_indices, neg_item_indices = inputs
        print(f"  Placeholder NGCFModel.call(): Processing BPR triplet...") #   플레이스홀더 NGCFModel.call(): BPR 삼중항 처리 중...

        final_user_embeddings_table, final_item_embeddings_table = self.get_all_layer_embeddings()

        # 배치에 대한 임베딩 수집
        user_embs = tf.gather(final_user_embeddings_table, user_indices)
        pos_item_embs = tf.gather(final_item_embeddings_table, pos_item_indices)
        neg_item_embs = tf.gather(final_item_embeddings_table, neg_item_indices)

        # 점수 계산 (내적)
        pos_scores = tf.reduce_sum(Multiply()([user_embs, pos_item_embs]), axis=1, keepdims=True)
        neg_scores = tf.reduce_sum(Multiply()([user_embs, neg_item_embs]), axis=1, keepdims=True)

        print(f"  Placeholder NGCFModel.call(): Scores calculated.") #   플레이스홀더 NGCFModel.call(): 점수 계산됨.
        return pos_scores - neg_scores # BPR 손실에 대한 차이

    def get_final_embeddings_for_recommendation(self):
        """For generating recommendations, get the final concatenated embeddings.""" # 추천 생성을 위해 최종 연결된 임베딩을 가져옵니다.
        print("Placeholder NGCFModel.get_final_embeddings_for_recommendation(): Fetching final embeddings...") # 플레이스홀더 NGCFModel.get_final_embeddings_for_recommendation(): 최종 임베딩 가져오는 중...
        return self.get_all_layer_embeddings()


# --- 메인 실행 블록 ---
if __name__ == "__main__":
    print("NGCF (Neural Graph Collaborative Filtering) Example - Conceptual Outline") # NGCF (Neural Graph Collaborative Filtering) 예제 - 개념적 개요
    print("="*70)
    print("This script provides a conceptual overview and structural outline of an NGCF model.") # 이 스크립트는 NGCF 모델의 개념적 개요와 구조적 윤곽을 제공합니다.
    print("It is NOT a runnable or fully implemented NGCF model. Key components like graph operations") # 실행 가능하거나 완전히 구현된 NGCF 모델이 아닙니다. 그래프 연산과 같은 주요 구성 요소
    print("within NGCFLayer and proper data handling for graph structures are simplified placeholders.") # NGCFLayer 내 및 그래프 구조에 대한 적절한 데이터 처리는 단순화된 플레이스홀더입니다.
    print("Refer to the original paper and established libraries for complete implementations.") # 완전한 구현을 위해서는 원본 논문 및 확립된 라이브러리를 참조하십시오.
    print("="*70 + "\n")

    # 1. 플레이스홀더 데이터 로드 및 전처리
    print("Step 1: Loading and preprocessing data (conceptual)...") # 단계 1: 개념적 데이터 로드 및 전처리...
    data_dict = load_and_preprocess_ngcf_data()

    if data_dict:
        df_interactions = data_dict['df_interactions']
        user_encoder = data_dict['user_encoder']
        item_encoder = data_dict['item_encoder']
        num_users = data_dict['num_users']
        num_items = data_dict['num_items']
        adj_matrix_sp_placeholder = data_dict['adj_matrix_sp'] # 현재 플레이스홀더에서는 None입니다.

        # 2. NGCF 모델 구축 (플레이스홀더 구조)
        print("\nStep 2: Building NGCF Model structure (conceptual)...") # 단계 2: NGCF 모델 구조 구축 (개념적)...
        ngcf_model_placeholder = NGCFModel(
            num_users, num_items, adj_matrix_sp_placeholder,
            embedding_dim=EMBEDDING_DIM,
            propagation_layer_dims=PROPAGATION_LAYERS
        )

        # 개념적으로 모델 컴파일
        ngcf_model_placeholder.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
            loss=bpr_loss_placeholder # 플레이스홀더 손실 사용
        )
        print("Conceptual NGCF model built and compiled with placeholder loss.") # 개념적 NGCF 모델 구축 및 플레이스홀더 손실로 컴파일됨.

        # 요약 출력 시도 (서브클래스 모델의 경우 빌드 전에 제한될 수 있음)
        # 요약을 얻으려면 호출하거나 입력 모양으로 build()를 정의해야 합니다.
        # 지금은 설명만 합니다.
        print("\nConceptual Model Structure:") # 개념적 모델 구조:
        print(f"  - Initial User Embeddings E0: (num_users, {EMBEDDING_DIM})") #   - 초기 사용자 임베딩 E0: (사용자_수, {EMBEDDING_DIM})
        print(f"  - Initial Item Embeddings E0: (num_items, {EMBEDDING_DIM})") #   - 초기 아이템 임베딩 E0: (아이템_수, {EMBEDDING_DIM})
        for i, p_dim in enumerate(PROPAGATION_LAYERS):
            print(f"  - NGCF Layer {i+1}: Outputs {p_dim}-dim embeddings") #   - NGCF 레이어 {i+1}: {p_dim}차원 임베딩 출력
        final_emb_dim = EMBEDDING_DIM + sum(PROPAGATION_LAYERS)
        print(f"  - Final User/Item Embeddings: Concatenated from all layers, total dim approx {final_emb_dim}") #   - 최종 사용자/아이템 임베딩: 모든 레이어에서 연결됨, 총 차원 약 {final_emb_dim}
        print(f"  - Prediction: Dot product of final User and Item embeddings.") #   - 예측: 최종 사용자 및 아이템 임베딩의 내적.

        # 3. BPR 학습 삼중항 생성 (플레이스홀더)
        print("\nStep 3: Generating BPR training triplets (conceptual)...") # 단계 3: BPR 학습 삼중항 생성 (개념적)...
        train_users, train_pos_items, train_neg_items = generate_bpr_triplets_placeholder(df_interactions, num_items)
        print(f"Placeholder BPR triplets generated (Users: {train_users.shape}, PosItems: {train_pos_items.shape}, NegItems: {train_neg_items.shape})") # 플레이스홀더 BPR 삼중항 생성됨 (사용자: {train_users.shape}, 긍정적 아이템: {train_pos_items.shape}, 부정적 아이템: {train_neg_items.shape})

        # 4. 모델 학습 (개념적 - 실제로 플레이스홀더를 학습하지 않음)
        print("\nStep 4: Model Training (conceptual)...") # 단계 4: 모델 학습 (개념적)...
        print(f"  (This would involve feeding triplets to model.fit() for {EPOCHS} epochs with BPR loss)") #   (이는 BPR 손실을 사용하여 {EPOCHS} 에포크 동안 model.fit()에 삼중항을 공급하는 것을 포함합니다)
        print("  Skipping actual training for this placeholder script.") #   이 플레이스홀더 스크립트에 대한 실제 학습 건너뛰기.

        # 5. 추천 생성 (개념적)
        print("\nStep 5: Generating Recommendations (conceptual)...") # 단계 5: 추천 생성 (개념적)...
        if num_users > 0:
            sample_user_original_id = user_encoder.classes_[0] # 예제 원본 사용자 ID 가져오기
            print(f"  (Conceptual: For user {sample_user_original_id}, compute their final embedding,") #   (개념적: 사용자 {sample_user_original_id}의 경우 최종 임베딩을 계산하고,
            print(f"   compute all item final embeddings, calculate dot products, and rank.)") #    모든 아이템 최종 임베딩을 계산하고, 내적을 계산하고, 순위를 매깁니다.)
            print(f"  (Skipping actual recommendation ranking for this placeholder script.)") #   (이 플레이스홀더 스크립트에 대한 실제 추천 순위 지정 건너뛰기.)
        else:
            print("  No users available in dummy data to generate example recommendations for.") #   더미 데이터에 예제 추천을 생성할 사용자가 없습니다.

    else:
        print("\nData loading placeholder failed. Cannot proceed with conceptual outline.") # 데이터 로딩 플레이스홀더 실패. 개념적 개요를 진행할 수 없습니다.

    print("\n" + "="*70)
    print("NGCF Conceptual Outline Example Finished.") # NGCF 개념적 개요 예제 완료.
    print("Reminder: This is a structural guide, not a working implementation.") # 알림: 이것은 구조적 가이드이며 작동하는 구현이 아닙니다.
    print("="*70)
