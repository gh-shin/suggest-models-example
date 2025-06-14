# examples/gnn/gat_example.py
import pandas as pd
import numpy as np
import os
import sys
import tensorflow as tf
# 실제 GAT의 경우 일반적으로 다음을 사용합니다:
# from tensorflow.keras.layers import Layer, Dense, Dropout, LeakyReLU
# from tensorflow.keras.models import Model
# from sklearn.preprocessing import LabelEncoder
# import collections

# --- GAT (Graph Attention Network): 상세 설명 ---
# GAT (Graph Attention Network)은 그래프 컨볼루션 과정에 어텐션 메커니즘을 통합한 그래프 신경망의 한 유형입니다.
# GCN처럼 이웃에 고정된 균일한 가중치를 할당하거나 단순한 합/평균을 학습하는 대신,
# GAT는 노드가 특징을 집계할 때 다른 이웃에 다른 수준의 중요도(어텐션 점수)를 할당할 수 있도록 합니다.
# 이는 모델을 더욱 표현력 있게 만들고, 특히 차수가 다양하거나 일부 이웃이 다른 이웃보다 더 관련성이 높은 노드에서
# 종종 더 나은 성능으로 이어집니다.
#
# 참고 자료:
# - 원본 논문: Veličković, P., Cucurull, G., Casanova, A., Romero, A., Liò, P., & Bengio, Y. (2018).
#   Graph Attention Networks. In International Conference on Learning Representations (ICLR).
#   링크: https://arxiv.org/abs/1710.10903
#
# 핵심 구성 요소 및 개념:
# 1. 그래프 표현:
#    - 노드 (사용자/아이템), 엣지 (상호작용), 특징 Matrix (초기 노드 특징의 경우 X 또는 H^(0)).
#    - 인접성 정보 (예: 엣지 리스트 또는 인접 행렬)는 이웃을 식별하는 데 중요합니다.
#
# 2. 어텐션 메커니즘 (GAT 레이어의 핵심):
#    - 대상 노드 'i'와 각 이웃 'j'에 대해:
#        a. 특징 변환: 노드 특징 (이전 레이어의 h_i 및 h_j)은 먼저 공유 선형 변환 (가중치 행렬 W)에 의해 변환됩니다: Wh_i 및 Wh_j.
#        b. 어텐션 계수 (e_ij): 어텐션 메커니즘 'a' (예: 단일 레이어 피드포워드 네트워크)는
#           노드 j의 특징이 노드 i에 미치는 중요성을 나타내는 정규화되지 않은 어텐션 점수 e_ij를 계산합니다:
#           e_ij = a(Wh_i, Wh_j)
#           일반적으로 다음과 같이 구현됩니다: e_ij = LeakyReLU( att_kernel^T * Concat(Wh_i, Wh_j) ),
#           여기서 att_kernel은 어텐션 메커니즘을 위한 학습 가능한 가중치 벡터입니다.
#        c. 마스크된 어텐션: 이러한 어텐션 계수 e_ij는 그래프에서 노드 i의 실제 이웃인 노드 j에 대해서만 계산됩니다 (즉, (j,i)가 엣지임).
#        d. Softmax 정규화: 어텐션 점수 e_ij는 노드 'i'의 모든 이웃 'j'에 대해 소프트맥스 함수를 사용하여 정규화되어
#           정규화된 어텐션 계수 α_ij를 얻습니다:
#           α_ij = softmax_j(e_ij) = exp(e_ij) / sum_{k in Neighbors(i)} (exp(e_ik))
#
# 3. 가중 집계:
#    - 노드 'i'에 대한 새로운 표현 h'_i는 이웃의 변환된 특징을 정규화된 어텐션 계수 α_ij로 가중치를 부여하여 집계함으로써 계산됩니다:
#      h'_i = σ( sum_{j in Neighbors(i)} (α_ij * Wh_j) )
#      여기서 σ는 비선형 활성화 함수입니다 (예: ReLU, ELU).
#
# 4. 멀티-헤드 어텐션:
#    - 학습을 안정화하고 이웃 중요도의 다양한 측면을 포착하기 위해 GAT는 일반적으로 멀티-헤드 어텐션을 사용합니다.
#    - K개의 독립적인 어텐션 메커니즘(헤드)이 위에서 설명한 프로세스를 실행합니다 (단계 2a-2c, 최종 σ 없는 3단계).
#    - 그들의 출력 (각 노드에 대한 K개의 임베딩 벡터)은 다음 중 한 가지 방법으로 결합됩니다:
#        - 연결 (Concatenation): h'_i = Concat(h'_i_head1, ..., h'_i_headK) 후 최종 선형 변환 및 활성화.
#        - 평균화 (Averaging): h'_i = σ( (1/K) * sum_{k=1 to K} (sum_{j in N(i)} (α_ij_headk * W_headk * h_j)) ) (마지막 레이어에서).
#
# 5. 레이어 스태킹:
#    - 여러 GAT 레이어를 쌓을 수 있습니다. 한 레이어의 출력 h'가 다음 레이어의 입력 h가 됩니다.
#    - 이를 통해 모델은 더 큰 이웃(k-hop 이웃)으로부터 정보를 포착할 수 있습니다.
#
# 6. 추천에의 적용:
#    - 사용자와 아이템은 노드입니다. 상호작용은 엣지입니다. 초기 특징은 학습된 ID 임베딩 또는 콘텐츠 특징일 수 있습니다.
#    - GAT는 각 임베딩이 상호작용 그래프에서 더 "중요한" 이웃에 의해 더 큰 영향을 받는 사용자 및 아이템 임베딩을 학습합니다.
#    - 이러한 최종 임베딩은 예측에 사용됩니다 (예: 순위 지정을 위한 내적).
#
# Pros (장점):
# - 차등적 중요도 할당: 이웃에 차등적으로 가중치를 부여하는 방법을 학습하여, 노드 차수를 기반으로 고정된 정규화를 사용하는
#   GCN보다 표현력이 뛰어납니다.
# - 다양한 차수를 암묵적으로 처리: 어텐션 메커니즘은 이웃 수가 다른 노드에 적응합니다.
# - 우수한 성능: 다양한 그래프 학습 벤치마크에서 종종 최첨단 결과를 달성합니다.
# - 귀납적 능력: GraphSAGE와 마찬가지로 GAT 레이어는 로컬 이웃과 공유 가중치에 대해 작동하므로
#   본질적으로 귀납적입니다 (특징과 이웃이 제공되면 보이지 않는 노드에 대한 임베딩을 생성할 수 있음).
#
# Cons (단점):
# - 계산 비용이 더 많이 듬: 특히 멀티-헤드 어텐션을 사용할 경우 노드당 모든 이웃 쌍에 대한 어텐션 계수 계산은
#   GCN의 더 간단한 집계보다 계산 집약적일 수 있습니다.
# - 더 많은 하이퍼파라미터: 헤드 수, 어텐션 드롭아웃 등은 모델 튜닝 복잡성을 증가시킵니다.
# - 과적합 가능성: 모델 복잡성이 증가하면 더 많은 데이터 또는 더 강력한 정규화가 필요할 수 있습니다.
#
# 일반적인 사용 사례:
# - 다양한 도메인에서의 노드 분류, 그래프 분류 및 링크 예측.
# - 사용자와 아이템 간의 미묘한 영향을 모델링하는 것이 유익한 추천 시스템.
# - 다른 이웃이 노드 표현에 다르게 기여해야 하는 상황.
# ---

# 프로젝트 루트를 sys.path에 동적으로 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- 모델 하이퍼파라미터 (개념적) ---
INPUT_FEATURE_DIM = 64      # 초기 노드 특징의 차원.
GAT_LAYER_UNITS = [64, 64]  # 각 GAT 레이어의 출력 유닛 (각 헤드별).
NUM_HEADS_PER_LAYER = [4, 1] # GAT 레이어 1, GAT 레이어 2의 어텐션 헤드 수. 마지막 레이어는 종종 1개의 헤드를 사용하거나 평균화합니다.
ACTIVATION_GAT = 'elu'      # GAT 레이어에 일반적인 활성화 함수 (ELU 또는 LeakyReLU).
DROPOUT_GAT_FEATURES = 0.1  # GAT 레이어 입력 특징에 대한 Dropout.
DROPOUT_GAT_ATTENTION = 0.1 # 어텐션 계수에 대한 Dropout.
LEARNING_RATE = 0.001
EPOCHS = 3 # 예제를 위해 작게 설정
# BATCH_SIZE (학습 설정에 따라 다름, 예: 노드 수준 또는 그래프 수준 작업)

# --- GAT를 위한 플레이스홀더 데이터 로딩 ---
def load_gat_data(interactions_filepath_rel='data/dummy_interactions.csv',
                  features_filepath_rel='data/dummy_item_features_gat.csv'): # 아이템 특징을 가정함
    """
    한국어: GAT를 위한 상호작용 그래프 데이터 및 노드 특징 로딩을 위한 플레이스홀더입니다.
    GAT 요구 사항:
    1. 그래프 구조 (이웃 식별을 위한 엣지 리스트 또는 인접 행렬).
    2. 모든 노드에 대한 초기 특징.

    Placeholder for loading interaction graph data and node features for GAT.
    GAT requires:
    1. Graph structure (edge list or adjacency matrix for identifying neighbors).
    2. Initial features for all nodes.
    """
    print(f"Attempting to load data for GAT...") # GAT 데이터 로드 시도 중...
    interactions_filepath = os.path.join(project_root, interactions_filepath_rel)
    features_filepath = os.path.join(project_root, features_filepath_rel) # 아이템 특징 예시

    files_exist = True
    if not os.path.exists(interactions_filepath):
        print(f"Warning: Interactions file not found at {interactions_filepath}.") # 경고: {interactions_filepath}에서 상호작용 파일을 찾을 수 없습니다.
        files_exist = False
    if not os.path.exists(features_filepath):
        print(f"Warning: Item features file not found at {features_filepath}.") # 경고: {features_filepath}에서 아이템 특징 파일을 찾을 수 없습니다.
        files_exist = False

    if not files_exist:
        print("Attempting to generate dummy data for GAT (interactions and item features)...") # GAT용 더미 데이터(상호작용 및 아이템 특징) 생성 시도 중...
        try:
            from data.generate_dummy_data import generate_dummy_data
            generate_dummy_data(num_users=50, num_items=30, num_interactions=300,
                                generate_sequences=False,
                                generate_generic_item_features=True, item_feature_dim=INPUT_FEATURE_DIM)
            print("Dummy data generation script executed.") # 더미 데이터 생성 스크립트 실행됨.
        except Exception as e:
            print(f"Error during dummy data generation: {e}") # 더미 데이터 생성 중 오류: {e}
            return None
        if not os.path.exists(interactions_filepath) or not os.path.exists(features_filepath):
            print("Error: One or both dummy data files still not found after generation attempt.") # 오류: 생성 시도 후에도 하나 또는 두 개의 더미 데이터 파일을 찾을 수 없습니다.
            return None
        print("Dummy data files should now be available.") # 이제 더미 데이터 파일을 사용할 수 있습니다.

    df_interactions = pd.read_csv(interactions_filepath)
    df_item_features = pd.read_csv(features_filepath)

    if df_interactions.empty or df_item_features.empty:
        print("Error: Interaction or item feature data is empty.") # 오류: 상호작용 또는 아이템 특징 데이터가 비어 있습니다.
        return None

    # 사용자 및 아이템 ID 인코딩 (개념적, 잠재적 OOV 처리를 위해 StringLookup 사용)
    user_encoder = tf.keras.layers.StringLookup(mask_token=None)
    user_encoder.adapt(df_interactions['user_id'].astype(str).unique())
    num_users = user_encoder.vocabulary_size()

    item_encoder = tf.keras.layers.StringLookup(mask_token=None)
    all_item_ids = pd.concat([
        df_interactions['item_id'].astype(str),
        df_item_features['item_id'].astype(str)
    ]).unique()
    item_encoder.adapt(all_item_ids)
    num_items = item_encoder.vocabulary_size()

    num_total_nodes = num_users + num_items
    print(f"Data loaded: {num_users} users, {num_items} items (potential OOV included), {len(df_interactions)} interactions.") # 데이터 로드됨: 사용자 {num_users}명, 아이템 {num_items}개 (잠재적 OOV 포함), 상호작용 {len(df_interactions)}개.

    # 엣지 인덱스 플레이스홀더 (소스_노드_인덱스, 타겟_노드_인덱스)
    # 이는 df_interactions로부터 구성되며, 원본 ID를 전역 인코딩된 ID에 매핑합니다.
    # 사용자: 0 ~ num_users-1. 아이템: num_users ~ num_users + num_items - 1.
    print("Placeholder: Edge index construction (for neighbor lookup) would happen here.") # 플레이스홀더: 엣지 인덱스 구성 (이웃 조회를 위해)이 여기서 수행됩니다.
    # 예시: edge_index = [[u1,u2,...], [i1,i2,...]] 여기서 u1은 i1과 상호작용합니다.
    # GAT의 경우, 종종 각 노드의 이웃이 누구인지 알기만 하면 됩니다.
    # 리스트의 리스트 또는 인접 리스트 딕셔너리가 이웃 조회를 위해 일반적으로 사용됩니다.
    dummy_edge_index_placeholder = np.random.randint(0, num_total_nodes, size=(2, len(df_interactions)))

    # 결합된 사용자 및 아이템 특징 플레이스홀더 (H^(0))
    print(f"Placeholder: Initial node feature matrix H0 (shape: ({num_total_nodes}, {INPUT_FEATURE_DIM})) would be prepared here.") # 플레이스홀더: 초기 노드 특징 Matrix H0 (모양: ({num_total_nodes}, {INPUT_FEATURE_DIM}))이 여기서 준비됩니다.
    # 사용자 특징은 학습 가능한 임베딩 또는 명시적 특징일 수 있습니다.
    user_features_placeholder = np.random.rand(num_users, INPUT_FEATURE_DIM).astype(np.float32)
    # 로드된 맵의 아이템 특징, num_items까지 모든 아이템에 항목이 있는지 확인합니다.
    df_item_features['item_idx_encoded'] = item_encoder(df_item_features['item_id'].astype(str)).numpy()
    item_features_map = {
        row['item_idx_encoded']: row.drop(['item_id', 'item_idx_encoded']).values.astype(np.float32)
        for _, row in df_item_features.iterrows()
    }
    item_features_list_for_stacking = []
    for i in range(num_items): # item_encoder가 0..num_items-1에 매핑한다고 가정
        if i in item_features_map:
            item_features_list_for_stacking.append(item_features_map[i][:INPUT_FEATURE_DIM]) # 올바른 차원 확인
        else:
            item_features_list_for_stacking.append(np.zeros(INPUT_FEATURE_DIM, dtype=np.float32))
    item_features_array_placeholder = np.array(item_features_list_for_stacking)

    if item_features_array_placeholder.shape[1] != INPUT_FEATURE_DIM: # 특징 차원 일관성 확인
        print(f"Warning: Item feature dimension mismatch. Expected {INPUT_FEATURE_DIM}, got {item_features_array_placeholder.shape[1]}. Adjusting...") # 경고: 아이템 특징 차원 불일치. 예상: {INPUT_FEATURE_DIM}, 실제: {item_features_array_placeholder.shape[1]}. 조정 중...
        # 특징 차원이 정렬되지 않는 경우 대체 (예: 더미 데이터 문제)
        item_features_array_placeholder = np.random.rand(num_items, INPUT_FEATURE_DIM).astype(np.float32)


    initial_node_features_placeholder = np.vstack([
        user_features_placeholder,
        item_features_array_placeholder
    ]).astype(np.float32) if item_features_array_placeholder.size > 0 else user_features_placeholder

    # 총 노드 대 특징에 대한 안전 점검
    if initial_node_features_placeholder.shape[0] != num_total_nodes:
         print(f"Warning: Feature matrix node count ({initial_node_features_placeholder.shape[0]}) " # 경고: 특징 Matrix 노드 수 ({initial_node_features_placeholder.shape[0]})
               f"differs from total encoded nodes ({num_total_nodes}). Using random features for safety.") # 가 총 인코딩된 노드 수 ({num_total_nodes})와 다릅니다. 안전을 위해 임의 특징 사용 중.
         initial_node_features_placeholder = np.random.rand(num_total_nodes, INPUT_FEATURE_DIM).astype(np.float32)


    return {
        'df_interactions': df_interactions,
        'user_encoder': user_encoder,
        'item_encoder': item_encoder,
        'num_users': num_users,
        'num_items': num_items,
        'num_total_nodes': num_total_nodes,
        'edge_index_placeholder': dummy_edge_index_placeholder,
        'initial_node_features_placeholder': initial_node_features_placeholder
    }

# --- GAT 모델 구성 요소 (플레이스홀더) ---
class GraphAttentionLayerPlaceholder(tf.keras.layers.Layer):
    """Conceptual placeholder for a Graph Attention (GAT) layer.""" # 그래프 어텐션(GAT) 레이어의 개념적 플레이스홀더입니다.
    def __init__(self, output_dim, num_heads, activation='elu', feature_dropout_rate=0.1, attention_dropout_rate=0.1, **kwargs):
        super(GraphAttentionLayerPlaceholder, self).__init__(**kwargs)
        self.output_dim = output_dim # 헤드당 출력 특징 차원
        self.num_heads = num_heads
        self.activation = tf.keras.layers.Activation(activation)
        self.feature_dropout = tf.keras.layers.Dropout(feature_dropout_rate)
        self.attention_dropout = tf.keras.layers.Dropout(attention_dropout_rate)

        # 특징 변환을 위한 가중치 행렬 플레이스홀더 (헤드당 하나)
        # 그리고 어텐션 메커니즘 가중치 (또한 헤드당 한 세트)
        self.W_heads = []
        self.att_kernels_self_heads = [] # e_ij = LeakyReLU(a^T [Wh_i || Wh_j]) 에서 a_l 용
        self.att_kernels_neigh_heads = []# e_ij 에서 a_r 용
        for i in range(num_heads):
            self.W_heads.append(tf.keras.layers.Dense(output_dim, use_bias=False, name=f"{self.name}_W_head_{i}"))
            # 어텐션 메커니즘: 일반적으로 a_l_T * Wh_i 와 a_r_T * Wh_j 를 합산한 후 LeakyReLU 적용
            self.att_kernels_self_heads.append(tf.keras.layers.Dense(1, use_bias=False, name=f"{self.name}_att_self_head_{i}"))
            self.att_kernels_neigh_heads.append(tf.keras.layers.Dense(1, use_bias=False, name=f"{self.name}_att_neigh_head_{i}"))

        self.leaky_relu = tf.keras.layers.LeakyReLU(alpha=0.2)
        print(f"GraphAttentionLayerPlaceholder initialized (output_dim_per_head={output_dim}, num_heads={num_heads}).") # GraphAttentionLayerPlaceholder 초기화됨 (헤드당_출력_차원={output_dim}, 헤드_수={num_heads}).

    def call(self, inputs, training=False):
        """
        Conceptual call method for a GAT layer. # GAT 레이어의 개념적 호출 메소드입니다.
        inputs: A list/tuple [node_features, edge_index]
                - node_features (H_in): Tensor of shape (num_nodes, input_feature_dim)
                - edge_index: Tensor of shape (2, num_edges) representing source and target nodes of edges.
                              Used to determine neighbors for attention calculation.
        """
        node_features, edge_index = inputs
        print(f"  GraphAttentionLayerPlaceholder.call(): Processing features for {tf.shape(node_features)[0]} nodes...") # GraphAttentionLayerPlaceholder.call(): {tf.shape(node_features)[0]}개 노드에 대한 특징 처리 중...

        # 입력 특징에 드롭아웃 적용
        node_features_dropped = self.feature_dropout(node_features, training=training)

        head_outputs = []
        for i in range(self.num_heads):
            # 1. 노드 특징 선형 변환 (모든 i에 대해 Wh_i)
            transformed_features_head_i = self.W_heads[i](node_features_dropped) # (노드_수, 출력_차원)

            # 2. 어텐션 계수 e_ij 계산 (개념적으로)
            #    이를 위해서는 엣지를 반복하거나 고급 희소 연산을 사용해야 합니다.
            #    e_ij = LeakyReLU( att_self_head(Wh_i) + att_neigh_head(Wh_j) )
            #    플레이스홀더의 경우, 일부 어텐션 점수를 얻는다고 가정하여 이를 시뮬레이션합니다.
            #    실제 구현에서는 이 부분이 가장 복잡합니다.
            #    엣지의 소스 노드: edge_index[0], 엣지의 타겟 노드: edge_index[1]
            #    att_self_terms = self.att_kernels_self_heads[i](transformed_features_head_i) # (노드_수, 1)
            #    att_neigh_terms = self.att_kernels_neigh_heads[i](transformed_features_head_i) # (노드_수, 1)
            #
            #    # 엣지에 대한 용어 수집
            #    edge_att_self = tf.gather(att_self_terms, edge_index[0]) # 엣지의 소스 노드와 관련된 어텐션 점수
            #    edge_att_neigh = tf.gather(att_neigh_terms, edge_index[1]) # 엣지의 타겟 노드와 관련된 어텐션 점수
            #    edge_attention_unnorm = self.leaky_relu(edge_att_self + edge_att_neigh) # (엣지_수, 1)

            # 3. 소프트맥스를 사용하여 어텐션 계수 정규화 (개념적으로, 노드 이웃별)
            #    alpha_ij = softmax_j(e_ij)
            #    이는 희소 이웃에 대해 tf.math.segment_softmax 또는 유사한 것을 사용합니다.
            #    플레이스홀더: 집계의 단순성을 위해 균일한 어텐션을 가정합니다.

            # 4. 어텐션으로 가중치 부여된 이웃 특징 집계 (개념적으로)
            #    h'_i = sum_j (alpha_ij * Wh_j)
            #    플레이스홀더: 변환된 특징의 단순 평균 집계 (알파가 균일한 경우 GCN과 유사한 평균 집계 시뮬레이션)
            #    실제 GAT는 `alpha_ij * Wh_j`와 함께 `tf.math.segment_sum`을 사용합니다.
            #    이 플레이스홀더의 경우, 집계된 것처럼 변환된 특징을 반환합니다.
            #    이는 주요 단순화입니다.
            if edge_index is not None and tf.size(edge_index) > 0: # 그래프에 엣지가 있는 경우
                 # 매우 조잡한 집계 플레이스홀더: 엣지 대상의 변환된 특징을 평균냅니다.
                 # 이는 특정 이웃으로부터의 GAT의 어텐션 가중 합계를 정확하게 모델링하지 않습니다.
                aggregated_features_head_i = tf.math.unsorted_segment_mean(
                    data=tf.gather(transformed_features_head_i, edge_index[1]), # 엣지 대상(이웃)의 특징
                    segment_ids=edge_index[0], # 엣지 소스(업데이트되는 노드)별 그룹화
                    num_segments=tf.shape(node_features)[0] # 총 노드 수
                )
            else: # 엣지가 없거나 격리된 노드 - 자체 변환된 특징 사용
                aggregated_features_head_i = transformed_features_head_i # 또는 0

            head_outputs.append(aggregated_features_head_i)
            print(f"    Head {i+1} conceptual output shape: {aggregated_features_head_i.shape if isinstance(aggregated_features_head_i, tf.Tensor) else 'N/A'}") # 헤드 {i+1} 개념적 출력 모양: ...

        # 5. 여러 헤드의 출력 결합
        if self.num_heads > 1:
            # 다른 헤드의 특징 연결
            final_output_embeddings = Concatenate(axis=-1)(head_outputs) # (노드_수, 헤드_수 * 출력_차원)
        else:
            # 단일 헤드, 해당 출력 사용
            final_output_embeddings = head_outputs[0] # (노드_수, 출력_차원)

        final_output_embeddings = self.activation(final_output_embeddings)
        print(f"    -> Combined output embeddings shape from layer: {final_output_embeddings.shape if isinstance(final_output_embeddings, tf.Tensor) else 'N/A'}") # -> 레이어의 결합된 출력 임베딩 모양: ...
        return final_output_embeddings

    def get_config(self):
        config = super().get_config()
        config.update({
            "output_dim": self.output_dim,
            "num_heads": self.num_heads,
            "activation": self.activation.name if hasattr(self.activation, 'name') else str(self.activation),
        })
        return config

class GATModelPlaceholder(tf.keras.Model):
    """Conceptual placeholder for a GAT-based recommendation model.""" # GAT 기반 추천 모델의 개념적 플레이스홀더입니다.
    def __init__(self, num_total_nodes, initial_feature_dim,
                 gat_layer_units=GAT_LAYER_UNITS, num_heads_per_layer=NUM_HEADS_PER_LAYER,
                 use_initial_node_embeddings=False, final_embedding_dim=None, **kwargs):
        super(GATModelPlaceholder, self).__init__(**kwargs)
        self.num_total_nodes = num_total_nodes
        self.use_initial_node_embeddings = use_initial_node_embeddings

        if self.use_initial_node_embeddings:
            # 참이면 초기 특징은 학습 가능한 ID 임베딩입니다.
            self.initial_node_embeddings = Embedding(num_total_nodes, initial_feature_dim, name="initial_node_embeddings")
            current_dim = initial_feature_dim
        else:
            # 거짓이면 `initial_feature_dim`이 외부에서 제공된 특징의 차원이라고 가정합니다.
            self.initial_node_embeddings = None
            current_dim = initial_feature_dim

        self.gat_layers = []
        for i, units in enumerate(gat_layer_units):
            num_heads = num_heads_per_layer[i]
            # 헤드를 연결하는 경우 GAT 레이어의 출력 차원은 units * num_heads입니다 (마지막 레이어이고 평균화하는 경우가 아니면).
            # 이 플레이스홀더의 경우 `units`가 헤드 조합 후의 차원이라고 가정합니다 (예: 최종 Dense 또는 평균화 헤드의 경우).
            # 또는 더 일반적으로 `units`는 헤드당 출력 차원이고 연결이 발생합니다.
            # 여기서 `units`는 해당되는 경우 헤드 조합 후 레이어의 대상 출력 차원을 의미한다고 가정합니다.
            # 연결하는 경우 GATLayer 내부의 실제 Dense 레이어는 W에 대해 units/num_heads를 대상으로 합니다.
            # 단순화를 위해 GATLayerPlaceholder의 output_dim을 해당 레이어의 최종 출력 차원으로 가정합니다.
            self.gat_layers.append(
                GraphAttentionLayerPlaceholder(output_dim=units, num_heads=num_heads,
                                               activation=ACTIVATION_GAT if i < len(gat_layer_units) -1 else 'linear', # 일반적으로 마지막 GAT 레이어는 선형
                                               name=f"gat_layer_{i+1}")
            )
            current_dim = units # GAT 레이어에서 연결/평균 및 잠재적 최종 Dense 후

        self.final_embedding_dim = current_dim
        if final_embedding_dim and current_dim != final_embedding_dim:
            self.final_projection = Dense(final_embedding_dim, name="final_embedding_projection_gat")
            self.final_embedding_dim = final_embedding_dim
            print(f"  Added final projection layer to {final_embedding_dim}-dim for GAT model.") # GAT 모델에 {final_embedding_dim}차원으로 최종 프로젝션 레이어 추가됨.
        else:
            self.final_projection = None


        print(f"Placeholder GATModel initialized with {len(self.gat_layers)} GAT layers.") # 플레이스홀더 GATModel이 {len(self.gat_layers)}개의 GAT 레이어로 초기화되었습니다.
        print(f"  Layer output units (final dim after head combination): {gat_layer_units}") # 레이어 출력 유닛 (헤드 조합 후 최종 차원): {gat_layer_units}
        print(f"  Attention heads per layer: {num_heads_per_layer}") # 레이어당 어텐션 헤드: {num_heads_per_layer}
        print(f"  Final output embedding dimension: {self.final_embedding_dim}") # 최종 출력 임베딩 차원: {self.final_embedding_dim}


    def call(self, inputs, training=False):
        """
        Conceptual forward pass for GAT. # GAT의 개념적 순방향 패스입니다.
        inputs: A list/tuple [initial_node_features_or_ids, edge_index]
        """
        print(f"  Placeholder GATModel.call(): Processing graph data...") # 플레이스홀더 GATModel.call(): 그래프 데이터 처리 중...
        initial_node_repr, edge_index = inputs # Repr은 use_initial_node_embeddings인 경우 ID일 수 있고, 그렇지 않으면 특징 Matrix입니다.

        if self.use_initial_node_embeddings:
            # initial_node_repr이 노드 ID [0...num_total_nodes-1]를 포함한다고 가정
            # 이는 모든 노드 ID의 텐서가 되어 초기 임베딩을 얻습니다.
            h_current = self.initial_node_embeddings(initial_node_repr)
        else:
            # initial_node_repr이 실제 특징 Matrix H^(0)이라고 가정
            h_current = initial_node_repr

        print(f"    Initial node features/embeddings shape: {h_current.shape}") # 초기 노드 특징/임베딩 모양: {h_current.shape}

        for i, gat_layer in enumerate(self.gat_layers):
            h_current = gat_layer([h_current, edge_index], training=training)
            print(f"    -> Output shape after GAT Layer {i+1} ({gat_layer.name}): {h_current.shape}") # -> GAT 레이어 {i+1} ({gat_layer.name}) 후 출력 모양: {h_current.shape}

        final_node_embeddings = h_current
        if self.final_projection:
            final_node_embeddings = self.final_projection(final_node_embeddings)
            print(f"    -> Output shape after final_projection: {final_node_embeddings.shape}") # -> final_projection 후 출력 모양: {final_node_embeddings.shape}

        print(f"  Placeholder GATModel.call(): Final node embeddings computed (shape: {final_node_embeddings.shape}).") # 플레이스홀더 GATModel.call(): 최종 노드 임베딩 계산됨 (모양: {final_node_embeddings.shape}).
        return final_node_embeddings

# --- 메인 실행 블록 ---
if __name__ == "__main__":
    print("GAT (Graph Attention Network) for Recommendations - Conceptual Outline") # GAT (Graph Attention Network) 추천 - 개념적 개요
    print("="*80)
    print("This script provides a conceptual overview and structural outline of a GAT model.") # 이 스크립트는 GAT 모델의 개념적 개요와 구조적 윤곽을 제공합니다.
    print("It is NOT a runnable or fully implemented GAT model. Key components like efficient computation") # 실행 가능하거나 완전히 구현된 GAT 모델이 아닙니다. 효율적인 계산과 같은 주요 구성 요소
    print("of attention coefficients over sparse neighborhoods and multi-head combination details") # 희소 이웃에 대한 어텐션 계수 및 멀티-헤드 조합 세부 정보
    print("are significantly simplified placeholders.") # 는 상당히 단순화된 플레이스홀더입니다.
    print("Refer to the original GAT paper and specialized graph learning libraries for robust implementations.") # 강력한 구현을 위해서는 원본 GAT 논문 및 특수 그래프 학습 라이브러리를 참조하십시오.
    print("="*80 + "\n")

    # 1. 개념적 데이터 로드
    print("Step 1: Loading conceptual data (interactions and node features)...") # 단계 1: 개념적 데이터 로드 (상호작용 및 노드 특징)...
    data_dict = load_gat_data()

    if data_dict:
        num_users = data_dict['num_users']
        num_items = data_dict['num_items']
        num_total_nodes = data_dict['num_total_nodes']
        edge_index_placeholder = data_dict['edge_index_placeholder']
        initial_node_features_placeholder = data_dict['initial_node_features_placeholder']

        # 2. GAT 모델 구축 (플레이스홀더 구조)
        print("\nStep 2: Building GAT Model structure (conceptual)...") # 단계 2: GAT 모델 구조 구축 (개념적)...
        gat_model_placeholder = GATModelPlaceholder(
            num_total_nodes=num_total_nodes,
            initial_feature_dim=INPUT_FEATURE_DIM,
            gat_layer_units=GAT_LAYER_UNITS,
            num_heads_per_layer=NUM_HEADS_PER_LAYER,
            use_initial_node_embeddings=False, # 특징이 직접 전달된다고 가정
            final_embedding_dim=GAT_LAYER_UNITS[-1] # 마지막 GAT 레이어의 출력
        )
        # 개념적 컴파일 (손실은 다운스트림 작업에 따라 다름, 예: 링크 예측, 노드 분류)
        # gat_model_placeholder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE), loss="some_graph_loss")
        print("Conceptual GAT model built.") # 개념적 GAT 모델 구축됨.
        print("\nConceptual Model Structure:") # 개념적 모델 구조:
        if gat_model_placeholder.use_initial_node_embeddings:
            print(f"  - Initial Node Embeddings (learnable): ({num_total_nodes}, {INPUT_FEATURE_DIM})") #   - 초기 노드 임베딩 (학습 가능): ({num_total_nodes}, {INPUT_FEATURE_DIM})
        else:
            print(f"  - Initial Node Features (provided): ({num_total_nodes}, {INPUT_FEATURE_DIM})") #   - 초기 노드 특징 (제공됨): ({num_total_nodes}, {INPUT_FEATURE_DIM})

        current_input_dim_desc = INPUT_FEATURE_DIM
        for i, units in enumerate(GAT_LAYER_UNITS):
            heads = NUM_HEADS_PER_LAYER[i]
            # 참고: 헤드 조합 후 GAT 레이어의 출력 차원은 `units` (평균화 또는 프로젝션하는 경우)
            # 또는 `units * heads` (연결하고 `units`가 헤드당 차원인 경우)일 수 있습니다.
            # 플레이스홀더 GATLayer는 단순화되었습니다. `units`가 레이어의 최종 출력 차원이라고 가정합니다.
            print(f"  - GAT Layer {i+1}: Input ({current_input_dim_desc}-dim), Heads: {heads}, Output: ({units}-dim) features, Activation: {ACTIVATION_GAT}") #   - GAT 레이어 {i+1}: 입력 ({current_input_dim_desc}차원), 헤드: {heads}, 출력: ({units}차원) 특징, 활성화: {ACTIVATION_GAT}
            current_input_dim_desc = units
        if gat_model_placeholder.final_projection:
             print(f"  - Final Projection Layer to: {gat_model_placeholder.final_embedding_dim}-dim") #   - 최종 프로젝션 레이어 대상: {gat_model_placeholder.final_embedding_dim}차원


        # 3. 모델 학습 / 임베딩 생성 (개념적 - 전체 그래프 패스)
        print("\nStep 3: Generating Node Embeddings (conceptual full graph forward pass)...") # 단계 3: 노드 임베딩 생성 (개념적 전체 그래프 순방향 패스)...
        print(f"  (This would involve defining a loss based on a downstream task, e.g., link prediction for interactions,") #   (이는 다운스트림 작업(예: 상호작용에 대한 링크 예측)을 기반으로 손실을 정의하고,
        print(f"   and using model.fit() or a custom training loop with the graph structure.)") #    그래프 구조와 함께 model.fit() 또는 사용자 정의 학습 루프를 사용하는 것을 포함합니다.)

        # 모든 노드 임베딩을 얻기 위한 개념적 순방향 패스
        # 실제 시나리오에서는 `initial_node_features_placeholder`가 실제 특징이고,
        # `edge_index_placeholder`가 실제 그래프 연결성입니다.
        all_node_embeddings_conceptual = gat_model_placeholder(
            [initial_node_features_placeholder, edge_index_placeholder]
        )
        print(f"  Conceptual all_node_embeddings computed (shape: {all_node_embeddings_conceptual.shape}).") #   개념적 all_node_embeddings 계산됨 (모양: {all_node_embeddings_conceptual.shape}).
        print("  Skipping actual training for this placeholder script.") #   이 플레이스홀더 스크립트에 대한 실제 학습 건너뛰기.

        # 4. 추천을 위한 임베딩 사용 (개념적)
        print("\nStep 4: Using Embeddings for Recommendations (conceptual)...") # 단계 4: 추천을 위한 임베딩 사용 (개념적)...
        if num_users > 0 and num_items > 0 and all_node_embeddings_conceptual is not None:
            # 원본 노드 인덱싱을 기반으로 최종 사용자 및 아이템 임베딩 추출
            # 이는 사용자가 노드 0부터 num_users-1까지, 아이템이 num_users부터 num_total_nodes-1까지라고 가정합니다.
            final_user_embeddings_conceptual = all_node_embeddings_conceptual[:num_users, :]
            final_item_embeddings_conceptual = all_node_embeddings_conceptual[num_users:, :]
            print(f"  Conceptual final user embeddings shape: {final_user_embeddings_conceptual.shape}") #   개념적 최종 사용자 임베딩 모양: {final_user_embeddings_conceptual.shape}
            print(f"  Conceptual final item embeddings shape: {final_item_embeddings_conceptual.shape}") #   개념적 최종 아이템 임베딩 모양: {final_item_embeddings_conceptual.shape}

            sample_user_idx = 0 # 예시: 첫 번째 사용자 (전역 인덱스)
            print(f"  (Conceptual: For user_idx {sample_user_idx}, take their embedding from final_user_embeddings_conceptual,") #   (개념적: user_idx {sample_user_idx}의 경우 final_user_embeddings_conceptual에서 해당 임베딩을 가져와,
            print(f"   compute dot products with all item embeddings from final_item_embeddings_conceptual, and rank.)") #    final_item_embeddings_conceptual의 모든 아이템 임베딩과 내적을 계산하고 순위를 매깁니다.)
            print(f"  (Skipping actual recommendation ranking for this placeholder script.)") #   (이 플레이스홀더 스크립트에 대한 실제 추천 순위 지정 건너뛰기.)
        else:
            print("  Not enough user/item data or embeddings not generated to demonstrate conceptual recommendations.") #   개념적 추천을 시연하기에 충분한 사용자/아이템 데이터 또는 임베딩이 생성되지 않았습니다.

    else:
        print("\nData loading placeholder failed. Cannot proceed with GAT conceptual outline.") # 데이터 로딩 플레이스홀더 실패. GAT 개념적 개요를 진행할 수 없습니다.

    print("\n" + "="*80)
    print("GAT for Recommendations Conceptual Outline Example Finished.") # GAT 추천 개념적 개요 예제 완료.
    print("Reminder: This is a structural guide, not a working implementation of GAT.") # 알림: 이것은 구조적 가이드이며 GAT의 작동하는 구현이 아닙니다.
    print("="*80)
