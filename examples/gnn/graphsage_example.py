# examples/gnn/graphsage_example.py
import pandas as pd
import numpy as np
import os
import sys
import tensorflow as tf
# 실제 GraphSAGE의 경우 일반적으로 다음을 사용합니다:
# from tensorflow.keras.layers import Layer, Dense, Activation
# from tensorflow.keras.models import Model
# from sklearn.preprocessing import LabelEncoder
# import collections # 이웃 샘플링용

# --- GraphSAGE (Graph SAmple and aggreGatE): 상세 설명 ---
# GraphSAGE는 대규모 그래프에서 노드 임베딩을 학습하기 위한 귀납적 프레임워크입니다.
# "귀납적"이라는 것은 학습 중에 보지 못한 노드에 대해서도 특징과 로컬 이웃 구조가 주어지면 임베딩을 생성할 수 있음을 의미합니다.
# 이는 고정된 노드 집합에 대한 임베딩을 학습하는 전이적 방법(예: 원본 GCN)과 대조됩니다.
#
# 참고 자료:
# - 원본 논문: Hamilton, W. L., Ying, R., & Leskovec, J. (2017).
#   Inductive Representation Learning on Large Graphs. In Advances in Neural Information Processing Systems (NIPS).
#   링크: https://arxiv.org/abs/1706.02216
#
# 핵심 구성 요소 및 개념:
# 1. 귀납적 학습: GraphSAGE는 모든 노드에 대한 임베딩을 생성할 수 있는 함수를 학습하므로,
#    노드가 자주 추가되거나 변경되는 동적 그래프에 적합합니다.
#
# 2. 이웃 샘플링:
#    - 임베딩을 계산해야 하는 각 노드(주어진 레이어에서)에 대해 GraphSAGE는 해당 이웃의 고정 크기 집합을 균일하게 샘플링합니다.
#    - 이 고정 크기 샘플링은 노드의 실제 차수에 관계없이 각 집계 단계가 일관된 계산 공간을 갖도록 보장합니다.
#      이는 확장성 및 배치 처리에 중요합니다.
#    - 샘플링을 통해 전체 이웃(또는 전체 그래프 인접 행렬)에 대한 연산을 피하며, 이는 대규모 그래프의 병목 현상입니다.
#
# 3. 집계 함수:
#    - GraphSAGE는 샘플링된 이웃으로부터 정보를 결합하기 위해 여러 집계 함수를 제안합니다:
#        a. 평균 집계기 (Mean Aggregator): 이전 레이어에서 샘플링된 이웃의 특징 벡터의 요소별 평균을 취합니다.
#           그런 다음 이 집계된 벡터는 종종 이전 레이어의 대상 노드 자체 특징 벡터와 연결됩니다.
#        b. LSTM 집계기 (LSTM Aggregator): 이웃 순서가 중요할 수 있는 그래프(종종 이웃은 순서 없는 집합으로 처리되지만)의 경우,
#           LSTM을 이웃 특징의 임의 순열에 적용할 수 있습니다.
#        c. 풀링 집계기 (Pooling Aggregator): 신경망으로 변환한 후 이웃 특징에 대해 요소별 최대 풀링(또는 평균 풀링) 연산을 적용합니다.
#
# 4. 임베딩 업데이트 (레이어에서의 순방향 전파):
#    - 레이어 'k'의 대상 노드 'v'에 대해:
#        i. 'v'의 N_k개 이웃을 샘플링합니다.
#        ii. 이전 레이어(k-1)에서 이러한 N_k개 이웃의 임베딩을 가져옵니다: {h_u^(k-1) for u in Sampled_Neighbors(v)}.
#        iii. 집계 함수를 사용하여 이러한 이웃 임베딩을 집계하여 h_Neighbors(v)^(k)를 얻습니다.
#        iv. 이전 레이어의 대상 노드 자체 임베딩 h_v^(k-1)을 집계된 이웃 벡터와 연결합니다: Concat(h_v^(k-1), h_Neighbors(v)^(k)).
#        v. 이 연결된 벡터를 비선형 활성화 함수(예: ReLU)가 있는 완전 연결 레이어를 통과시켜
#           레이어 'k'에서 노드 'v'에 대한 새 임베딩 h_v^(k)를 얻습니다.
#           h_v^(k) = σ( W^(k) * Concat(h_v^(k-1), h_Neighbors(v)^(k)) + b^(k) )
#
# 5. 레이어 스태킹 (깊이):
#    - 여러 GraphSAGE 레이어(샘플링 및 집계 단계)가 쌓입니다.
#    - K개의 레이어가 쌓이면 노드의 최종 임베딩은 K-홉 이웃으로부터 정보를 포착합니다 (샘플링으로 인해 확률적으로).
#    - 레이어 (k-1)의 출력은 레이어 'k'의 입력 특징으로 사용됩니다.
#
# 6. 추천에의 적용:
#    - 사용자와 아이템은 그래프의 노드로 처리됩니다 (종종 이분형이지만 GraphSAGE는 일반 그래프를 처리할 수 있음).
#    - 상호작용은 엣지를 형성합니다. 노드는 특징을 가질 수 있습니다 (예: 사용자 인구 통계, 아이템 콘텐츠).
#    - 모델은 사용자 및 아이템 임베딩을 생성하는 함수를 학습합니다.
#    - 이러한 임베딩은 상호작용 확률 예측(예: 내적 및 시그모이드를 통해) 또는 사용자를 위한 아이템 순위 지정과 같은
#      다운스트림 작업에 사용됩니다.
#
# Pros (장점):
# - 귀납적: 전체 모델을 재학습하지 않고도 새 사용자/아이템에 대한 임베딩을 생성할 수 있습니다.
#   (특징과 (사용 가능한 경우) 로컬 이웃을 샘플링할 수 있는 한).
# - 확장성: 고정 크기 이웃 샘플링은 GCN과 같이 전체 이웃 집계가 너무 비용이 많이 드는 대규모 그래프에 효율적입니다.
#   미니 배치 학습을 허용합니다.
# - 유연한 집계기: 집계기(평균, LSTM, 풀링) 선택은 다양한 그래프 속성 및 계산 예산에 적응할 수 있는 유연성을 제공합니다.
# - 노드 특징 활용 가능: 노드 특징을 임베딩 프로세스에 자연스럽게 통합합니다.
#
# Cons (단점):
# - 성능 의존성: 샘플링 전략(이웃 수, 레이어 수) 및 집계 함수 선택은 성능에 큰 영향을 미치며 신중한 튜닝이 필요합니다.
# - 구현 복잡성: 특히 미니 배치 학습을 위한 이웃 샘플링 부분은 GCN 또는 LightGCN과 같은 더 간단한 모델보다
#   효율적으로 구현하기가 더 복잡할 수 있습니다.
# - 귀납성을 위한 특징 의존성: 귀납적이지만 새 노드에 대한 좋은 임베딩을 생성하는 능력은
#   종종 해당 새 노드에 대한 의미 있는 노드 특징의 존재에 의존합니다.
#
# 일반적인 사용 사례:
# - 새 노드가 자주 추가되는 대규모 동적 그래프에서의 노드 표현 학습.
# - 귀납적 기능이 중요한 추천 시스템 (예: 새 사용자/아이템).
# - 미니 배치로 학습할 수 있는 확장 가능한 GNN이 필요한 애플리케이션.
# ---

# 프로젝트 루트를 sys.path에 동적으로 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- 모델 하이퍼파라미터 (개념적) ---
INPUT_FEATURE_DIM = 64      # 초기 노드 특징의 차원.
SAGE_LAYER_UNITS = [64, 64] # 각 GraphSAGE 레이어의 출력 유닛.
NUM_NEIGHBORS_PER_LAYER = [10, 5] # 레이어 1, 레이어 2 등에 대해 샘플링할 이웃 수.
AGGREGATOR_TYPE = 'mean'    # 'mean', 'gcn', 'lstm', 'pool' (개념적)
ACTIVATION_SAGE = 'relu'
LEARNING_RATE = 0.001
EPOCHS = 3 # 예제를 위해 작게 설정
BATCH_SIZE = 512

# --- GraphSAGE를 위한 플레이스홀더 데이터 로딩 ---
def load_graphsage_data(interactions_filepath_rel='data/dummy_interactions.csv',
                        features_filepath_rel='data/dummy_item_features_graphsage.csv'): # 현재는 아이템 특징을 가정함
    """
    한국어: GraphSAGE를 위한 상호작용 그래프 데이터 및 노드 특징 로딩을 위한 플레이스홀더입니다.
    GraphSAGE 요구 사항:
    1. 그래프 구조 (이웃 샘플링을 위한 인접 리스트).
    2. 모든 노드(사용자 및 아이템)에 대한 초기 특징.

    Placeholder for loading interaction graph data and node features for GraphSAGE.
    GraphSAGE needs:
    1. Graph structure (adjacency list for neighbor sampling).
    2. Initial features for all nodes (users and items).
    """
    print(f"Attempting to load data for GraphSAGE...") # GraphSAGE 데이터 로드 시도 중...
    interactions_filepath = os.path.join(project_root, interactions_filepath_rel)
    # GraphSAGE의 경우 사용자 와 아이템 모두 이상적으로 특징을 가져야 합니다.
    # 이 예제는 아이템 특징으로 단순화하지만 실제 사례에는 별도의 user_features.csv가 있을 수 있습니다.
    features_filepath = os.path.join(project_root, features_filepath_rel)

    files_exist = True
    if not os.path.exists(interactions_filepath):
        print(f"Warning: Interactions file not found at {interactions_filepath}.") # 경고: {interactions_filepath}에서 상호작용 파일을 찾을 수 없습니다.
        files_exist = False
    if not os.path.exists(features_filepath): # 이 플레이스홀더에 대해 아이템 특징을 가정함
        print(f"Warning: Item features file not found at {features_filepath}.") # 경고: {features_filepath}에서 아이템 특징 파일을 찾을 수 없습니다.
        files_exist = False

    if not files_exist:
        print("Attempting to generate dummy data for GraphSAGE (interactions and item features)...") # GraphSAGE용 더미 데이터(상호작용 및 아이템 특징) 생성 시도 중...
        try:
            from data.generate_dummy_data import generate_dummy_data
            generate_dummy_data(num_users=50, num_items=30, num_interactions=300,
                                generate_sequences=False, generate_item_features_pinsage=False, # PinSage 특정 없음
                                generate_generic_item_features=True, item_feature_dim=INPUT_FEATURE_DIM) # 일반 특징
            print("Dummy data generation script executed.") # 더미 데이터 생성 스크립트 실행됨.
        except Exception as e:
            print(f"Error during dummy data generation: {e}") # 더미 데이터 생성 중 오류: {e}
            return None
        # 파일 다시 확인
        if not os.path.exists(interactions_filepath) or not os.path.exists(features_filepath):
            print("Error: One or both dummy data files still not found after generation attempt.") # 오류: 생성 시도 후에도 하나 또는 두 개의 더미 데이터 파일을 찾을 수 없습니다.
            return None
        print("Dummy data files should now be available.") # 이제 더미 데이터 파일을 사용할 수 있습니다.

    df_interactions = pd.read_csv(interactions_filepath)
    df_item_features = pd.read_csv(features_filepath) # item_id, feature_1, ..., feature_N을 가정함

    if df_interactions.empty or df_item_features.empty:
        print("Error: Interaction or item feature data is empty.") # 오류: 상호작용 또는 아이템 특징 데이터가 비어 있습니다.
        return None

    # 사용자 및 아이템 ID 인코딩
    user_encoder = tf.keras.layers.StringLookup(mask_token=None)
    user_encoder.adapt(df_interactions['user_id'].astype(str).unique())
    num_users = user_encoder.vocabulary_size()

    item_encoder = tf.keras.layers.StringLookup(mask_token=None)
    # 완전한 아이템 어휘를 갖기 위해 상호작용 및 특징의 모든 아이템 ID에 대해 적합
    all_item_ids = pd.concat([
        df_interactions['item_id'].astype(str),
        df_item_features['item_id'].astype(str)
    ]).unique()
    item_encoder.adapt(all_item_ids)
    num_items = item_encoder.vocabulary_size()

    print(f"Data loaded: {num_users} users (incl. OOV if any), {num_items} items (incl. OOV if any), {len(df_interactions)} interactions.") # 데이터 로드됨: 사용자 {num_users}명(OOV 포함 시), 아이템 {num_items}개(OOV 포함 시), 상호작용 {len(df_interactions)}개.

    # 인접 리스트 플레이스홀더 (이웃 샘플링용)
    # 실제 구현에서는 예를 들어 df_interactions로부터 효율적으로 구성됩니다.
    # 형식: 키가 node_id(전역, 예: 사용자 0..N-1, 아이템 N..N+M-1)이고
    # 값은 neighbor_id 목록인 딕셔너리.
    adj_list_placeholder = {i: np.random.randint(0, num_users + num_items, 10).tolist() for i in range(num_users + num_items)}
    print(f"Placeholder: Adjacency list created for {len(adj_list_placeholder)} total conceptual nodes.") # 플레이스홀더: 총 {len(adj_list_placeholder)}개의 개념적 노드에 대한 인접 리스트 생성됨.

    # 결합된 사용자 및 아이템 특징 플레이스홀더 (H^(0))
    # 이 예제는 현재 사용자가 명시적인 특징을 가지고 있지 않으며 학습 가능한 임베딩으로 시작할 수 있다고 가정합니다.
    # 반면 아이템은 로드된 특징을 사용합니다. 더 일반적인 GraphSAGE는 모든 노드에 대한 임의의 특징을 처리합니다.
    # 단순화를 위해 아이템 특징을 직접 사용할 수 있다고 가정합니다. 사용자 특징은 임베딩으로 초기화될 수 있습니다.
    # 이 부분은 실제 시스템에서 신중한 설계가 필요합니다.

    # 아이템 특징 df를 조회 딕셔너리로 변환 (인코딩된 item_idx -> feature_vector)
    df_item_features['item_idx'] = item_encoder(df_item_features['item_id'].astype(str)).numpy()
    item_features_map = {
        row['item_idx']: row.drop(['item_id', 'item_idx']).values.astype(np.float32)
        for _, row in df_item_features.iterrows()
    }
    print(f"  Item features processed for {len(item_features_map)} items.") #   {len(item_features_map)}개 아이템에 대한 아이템 특징 처리됨.
    # 명시적인 특징이 제공되지 않으면 사용자 특징을 학습 가능한 임베딩으로 초기화할 수 있습니다.
    # 플레이스홀더의 경우 모델이 처리하도록 맵과 num_users만 전달합니다.

    return {
        'df_interactions': df_interactions,
        'user_encoder': user_encoder,
        'item_encoder': item_encoder,
        'num_users': num_users,
        'num_items': num_items,
        'adj_list_placeholder': adj_list_placeholder,
        'item_features_map_placeholder': item_features_map # 맵: 인코딩된 item_idx -> feature_vector
        # 'user_features_placeholder': ... (유사하게 추가될 수 있음)
    }

# --- GraphSAGE 모델 구성 요소 (플레이스홀더) ---
class NeighborSamplerPlaceholder:
    """Conceptual placeholder for a fixed-size neighbor sampler.""" # 고정 크기 이웃 샘플러의 개념적 플레이스홀더입니다.
    def __init__(self, adj_list, num_neighbors_to_sample):
        self.adj_list = adj_list
        self.num_neighbors = num_neighbors_to_sample
        print(f"NeighborSamplerPlaceholder initialized (sampling {num_neighbors_to_sample} neighbors).") # NeighborSamplerPlaceholder 초기화됨 ({num_neighbors_to_sample}개 이웃 샘플링).
        print("  Note: This sampler is a placeholder and uses simplified random sampling.") #   참고: 이 샘플러는 플레이스홀더이며 단순화된 무작위 샘플링을 사용합니다.

    def sample(self, node_ids_batch):
        """Samples fixed-size neighborhoods for a batch of node IDs.""" # 노드 ID 배치에 대해 고정 크기 이웃을 샘플링합니다.
        batch_sampled_neighbors = []
        print(f"  NeighborSamplerPlaceholder.sample called for {len(node_ids_batch)} nodes.") #   NeighborSamplerPlaceholder.sample이 {len(node_ids_batch)}개 노드에 대해 호출됨.
        for node_id in node_ids_batch:
            neighbors = self.adj_list.get(node_id, [])
            if len(neighbors) >= self.num_neighbors:
                sampled = np.random.choice(neighbors, self.num_neighbors, replace=False).tolist()
            else: # 고유한 이웃이 충분하지 않으면 교체 샘플링 또는 패딩
                sampled = np.random.choice(neighbors, self.num_neighbors, replace=True).tolist() if neighbors else [-1] * self.num_neighbors # -1은 패딩용
            batch_sampled_neighbors.append(sampled)
        print(f"    -> Sampled neighborhoods (example for first node if batch > 0): {batch_sampled_neighbors[0] if node_ids_batch else 'N/A'}") #     -> 샘플링된 이웃 (배치가 0보다 큰 경우 첫 번째 노드 예시): ...
        return batch_sampled_neighbors # 이웃 ID 목록의 목록

class GraphSAGELayerPlaceholder(tf.keras.layers.Layer):
    """Conceptual placeholder for a GraphSAGE layer.""" # GraphSAGE 레이어의 개념적 플레이스홀더입니다.
    def __init__(self, output_dim, aggregator_type='mean', activation='relu', **kwargs):
        super(GraphSAGELayerPlaceholder, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.aggregator_type = aggregator_type # 'mean', 'pool', 'lstm', 'gcn' (개념적)
        self.activation = tf.keras.layers.Activation(activation)

        # 집계된 이웃 특징 + 자체 특징 변환을 위한 Dense 레이어
        # 이 레이어의 입력은 2 * input_feature_dim (자체 특징 연결 시)
        # 또는 자체 특징이 별도로 변환된 후 추가/연결되는 경우 input_feature_dim
        self.dense_transform = tf.keras.layers.Dense(output_dim, name=f"{self.name}_dense_transform")
        print(f"GraphSAGELayerPlaceholder initialized (output_dim={output_dim}, aggregator='{aggregator_type}').") # GraphSAGELayerPlaceholder 초기화됨 (output_dim={output_dim}, aggregator='{aggregator_type}').

    def call(self, self_features, neighbor_features):
        """
        Conceptual call method for a GraphSAGE layer. # GraphSAGE 레이어의 개념적 호출 메소드입니다.
        - self_features: Features of the target nodes. Shape: (batch_size, input_feature_dim)
        - neighbor_features: Features of the sampled neighbors. Shape: (batch_size, num_sampled_neighbors, input_feature_dim)
        """
        print(f"  GraphSAGELayerPlaceholder.call(): Processing features...") #   GraphSAGELayerPlaceholder.call(): 특징 처리 중...
        print(f"    Self features shape: {self_features.shape if isinstance(self_features, tf.Tensor) else 'N/A (Placeholder)'}") #     자체 특징 모양: ...
        print(f"    Neighbor features shape: {neighbor_features.shape if isinstance(neighbor_features, tf.Tensor) else 'N/A (Placeholder)'}") #     이웃 특징 모양: ...

        # 1. 이웃 특징 집계 (개념적)
        if self.aggregator_type == 'mean':
            # 이웃 특징 벡터의 평균 (요소별)
            aggregated_neighbors = tf.reduce_mean(neighbor_features, axis=1) # (batch_size, input_feature_dim)
        elif self.aggregator_type == 'pool': # 예시: 최대 풀링
            aggregated_neighbors = tf.reduce_max(neighbor_features, axis=1)
        # elif self.aggregator_type == 'lstm': # LSTM 집계기는 정렬된 이웃과 LSTM 레이어가 필요합니다.
        #    aggregated_neighbors = self.lstm_layer(neighbor_features) # LSTM의 마지막 상태 출력
        else: # 기본값은 평균
            aggregated_neighbors = tf.reduce_mean(neighbor_features, axis=1)
        print(f"    -> Aggregated neighbor features shape: {aggregated_neighbors.shape if isinstance(aggregated_neighbors, tf.Tensor) else 'N/A (Placeholder)'}") #     -> 집계된 이웃 특징 모양: ...

        # 2. 대상 노드의 특징과 집계된 이웃 특징 연결
        combined_representation = tf.concat([self_features, aggregated_neighbors], axis=1) # (batch_size, 2 * input_feature_dim)

        # 3. 변환, 활성화 및 정규화 (정규화는 기본 GraphSAGE 다이어그램에서는 종종 생략되지만 좋은 관행임)
        output_embeddings = self.dense_transform(combined_representation) # (batch_size, output_dim)
        output_embeddings = self.activation(output_embeddings)
        # output_embeddings = tf.linalg.l2_normalize(output_embeddings, axis=1) # 선택적 L2 정규화

        print(f"    -> Output embeddings shape from layer: {output_embeddings.shape if isinstance(output_embeddings, tf.Tensor) else 'N/A (Placeholder)'}") #     -> 레이어의 출력 임베딩 모양: ...
        return output_embeddings

    def get_config(self):
        config = super().get_config()
        config.update({"output_dim": self.output_dim, "aggregator_type": self.aggregator_type})
        return config

class GraphSAGEModelPlaceholder(tf.keras.Model):
    """Conceptual placeholder for the multi-layer GraphSAGE model.""" # 다중 레이어 GraphSAGE 모델의 개념적 플레이스홀더입니다.
    def __init__(self, initial_feature_dim, sage_layer_units=SAGE_LAYER_UNITS, num_neighbors_per_layer=NUM_NEIGHBORS_PER_LAYER,
                 aggregator_type='mean', final_embedding_dim=None, **kwargs):
        super(GraphSAGEModelPlaceholder, self).__init__(**kwargs)
        self.initial_feature_dim = initial_feature_dim
        self.num_layers = len(sage_layer_units)

        # 입력 특징을 위한 초기 변환 레이어 (선택 사항이지만 일반적)
        # self.initial_transform = Dense(initial_feature_dim, activation='relu', name="initial_feature_projector")

        self.sage_layers = []
        current_dim = initial_feature_dim
        for i, units in enumerate(sage_layer_units):
            self.sage_layers.append(
                GraphSAGELayerPlaceholder(output_dim=units, aggregator_type=aggregator_type, name=f"graphsage_layer_{i+1}")
            )
            current_dim = units # 다음 레이어의 self_features에 대한 입력 차원

        # final_embedding_dim이 지정되고 마지막 SAGE 레이어의 출력과 다른 경우
        if final_embedding_dim and current_dim != final_embedding_dim:
            self.final_projection = Dense(final_embedding_dim, name="final_embedding_projection")
            print(f"  Added final projection layer to {final_embedding_dim}-dim.") #   {final_embedding_dim}차원으로 최종 프로젝션 레이어 추가됨.
        else:
            self.final_projection = None

        print(f"GraphSAGEModelPlaceholder initialized with {self.num_layers} SAGE layers.") # GraphSAGEModelPlaceholder가 {self.num_layers}개의 SAGE 레이어로 초기화되었습니다.
        print(f"  Layer output units: {sage_layer_units}") #   레이어 출력 유닛: {sage_layer_units}
        print(f"  Number of neighbors to sample at each layer: {num_neighbors_per_layer}") #   각 레이어에서 샘플링할 이웃 수: {num_neighbors_per_layer}

    def call(self, inputs, training=False):
        """
        Conceptual forward pass for generating embeddings for a batch of target nodes. # 대상 노드 배치에 대한 임베딩 생성을 위한 개념적 순방향 패스입니다.
        - inputs: A dictionary or tuple, conceptually containing:
            - 'target_node_ids': Batch of node IDs to generate embeddings for.
            - 'all_node_features': A lookup mechanism (e.g., TF Embedding or dict) for initial features.
            - 'neighbor_samplers': A list of NeighborSamplerPlaceholder instances, one for each layer.
        """
        target_node_ids, all_node_features_lookup, neighbor_samplers = inputs

        print(f"  GraphSAGEModelPlaceholder.call(): Generating embeddings for {len(target_node_ids)} target nodes.") #   GraphSAGEModelPlaceholder.call(): {len(target_node_ids)}개 대상 노드에 대한 임베딩 생성 중.

        # `h_layer_k_minus_1`은 레이어 `k`에 필요한 노드의 특징/임베딩을 저장합니다.
        # 처음에는 레이어 1의 경우 대상 노드와 1차 이웃의 원시 입력 특징이 됩니다.
        # 이는 단순화된 표현입니다. 실제 구현은 반복적인 샘플링을 사용합니다.

        # `target_node_ids`가 최종 임베딩을 원하는 노드라고 가정합니다.
        # 프로세스는 가장 바깥쪽 이웃 레이어에서 안쪽으로 반복됩니다.
        # K개 레이어의 경우 K-홉 이웃을 샘플링해야 합니다.

        # 레이어 0 특징 (초기 입력 특징)
        # 이는 미니 배치에서 특징이 수집되는 방식에 대한 개념적 단순화입니다.
        # 실제 시스템에서는 대상 노드 배치가 있고, K-홉 이웃을 샘플링한 다음,
        # 이러한 K-홉 계산 그래프에 관련된 모든 고유 노드에 대한 특징을 가져옵니다.
        # 그런 다음 레이어별로 전파합니다.

        # 이 플레이스홀더의 경우 `target_node_ids`에 대한 최종 레이어의 계산을 시뮬레이션합니다.
        # `h_of_target_nodes_prev_layer`와 `h_of_neighbors_prev_layer`가 어떻게든 사용 가능하다고 가정합니다.

        # `current_target_node_features`가 현재 임베딩을 계산 중인 노드의 특징이라고 가정합니다.
        # 처음에는 대상 노드 배치의 원시 특징입니다.
        # ID를 기반으로 특징을 조회하는 방법이 필요합니다.
        # 플레이스홀더의 경우 `all_node_features_lookup`이 [총_노드_수, 특징_차원] 텐서라고 가정합니다.
        current_target_node_features = tf.gather(all_node_features_lookup, target_node_ids)

        # initial_transform을 사용하는 경우:
        # current_target_node_features = self.initial_transform(current_target_node_features)

        for i in range(self.num_layers):
            print(f"    Executing GraphSAGELayerPlaceholder {i+1}...") #     GraphSAGELayerPlaceholder {i+1} 실행 중...
            # 1. 현재 target_node_ids에 대한 이웃 샘플링 (개념적으로)
            #    실제 배치 시스템에서는 이 샘플링이 이 호출 전에 배치의 모든 노드에 대해 발생합니다.
            #    `neighbor_samplers[i]`는 이 레이어에 대해 미리 샘플링된 이웃 ID를 제공합니다.
            #    `target_node_ids`에 대해 `sampled_neighbor_ids_for_layer_i`가 사용 가능하다고 가정합니다.
            #    플레이스홀더:
            sampled_neighbor_ids_at_layer_i = neighbor_samplers[i].sample(target_node_ids) # 목록의 목록

            # 2. 샘플링된 이웃에 대한 특징 가져오기 (개념적으로, 이전 레이어의 출력 또는 초기 특징으로부터)
            #    이는 실제 미니 배치 GraphSAGE에서 가장 복잡한 부분입니다.
            #    플레이스홀더의 경우 더미 이웃 특징을 만듭니다.
            #    `all_node_features_lookup`이 이전 레이어 또는 초기의 *모든* 노드에 대한 특징을 포함한다고 가정합니다.

            batch_neighbor_features_list = []
            for neighbor_id_list_for_one_node in sampled_neighbor_ids_at_layer_i:
                # 샘플러가 반환하는 경우 패딩(-1) 처리
                valid_neighbor_ids = [idx for idx in neighbor_id_list_for_one_node if idx != -1 and idx < tf.shape(all_node_features_lookup)[0]]
                if not valid_neighbor_ids: # 모두 패딩이거나 유효한 이웃이 없는 경우
                    # 유효한 이웃이 없으면 특징 차원과 일치하는 0 벡터 사용
                    neighbor_feats_for_node = tf.zeros((len(neighbor_id_list_for_one_node), tf.shape(current_target_node_features)[-1]), dtype=tf.float32)
                else:
                    neighbor_feats_for_node = tf.gather(all_node_features_lookup, valid_neighbor_ids)
                    # 샘플러가 고정 크기를 만들기 위해 패딩을 사용했고 일부가 유효하지 않은 경우 다시 패딩
                    padding_needed = len(neighbor_id_list_for_one_node) - tf.shape(neighbor_feats_for_node)[0]
                    if padding_needed > 0:
                        padding_tensor = tf.zeros((padding_needed, tf.shape(current_target_node_features)[-1]), dtype=tf.float32)
                        neighbor_feats_for_node = tf.concat([neighbor_feats_for_node, padding_tensor], axis=0)
                batch_neighbor_features_list.append(neighbor_feats_for_node)

            # 배치 생성: (batch_size, num_sampled_neighbors, feature_dim)
            batched_neighbor_features_for_layer_i = tf.stack(batch_neighbor_features_list)

            # 3. 현재 대상 노드 특징과 샘플링된 이웃 특징을 SAGE 레이어에 전달
            current_target_node_features = self.sage_layers[i](current_target_node_features, batched_neighbor_features_for_layer_i)
            # 출력 `current_target_node_features`는 이제 배치의 h_v^(k)입니다.
            # 다음 레이어의 경우 이는 *동일한* 대상 노드에 대한 h_v^(k-1)이 됩니다.
            # 다음 레이어의 `all_node_features_lookup`은 개념적으로 이러한 업데이트된 임베딩이어야 합니다.
            # 이 플레이스홀더는 이웃에 대해 항상 초기 `all_node_features_lookup`을 사용하여 이를 단순화합니다.
            # 실제 구현은 이전 SAGE 레이어 출력을 기반으로 이웃의 특징 소스를 업데이트합니다.

        final_embeddings = current_target_node_features
        if self.final_projection:
            final_embeddings = self.final_projection(final_embeddings)

        print(f"  GraphSAGEModelPlaceholder.call(): Final embeddings generated (shape: {final_embeddings.shape if isinstance(final_embeddings, tf.Tensor) else 'N/A (Placeholder)'}).") #   GraphSAGEModelPlaceholder.call(): 최종 임베딩 생성됨 (모양: ...).
        return final_embeddings

# --- 메인 실행 블록 ---
if __name__ == "__main__":
    print("GraphSAGE (Inductive Representation Learning on Large Graphs) - Conceptual Outline") # GraphSAGE (대규모 그래프에서의 귀납적 표현 학습) - 개념적 개요
    print("="*80)
    print("This script provides a conceptual overview and structural outline of a GraphSAGE model.") # 이 스크립트는 GraphSAGE 모델의 개념적 개요와 구조적 윤곽을 제공합니다.
    print("It is NOT a runnable or fully implemented GraphSAGE model. Key components like efficient") # 실행 가능하거나 완전히 구현된 GraphSAGE 모델이 아닙니다. 효율적인
    print("multi-layer neighbor sampling for mini-batches, actual aggregator implementations (LSTM/Pooling),") # 미니 배치를 위한 다중 레이어 이웃 샘플링, 실제 집계기 구현(LSTM/풀링),
    print("and specific loss functions for training are simplified placeholders.") # 및 학습을 위한 특정 손실 함수는 단순화된 플레이스홀더입니다.
    print("Refer to the original paper and specialized graph learning libraries for robust implementations.") # 강력한 구현을 위해서는 원본 논문 및 특수 그래프 학습 라이브러리를 참조하십시오.
    print("="*80 + "\n")

    # 1. 개념적 데이터 로드 (상호작용 그래프용, 노드 특징)
    print("Step 1: Loading conceptual data (interactions and node features)...") # 단계 1: 개념적 데이터 로드 (상호작용 및 노드 특징)...
    data_dict = load_graphsage_data()

    if data_dict:
        df_interactions = data_dict['df_interactions']
        user_encoder = data_dict['user_encoder'] # StringLookup
        item_encoder = data_dict['item_encoder'] # StringLookup
        num_users = data_dict['num_users']
        num_items = data_dict['num_items']
        adj_list_placeholder = data_dict['adj_list_placeholder']
        item_features_map_placeholder = data_dict['item_features_map_placeholder']

        # 개념적: 모든 노드(사용자 + 아이템)에 대한 결합된 특징 Matrix 생성
        # 명시적인 특징이 제공되지 않으면 사용자 특징은 학습 가능한 임베딩일 수 있습니다.
        # 아이템 특징은 item_features_map_placeholder에서 가져옵니다.
        # 이는 플레이스홀더를 위해 매우 단순화되었습니다.

        # 사용자 특징 (플레이스홀더: 임의 또는 학습 가능한 임베딩)
        # Keras에서는 user_id가 모델에 전달되면 임베딩 레이어가 이를 처리합니다.
        # 지금은 개념적인 결합 특징 Matrix를 만듭니다.
        user_features_placeholder = np.random.rand(num_users, INPUT_FEATURE_DIM).astype(np.float32)

        # 아이템 특징 (로드된 맵에서, num_items까지 모든 아이템에 항목이 있는지 확인)
        all_item_features_list = []
        for i in range(num_items): # item_encoder가 0..num_items-1에 매핑한다고 가정
            if i in item_features_map_placeholder:
                all_item_features_list.append(item_features_map_placeholder[i])
            else: # 맵에 없는 아이템에 대한 기본 특징 (예: StringLookup이 더 많은 OOV 아이템을 생성한 경우)
                all_item_features_list.append(np.zeros(INPUT_FEATURE_DIM, dtype=np.float32))
        item_features_tensor_placeholder = np.array(all_item_features_list)

        # 결합 (개념적으로, 사용자 특징 먼저, 그 다음 아이템 특징)
        # 실제 시나리오에서는 노드 ID가 전역적으로 0부터 (num_users + num_items - 1)까지 매핑되고
        # 특징이 그에 따라 정렬됩니다.
        # 이 플레이스홀더는 필요한 경우 모델이 내부적으로 별도의 사용자/아이템 처리를 처리한다고 가정합니다.
        # 모델 호출의 `all_node_features_lookup`에 대해 개념적인 결합된 것을 사용합니다.
        # 이 부분은 플레이스홀더에서 매우 모호합니다.
        all_nodes_initial_features_placeholder = np.vstack([
            user_features_placeholder,
            item_features_tensor_placeholder
        ]).astype(np.float32) if item_features_tensor_placeholder.size > 0 else user_features_placeholder

        if all_nodes_initial_features_placeholder.shape[0] != num_users + num_items:
            print(f"Warning: Shape mismatch in combined features {all_nodes_initial_features_placeholder.shape[0]} vs {num_users + num_items}") # 경고: 결합된 특징의 모양 불일치 {all_nodes_initial_features_placeholder.shape[0]} 대 {num_users + num_items}
            # OOV 아이템 등으로 인해 모양이 정렬되지 않는 경우 플레이스홀더의 안전을 위한 대체
            all_nodes_initial_features_placeholder = np.random.rand(num_users + num_items, INPUT_FEATURE_DIM).astype(np.float32)


        print(f"  Placeholder: Created dummy 'all_nodes_initial_features' of shape {all_nodes_initial_features_placeholder.shape}") #   플레이스홀더: {all_nodes_initial_features_placeholder.shape} 모양의 더미 'all_nodes_initial_features' 생성됨

        # 2. 플레이스홀더 샘플러 인스턴스화 (각 SAGE 레이어당 하나)
        print("\nStep 2: Initializing Neighbor Samplers (conceptual)...") # 단계 2: 이웃 샘플러 초기화 (개념적)...
        neighbor_samplers_placeholders = [
            NeighborSamplerPlaceholder(adj_list_placeholder, num_neighbors)
            for num_neighbors in NUM_NEIGHBORS_PER_LAYER
        ]

        # 3. GraphSAGE 모델 구축 (플레이스홀더 구조)
        print("\nStep 3: Building GraphSAGE Model structure (conceptual)...") # 단계 3: GraphSAGE 모델 구조 구축 (개념적)...
        graphsage_model_placeholder = GraphSAGEModelPlaceholder(
            initial_feature_dim=INPUT_FEATURE_DIM,
            sage_layer_units=SAGE_LAYER_UNITS,
            num_neighbors_per_layer=NUM_NEIGHBORS_PER_LAYER, # 정보용으로 전달하지만 모델이 직접 사용하지는 않음
            aggregator_type=AGGREGATOR_TYPE,
            final_embedding_dim=SAGE_LAYER_UNITS[-1] # 마지막 SAGE 레이어의 출력
        )
        # 개념적 컴파일 (손실은 작업에 따라 다름, 예: BPR, 감독 노드 분류)
        # graphsage_model_placeholder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE), loss="some_loss_for_sage")
        print("Conceptual GraphSAGE model built.") # 개념적 GraphSAGE 모델 구축됨.
        print("\nConceptual Model Structure:") # 개념적 모델 구조:
        print(f"  - Initial Node Features Dim: {INPUT_FEATURE_DIM}") #   - 초기 노드 특징 차원: {INPUT_FEATURE_DIM}
        for i, units in enumerate(SAGE_LAYER_UNITS):
            print(f"  - GraphSAGE Layer {i+1}: Samples {NUM_NEIGHBORS_PER_LAYER[i]} neighbors, Aggregator: '{AGGREGATOR_TYPE}', Output: {units}-dim") #   - GraphSAGE 레이어 {i+1}: {NUM_NEIGHBORS_PER_LAYER[i]}개 이웃 샘플링, 집계기: '{AGGREGATOR_TYPE}', 출력: {units}차원
        if graphsage_model_placeholder.final_projection:
             print(f"  - Final Projection Layer to: {graphsage_model_placeholder.final_projection.units}-dim") #   - 최종 프로젝션 레이어 대상: {graphsage_model_placeholder.final_projection.units}차원
        else:
             print(f"  - Final Output Dimension: {SAGE_LAYER_UNITS[-1]}-dim item/user embeddings.") #   - 최종 출력 차원: {SAGE_LAYER_UNITS[-1]}차원 아이템/사용자 임베딩.


        # 4. 모델 학습 / 임베딩 생성 (개념적)
        print("\nStep 4: Generating Node Embeddings (conceptual full-graph pass)...") # 단계 4: 노드 임베딩 생성 (개념적 전체 그래프 패스)...
        print(f"  (This would involve iterative sampling and aggregation for each node in mini-batches.") #   (이는 미니 배치에서 각 노드에 대한 반복적인 샘플링 및 집계를 포함합니다.
        print(f"   For simplicity, a conceptual 'full pass' for a sample of nodes is shown.)") #    단순화를 위해 노드 샘플에 대한 개념적 '전체 패스'가 표시됩니다.)

        # 예시: 일부 사용자 및 아이템 노드에 대한 임베딩 가져오기 (개념적)
        sample_user_ids_original = df_interactions['user_id'].astype(str).unique()[:2]
        sample_item_ids_original = df_interactions['item_id'].astype(str).unique()[:2]

        # 원본 ID를 인코딩된 정수 인덱스로 변환
        # StringLookup은 OOV에 대해 0을 반환하며, 0이 유효한 ID인 경우 문제가 될 수 있습니다.
        # 이 플레이스홀더의 경우 StringLookup이 알려진 ID에 대해 이를 잘 처리한다고 가정합니다.
        sample_user_indices_encoded = user_encoder(sample_user_ids_original).numpy()
        sample_item_indices_encoded = item_encoder(sample_item_ids_original).numpy()

        # GraphSAGE의 경우 사용자 및 아이템 노드는 종종 특징 조회를 위한 하나의 전역 노드 ID 공간의 일부입니다.
        # 사용자 인덱스: 0 ~ num_users-1
        # 아이템 인덱스: num_users ~ num_users + num_items -1
        # 이 매핑은 `all_nodes_initial_features_placeholder` 및 `adj_list_placeholder`와 일치해야 합니다.
        # 현재 플레이스홀더 `adj_list` 및 `features`는 이와 완벽하게 일치하지 않으므로
        # 이 부분은 매우 개념적입니다.

        # 개념적 대상 노드 (예: 인코딩된 인덱스의 첫 번째 사용자와 첫 번째 아이템)
        # 이는 user_indices가 0..N-1이고 item_indices가 인코딩 후 0..M-1이라고 가정하며,
        # `all_nodes_initial_features_placeholder`는 사용자 특징 위에 아이템 특징이 쌓인 것입니다.
        target_node_batch_indices_placeholder = []
        if len(sample_user_indices_encoded)>0: target_node_batch_indices_placeholder.append(sample_user_indices_encoded[0])
        if len(sample_item_indices_encoded)>0: target_node_batch_indices_placeholder.append(sample_item_indices_encoded[0] + num_users) # 아이템 인덱스 오프셋

        if target_node_batch_indices_placeholder:
            target_node_batch_indices_placeholder = np.array(target_node_batch_indices_placeholder, dtype=np.int32)

            conceptual_inputs_for_model = (
                target_node_batch_indices_placeholder,
                all_nodes_initial_features_placeholder, # 모든 노드 특징에 대한 조회 테이블
                neighbor_samplers_placeholders         # 각 레이어에 대한 샘플러 목록
            )
            final_embeddings_conceptual = graphsage_model_placeholder(conceptual_inputs_for_model)
            print(f"  Conceptual final embeddings for sample nodes computed (shape: {final_embeddings_conceptual.shape}).") #   샘플 노드에 대한 개념적 최종 임베딩 계산됨 (모양: {final_embeddings_conceptual.shape}).
        else:
            print("  Skipping conceptual forward pass as no sample user/item IDs were available.") #   사용 가능한 샘플 사용자/아이템 ID가 없어 개념적 순방향 패스를 건너<0xEB><0x9B><0x84>니다.

        print("  Skipping actual training for this placeholder script.") #   이 플레이스홀더 스크립트에 대한 실제 학습 건너뛰기.

        # 5. 추천을 위한 임베딩 사용 (개념적)
        print("\nStep 5: Using Embeddings for Recommendations (conceptual)...") # 단계 5: 추천을 위한 임베딩 사용 (개념적)...
        print(f"  (Once final user and item embeddings are generated for all relevant nodes,") #   (모든 관련 노드에 대한 최종 사용자 및 아이템 임베딩이 생성되면,
        print(f"   they can be used for similarity calculations (e.g., dot product) to find") #    유사도 계산(예: 내적)에 사용되어
        print(f"   top-N recommendations, similar to other embedding-based models.)") #    다른 임베딩 기반 모델과 유사하게 상위 N개 추천을 찾을 수 있습니다.)

    else:
        print("\nData loading placeholder failed. Cannot proceed with GraphSAGE conceptual outline.") # 데이터 로딩 플레이스홀더 실패. GraphSAGE 개념적 개요를 진행할 수 없습니다.

    print("\n" + "="*80)
    print("GraphSAGE Conceptual Outline Example Finished.") # GraphSAGE 개념적 개요 예제 완료.
    print("Reminder: This is a structural guide, not a working implementation of GraphSAGE.") # 알림: 이것은 구조적 가이드이며 GraphSAGE의 작동하는 구현이 아닙니다.
    print("="*80)
