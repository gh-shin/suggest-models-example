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

# --- SASRec (Self-Attentive Sequential Recommendation): 기본 설명 ---
# SASRec (Self-Attentive Sequential Recommendation)은 사용자의 과거 상호작용 시퀀스를 기반으로 사용자 선호도를 포착하도록 설계된 추천 모델입니다.
# Transformer 모델에서 영감을 받은 자기 어텐션 메커니즘을 활용하여 사용자의 과거 시퀀스에서 어떤 아이템이
# 다음 아이템을 예측하는 데 가장 큰 영향을 미치는지 이해합니다.
#
# 작동 방식:
# 1. 입력 표현:
#    - 사용자의 상호작용 기록은 아이템 ID 시퀀스로 표현됩니다 (예: [item_A, item_B, item_C]).
#    - 작업은 사용자가 다음에 상호작용할 가능성이 있는 아이템(예: item_D)을 예측하는 것입니다.
#
# 2. 아이템 임베딩:
#    - 입력 시퀀스의 각 아이템 ID는 밀집 벡터 표현(아이템 임베딩)에 매핑됩니다.
#    - 이 임베딩은 아이템의 잠재적 특성을 포착합니다.
#    - MAX_SEQ_LENGTH보다 짧은 시퀀스에는 특수 패딩 토큰(종종 0)이 사용됩니다.
#
# 3. 위치 임베딩:
#    - 자기 어텐션 자체는 본질적으로 아이템 순서를 이해하지 못하므로 위치 정보가 중요합니다.
#    - 위치 임베딩은 학습된 벡터이며, 시퀀스의 각 위치(1부터 MAX_SEQ_LENGTH까지)에 대해 하나씩 존재합니다.
#    - 이는 해당 아이템 임베딩에 추가되어 모델에 아이템 순서 감각을 부여합니다.
#
# 4. 자기 어텐션 블록: 이것이 SASRec의 핵심이며, 일반적으로 하나 이상의 동일한 블록으로 구성됩니다.
#    각 블록은 컨텍스트를 고려하여 시퀀스의 각 아이템 표현을 정제하는 것을 목표로 합니다.
#    블록은 일반적으로 다음을 포함합니다:
#    a. 인과적 멀티-헤드 자기 어텐션 (Causal Multi-Head Self-Attention):
#       - "자기 어텐션": 시퀀스의 각 아이템은 표현의 가중 합계를 계산하기 위해 *그 이전의* 모든 다른 아이템(및 자신)에 주목합니다.
#         이는 모델이 시퀀스에서 현재 아이템의 역할을 이해하는 데 가장 관련성이 높은 과거 아이템을 학습한다는 의미입니다.
#       - "인과적": 위치 't'의 아이템을 예측할 때 모델이 위치 'j <= t'의 아이템에만 주목하도록 보장합니다.
#         이는 미래 아이템으로부터의 데이터 유출을 방지합니다.
#       - "멀티-헤드": 어텐션 메커니즘은 서로 다른 학습된 선형 프로젝션(헤드)을 사용하여 병렬로 여러 번 실행됩니다.
#         이를 통해 모델은 다른 위치에서 다른 표현 하위 공간의 정보에 공동으로 주목할 수 있습니다.
#         그런 다음 출력은 연결되고 선형적으로 변환됩니다.
#    b. 점별 피드-포워드 네트워크 (Point-wise Feed-Forward Network, FFN):
#       - 자기 어텐션 단계 후 각 아이템의 표현에 독립적으로 적용됩니다.
#       - 일반적으로 중간에 비선형 활성화(예: ReLU)가 있는 두 개의 Dense 레이어로 구성됩니다.
#       - 이를 통해 각 아이템 표현의 더 복잡한 변환이 가능합니다.
#    c. 추가 및 정규화 (Residual Connections and Layer Normalization):
#       - 잔차 연결(하위 레이어의 입력을 출력에 추가)은 기울기 소실 문제를 완화하여 더 깊은 모델 학습에 도움이 됩니다.
#       - 레이어 정규화는 학습 과정을 안정화합니다.
#
# 5. 예측 출력:
#    - 입력 시퀀스가 모든 Transformer 블록을 통과한 후 (입력) 시퀀스의 *마지막 아이템*의 출력 표현이 사용됩니다.
#      이 벡터는 기록을 기반으로 한 사용자의 현재 관심사를 요약하는 것으로 간주됩니다.
#    - 이 최종 벡터는 `num_items_for_embedding` 단위(활성화 또는 소프트맥스 없음, 손실 함수가 로짓을 처리함)를 가진
#      Dense 레이어를 통과하여 가능한 모든 다음 아이템에 대한 점수(로짓)를 생성합니다.
#    - 학습 중에는 `SparseCategoricalCrossentropy` 손실이 사용되어 이러한 로짓을 시퀀스의 실제 다음 아이템과 비교합니다.
#
# Pros (장점):
# - 순차적 동역학 포착: 사용자 상호작용 시퀀스 내의 순서와 종속성을 효과적으로 모델링합니다.
#   자기 어텐션은 장거리 종속성을 식별할 수 있습니다.
# - 문맥적 이해: 다음 아이템을 예측할 때 현재 컨텍스트를 기반으로 다른 과거 아이템의 중요도를 동적으로 가중하는 방법을 학습합니다.
# - 병렬화 가능한 학습: Transformer 블록 내의 계산(특히 자기 어텐션)은 고도로 병렬화될 수 있어
#   GPU/TPU와 같은 최신 하드웨어에서 효율적인 학습이 가능합니다.
#
# Cons (단점):
# - 데이터 요구 사항: 의미 있는 패턴을 학습하기 위해 충분한 양의 사용자 시퀀스 데이터가 있을 때 최상의 성능을 발휘합니다.
# - 계산 비용: 자기 어텐션은 O(sequence_length^2 * embedding_dim)의 복잡도를 가지며, 이는 매우 긴 시퀀스의 경우
#   부담스러울 수 있습니다. 그러나 추천에서의 시퀀스 길이는 종종 적당합니다.
# - 아이템 Cold-Start: 학습 데이터에 없었던 새 아이템(임베딩이 없으므로)을 직접 추천할 수 없습니다.
#   새 아이템에 대해서는 재학습 또는 아이템 콘텐츠 특징 사용과 같은 전략이 필요합니다.
# ---

# 프로젝트 루트를 sys.path에 동적으로 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- 모델 하이퍼파라미터 (빠른 실행을 위해 이 예제에서는 작게 유지) ---
MAX_SEQ_LENGTH = 10  # 모델이 고려하는 사용자 아이템 상호작용 시퀀스의 최대 길이.
EMBEDDING_DIM = 32   # 아이템 및 위치 임베딩 벡터의 차원.
NUM_HEADS = 2        # MultiHeadAttention 레이어의 어텐션 헤드 수.
NUM_BLOCKS = 1       # 스택할 Transformer 블록 수. SASRec 논문은 일부 데이터셋에 대해 2개를 제안합니다.
FFN_UNITS = 64       # 피드-포워드 네트워크의 은닉 레이어 유닛 수.
DROPOUT_RATE = 0.2   # 다양한 레이어의 정규화를 위한 Dropout 비율.
LEARNING_RATE = 0.001# Adam 옵티마이저의 학습률.
EPOCHS = 5           # 학습 에포크 수 (예제를 위해 작게 설정).
BATCH_SIZE = 32      # 학습 배치당 시퀀스 수.

# --- 데이터 로딩 및 전처리 ---
# Time Complexity:
# - CSV 읽기: O(N_interactions_total)
# - LabelEncoding: O(N_total_items_in_sequences)
# - 시퀀스 패딩 및 분할: O(N_sequences * MAX_SEQ_LENGTH)
def load_sequences(base_filepath='data/dummy_sequences.csv', max_seq_len=MAX_SEQ_LENGTH):
    """
    한국어: 아이템 상호작용 시퀀스를 로드하고 SASRec을 위해 전처리하며 학습 샘플을 생성합니다.
    - CSV 파일에서 시퀀스를 읽습니다.
    - 아이템 ID를 0부터 시작하는 정수로 인코딩합니다.
    - (입력_시퀀스, 대상_아이템) 쌍을 생성합니다.
    - 입력 시퀀스를 고정 길이로 패딩합니다.

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
        print(f"Error: Sequence data file not found at {filepath}.") # 오류: {filepath}에서 시퀀스 데이터 파일을 찾을 수 없습니다.
        print("Attempting to generate dummy sequence data using 'data/generate_dummy_data.py'...") # 'data/generate_dummy_data.py'를 사용하여 더미 시퀀스 데이터 생성 시도 중...
        try:
            from data.generate_dummy_data import generate_dummy_data
            # 더미 데이터 생성을 위해 generate_sequences=True가 호출되었는지 확인
            generate_dummy_data(num_users=200, num_items=100, num_interactions=2000, generate_sequences=True)
            print("Dummy sequence data generation script executed.") # 더미 시퀀스 데이터 생성 스크립트 실행됨.
            if not os.path.exists(filepath): # 생성 시도 후 다시 확인
                print(f"Error: Dummy sequence file still not found at {filepath} after generation.") # 오류: 생성 후에도 {filepath}에서 더미 시퀀스 파일을 찾을 수 없습니다.
                return None, None, None, None
            print("Dummy sequence data file should now be available.") # 이제 더미 시퀀스 데이터 파일을 사용할 수 있습니다.
        except ImportError as e_import:
            print(f"ImportError: Failed to import 'generate_dummy_data'. Error: {e_import}") # ImportError: 'generate_dummy_data'를 임포트하지 못했습니다. 오류: {e_import}
            return None, None, None, None
        except Exception as e_general:
            print(f"Error during dummy data generation: {e_general}") # 더미 데이터 생성 중 오류: {e_general}
            return None, None, None, None

    df_sequences = pd.read_csv(filepath)
    if df_sequences.empty or 'item_ids_sequence' not in df_sequences.columns:
        print(f"Error: File {filepath} is empty or missing 'item_ids_sequence' column.") # 오류: 파일 {filepath}이(가) 비어 있거나 'item_ids_sequence' 열이 없습니다.
        return None, None, None, None

    # CSV의 공백으로 구분된 아이템 ID 문자열을 정수 목록으로 변환
    sequences_orig_ids = df_sequences['item_ids_sequence'].apply(lambda s: [int(i) for i in str(s).split(' ')]).tolist()

    # 시퀀스에 있는 모든 고유 아이템의 어휘 생성
    all_items_flat_list = [item for seq in sequences_orig_ids for item in seq]
    if not all_items_flat_list:
        print("Error: No items found in the sequences.") # 오류: 시퀀스에서 아이템을 찾을 수 없습니다.
        return None, None, None, None

    # LabelEncoder를 사용하여 원본 아이템 ID를 0부터 시작하는 정수로 매핑
    item_encoder = LabelEncoder()
    item_encoder.fit(all_items_flat_list)

    # 임베딩 레이어의 어휘 크기: 고유 아이템 수 + 패딩 토큰(0) 1개
    # 패딩 토큰은 인덱스 0을 가집니다. 실제 아이템은 인덱스 1부터 N까지 가집니다.
    num_items_for_embedding = len(item_encoder.classes_) + 1

    # 시퀀스를 인코딩된 아이템 인덱스로 변환. 모든 인코딩된 인덱스에 1을 더하여
    # 실제 아이템 인덱스가 1부터 시작하도록 하고 0은 패딩용으로만 예약합니다.
    encoded_sequences = [item_encoder.transform(s) + 1 for s in sequences_orig_ids]

    # 학습을 위한 (입력, 대상) 쌍 생성
    # 시퀀스 [i1, i2, i3, i4]의 경우:
    #   입력: [i1], 대상: i2
    #   입력: [i1, i2], 대상: i3
    #   입력: [i1, i2, i3], 대상: i4
    X_train_seqs, y_train_targets = [], []
    for seq in encoded_sequences:
        for i in range(1, len(seq)): # 입력으로 최소 하나의 아이템이 필요하므로 1부터 시작
            input_sub_sequence = seq[:i]    # 아이템 i-1까지의 시퀀스 (i 제외)
            target_item_label = seq[i]      # 인덱스 i의 아이템이 대상
            X_train_seqs.append(input_sub_sequence)
            y_train_targets.append(target_item_label)

    if not X_train_seqs:
        print("Error: Could not generate any training samples (X, y pairs). " # 오류: 학습 샘플(X, y 쌍)을 생성할 수 없습니다.
              "This might happen if all sequences have length 1 or less.") # 모든 시퀀스 길이가 1 이하인 경우 발생할 수 있습니다.
        return None, None, None, None

    # 입력 시퀀스를 `max_seq_len`으로 패딩
    # 'pre' 패딩: max_seq_len보다 짧은 시퀀스의 시작 부분에 0 추가.
    # 'pre' 자르기: max_seq_len보다 긴 시퀀스의 시작 부분에서 요소 제거.
    # `value=0`: 패딩 값으로 0 사용.
    X_padded = pad_sequences(X_train_seqs, maxlen=max_seq_len, padding='pre', truncating='pre', value=0)
    y_array = np.array(y_train_targets) # 대상 아이템은 이미 1부터 시작하는 인코딩된 아이템 인덱스임

    print(f"Data loaded: {len(df_sequences)} original sequences, resulting in {len(X_padded)} training samples.") # 데이터 로드됨: 원본 시퀀스 {len(df_sequences)}개, 학습 샘플 {len(X_padded)}개 생성됨.
    return X_padded, y_array, num_items_for_embedding, item_encoder

# --- SASRec 모델 구성 요소 ---
class PositionalEmbedding(Layer):
    """
    한국어: 아이템 임베딩에 위치 임베딩을 추가하기 위한 사용자 정의 Keras 레이어입니다.
    시퀀스에서 아이템의 위치는 순차 모델에 중요합니다.

    Custom Keras layer for adding positional embeddings to item embeddings.
    The position of an item in a sequence is important for sequential models.
    """
    def __init__(self, max_seq_len, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim
        # 위치 임베딩 레이어. 입력 차원은 max_seq_len + 1입니다.
        # 위치는 1부터 시작하는 인덱스(1부터 max_seq_len까지)이기 때문입니다.
        # 0번째 임베딩은 필요한 경우 패딩 위치에 대해 고려될 수 있지만,
        # 패딩 위치의 아이템이 마스크 처리되는 경우 일반적으로 명시적으로 사용되지 않습니다.
        self.pos_embeddings = Embedding(input_dim=max_seq_len + 1, output_dim=embed_dim, name=f"{self.name}_pos_emb_lookup")

    def call(self, x_item_embeddings): # x_item_embeddings의 모양은 (batch_size, seq_len, embed_dim)입니다.
        # 입력 아이템 임베딩 텐서에서 실제 시퀀스 길이 가져오기
        # (패딩되지 않았거나 마스킹이 다운스트림에서 처리되는 경우 max_seq_len보다 짧을 수 있음)
        seq_len = tf.shape(x_item_embeddings)[1]

        # 위치 인덱스 생성: [1, 2, ..., seq_len]
        positions = tf.range(start=1, limit=seq_len + 1, delta=1)
        # 이러한 위치에 대한 임베딩 조회
        embedded_positions = self.pos_embeddings(positions) # 모양: (seq_len, embed_dim)
        # 이는 배치 차원에 걸쳐 브로드캐스팅되어 item_embeddings에 추가됩니다.
        return embedded_positions

    def get_config(self): # 모델 저장 및 로딩용
        config = super().get_config()
        config.update({
            "max_seq_len": self.max_seq_len,
            "embed_dim": self.embed_dim,
        })
        return config

class TransformerBlock(Layer):
    """
    한국어: 멀티-헤드 자기 어텐션과 피드-포워드 네트워크로 구성된 단일 Transformer 블록입니다.
    잔차 연결과 레이어 정규화를 포함합니다.

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

        # 멀티-헤드 자기 어텐션 레이어
        # key_dim = embed_dim // num_heads는 프로젝션 헤드가 호환 가능한 차원을 갖도록 보장합니다.
        self.mha = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim // num_heads, dropout=dropout_rate, name=f"{self.name}_mha")

        # 점별 피드-포워드 네트워크 (FFN)
        self.ffn = tf.keras.Sequential([
            Dense(ffn_units, activation="relu", name=f"{self.name}_ffn_dense1"),
            Dropout(dropout_rate, name=f"{self.name}_ffn_dropout"),
            Dense(embed_dim, name=f"{self.name}_ffn_dense2") # 임베딩 차원으로 다시 프로젝션
        ], name=f"{self.name}_ffn")

        # 레이어 정규화 레이어
        self.layernorm1 = LayerNormalization(epsilon=1e-6, name=f"{self.name}_layernorm1")
        self.layernorm2 = LayerNormalization(epsilon=1e-6, name=f"{self.name}_layernorm2")

        # MHA 후 Dropout 레이어 (잔차 연결 추가 전)
        self.dropout_mha_output = Dropout(dropout_rate, name=f"{self.name}_mha_output_dropout")

    def call(self, inputs, training=False, attention_mask=None): # 명확성을 위해 causal_mask를 attention_mask로 이름 변경
        # `inputs` 모양: (batch_size, seq_len, embed_dim)
        # `attention_mask`는 미래 위치에 주목하는 것을 방지하는 인과적 마스크입니다.
        # 모양은 (batch_size, seq_len, seq_len)이어야 하며, 여기서 mask[b, i, j]가 True이면 배치 b에서 아이템 i에 대한 표현을 계산할 때
        # 아이템 j가 마스크 처리되어야 함(주목하지 않아야 함)을 의미합니다.

        # 멀티-헤드 자기 어텐션 하위 레이어
        # Query, Value, Key는 모두 자기 어텐션을 위한 동일한 `inputs`입니다.
        # `attention_mask`는 인과성을 보장합니다.
        attn_output = self.mha(query=inputs, value=inputs, key=inputs, attention_mask=attention_mask, training=training)
        attn_output_dropped = self.dropout_mha_output(attn_output, training=training)
        # 잔차 연결 추가 및 레이어 정규화 적용
        out1 = self.layernorm1(inputs + attn_output_dropped)

        # 피드-포워드 네트워크 하위 레이어
        ffn_output = self.ffn(out1, training=training)
        # 잔차 연결 추가 및 레이어 정규화 적용
        # (Dropout은 필요한 경우 Dense 레이어 뒤에 self.ffn 정의의 일부임)
        out2 = self.layernorm2(out1 + ffn_output) # 수정됨: out1 + ffn_output이어야 함
        return out2

    def get_config(self): # 모델 저장 및 로딩용
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ffn_units": self.ffn_units,
            "dropout_rate": self.dropout_rate,
        })
        return config

# --- SASRec 모델 구축 함수 ---
# 단일 순방향 패스에 대한 Time Complexity: Transformer 블록에 의해 지배됨.
# 각 블록: O(MAX_SEQ_LENGTH^2 * EMBEDDING_DIM + MAX_SEQ_LENGTH * EMBEDDING_DIM * FFN_UNITS)
# NUM_BLOCKS에 대한 총계: O(NUM_BLOCKS * (MAX_SEQ_LENGTH^2 * EMBEDDING_DIM + MAX_SEQ_LENGTH * EMBEDDING_DIM * FFN_UNITS))
def build_sasrec_model(max_seq_len, num_items_for_embedding, embed_dim, num_blocks, num_heads, ffn_units, dropout_rate):
    """
    한국어: Keras Functional API를 사용하여 SASRec 모델 아키텍처를 구축합니다.

    Builds the SASRec model architecture using Keras Functional API.
    """
    # 아이템 인덱스 시퀀스(0으로 패딩됨)를 위한 입력 레이어
    input_seq = Input(shape=(max_seq_len,), name="input_sequence", dtype='int32')

    # 아이템 임베딩 레이어
    # `mask_zero=True`는 후속 레이어가 패딩 토큰(0)을 무시하도록 지시합니다.
    item_embedding_layer = Embedding(
        input_dim=num_items_for_embedding, # 어휘 크기 (고유 아이템 수 + 패딩용 1)
        output_dim=embed_dim,
        name="item_embedding",
        mask_zero=True # 패딩된 시퀀스 처리에 중요
    )
    item_embs = item_embedding_layer(input_seq) # 모양: (batch_size, max_seq_len, embed_dim)

    # 위치 임베딩 레이어
    pos_embedding_layer = PositionalEmbedding(max_seq_len, embed_dim, name="positional_embedding")
    pos_embs = pos_embedding_layer(item_embs) # 모양: (max_seq_len, embed_dim)

    # 아이템 임베딩과 위치 임베딩 추가
    # 위치 임베딩은 배치 차원에 걸쳐 브로드캐스팅됩니다.
    seq_embs = item_embs + pos_embs
    seq_embs = Dropout(dropout_rate, name="input_dropout")(seq_embs)

    # --- 인과적 자기 어텐션 마스크 ---
    # 이 마스크는 위치 'i'의 아이템이 위치 'j <= i'의 아이템에만 주목할 수 있도록 보장합니다.
    # 주어진 대상에 대해 학습하는 동안 모델이 미래 아이템을 "보는" 것을 방지합니다.
    # 마스크는 *마스크 처리되어야 하는* (즉, 주목하지 않아야 하는) 위치에 대해 True를 가져야 합니다.
    # Keras MHA 레이어는 attention_mask 모양 (batch_size, Tq, Tv) 또는 (batch_size, num_heads, Tq, Tv)를 예상합니다.
    # 자기 어텐션의 경우 Tq (대상/쿼리 시퀀스 길이) = Tv (소스/값 시퀀스 길이) = max_seq_len입니다.

    # 1. (max_seq_len, max_seq_len) 모양의 부울 행렬 생성
    #    `tf.linalg.band_part(tf.ones((L, L)), -1, 0)`은 하삼각 행렬(대각선 및 그 아래는 1)을 만듭니다.
    #    `1.0 - lower_triangular_matrix`는 이를 반전시켜 미래 위치(상삼각)가 1이 되도록 합니다.
    #    `tf.cast(..., dtype=tf.bool)`은 부울로 변환합니다 (미래 위치의 경우 True).
    causal_mask_matrix = tf.cast(
        1.0 - tf.linalg.band_part(tf.ones((max_seq_len, max_seq_len)), -1, 0),
        dtype=tf.bool
    )
    # 브로드캐스팅을 위한 배치 차원 추가: (1, max_seq_len, max_seq_len)
    # 이 단일 마스크는 배치의 모든 샘플과 모든 어텐션 헤드에 걸쳐 브로드캐스팅됩니다.
    causal_attention_mask_for_mha = causal_mask_matrix[tf.newaxis, :, :]

    # 임베딩 레이어의 `mask_zero=True`는 자동으로 패딩 마스크를 생성합니다.
    # Keras의 MultiHeadAttention 레이어는 명시적 `attention_mask`(인과적 마스크와 같은)와
    # 임베딩 레이어에서 전파된 암시적 패딩 마스크를 올바르게 결합하도록 설계되었습니다.

    # Transformer 블록
    transformer_output = seq_embs
    for i in range(num_blocks):
        transformer_output = TransformerBlock(
            embed_dim, num_heads, ffn_units, dropout_rate, name=f"transformer_block_{i+1}"
        )(transformer_output, attention_mask=causal_attention_mask_for_mha) # 인과적 마스크 전달

    # 예측을 위해 입력 시퀀스의 *마지막 아이템* 표현 사용.
    # 'pre' 패딩으로 인해 마지막 실제 아이템은 시퀀스의 -1 인덱스에 있습니다.
    # 해당 표현은 자기 어텐션을 통해 모든 이전 아이템의 정보를 포착했습니다.
    last_item_representation = transformer_output[:, -1, :] # 모양: (batch_size, embed_dim)

    # 출력 레이어: 가능한 모든 다음 아이템에 대한 점수(로짓) 예측.
    # 유닛 수는 `num_items_for_embedding`(패딩 포함 어휘 크기)입니다.
    # `SparseCategoricalCrossentropy(from_logits=True)`가 원시 로짓을 예상하므로 여기에는 활성화 함수가 없습니다.
    output_logits = Dense(num_items_for_embedding, activation=None, name="output_logits")(last_item_representation)

    model = Model(inputs=input_seq, outputs=output_logits)
    return model

# --- 주어진 시퀀스에 대한 추천 생성 ---
def get_sasrec_recommendations(model, input_sequence_encoded_1based, item_encoder, num_items_for_embedding, num_recommendations=5):
    """
    한국어: 주어진 1부터 시작하는 인코딩된 입력 시퀀스에 대한 다음 아이템 추천을 생성합니다.

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
    if not input_sequence_encoded_1based: # 입력 목록이 비어 있는지 확인
        print("Input sequence is empty. Cannot generate recommendations.") # 입력 시퀀스가 비어 있습니다. 추천을 생성할 수 없습니다.
        return []

    # 입력 시퀀스를 MAX_SEQ_LENGTH로 패딩하고, 값 0으로 'pre' 패딩 사용.
    padded_input_sequence = pad_sequences(
        [input_sequence_encoded_1based], maxlen=MAX_SEQ_LENGTH, padding='pre', truncating='pre', value=0
    )

    # 모델 예측 가져오기 (어휘의 모든 아이템에 대한 로짓)
    # 단일 시퀀스에 대한 예측 Time Complexity: O(MAX_SEQ_LENGTH^2 * EMBEDDING_DIM + ...) (Transformer에 의해 지배됨)
    predicted_logits = model.predict(padded_input_sequence, verbose=0)[0] # 단일 입력 시퀀스에 대한 점수 가져오기

    # 추천해서는 안 되는 아이템 마스크 처리:
    # 1. 패딩 토큰 (인덱스 0)
    predicted_logits[0] = -np.inf

    # 2. 입력 시퀀스에 이미 있는 아이템 (사용자가 이미 상호작용한 아이템)
    for item_idx_1based_in_input in input_sequence_encoded_1based:
        if 0 < item_idx_1based_in_input < len(predicted_logits): # 경계 확인
             predicted_logits[item_idx_1based_in_input] = -np.inf # 점수를 음의 무한대로 설정하여 마스크 처리

    # 가장 높은 점수(로짓)를 가진 상위 N개 아이템의 인덱스 가져오기
    # `np.argsort`는 배열을 오름차순으로 정렬하는 인덱스를 반환합니다.
    # `[-num_recommendations:]` 슬라이싱은 상위 N개 최대값을 가져옵니다.
    # `[::-1]`은 점수 내림차순으로 정렬하기 위해 역순으로 만듭니다.
    top_n_item_indices_1based = np.argsort(predicted_logits)[-num_recommendations:][::-1]

    recommended_items = []
    for item_idx_1based in top_n_item_indices_1based:
        if item_idx_1based == 0: # -np.inf로 효과적으로 필터링되어야 하지만 이중 확인
            continue

        # 1부터 시작하는 인덱스(모델에서 0 패딩으로 인해 사용됨)를 item_encoder를 위해 0부터 시작하는 인덱스로 다시 변환
        item_idx_0based_for_encoder = item_idx_1based - 1

        # 인덱스가 인코더에 유효한지 확인
        if 0 <= item_idx_0based_for_encoder < len(item_encoder.classes_):
            original_item_id = item_encoder.inverse_transform([item_idx_0based_for_encoder])[0]
            recommended_items.append({'item_id': original_item_id, 'score': predicted_logits[item_idx_1based]})
        else:
            print(f"Warning: Skipping recommended item index {item_idx_1based} as it's out of bounds for item_encoder after adjustment.") # 경고: 권장 아이템 인덱스 {item_idx_1based}는 조정 후 item_encoder의 범위를 벗어나므로 건너<0xEB><0x9B><0x84>니다.


    return recommended_items

# --- 메인 실행 ---
# 학습 Time Complexity: O(EPOCHS * (N_train_samples / BATCH_SIZE) * SASRec_순방향_패스_복잡도)
if __name__ == "__main__":
    print("--- SASRec (Self-Attentive Sequential Recommendation) Example ---") # --- SASRec (자기 어텐션 순차 추천) 예제 ---

    print("\nLoading and preprocessing sequence data...") # 시퀀스 데이터 로드 및 전처리 중...
    X_padded, y_array, num_items_for_embedding, item_encoder = load_sequences()

    if X_padded is not None and item_encoder is not None:
        print(f"Padded input sequences shape: {X_padded.shape}") # 패딩된 입력 시퀀스 모양: {X_padded.shape}
        print(f"Target item array shape: {y_array.shape}") # 대상 아이템 배열 모양: {y_array.shape}
        print(f"Number of unique items for embedding (incl. padding): {num_items_for_embedding}") # 임베딩을 위한 고유 아이템 수 (패딩 포함): {num_items_for_embedding}

        print("\nBuilding SASRec model...") # SASRec 모델 구축 중...
        model = build_sasrec_model(
            MAX_SEQ_LENGTH,
            num_items_for_embedding,
            EMBEDDING_DIM,
            NUM_BLOCKS,
            NUM_HEADS,
            FFN_UNITS,
            DROPOUT_RATE
        )

        # 모델 컴파일
        # 다음 아이템(N개 아이템 중 클래스)을 예측하기 위해 SparseCategoricalCrossentropy 사용.
        # 모델이 원시 점수(로짓)를 출력하므로 `from_logits=True` (예: 소프트맥스에서).
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'] # 정확한 다음 아이템 예측 정확도.
        )
        model.summary() # 모델 아키텍처 출력

        print("\nTraining the model...") # 모델 학습 중...
        history = model.fit(
            X_padded,
            y_array,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=1 # 학습 진행 상황 표시
        )
        print("Model training completed.") # 모델 학습 완료.

        # 예시: 샘플 시퀀스에 대한 추천 생성
        if hasattr(item_encoder, 'classes_') and len(item_encoder.classes_) > 0:
            # 실제 테스트를 위해 원본 더미 데이터에서 예제 시퀀스를 가져오려고 시도
            raw_sequences_df_path = os.path.join(project_root, 'data/dummy_sequences.csv')
            example_input_sequence_orig_ids = []
            if os.path.exists(raw_sequences_df_path):
                raw_sequences_df = pd.read_csv(raw_sequences_df_path)
                if not raw_sequences_df.empty and 'item_ids_sequence' in raw_sequences_df.columns:
                    # 더미 데이터의 첫 번째 시퀀스를 예제로 사용
                    example_raw_sequence_str = raw_sequences_df['item_ids_sequence'].iloc[0]
                    example_input_sequence_orig_ids = [int(i) for i in example_raw_sequence_str.split(' ')]

            if not example_input_sequence_orig_ids: # 파일 로드 실패 또는 비어 있는 경우 대체
                print("\nWarning: Could not load example sequence from file, using a fallback sequence.") # 경고: 파일에서 예제 시퀀스를 로드할 수 없어 대체 시퀀스를 사용합니다.
                # 대체 아이템이 item_encoder.classes_에 있는지 확인하거나 인코딩된 값을 직접 사용
                if len(item_encoder.classes_) >= 3: # 인코더가 최소 3개 아이템을 알고 있는지 확인
                   example_input_sequence_orig_ids = item_encoder.classes_[:min(3, MAX_SEQ_LENGTH-1)].tolist() # 알려진 처음 3개 아이템 가져오기
                else:
                   example_input_sequence_orig_ids = [] # 의미 있는 시퀀스를 형성할 수 없음

            if example_input_sequence_orig_ids:
                 # 인코더에 알려지지 않은 아이템 필터링 (예: 더미 데이터가 변경된 경우)
                known_items_in_sequence_orig_ids = [item for item in example_input_sequence_orig_ids if item in item_encoder.classes_]

                if known_items_in_sequence_orig_ids:
                    # 시퀀스의 일부(예: MAX_SEQ_LENGTH-1까지의 마지막 몇 개 아이템)를 다음 예측을 위한 입력으로 사용
                    # 모델은 다음을 예측하기 위해 입력 시퀀스를 예상합니다.
                    input_for_recs_orig_ids = known_items_in_sequence_orig_ids[:MAX_SEQ_LENGTH-1] # 최대 입력 길이는 다음 예측을 위해 MAX_SEQ_LENGTH-1
                    if input_for_recs_orig_ids: # 필터링 및 슬라이싱 후 비어 있지 않은지 확인
                        input_for_recs_encoded_1based = item_encoder.transform(input_for_recs_orig_ids) + 1

                        print(f"\nExample: Input sequence for recommendation (Original IDs): {input_for_recs_orig_ids}") # 예시: 추천을 위한 입력 시퀀스 (원본 ID): {input_for_recs_orig_ids}
                        print(f"Encoded 1-based input sequence: {input_for_recs_encoded_1based.tolist()}") # 인코딩된 1부터 시작하는 입력 시퀀스: {input_for_recs_encoded_1based.tolist()}

                        recommendations = get_sasrec_recommendations(
                            model,
                            input_for_recs_encoded_1based.tolist(),
                            item_encoder,
                            num_items_for_embedding
                        )

                        print("\nRecommended next items:") # 추천된 다음 아이템:
                        if recommendations:
                            for rec in recommendations:
                                print(f"- Item ID (original): {rec['item_id']}, Predicted Score (logit): {rec['score']:.3f}")
                        else:
                            print("No recommendations could be generated for the example sequence.") # 예제 시퀀스에 대한 추천을 생성할 수 없습니다.
                    else:
                        print("\nWarning: Example input sequence became empty after filtering for known items or slicing.") # 경고: 알려진 아이템 필터링 또는 슬라이싱 후 예제 입력 시퀀스가 비어 있게 되었습니다.
                else:
                    print("\nWarning: None of the items in the example sequence are known to the item encoder.") # 경고: 예제 시퀀스의 아이템 중 어느 것도 아이템 인코더에 알려져 있지 않습니다.
            else:
                print("\nCould not prepare a valid example input sequence for recommendation.") # 추천을 위한 유효한 예제 입력 시퀀스를 준비할 수 없습니다.
        else:
            print("\nItem encoder not available or has no classes; cannot generate example recommendations.") # 아이템 인코더를 사용할 수 없거나 클래스가 없어 예제 추천을 생성할 수 없습니다.
    else:
        print("\nData loading failed. Cannot proceed with the SASRec example.") # 데이터 로드 실패. SASRec 예제를 진행할 수 없습니다.

    print("\n--- SASRec Sequential Recommendation Model example execution complete ---") # --- SASRec 순차 추천 모델 예제 실행 완료 ---
