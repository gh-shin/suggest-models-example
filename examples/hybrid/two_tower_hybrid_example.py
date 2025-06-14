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

# --- 2-타워 하이브리드 추천 모델: 기본 설명 ---
# 2-타워 모델은 추천 시스템 구축을 위한 매우 효과적이고 확장 가능한 아키텍처입니다.
# 대규모 산업 추천 시스템(예: YouTube, Spotify)에서 후보 생성에 특히 인기가 많습니다.
# 핵심 아이디어는 두 개의 독립적인 "타워"에서 사용자와 아이템에 대한 별도의 신경망 표현(임베딩)을 학습하는 것입니다.
# 이러한 임베딩은 일반적으로 내적을 통해 결합되어 사용자-아이템 상호작용 가능성을 예측합니다.
#
# 작동 방식:
# 1. 사용자 타워:
#    - 입력: 다양한 사용자 특징을 입력으로 받습니다. 여기에는 다음이 포함될 수 있습니다:
#        - 사용자 ID (범주형 특징, 임베딩 레이어를 통해 학습됨).
#        - 사용자 인구 통계 데이터 (예: 연령, 위치; 범주형 또는 수치형).
#        - 사용자 활동 기록 (예: 이전에 상호작용한 아이템의 임베딩, 풀링되거나 RNN/Transformer를 통해 처리됨).
#    - 아키텍처: 일반적으로 범주형 특징은 임베딩 레이어를 통과합니다. 수치형 특징은 정규화될 수 있습니다.
#      처리된 특징은 연결된 후 하나 이상의 Dense (완전 연결) 레이어를 통과하여 최종 고정 크기 사용자 임베딩 벡터를 생성합니다.
#    - 이 단순화된 예제에서는: 사용자 ID -> 사용자 임베딩 레이어 -> 사용자 벡터.
#
# 2. 아이템 타워:
#    - 입력: 다양한 아이템 특징을 입력으로 받습니다. 여기에는 다음이 포함될 수 있습니다:
#        - 아이템 ID (범주형 특징, 임베딩 레이어를 통해 학습됨).
#        - 아이템 메타데이터 (예: 장르, 카테고리, 브랜드; 종종 임베딩되는 범주형 특징).
#        - 아이템 콘텐츠 (예: TF-IDF로 처리된 텍스트 설명, 단어 임베딩 + CNN/RNN; CNN의 이미지 특징).
#    - 아키텍처: 사용자 타워와 유사하게 아이템 특징은 처리되고(범주형의 경우 임베딩, 콘텐츠의 경우 특수 네트워크)
#      일반적으로 연결된 후 Dense 레이어를 통과하여 최종 고정 크기 아이템 임베딩 벡터를 생성합니다.
#    - 이 예제에서는: 아이템 ID -> 아이템 임베딩; 아이템 장르 (multi-hot) -> 장르 임베딩 (풀링됨) -> 아이템 ID 임베딩과 연결 -> Dense 레이어 -> 아이템 벡터.
#
# 3. 상호작용 및 예측 (유사성 학습):
#    - 사용자 타워의 사용자 임베딩(U_vector)과 아이템 타워의 아이템 임베딩(I_vector)이 생성됩니다.
#    - (사용자, 아이템) 쌍에 대한 예측된 선호도 또는 상호작용 점수는 일반적으로 내적을 사용하여 계산됩니다:
#      score = dot(U_vector, I_vector).
#    - 높은 내적은 높은 유사성 또는 선호도를 나타냅니다. 코사인 유사도(정규화된 내적)도 일반적으로 사용됩니다.
#    - 이진 레이블(상호작용함/상호작용 안 함)로 학습하는 경우 이 점수는 종종 시그모이드 함수를 통과하여 확률을 얻습니다:
#      P(interaction | user, item) = sigmoid(score).
#
# 4. 학습 (임베딩 학습):
#    - 모델은 관찰된 사용자-아이템 상호작용(긍정적 쌍)에 대해 학습됩니다.
#    - 암시적 피드백 데이터에는 종종 명시적인 부정적 신호가 없으므로 부정적 샘플(사용자가 상호작용하지 않은 아이템)이 중요합니다.
#      이는 종종 무작위 샘플링 또는 더 정교한 방법(예: 다른 사용자에게는 인기가 있지만 현재 사용자는 상호작용하지 않은 아이템 샘플링)을 통해 생성됩니다.
#    - 손실 함수:
#        - 명시적 피드백 (평점)의 경우: 평균 제곱 오차 (내적이 직접 평점을 예측하는 경우).
#        - 암시적 피드백 (클릭, 조회)의 경우: 이진 교차 엔트로피가 시그모이드 출력과 함께 일반적으로 사용됩니다.
#          순위 손실(예: Hinge 손실, BPR 손실, Triplet 손실)도 상대적 선호도를 학습하는 데 효과적입니다.
#
# 5. 서빙 / 추천 생성:
#    - 대규모 환경에서의 효율성이 주요 장점입니다.
#    - 사용자 타워: 주어진 사용자에 대해 사용자 임베딩 벡터 U_vector를 계산합니다.
#    - 아이템 타워: 코퍼스의 모든 아이템에 대한 아이템 임베딩 벡터 I_vector를 미리 계산합니다. 이는 오프라인으로 수행할 수 있습니다.
#    - 후보 검색: Approximate Nearest Neighbor (ANN) 검색 시스템(예: FAISS, ScaNN)을 사용하여 미리 계산된 코퍼스에서
#      사용자의 임베딩 U_vector와 가장 높은 내적(또는 코사인 유사도)을 갖는 상위 K개 아이템 임베딩을 효율적으로 찾습니다.
#      이러한 아이템이 추천 후보를 형성합니다.
#    - 이 "후보 생성" 단계는 종종 후보를 다시 순위 매기는 더 복잡한 "순위 지정" 모델이 뒤따릅니다.
#
# Pros (장점):
# - 서빙 확장성: 타워 분리를 통해 아이템 임베딩을 미리 계산하고 인덱싱할 수 있어 수백만 개의 아이템이 있더라도
#   실시간 추천 검색이 매우 빠릅니다.
# - 특징 공학 유연성: 다양한 사용자 및 아이템 특징(범주형, 수치형, 텍스트, 시각적)을 각 타워에 쉽게 통합할 수 있습니다.
#   각 타워는 자체 특화된 아키텍처를 가질 수 있습니다.
# - 후보 생성에 효과적: 방대한 아이템 코퍼스를 추가 순위 지정 또는 직접 표시를 위한 더 작고 관련성 있는 후보 집합으로
#   좁히는 데 매우 효율적입니다.
# - 우수한 일반화: 복잡한 관계를 포착할 수 있는 의미 있는 밀집 표현을 학습합니다.
#
# Cons (단점):
# - 제한적인 상호작용 모델링: 최종 사용자-아이템 상호작용은 종종 내적(또는 유사한 얕은 상호작용)으로 단순화됩니다.
#   이는 네트워크 초기에 더 깊고 명시적인 교차 특징 상호작용을 허용하는 모델(예: DCN, xDeepFM)만큼
#   사용자-아이템 선호도의 모든 미묘한 차이를 효과적으로 포착하지 못할 수 있습니다.
# - 타워 내 특징 공학: 유연하지만 성능은 여전히 각 타워에 대해 공학된 특징의 품질과 정보성에 크게 의존합니다.
# - Cold-Start: 많은 임베딩 기반 모델과 마찬가지로 상호작용 데이터가 희소하거나 존재하지 않는 새 사용자 또는 아이템에 대해
#   어려움을 겪을 수 있으며, 충분한 데이터 없이는 의미 있는 임베딩을 쉽게 학습할 수 없습니다.
#   타워의 콘텐츠 특징은 새 아이템에 대해 어느 정도 이를 완화하는 데 도움이 될 수 있습니다.
# ---

# 프로젝트 루트를 sys.path에 동적으로 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- 모델 하이퍼파라미터 ---
USER_EMBEDDING_DIM = 32       # 사용자 임베딩 벡터의 차원.
ITEM_EMBEDDING_DIM = 32       # 아이템 ID 임베딩 벡터의 차원.
GENRE_EMBEDDING_DIM = 16      # 장르 임베딩 벡터의 차원.
MAX_GENRES_PER_ITEM = 5       # 아이템당 고려할 최대 장르 수 (장르 시퀀스 패딩용).
DENSE_UNITS = [64, USER_EMBEDDING_DIM] # 아이템 타워의 Dense 레이어. 마지막 레이어는 내적을 위해 사용자 임베딩 차원과 일치해야 합니다.
LEARNING_RATE = 0.001         # Adam 옵티마이저의 학습률.
EPOCHS = 5                    # 학습 에포크 수 (예제를 위해 작게 설정).
BATCH_SIZE = 64               # 학습을 위한 배치 크기.
NEGATIVE_SAMPLES = 4          # 학습 중 각 긍정적 상호작용에 대해 생성할 부정적 샘플 수.

# --- 데이터 로딩 및 전처리 ---
# Time Complexity: CSV 읽기(O(N_interactions + N_items_metadata)) 및 네거티브 샘플링에 의해 지배됨.
# 네거티브 샘플링은 O(N_positive_samples * NEGATIVE_SAMPLES * Avg_time_to_find_negative)가 될 수 있음.
def load_and_preprocess_data(
    base_interactions_filepath='data/dummy_interactions.csv',
    base_metadata_filepath='data/dummy_item_metadata.csv'
):
    """
    한국어: 2-타워 모델을 위해 상호작용 및 아이템 메타데이터를 로드하고 전처리합니다.
    - 파일이 누락된 경우 더미 데이터 생성을 처리합니다.
    - 사용자 및 아이템 ID를 인코딩합니다.
    - 아이템에 대한 장르 정보를 토큰화하고 패딩합니다.
    - 학습을 위한 부정적 샘플을 생성합니다.
    - 데이터를 학습 및 테스트 세트로 분할합니다.

    Loads interaction and item metadata, preprocesses them for the Two-Tower model.
    - Handles dummy data generation if files are missing.
    - Encodes user and item IDs.
    - Tokenizes and pads genre information for items.
    - Generates negative samples for training.
    - Splits data into training and testing sets.
    """
    interactions_filepath = os.path.join(project_root, base_interactions_filepath)
    metadata_filepath = os.path.join(project_root, base_metadata_filepath)

    # 데이터 파일이 존재하는지 확인하고, 없으면 더미 데이터 생성을 시도합니다.
    files_missing = False
    for fp_abs, fp_rel_for_check in [(interactions_filepath, 'data/dummy_interactions.csv'),
                                     (metadata_filepath, 'data/dummy_item_metadata.csv')]:
        if not os.path.exists(fp_abs):
            print(f"Warning: Data file not found at {fp_abs} (expected relative: {fp_rel_for_check}).") # 경고: {fp_abs}에서 데이터 파일을 찾을 수 없습니다 (예상 상대 경로: {fp_rel_for_check}).
            if fp_abs.endswith(fp_rel_for_check): # 특정 더미 파일인 경우에만 생성 시도
                 files_missing = True
            else: # 특정 비-더미 파일이 누락되어 진행할 수 없음
                print(f"Error: Required file {fp_abs} is missing and is not a default dummy file.") # 오류: 필수 파일 {fp_abs}이(가) 누락되었으며 기본 더미 파일이 아닙니다.
                return None


    if files_missing:
        print("Attempting to generate dummy data (generate_sequences=False)...") # 더미 데이터 생성 시도 중 (generate_sequences=False)...
        try:
            from data.generate_dummy_data import generate_dummy_data
            # 2-타워 모델은 기본적으로 SASRec과 같이 미리 생성된 시퀀스를 필요로 하지 않습니다.
            generate_dummy_data(num_users=200, num_items=100, num_interactions=2000, generate_sequences=False)
            print("Dummy data generation script executed.") # 더미 데이터 생성 스크립트 실행됨.
            # 파일 다시 확인
            if not os.path.exists(interactions_filepath) or not os.path.exists(metadata_filepath):
                print("Error: One or both dummy data files still not found after generation attempt.") # 오류: 생성 시도 후에도 하나 또는 두 개의 더미 데이터 파일을 찾을 수 없습니다.
                return None
            print("Dummy data files should now be available.") # 이제 더미 데이터 파일을 사용할 수 있습니다.
        except ImportError as e_import:
            print(f"ImportError: Failed to import 'generate_dummy_data'. Error: {e_import}") # ImportError: 'generate_dummy_data'를 임포트하지 못했습니다. 오류: {e_import}
            return None
        except Exception as e:
            print(f"Error during dummy data generation: {e}") # 더미 데이터 생성 중 오류: {e}
            return None

    df_interactions = pd.read_csv(interactions_filepath)
    df_items_meta = pd.read_csv(metadata_filepath)

    if df_interactions.empty or df_items_meta.empty:
        print("Error: Interaction or item metadata is empty after loading.") # 오류: 로드 후 상호작용 또는 아이템 메타데이터가 비어 있습니다.
        return None

    # ID 유형 표준화 (일관된 인코딩 및 병합에 중요)
    df_interactions['user_id'] = df_interactions['user_id'].astype(str)
    df_interactions['item_id'] = df_interactions['item_id'].astype(str)
    df_items_meta['item_id'] = df_items_meta['item_id'].astype(str)

    # LabelEncoder를 사용하여 사용자 ID 인코딩 (원본 ID를 0부터 시작하는 정수로 매핑)
    user_encoder = LabelEncoder()
    df_interactions['user_idx'] = user_encoder.fit_transform(df_interactions['user_id'])
    num_users = len(user_encoder.classes_)

    # 두 DataFrame 모두에서 일관되게 아이템 ID 인코딩
    # LabelEncoder를 적합시키기 전에 상호작용 및 메타데이터의 모든 고유 아이템 ID 수집
    all_item_ids_from_interactions = df_interactions['item_id'].unique()
    all_item_ids_from_metadata = df_items_meta['item_id'].unique()
    combined_item_ids = pd.Series(list(set(all_item_ids_from_interactions) | set(all_item_ids_from_metadata))).astype(str)

    item_encoder = LabelEncoder()
    item_encoder.fit(combined_item_ids)

    df_interactions['item_idx'] = item_encoder.transform(df_interactions['item_id'])
    df_items_meta['item_idx'] = item_encoder.transform(df_items_meta['item_id'])
    num_items = len(item_encoder.classes_)

    # 장르 전처리:
    # 1. 장르 문자열을 목록으로 분할. 누락된 장르는 빈 문자열로 채워서 처리.
    df_items_meta['genres_list'] = df_items_meta['genres'].fillna('').astype(str).apply(lambda x: x.split(';') if x else [])

    # 2. Keras Tokenizer를 사용하여 장르 이름을 정수 시퀀스로 변환.
    #    `oov_token`은 학습에는 없지만 테스트/새 데이터에 있는 장르를 처리.
    genre_tokenizer = Tokenizer(oov_token="<unk>")
    genre_tokenizer.fit_on_texts(df_items_meta['genres_list'])

    # 3. 장르 문자열 목록을 정수 토큰 시퀀스 목록으로 변환.
    item_genre_sequences = genre_tokenizer.texts_to_sequences(df_items_meta['genres_list'])

    # 4. 장르 시퀀스를 고정 길이(MAX_GENRES_PER_ITEM)로 패딩.
    #    `padding='post'`: 끝에 패딩 토큰 추가.
    #    `truncating='post'`: 너무 길면 끝에서 자름.
    #    `value=0`: 패딩 토큰으로 0 사용. Tokenizer는 지정하지 않으면 패딩용으로 0을 예약하지만,
    #               명시하는 것이 좋음. `word_index`는 1부터 시작.
    item_genre_padded = pad_sequences(item_genre_sequences, maxlen=MAX_GENRES_PER_ITEM, padding='post', truncating='post', value=0)

    # 장르 임베딩 레이어의 어휘 크기 (+1은 패딩 토큰 0용, Tokenizer가 암묵적으로 고려하지 않는 경우,
    # pad_sequences에서 value=0을 사용하고 0이 word_index에 없는 경우).
    # `len(word_index)`는 고유 단어 수. `+1`은 패딩/OOV용으로 예약된 인덱스 0을 위한 공간을 만듦.
    # Keras Tokenizer의 word_index는 1부터 시작. 따라서 max_index = len(word_index). input_dim = max_index + 1이 필요.
    num_genres_vocab = len(genre_tokenizer.word_index) + 1 # +1은 word_index가 1부터 시작하고 임베딩에 패딩용 0이 필요하기 때문.

    # 인코딩된 item_idx에서 해당 패딩된 장르 시퀀스로의 매핑 생성 (쉬운 조회용).
    item_idx_to_genres_padded_seq = {row['item_idx']: item_genre_padded[i] for i, row in df_items_meta.iterrows()}

    # --- 네거티브 샘플링으로 학습 데이터 생성 ---
    # 긍정적 샘플: 실제 사용자-아이템 상호작용. 레이블 = 1.0
    positive_samples_df = df_interactions[['user_idx', 'item_idx']].copy()
    positive_samples_df['label'] = 1.0 # Keras 손실 함수와 호환되도록 레이블에 float 사용.

    # 부정적 샘플: 상호작용하지 않은 사용자-아이템 쌍. 레이블 = 0.0
    # 이는 암시적 피드백으로 모델을 학습시키는 데 중요한 단계임.
    negative_samples_list = []
    all_possible_item_indices_encoded = np.arange(num_items) # 모든 인코딩된 아이템 인덱스: [0, 1, ..., num_items-1]

    # 네거티브 샘플링 중 빠른 조회를 위해 기존 상호작용 세트 생성
    user_item_interaction_set = set(zip(positive_samples_df['user_idx'], positive_samples_df['item_idx']))

    print(f"Generating {NEGATIVE_SAMPLES} negative samples for each of {len(positive_samples_df)} positive interactions...") # {len(positive_samples_df)}개의 긍정적 상호작용 각각에 대해 {NEGATIVE_SAMPLES}개의 부정적 샘플 생성 중...
    for _, row in positive_samples_df.iterrows():
        user_idx = row['user_idx']
        num_neg_generated = 0
        attempts = 0 # 사용자가 거의 모든 아이템을 평가한 경우 무한 루프 방지용
        while num_neg_generated < NEGATIVE_SAMPLES and attempts < num_items * 2: # 안전 브레이크
            negative_item_idx = np.random.choice(all_possible_item_indices_encoded)
            if (user_idx, negative_item_idx) not in user_item_interaction_set:
                negative_samples_list.append({'user_idx': user_idx, 'item_idx': negative_item_idx, 'label': 0.0})
                num_neg_generated += 1
            attempts +=1
        if num_neg_generated < NEGATIVE_SAMPLES:
            print(f"Warning: Could only generate {num_neg_generated}/{NEGATIVE_SAMPLES} negative samples for user_idx {user_idx}.") # 경고: 사용자 ID {user_idx}에 대해 {NEGATIVE_SAMPLES}개 중 {num_neg_generated}개의 부정적 샘플만 생성할 수 있었습니다.


    df_negative_samples = pd.DataFrame(negative_samples_list)

    # 긍정적 샘플과 부정적 샘플 결합 후 셔플
    training_data_df = pd.concat([positive_samples_df, df_negative_samples]).sample(frac=1, random_state=42).reset_index(drop=True)

    # 학습 및 테스트 세트로 분할
    train_df, test_df = train_test_split(training_data_df, test_size=0.2, random_state=42, stratify=training_data_df['label'])

    print(f"Data preprocessing complete. Training samples: {len(train_df)}, Test samples: {len(test_df)}") # 데이터 전처리 완료. 학습 샘플: {len(train_df)}, 테스트 샘플: {len(test_df)}
    print(f"Unique users: {num_users}, Unique items: {num_items}, Unique genre tokens (incl. padding/OOV): {num_genres_vocab}") # 고유 사용자: {num_users}, 고유 아이템: {num_items}, 고유 장르 토큰 (패딩/OOV 포함): {num_genres_vocab}

    return (train_df, test_df,
            user_encoder, item_encoder, genre_tokenizer,
            num_users, num_items, num_genres_vocab, # num_genres_vocab 사용
            item_idx_to_genres_padded_seq, df_items_meta)

# --- 모델 정의 ---
def build_two_tower_model(num_users, num_items, num_genres_vocab):
    """
    한국어: 2-타워 Keras 모델을 구축합니다.
    사용자 타워, 아이템 타워를 생성한 다음 이를 결합하여
    사용자-아이템 상호작용을 예측하도록 학습합니다.

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
    # --- 사용자 타워 ---
    # 입력: 사용자의 인코딩된 정수 인덱스.
    user_input_idx = Input(shape=(1,), name='user_input_idx_tower', dtype='int32')
    # 임베딩 레이어: 사용자 인덱스를 USER_EMBEDDING_DIM 크기의 밀집 벡터에 매핑.
    user_embedding_layer = Embedding(input_dim=num_users, output_dim=USER_EMBEDDING_DIM, name='user_embedding')
    user_embedding_vector = user_embedding_layer(user_input_idx)
    # 임베딩 출력을 1D 벡터로 평탄화 (batch_size, USER_EMBEDDING_DIM).
    user_vector = Flatten(name='flatten_user_embedding')(user_embedding_vector)
    # 선택 사항: 필요한 경우 사용자 임베딩을 추가로 변환하기 위해 여기에 Dense 레이어를 더 추가.
    # 이 예제에서는 단순화를 위해 직접 임베딩을 사용자의 최종 표현으로 사용.
    user_model = Model(inputs=user_input_idx, outputs=user_vector, name='user_tower')

    # --- 아이템 타워 ---
    # 입력 1: 아이템의 인코딩된 정수 인덱스.
    item_id_input_idx = Input(shape=(1,), name='item_id_input_idx_tower', dtype='int32')
    # 입력 2: 아이템의 패딩된 인코딩된 장르 토큰 시퀀스.
    item_genre_input_seq = Input(shape=(MAX_GENRES_PER_ITEM,), name='item_genre_input_seq_tower', dtype='int32')

    # 아이템 ID 임베딩
    item_id_embedding_layer = Embedding(input_dim=num_items, output_dim=ITEM_EMBEDDING_DIM, name='item_id_embedding')
    item_id_vector = Flatten(name='flatten_item_id_embedding')(item_id_embedding_layer(item_id_input_idx))

    # 장르 임베딩
    # `mask_zero=False`는 Tokenizer에서 0이 명시적으로 다르게 처리되지 않는 한 유효한 토큰이기 때문에 사용됨.
    # 그러나 pad_sequences에서 0이 엄격하게 패딩용인 경우, 원한다면 mask_zero=True를 사용할 수 있음.
    # GlobalAveragePooling1D는 마스크되지 않은 단계에 대해 평균을 내어 가변 길이를 처리함.
    genre_embedding_layer = Embedding(input_dim=num_genres_vocab, output_dim=GENRE_EMBEDDING_DIM, name='genre_embedding', mask_zero=False)
    genre_embedding_vectors = genre_embedding_layer(item_genre_input_seq) # 모양: (batch, MAX_GENRES_PER_ITEM, GENRE_EMBEDDING_DIM)
    # 장르 임베딩을 아이템에 대한 단일 고정 크기 벡터로 풀링.
    # GlobalAveragePooling1D는 MAX_GENRES_PER_ITEM 차원에 걸쳐 임베딩을 평균냄.
    genre_pooled_vector = GlobalAveragePooling1D(name='pool_genre_embeddings')(genre_embedding_vectors)

    # 아이템 ID 임베딩과 풀링된 장르 임베딩 연결.
    concatenated_item_features = Concatenate(name='concat_item_features')([item_id_vector, genre_pooled_vector])

    # 연결된 아이템 특징을 Dense 레이어를 통과시켜 최종 아이템 임베딩을 얻음.
    current_item_dense_layer = concatenated_item_features
    for i, units in enumerate(DENSE_UNITS):
        # 마지막 Dense 레이어는 USER_EMBEDDING_DIM(또는 내적을 위해 선택된 차원)의 벡터를 출력해야 함.
        # 내적 전 타워의 마지막 레이어에 'linear' 활성화를 사용하는 것이 일반적임.
        activation = 'relu' if i < len(DENSE_UNITS) - 1 else 'linear'
        current_item_dense_layer = Dense(units, activation=activation, name=f'item_dense_layer_{i+1}')(current_item_dense_layer)
    item_vector = current_item_dense_layer # 이것이 아이템 타워의 최종 아이템 임베딩임.

    # 아이템 타워 모델 (아이템 임베딩 미리 계산에 사용 가능)
    item_model = Model(inputs=[item_id_input_idx, item_genre_input_seq], outputs=item_vector, name='item_tower')

    # --- 학습 모델을 위해 타워 결합 ---
    # 결합된 학습 모델에 대한 입력 정의. 이는 `model.fit()` 중에 공급됨.
    # 여기서 이름('user_idx_input', 'item_idx_input', 'genre_seq_input')은
    # `model.fit(x=...)`에 전달된 딕셔너리의 키와 일치해야 함.
    user_idx_training_input = Input(shape=(1,), name='user_idx_input', dtype='int32')
    item_idx_training_input = Input(shape=(1,), name='item_idx_input', dtype='int32')
    genre_seq_training_input = Input(shape=(MAX_GENRES_PER_ITEM,), name='genre_seq_input', dtype='int32')

    # 각 타워에서 임베딩 가져오기
    user_embedding_for_training = user_model(user_idx_training_input)
    item_embedding_for_training = item_model([item_idx_training_input, genre_seq_training_input])

    # 사용자 및 아이템 임베딩 간의 내적 계산. 이것이 유사성 점수임.
    # `axes=1`은 임베딩 차원을 따라 내적을 의미함.
    # 표준 내적의 경우 `normalize=False`. 코사인 유사도를 계산하려면 `normalize=True`.
    interaction_score = Dot(axes=1, normalize=False, name='dot_product_interaction')([user_embedding_for_training, item_embedding_for_training])

    # 출력 레이어: 상호작용 확률(0 또는 1)을 예측하기 위한 시그모이드 활성화가 있는 단일 뉴런.
    output_probability = Dense(1, activation='sigmoid', name='interaction_probability_output')(interaction_score)

    # 학습에 사용되는 전체 모델
    training_model = Model(
        inputs={ # 입력 딕셔너리, 키는 Input 레이어 이름과 일치
            'user_idx_input': user_idx_training_input,
            'item_idx_input': item_idx_training_input,
            'genre_seq_input': genre_seq_training_input
        },
        outputs=output_probability,
        name='two_tower_training_model'
    )
    return training_model, user_model, item_model

# --- 추천 생성 ---
# 사용자 한 명에 대한 추천 제공 Time Complexity:
# - 사용자 임베딩: O(사용자_타워_순방향_패스_복잡도) (빠름)
# - 아이템 임베딩: O(N_items * 아이템_타워_순방향_패스_복잡도) (미리 계산하여 저장 가능)
# - 유사도 (내적) + Top-K: 브루트 포스의 경우 O(N_items * Embedding_Dim), ANN 사용 시 O(log N_items).
def get_two_tower_recommendations(user_id_original, user_encoder, item_encoder,
                                  user_model, item_model,
                                  item_idx_to_genres_padded_seq, all_possible_item_indices_encoded,
                                  num_recommendations=5):
    """
    한국어: 학습된 2-타워 모델을 사용하여 주어진 사용자에 대한 상위 N개 추천을 생성합니다.
    사용자의 임베딩을 가져온 다음 모든 아이템 임베딩과 비교하는 과정을 포함합니다.

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
        # 원본 사용자 ID를 인코딩된 정수 인덱스로 변환
        user_idx_encoded = user_encoder.transform(np.array([user_id_original]))[0] # 입력이 배열과 유사한지 확인
    except ValueError:
        print(f"Error: User ID '{user_id_original}' not found in user_encoder. Cannot generate recommendations.") # 오류: 사용자 ID '{user_id_original}'를 user_encoder에서 찾을 수 없습니다. 추천을 생성할 수 없습니다.
        return []

    # 1. user_model을 사용하여 대상 사용자에 대한 임베딩을 가져옵니다.
    # 입력은 NumPy 배열이어야 합니다.
    user_embedding = user_model.predict(np.array([user_idx_encoded]), verbose=0)

    # 2. 모든 아이템에 대한 임베딩을 얻기 위해 item_model에 대한 입력 준비.
    #    - all_item_ids_input: 모든 인코딩된 아이템 ID의 배열.
    #    - all_item_genres_input: 이러한 아이템에 대한 해당 패딩된 장르 시퀀스 배열.
    all_item_ids_input_for_model = all_possible_item_indices_encoded.reshape(-1, 1) # Keras 모델 입력을 위해 재구성

    # 모든 아이템에 대한 장르 시퀀스 검색. 누락된 항목은 패딩으로 처리.
    default_genre_padding = [0] * MAX_GENRES_PER_ITEM
    all_item_genres_input_for_model = np.array(
        [item_idx_to_genres_padded_seq.get(idx, default_genre_padding) for idx in all_possible_item_indices_encoded]
    )

    # 3. item_model을 사용하여 모든 아이템에 대한 임베딩 가져오기.
    #    실제 시스템에서는 효율성을 위해 이를 미리 계산하여 저장할 수 있습니다.
    print(f"Generating embeddings for all {len(all_possible_item_indices_encoded)} items...") # {len(all_possible_item_indices_encoded)}개 모든 아이템에 대한 임베딩 생성 중...
    all_item_embeddings = item_model.predict(
        [all_item_ids_input_for_model, all_item_genres_input_for_model],
        batch_size=BATCH_SIZE, # 예측을 위한 합리적인 배치 크기 사용
        verbose=0
    )
    print("Item embeddings generated.") # 아이템 임베딩 생성됨.

    # 4. 사용자의 임베딩과 모든 아이템 임베딩 간의 유사도(내적) 계산.
    #    user_embedding 모양: (1, USER_EMBEDDING_DIM)
    #    all_item_embeddings 모양: (num_items, USER_EMBEDDING_DIM)
    #    결과 유사도 모양: (num_items,)
    similarities = np.dot(all_item_embeddings, user_embedding.T).flatten()

    # 5. 가장 높은 유사도 점수를 가진 아이템의 인덱스 가져오기.
    #    `np.argsort`는 오름차순으로 정렬하는 인덱스를 반환합니다.
    #    `[-num_recommendations:]` 슬라이싱은 상위 N개 최대값을 가져옵니다.
    #    `[::-1]`은 점수 내림차순으로 정렬하기 위해 역순으로 만듭니다.
    top_n_indices_in_all_items_array = np.argsort(similarities)[-num_recommendations:][::-1]

    # 6. 추천 형식 지정: 인코딩된 아이템 인덱스를 원본 아이템 ID로 다시 변환.
    recommendations = []
    for array_idx in top_n_indices_in_all_items_array:
        # `array_idx`는 `all_possible_item_indices_encoded` 및 `similarities`에 대한 인덱스임
        item_idx_encoded = all_possible_item_indices_encoded[array_idx]
        original_item_id = item_encoder.inverse_transform([item_idx_encoded])[0] # inverse_transform은 배열을 예상함
        recommendations.append({
            'item_id': original_item_id,
            'similarity_score': similarities[array_idx]
        })
    return recommendations

# --- 메인 실행 ---
# 전체 학습 Time Complexity: O(EPOCHS * (N_train_samples / BATCH_SIZE) * Complexity_of_Forward_Backward_Pass)
# 순방향/역방향 패스의 복잡도는 두 타워 모두에서 임베딩 조회 및 Dense 레이어 연산의 영향을 받음.
if __name__ == "__main__":
    print("--- Two-Tower Hybrid Recommender Example ---") # --- 2-타워 하이브리드 추천 예제 ---

    print("\nLoading and preprocessing data...") # 데이터 로드 및 전처리 중...
    # 데이터 로드 및 전처리
    processed_data = load_and_preprocess_data()

    if processed_data:
        (train_df, test_df,
         user_encoder, item_encoder, genre_tokenizer,
         num_users, num_items, num_genres_vocab, # num_genres를 num_genres_vocab으로 이름 변경
         item_idx_to_genres_padded_seq, df_items_meta) = processed_data

        print(f"\nBuilding Two-Tower model (User features: ID; Item features: ID, Genres)...") # 2-타워 모델 구축 중 (사용자 특징: ID; 아이템 특징: ID, 장르)...
        training_model, user_model, item_model = build_two_tower_model(num_users, num_items, num_genres_vocab)

        # 학습 모델 컴파일
        # 상호작용 확률(0 또는 1)을 예측하므로 BinaryCrossentropy 사용.
        training_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
            loss='binary_crossentropy',
            metrics=['accuracy'] # 이진 레이블 예측 정확도.
        )

        print("\n--- Training Model Summary ---") # --- 학습 모델 요약 ---
        training_model.summary()
        print("\n--- User Tower Summary ---") # --- 사용자 타워 요약 ---
        user_model.summary()
        print("\n--- Item Tower Summary ---") # --- 아이템 타워 요약 ---
        item_model.summary()

        # Keras 모델을 위한 학습 및 테스트 데이터 입력 준비.
        # 입력은 Input 레이어의 `name` 속성과 일치하는 키를 가진 딕셔너리여야 함.
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

        print("\nTraining the Two-Tower model...") # 2-타워 모델 학습 중...
        history = training_model.fit(
            train_inputs_dict,
            train_labels_array,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_data=(test_inputs_dict, test_labels_array),
            verbose=1 # 학습 진행 상황 표시
        )
        print("Model training completed.") # 모델 학습 완료.

        # 예제 사용자에 대한 추천 생성
        if hasattr(user_encoder, 'classes_') and user_encoder.classes_.size > 0:
            # 인코더의 알려진 클래스에서 첫 번째 사용자를 예제로 선택
            target_user_original_id = user_encoder.classes_[0]

            # item_encoder가 알고 있는 모든 고유 인코딩된 아이템 인덱스 가져오기
            all_item_indices_encoded = item_encoder.transform(item_encoder.classes_)

            print(f"\nGenerating recommendations for User ID (original): {target_user_original_id}...") # 사용자 ID (원본): {target_user_original_id}에 대한 추천 생성 중...
            recommendations = get_two_tower_recommendations(
                target_user_original_id, user_encoder, item_encoder,
                user_model, item_model,
                item_idx_to_genres_padded_seq, # 수정된 변수 이름
                all_item_indices_encoded,
                num_recommendations=5
            )

            print("\nRecommended items:") # 추천 아이템:
            if recommendations:
                for rec in recommendations:
                    # 자세한 내용을 표시하기 위해 아이템 메타데이터와 병합
                    item_details = df_items_meta[df_items_meta['item_id'] == str(rec['item_id'])] # 조회를 위해 ID가 문자열인지 확인
                    description = item_details['description'].iloc[0] if not item_details.empty else "N/A"
                    genres_display = item_details['genres'].iloc[0] if not item_details.empty else "N/A"
                    print(f"- Item ID: {rec['item_id']} (Similarity Score: {rec['similarity_score']:.4f}) "
                          f"| Genres: {genres_display} | Description: {description[:60]}...")
            else:
                print("No recommendations could be generated for this user.") # 이 사용자에 대한 추천을 생성할 수 없습니다.
        else:
            print("\nCannot generate recommendations as no users were encoded or available.") # 인코딩되었거나 사용 가능한 사용자가 없어 추천을 생성할 수 없습니다.
    else:
        print("\nData loading and preprocessing failed. Cannot proceed with the Two-Tower example.") # 데이터 로드 및 전처리에 실패했습니다. 2-타워 예제를 진행할 수 없습니다.

    print("\n--- Two-Tower Hybrid Recommender Example Finished ---") # --- 2-타워 하이브리드 추천 예제 완료 ---
