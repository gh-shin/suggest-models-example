# examples/deep_learning/dnn_recommender.py
import pandas as pd
import numpy as np
import os
import sys
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Concatenate, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# --- 임베딩을 사용한 딥러닝(DNN) 추천: 기본 설명 ---
# 딥러닝 신경망(DNN)은 임베딩 레이어와 결합될 때 유연하고 강력한 추천 시스템 접근 방식을 제공합니다.
# 사용자-아이템 상호작용 데이터 내의 복잡한 비선형 관계를 포착하는 데 뛰어나며,
# 사용자와 아이템 모두에 대해 풍부하고 밀집된 벡터 표현(임베딩)을 학습할 수 있습니다.
#
# 일반적인 작동 방식 (2-타워 모델 구조를 사용한 평점 예측 작업의 경우):
# 1. 입력 데이터: 사용자 ID, 아이템 ID 및 해당 평점 (또는 암시적 피드백).
# 2. 임베딩 레이어:
#    - 사용자 ID는 사용자 임베딩 레이어를 사용하여 밀집 벡터에 매핑됩니다. 이 레이어는 각 사용자를
#      선호도 측면에서 유사한 사용자가 더 가까이 위치하는 저차원 공간에서 표현하는 방법을 학습합니다.
#      입력: 사용자 ID (정수), 출력: 사용자 임베딩 벡터 (예: 32차원).
#    - 아이템 ID도 유사하게 아이템 임베딩 레이어를 사용하여 매핑됩니다.
#      입력: 아이템 ID (정수), 출력: 아이템 임베딩 벡터.
#    이러한 임베딩은 상호작용 데이터로부터 잠재적인 특징이나 특성을 효과적으로 포착합니다.
# 3. 임베딩 결합: 학습된 사용자 및 아이템 임베딩 벡터는 상호작용을 모델링하기 위해 결합됩니다.
#    일반적인 방법은 다음과 같습니다:
#    - 연결 (Concatenation): 두 벡터를 끝과 끝으로 연결하여 더 긴 단일 벡터를 만듭니다.
#      이 결합된 벡터는 후속 Dense 레이어의 입력으로 사용됩니다.
#    - 내적 (Dot Product) (또는 요소별 곱): 이는 전통적인 Matrix Factorization이 평점을 예측하는 방식과 유사하게
#      상호작용을 명시적으로 모델링합니다. 이 또한 Dense 레이어에 입력될 수 있습니다.
# 4. 딥러닝 신경망 (DNN) 레이어: 결합된 (또는 상호작용한) 임베딩 벡터는 하나 이상의 Dense (완전 연결) 레이어를 통과합니다.
#    이러한 레이어는 (ReLU와 같은 활성화 함수를 사용하여) 비선형 변환을 적용하여 연결된 임베딩으로부터
#    더 높은 수준의 패턴을 학습합니다. 과적합을 방지하고 정규화를 제공하기 위해 Dropout 레이어가 종종 사이에 배치됩니다.
# 5. 출력 레이어: 최종 Dense 레이어가 예측값을 출력합니다.
#    - 평점 예측 (회귀)의 경우: 이 레이어는 일반적으로 단일 뉴런과 선형 활성화 함수를 가집니다
#      (또는 평점이 [0,1]로 스케일링된 경우 시그모이드).
#    - 클릭률 예측 (이진 분류)의 경우: 시그모이드 활성화를 가진 단일 뉴런.
# 6. 학습: 모델은 Adam 또는 RMSprop과 같은 옵티마이저를 사용하여 적절한 손실 함수
#    (예: 평점 예측의 경우 평균 제곱 오차(MSE), 분류의 경우 이진 교차 엔트로피)를 최소화하여
#    종단 간(end-to-end)으로 학습됩니다.
#
# Pros (장점):
# - 강력한 특징 표현: ID에 대한 수동 특징 공학 없이 사용자와 아이템에 대한 풍부하고 밀집된 임베딩을 자동으로 학습하여
#   미묘한 특성을 포착합니다.
# - 비선형 관계 포착: DNN은 기본 SVD 또는 선형 회귀와 같은 더 간단한 선형 모델이 놓칠 수 있는
#   복잡한 상호작용과 패턴을 모델링할 수 있습니다.
# - 높은 유연성: 아키텍처 사용자 정의가 매우 용이합니다. 추가 특징(예: 사용자 인구 통계, 자체 임베딩이 있는
#   더 많은 입력 분기 또는 직접적인 숫자 입력을 추가하여 아이템 메타데이터)을 쉽게 통합하고
#   네트워크의 깊이/너비를 조정할 수 있습니다.
# - 최첨단 성능: 특히 대규모 데이터셋에서 많은 추천 벤치마크에서 높은 정확도를 달성하는 경우가 많습니다.
#
# Cons (단점):
# - 복잡성 및 학습 시간: 특히 매우 큰 데이터셋, 큰 임베딩 차원 또는 깊거나 넓은 네트워크의 경우
#   학습에 계산 비용이 많이 들고 시간이 오래 걸릴 수 있습니다.
#   효율적인 학습을 위해 더 많은 계산 리소스(예: GPU)가 필요합니다.
# - 데이터 요구 사항 ("Data Hungry"): 효과적으로 학습하고 과적합을 피하기 위해 일반적으로 상당한 양의
#   상호작용 데이터가 필요합니다. 적절한 정규화나 아키텍처 선택 없이는 매우 희소한 데이터셋에서
#   성능이 최적이 아닐 수 있습니다.
# - 해석 가능성 문제: 많은 딥러닝 모델과 마찬가지로 학습된 관계와 임베딩 벡터의 개별 차원의 의미를
#   직접 해석하기 어려울 수 있습니다.
# - Cold-Start 문제: 학습 데이터에 상호작용이 없거나 매우 적은 신규 사용자 또는 신규 아이템에 대해서는
#   의미 있는 임베딩을 학습할 수 없으므로 여전히 어려움에 직면합니다.
#   이를 완화하기 위해 하이브리드 접근 방식이나 콘텐츠 특징 통합이 종종 필요합니다.
# ---

# 모듈 임포트를 위해 프로젝트 루트를 sys.path에 동적으로 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root) # 우선순위를 위해 프로젝트 루트가 처음에 오도록 보장

# --- 데이터 로딩 및 전처리 ---
# Time Complexity:
# - CSV 읽기: O(N_interactions)
# - LabelEncoding: 피팅 및 변환에 대략 O(N_interactions).
# - train_test_split: O(N_interactions).
# Overall: 상호작용 수에 비례하는 작업에 의해 지배됨.
def load_and_preprocess_data(base_filepath='data/dummy_interactions.csv', test_size=0.2, random_state=42):
    """
    한국어: CSV에서 상호작용 데이터를 로드하고, DNN 모델을 위해 전처리하며
    (사용자/아이템 ID 인코딩), 학습 및 테스트 세트로 분할합니다.

    Loads interaction data from a CSV, preprocesses it for the DNN model
    (encodes user/item IDs), and splits it into training and testing sets.

    Args:
        base_filepath (str): Path to the interaction data CSV, relative to project root.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Seed for random operations for reproducibility.

    Returns:
        tuple: Contains (X_train, X_test, y_train, y_test, num_users, num_items,
                 user_encoder, item_encoder, df_all_interactions) or Nones on failure.
    """
    filepath = os.path.join(project_root, base_filepath)

    if not os.path.exists(filepath):
        print(f"Error: Data file not found at {filepath}.") # 오류: {filepath}에서 데이터 파일을 찾을 수 없습니다.
        if base_filepath == 'data/dummy_interactions.csv': # 기본 더미 파일인지 확인
            print("Attempting to generate dummy data using 'data/generate_dummy_data.py'...") # 'data/generate_dummy_data.py'를 사용하여 더미 데이터를 생성하려고 시도 중...
            try:
                from data.generate_dummy_data import generate_dummy_data
                generate_dummy_data()
                print("Dummy data generation script executed.") # 더미 데이터 생성 스크립트가 실행되었습니다.
                if not os.path.exists(filepath): # 생성 후 다시 확인
                    print(f"Error: Dummy data file still not found at {filepath} after generation.") # 오류: 생성 후에도 {filepath}에서 더미 데이터 파일을 찾을 수 없습니다.
                    return None, None, None, None, None, None, None, None, None
                print("Dummy data file should now be available.") # 이제 더미 데이터 파일을 사용할 수 있습니다.
            except ImportError as e_import:
                print(f"ImportError: Failed to import 'generate_dummy_data'. Error: {e_import}") # ImportError: 'generate_dummy_data'를 임포트하지 못했습니다. 오류: {e_import}
                return None, None, None, None, None, None, None, None, None
            except Exception as e_general:
                print(f"Error during dummy data generation: {e_general}") # 더미 데이터 생성 중 오류: {e_general}
                return None, None, None, None, None, None, None, None, None
        else:
            # 누락된 파일은 생성 방법을 아는 파일이 아님
            return None, None, None, None, None, None, None, None, None

    df = pd.read_csv(filepath)
    if df.empty:
        print("Error: Data file is empty.") # 오류: 데이터 파일이 비어 있습니다.
        return None, None, None, None, None, None, None, None, None

    # 사용자 및 아이템 ID를 0부터 시작하는 정수 인덱스로 인코딩
    # Keras의 임베딩 레이어는 정수 입력을 예상하기 때문에 이것이 필요합니다.
    user_encoder = LabelEncoder()
    df['user_idx'] = user_encoder.fit_transform(df['user_id'])

    item_encoder = LabelEncoder()
    df['item_idx'] = item_encoder.fit_transform(df['item_id'])

    # 임베딩 레이어 input_dim 정의에 중요한 고유 사용자 및 아이템 수
    num_users = df['user_idx'].nunique()
    num_items = item_encoder.classes_.shape[0] # 고유하게 인코딩된 아이템 수를 얻는 더 강력한 방법

    # 특징(X)과 대상(y) 준비
    # X는 각 행이 [user_idx, item_idx]인 2D 배열이 됩니다.
    X = df[['user_idx', 'item_idx']].values
    # y는 평점이며, TensorFlow/Keras 모델 학습을 위해 float인지 확인합니다.
    y = df['rating'].values.astype(np.float32)

    # 데이터를 학습 세트와 테스트 세트로 분할
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    print(f"Data loaded: {len(df)} interactions.") # 데이터 로드됨: {len(df)}개 상호작용.
    print(f"Number of unique users: {num_users}, Number of unique items: {num_items}") # 고유 사용자 수: {num_users}, 고유 아이템 수: {num_items}
    return X_train, X_test, y_train, y_test, num_users, num_items, user_encoder, item_encoder, df


# --- DNN 모델 구축 ---
def build_dnn_model(num_users, num_items, embedding_dim=32, dense_layers=[64, 32], dropout_rate=0.1):
    """
    한국어: Keras를 사용하여 추천을 위한 딥러닝 신경망(DNN) 모델을 구축합니다.
    이 모델은 사용자와 아이템에 대해 별도의 임베딩 레이어를 사용하고, 이를 연결한 다음,
    Dense 레이어를 통과시켜 평점을 예측합니다.

    Builds a Deep Neural Network (DNN) model for recommendation using Keras.
    The model uses separate embedding layers for users and items, concatenates them,
    and passes them through dense layers to predict ratings.

    Args:
        num_users (int): Total number of unique users (for user embedding layer).
        num_items (int): Total number of unique items (for item embedding layer).
        embedding_dim (int): Dimensionality of the embedding vectors.
        dense_layers (list of int): List specifying the number of units in each dense layer.
        dropout_rate (float): Dropout rate for regularization.

    Returns:
        tensorflow.keras.models.Model: The compiled Keras model.
    """
    # 사용자 임베딩 경로
    # 사용자 인덱스 입력 레이어 (단일 정수)
    user_input = Input(shape=(1,), name='user_input')
    # 임베딩 레이어: 각 사용자 인덱스를 `embedding_dim` 크기의 밀집 벡터에 매핑합니다.
    # `input_dim`은 어휘 크기 (고유 사용자 수)입니다.
    user_embedding = Embedding(input_dim=num_users, output_dim=embedding_dim, name='user_embedding')(user_input)
    # 임베딩 출력을 1D 벡터로 평탄화
    user_vec = Flatten(name='flatten_user_embedding')(user_embedding)
    user_vec = Dropout(dropout_rate, name='dropout_user_vec')(user_vec) # 정규화를 위해 드롭아웃 적용

    # 아이템 임베딩 경로
    # 아이템 인덱스 입력 레이어
    item_input = Input(shape=(1,), name='item_input')
    # 아이템용 임베딩 레이어
    item_embedding = Embedding(input_dim=num_items, output_dim=embedding_dim, name='item_embedding')(item_input)
    # 임베딩 출력 평탄화
    item_vec = Flatten(name='flatten_item_embedding')(item_embedding)
    item_vec = Dropout(dropout_rate, name='dropout_item_vec')(item_vec) # 정규화를 위해 드롭아웃 적용

    # 사용자 및 아이템 임베딩 벡터 결합
    # 연결 작업은 두 벡터를 나란히 결합합니다.
    concat_embeddings = Concatenate(name='concatenate_embeddings')([user_vec, item_vec])
    concat_dropout = Dropout(dropout_rate, name='dropout_concatenated')(concat_embeddings)

    # 완전 연결 (Dense) 레이어
    # 연결된 (그리고 드롭아웃이 적용되었을 수 있는) 임베딩으로 시작
    current_dense_layer = concat_dropout
    for i, units in enumerate(dense_layers):
        current_dense_layer = Dense(units, activation='relu', name=f'dense_layer_{i+1}')(current_dense_layer)
        current_dense_layer = Dropout(dropout_rate, name=f'dropout_dense_{i+1}')(current_dense_layer) # 정규화를 위해 드롭아웃 적용

    # 출력 레이어
    # 단일 값(평점)을 예측합니다.
    # 'linear' 활성화는 회귀 작업(연속적인 평점 값 예측)에 사용됩니다.
    # 평점이 [0,1]로 정규화된 경우 'sigmoid'도 옵션이 될 수 있습니다.
    output_layer = Dense(1, activation='linear', name='rating_output')(current_dense_layer)

    # Keras 모델 생성 및 컴파일
    model = Model(inputs=[user_input, item_input], outputs=output_layer)
    return model

# --- 추천 생성 ---
def get_dnn_recommendations(model, user_id_original, user_encoder, item_encoder, df_all_interactions, num_total_items_encoded, num_recommendations=5):
    """
    한국어: 학습된 DNN 모델을 사용하여 특정 사용자에 대한 상위 N개 추천을 생성합니다.

    Generates top-N recommendations for a specific user using the trained DNN model.

    Args:
        model (tensorflow.keras.models.Model): The trained Keras DNN model.
        user_id_original: The original ID of the user for whom to generate recommendations.
        user_encoder (LabelEncoder): The fitted user ID encoder.
        item_encoder (LabelEncoder): The fitted item ID encoder.
        df_all_interactions (pd.DataFrame): DataFrame containing all user-item interactions.
        num_total_items_encoded (int): Total number of unique items known to item_encoder.
        num_recommendations (int): Number of recommendations to return.

    Returns:
        list: A list of dictionaries, each containing 'item_id' (original) and 'predicted_rating'.
    """
    try:
        # 원본 사용자 ID를 학습 중에 사용된 정수 인덱스로 변환
        user_idx_encoded = user_encoder.transform([user_id_original])[0]
    except ValueError:
        print(f"Error: User ID '{user_id_original}' was not found in the training data (new user). " # 오류: 사용자 ID '{user_id_original}'가 학습 데이터에 없습니다(신규 사용자).
              "Cannot generate recommendations for this user with this model.") # 이 모델로는 이 사용자에 대한 추천을 생성할 수 없습니다.
        return []

    # 사용자가 이미 상호작용한 아이템 식별 (명확성을 위해 원본 아이템 ID 사용)
    # 이러한 아이템은 추천에서 제외되어야 합니다.
    items_rated_by_user_original_ids = set()
    if 'user_id' in df_all_interactions.columns and 'item_id' in df_all_interactions.columns:
        items_rated_by_user_original_ids = set(
            df_all_interactions[df_all_interactions['user_id'] == user_id_original]['item_id'].unique()
        )
    print(f"User {user_id_original} has already interacted with {len(items_rated_by_user_original_ids)} items.") # 사용자 {user_id_original}는 이미 {len(items_rated_by_user_original_ids)}개 아이템과 상호작용했습니다.

    # 가능한 모든 아이템 인덱스 목록 생성 (0부터 num_total_items_encoded - 1까지)
    all_item_indices_encoded = np.arange(num_total_items_encoded)

    # 사용자가 이미 상호작용한 아이템 필터링.
    # 사용자가 보지 않은 아이템에 대해서만 평점을 예측해야 합니다.
    items_to_predict_encoded_indices = []
    for item_idx_encoded in all_item_indices_encoded:
        # 사용자가 평가했는지 확인하기 위해 인코딩된 아이템 인덱스를 원본 아이템 ID로 다시 변환
        original_item_id_for_check = item_encoder.inverse_transform([item_idx_encoded])[0]
        if original_item_id_for_check not in items_rated_by_user_original_ids:
            items_to_predict_encoded_indices.append(item_idx_encoded)

    items_to_predict_encoded_indices = np.array(items_to_predict_encoded_indices, dtype=np.int32)

    if len(items_to_predict_encoded_indices) == 0:
        print(f"User {user_id_original} has already rated all available items, or no new items to recommend.") # 사용자 {user_id_original}가 이미 사용 가능한 모든 아이템을 평가했거나 추천할 새 아이템이 없습니다.
        return []

    # Keras 모델 예측을 위한 입력 배열 준비
    # 예측하려는 각 아이템에 대해 user_idx_encoded를 반복해야 합니다.
    user_input_array = np.full(len(items_to_predict_encoded_indices), user_idx_encoded, dtype=np.int32)
    item_input_array = items_to_predict_encoded_indices # 이미 dtype=np.int32 확인됨

    # 평가되지 않은 아이템에 대한 평점 예측
    # N개 아이템에 대한 예측 Time Complexity: O(N * Forward_Pass의 복잡도)
    print(f"Predicting ratings for {len(items_to_predict_encoded_indices)} unrated items for user {user_id_original}...") # 사용자 {user_id_original}에 대해 평가되지 않은 {len(items_to_predict_encoded_indices)}개 아이템에 대한 평점 예측 중...
    predicted_ratings = model.predict([user_input_array, item_input_array], verbose=0).flatten()

    # 아이템 인덱스와 예측된 평점 결합
    recommendation_results = list(zip(items_to_predict_encoded_indices, predicted_ratings))
    # 예측된 평점을 기준으로 내림차순 정렬
    recommendation_results.sort(key=lambda x: x[1], reverse=True)

    # 상위 N개 추천 형식 지정
    top_n_recommendations = []
    for item_idx_encoded, score in recommendation_results[:num_recommendations]:
        original_item_id = item_encoder.inverse_transform([item_idx_encoded])[0] # 원본 ID로 다시 변환
        top_n_recommendations.append({'item_id': original_item_id, 'predicted_rating': score})

    return top_n_recommendations

# --- 메인 실행 블록 ---
# 학습 Time Complexity: model.fit()이 지배적이며 대략:
# O(N_epochs * (N_train_samples / Batch_size) * (역전파의 각 레이어 부분에 대한 (Input_features * Output_features)의 합계))
# 임베딩 레이어가 크게 기여: (num_users * emb_dim + num_items * emb_dim) 매개변수.
if __name__ == "__main__":
    print("--- DNN (Deep Neural Network) Based Recommendation Example (TensorFlow/Keras) ---") # --- DNN (딥러닝 신경망) 기반 추천 예제 (TensorFlow/Keras) ---

    # 1. 데이터 로드 및 전처리
    print("\nLoading and preprocessing data...") # 데이터 로드 및 전처리 중...
    X_train, X_test, y_train, y_test, num_users, num_items, user_encoder, item_encoder, df_all_interactions = \
        load_and_preprocess_data(base_filepath='data/dummy_interactions.csv')

    if X_train is not None and df_all_interactions is not None:
        print(f"Training data samples: {X_train.shape[0]}, Test data samples: {X_test.shape[0]}") # 학습 데이터 샘플: {X_train.shape[0]}, 테스트 데이터 샘플: {X_test.shape[0]}
        # num_users 및 num_items는 load_and_preprocess_data에서 이미 출력됨

        # 2. DNN 모델 구축
        print("\nBuilding DNN model...") # DNN 모델 구축 중...
        # 모델 하이퍼파라미터
        embedding_dimension = 50       # 임베딩 벡터의 차원
        dnn_dense_layers = [128, 64, 32] # 각 Dense 레이어의 유닛 수
        dnn_dropout_rate = 0.2         # 정규화를 위한 드롭아웃 비율
        learning_rate = 0.001          # Adam 옵티마이저의 학습률

        model = build_dnn_model(
            num_users,
            num_items,
            embedding_dim=embedding_dimension,
            dense_layers=dnn_dense_layers,
            dropout_rate=dnn_dropout_rate
        )
        # 모델 컴파일: 옵티마이저, 손실 함수 및 메트릭 구성
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='mean_squared_error', # 평점 예측(회귀)에 적합
            metrics=['mae']            # 평균 절대 오차, 또 다른 일반적인 회귀 메트릭
        )
        model.summary() # 모델 아키텍처 출력

        # 3. 모델 학습
        print("\nTraining the model... (This may take some time depending on epochs and data size)") # 모델 학습 중... (에포크 및 데이터 크기에 따라 시간이 걸릴 수 있음)
        # 학습 하이퍼파라미터
        num_epochs = 10                # 전체 학습 데이터셋을 반복하는 횟수
        training_batch_size = 64       # 그래디언트 업데이트당 샘플 수

        # `fit`에 대한 입력 데이터가 올바른 유형(임베딩을 위한 정수 인덱스)인지 확인
        # X_train[:, 0]은 user_idx, X_train[:, 1]은 item_idx입니다.
        history = model.fit(
            [X_train[:, 0].astype(np.int32), X_train[:, 1].astype(np.int32)],
            y_train,
            epochs=num_epochs,
            batch_size=training_batch_size,
            validation_data=([X_test[:, 0].astype(np.int32), X_test[:, 1].astype(np.int32)], y_test),
            verbose=1 # 학습 진행 상황 표시
        )
        print("Model training completed.") # 모델 학습 완료.

        # 테스트 세트에서 모델 평가
        loss, mae = model.evaluate(
            [X_test[:, 0].astype(np.int32), X_test[:, 1].astype(np.int32)],
            y_test,
            verbose=0
        )
        print(f"\nEvaluation on Test Dataset: Loss = {loss:.4f}, MAE = {mae:.4f}") # 테스트 데이터셋 평가: 손실 = {loss:.4f}, MAE = {mae:.4f}

        # 4. 특정 사용자에 대한 추천 생성
        if not df_all_interactions.empty and user_encoder.classes_.shape[0] > 0:
            # 데모를 위해 대상 사용자 선택 (예: 원본 데이터셋의 첫 번째 사용자)
            target_user_original_id = user_encoder.classes_[0] # 인코더에 알려진 첫 번째 원본 사용자 ID 가져오기

            print(f"\nGenerating recommendations for User ID (original): {target_user_original_id}...") # 사용자 ID (원본): {target_user_original_id}에 대한 추천 생성 중...

            # num_items는 item_encoder.classes_.shape[0]이며, 이는 num_total_items_encoded입니다.
            recommendations = get_dnn_recommendations(
                model,
                target_user_original_id,
                user_encoder,
                item_encoder,
                df_all_interactions, # 사용자가 이미 평가한 아이템을 찾는 데 사용됨
                num_items,             # 고유 아이템 총 수 (후보 생성용)
                num_recommendations=5
            )

            if recommendations:
                print(f"\nTop recommendations for User {target_user_original_id} (DNN-based):") # 사용자 {target_user_original_id}를 위한 상위 추천 (DNN 기반):
                for rec in recommendations:
                    print(f"- Item ID (original): {rec['item_id']}, Predicted Rating: {rec['predicted_rating']:.3f}")
            else:
                print(f"No new recommendations could be generated for User {target_user_original_id}.") # 사용자 {target_user_original_id}에 대한 새 추천을 생성할 수 없습니다.
        else:
            print("\nCannot generate recommendations as interaction data is empty or no users were encoded.") # 상호작용 데이터가 비어 있거나 인코딩된 사용자가 없어 추천을 생성할 수 없습니다.
    else:
        print("\nData loading and preprocessing failed. Cannot proceed with the DNN example.") # 데이터 로드 및 전처리에 실패했습니다. DNN 예제를 진행할 수 없습니다.

    print("\n--- DNN Based Recommendation Example Finished ---") # --- DNN 기반 추천 예제 완료 ---
