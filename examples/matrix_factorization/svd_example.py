# examples/matrix_factorization/svd_example.py
import pandas as pd
import os
import sys
from surprise import Dataset, Reader, SVD
# train_test_split은 이 특정 예제에서는 사용되지 않지만 종종 유용합니다.
# from surprise.model_selection import train_test_split
from collections import defaultdict

# --- 추천을 위한 특이값 분해(SVD): 기본 설명 ---
# Matrix Factorization, 특히 SVD에서 영감을 받은 변형들은 추천 시스템을 위한 인기 있는 알고리즘 클래스입니다.
# 핵심 아이디어는 희소 사용자-아이템 상호작용 행렬(예: 평점)을 사용자와 아이템의 잠재 요인(숨겨진 특징)을 나타내는
# 저차원 행렬로 분해하는 것입니다.
#
# 작동 방식 (특히 Surprise의 SVD, 종종 Funk SVD 또는 SVD++ 변형):
# 1. 사용자-아이템 상호작용 데이터: 입력은 일반적으로 (user_id, item_id, rating) 삼중항 목록입니다.
# 2. 잠재 요인 모델:
#    - 'k'개의 잠재 요인이 존재하며, 이는 근본적인 속성을 포착한다고 가정합니다 (예: 영화의 경우 액션 수준, 코미디 콘텐츠, 로맨틱 요소; 사용자의 경우 이러한 측면에 대한 선호도).
#    - 각 사용자 'u'는 k차원 벡터 p_u (사용자 요인)와 연관됩니다.
#    - 각 아이템 'i'는 k차원 벡터 q_i (아이템 요인)와 연관됩니다.
# 3. 평점 예측:
#    사용자 'u'와 아이템 'i'에 대한 예측 평점 r_ui는 종종 다음과 같이 모델링됩니다:
#    r_ui_predicted = μ + b_u + b_i + p_u^T * q_i
#    여기서:
#      - μ (mu): 모든 아이템에 대한 전역 평균 평점.
#      - b_u: 사용자 편향 (이 사용자가 평균에 비해 평점을 매기는 경향).
#      - b_i: 아이템 편향 (이 아이템이 평균에 비해 평점을 받는 경향).
#      - p_u^T * q_i: 사용자 및 아이템 요인 벡터의 내적으로, 잠재 공간에서 사용자 선호도와 아이템 특성 간의 상호작용을 포착합니다.
#    참고: Surprise의 SVD 알고리즘은 "순수" 수학적 SVD가 아니라 추천 작업(알려진 평점 예측)에 최적화된 모델입니다.
#
# 4. 학습 (매개변수 학습):
#    모델 매개변수(모든 사용자와 아이템에 대한 μ, b_u, b_i, p_u, q_i)는 목적 함수를 최소화하여 학습됩니다.
#    이 함수는 일반적으로 예측 평점과 실제 알려진 평점 간의 제곱 오차 합계와 과적합을 방지하기 위한 정규화 항으로 구성됩니다.
#    최적화는 일반적으로 확률적 경사 하강법(SGD) 또는 교대 최소 제곱법(ALS)을 사용하여 수행됩니다.
#    Surprise의 SVD는 기본적으로 SGD를 사용합니다.
#
# Pros (장점):
# - 희소성 처리 우수: 잠재 요인을 학습함으로써 SVD는 알려진 평점으로부터 일반화하여 알 수 없는 평점을 예측할 수 있으며,
#   종종 희소 데이터에서 이웃 기반 CF보다 성능이 뛰어납니다.
# - 간결한 표현: 사용자 및 아이템 요인은 저차원의 밀집 표현을 제공합니다.
# - 복잡한 관계 포착: 잠재 요인은 메타데이터에 명시적으로 존재하지 않더라도 복잡한 사용자 선호도와 아이템 속성을 암묵적으로 포착할 수 있습니다.
# - 확장성: 학습은 집약적일 수 있지만 요인이 학습되면 예측은 비교적 빠릅니다.
#
# Cons (단점):
# - 해석 가능성: 학습된 잠재 요인은 종종 인간의 용어로 직접 해석하기 어렵습니다.
# - Cold-Start 문제:
#   - 신규 사용자: 사용자가 평점을 매긴 적이 없으면 사용자 요인 벡터 p_u를 학습할 수 없습니다.
#   - 신규 아이템: 아이템에 평점이 없으면 아이템 요인 벡터 q_i를 학습할 수 없습니다.
#   이러한 경우 대체 방법(예: 콘텐츠 기반 특징, 평균 평점)이 필요합니다.
# - 학습 복잡성: 매우 큰 데이터셋의 경우 학습이 계산적으로 집약적일 수 있습니다.
#   SGD의 경우 복잡도는 대략 O(N_interactions * N_factors * N_epochs)입니다.
# ---

# 모듈 임포트를 위해 프로젝트 루트를 sys.path에 동적으로 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root) # 우선순위를 위해 처음에 삽입

# --- Surprise를 위한 데이터 로딩 ---
# Time Complexity: CSV 읽기 및 Surprise Dataset으로 로딩하는 데 O(N_interactions).
def load_data_for_surprise(base_filepath='data/dummy_interactions.csv'):
    """
    한국어: CSV 파일에서 상호작용 데이터를 로드하여 Surprise Dataset 객체로 만듭니다.
    CSV 파일에는 사용자 ID, 아이템 ID 및 평점 열이 포함되어야 합니다.
    프로젝트 루트에 상대적인 파일 경로를 처리하고 필요한 경우 더미 데이터 생성을 시도합니다.

    Loads interaction data from a CSV file into a Surprise Dataset object.
    The CSV file must contain columns for user ID, item ID, and rating.
    Handles file path relative to the project root and attempts dummy data generation if needed.

    Args:
        base_filepath (str): Path to the CSV data file, relative to the project root if starts with 'data/'.

    Returns:
        surprise.Dataset: A Dataset object ready for use with Surprise algorithms, or None on failure.
    """
    filepath = base_filepath
    # base_filepath가 알려진 상대 경로 패턴인 경우 절대 경로 구성
    if not os.path.isabs(base_filepath) and base_filepath.startswith('data/'):
        filepath = os.path.join(project_root, base_filepath)

    if not os.path.exists(filepath):
        print(f"Error: Data file not found at {filepath}.") # 오류: {filepath}에서 데이터 파일을 찾을 수 없습니다.
        # 기본 더미 데이터 경로인지 구체적으로 확인하여 생성 제안
        if filepath.endswith('data/dummy_interactions.csv'):
            print("Attempting to generate dummy data using 'data/generate_dummy_data.py'...") # 'data/generate_dummy_data.py'를 사용하여 더미 데이터를 생성하려고 시도 중...
            try:
                from data.generate_dummy_data import generate_dummy_data
                generate_dummy_data()
                print("Dummy data generation script executed.") # 더미 데이터 생성 스크립트 실행됨.
                if not os.path.exists(filepath): # 생성 시도 후 다시 확인
                    print(f"Error: Dummy data file still not found at {filepath} after generation attempt.") # 오류: 생성 시도 후에도 {filepath}에서 더미 데이터 파일을 찾을 수 없습니다.
                    return None
                print("Dummy data file should now be available.") # 이제 더미 데이터 파일을 사용할 수 있습니다.
            except ImportError as e_import:
                print(f"ImportError: Failed to import 'generate_dummy_data'. Error: {e_import}") # ImportError: 'generate_dummy_data'를 임포트하지 못했습니다. 오류: {e_import}
                return None
            except Exception as e_general:
                print(f"Error during dummy data generation: {e_general}") # 더미 데이터 생성 중 오류: {e_general}
                return None
        else:
            # 누락된 파일은 생성 방법을 아는 파일이 아님
            return None

    # Reader 객체는 파일 또는 DataFrame을 구문 분석하는 데 사용됩니다.
    # rating_scale을 정의해야 합니다 (예: 별 1-5개).
    reader = Reader(rating_scale=(1, 5)) # 평점 척도가 다른 경우 조정하십시오.

    try:
        df = pd.read_csv(filepath)
        # 필수 열이 있는지 확인합니다. Surprise는 사용자, 아이템, 평점 순서로 열을 예상합니다.
        required_cols = ['user_id', 'item_id', 'rating']
        if not all(col in df.columns for col in required_cols):
            print(f"Error: CSV file must contain the columns: {', '.join(required_cols)}.") # 오류: CSV 파일에는 다음 열이 포함되어야 합니다: {', '.join(required_cols)}.
            return None

        # DataFrame을 Surprise Dataset으로 로드합니다.
        # Surprise는 이 세 개의 열만 사용합니다.
        data = Dataset.load_from_df(df[['user_id', 'item_id', 'rating']], reader)
        return data
    except Exception as e:
        print(f"Error loading data into Surprise Dataset: {e}") # Surprise Dataset으로 데이터 로드 중 오류: {e}
        return None

# --- Top-N 추천 생성 ---
def get_top_n_recommendations(predictions, n=10):
    """
    한국어: Surprise 예측 목록에서 각 사용자에 대한 상위 N개 추천을 추출합니다.

    Extracts the top-N recommendations for each user from a list of Surprise predictions.

    Args:
        predictions (list of surprise.Prediction objects): A list of predictions,
            typically generated by an algorithm's `test` method or multiple `predict` calls.
        n (int): The number of recommendations to return for each user.

    Returns:
        defaultdict: A dictionary where keys are user (raw) IDs and values are lists of
                     (raw_item_id, estimated_rating) tuples, sorted by estimated_rating.
    """
    # 각 사용자에 예측 매핑.
    top_n = defaultdict(list)
    for pred in predictions:
        # pred 속성: uid (원본 사용자 ID), iid (원본 아이템 ID), r_ui (실제 평점, 예측 시 종종 None), est (예측 평점)
        top_n[pred.uid].append((pred.iid, pred.est))

    # 각 사용자에 대한 예측을 예측 평점(내림차순)으로 정렬하고 상위 n개 검색.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]
    return top_n

# --- 메인 실행 블록 ---
if __name__ == "__main__":
    print("--- SVD (Matrix Factorization) Example using Surprise library ---") # --- Surprise 라이브러리를 사용한 SVD (Matrix Factorization) 예제 ---

    data_file_path = 'data/dummy_interactions.csv' # 프로젝트 루트에 대한 상대 경로

    # 1. Surprise Reader 및 Dataset을 사용하여 데이터 로드
    print(f"\nLoading data from '{data_file_path}' for Surprise...") # Surprise를 위해 '{data_file_path}'에서 데이터 로드 중...
    data = load_data_for_surprise(base_filepath=data_file_path)

    if data:
        # 2. 전체 학습 세트 구축
        # 이 예제에서는 전체 데이터셋으로 학습합니다.
        # 실제 시나리오에서는 학습/테스트 세트로 분할합니다 (예: Surprise의 train_test_split 사용).
        trainset = data.build_full_trainset()
        print("Data loaded and full training set built successfully.") # 데이터 로드 및 전체 학습 세트 구축 완료.

        # 3. SVD 모델 학습
        # 학습 Time Complexity: O(N_interactions * N_factors * N_epochs)
        print("\nTraining SVD model...") # SVD 모델 학습 중...
        # SVD 매개변수:
        # - n_factors: 잠재 요인 수 (설명의 k). 기본값은 100.
        # - n_epochs: SGD 최적화 반복 횟수. 기본값은 20.
        # - biased: True (기본값)인 경우 모델에 편향(μ, b_u, b_i)이 포함됩니다. 일반적으로 권장됩니다.
        # - random_state: 재현성을 위해.
        algo = SVD(n_factors=50, n_epochs=20, biased=True, random_state=42)
        algo.fit(trainset)
        print("SVD model training completed.") # SVD 모델 학습 완료.

        # 4. 특정 사용자에 대한 Top-N 추천 생성
        # 다음을 포함합니다:
        #  a. 사용자가 아직 평가하지 않은 모든 아이템 식별.
        #  b. 학습된 모델을 사용하여 이러한 미평가 아이템에 대한 평점 예측.
        #  c. 예측 평점이 가장 높은 아이템 선택.

        target_user_raw_id = None # 로드된 데이터에서 결정됨
        try:
            # 고유 사용자 ID 목록을 얻기 위해 원본 CSV를 다시 로드합니다.
            # 이는 데모를 위해 데이터셋에 실제로 존재하는 사용자 ID를 선택하도록 보장합니다.
            # 실제 애플리케이션에서는 애플리케이션 컨텍스트에서 대상 사용자 ID를 가져옵니다.
            df_interactions_for_user_selection = pd.read_csv(os.path.join(project_root, data_file_path))
            if not df_interactions_for_user_selection.empty:
                target_user_raw_id = df_interactions_for_user_selection['user_id'].unique()[0]
                # ID가 문자열인 경우 문자열인지, 정수인 경우 정수인지 확인합니다.
                # Surprise는 제공된 원본 ID를 그대로 처리합니다 (str, int).
            else:
                print("Warning: Interaction data file is empty. Cannot select a target user for recommendations.") # 경고: 상호작용 데이터 파일이 비어 있습니다. 추천을 위한 대상 사용자를 선택할 수 없습니다.
        except FileNotFoundError:
             print(f"Error: Could not find '{data_file_path}' to select a target user. Please ensure dummy data exists.") # 오류: '{data_file_path}'를 찾을 수 없어 대상 사용자를 선택할 수 없습니다. 더미 데이터가 있는지 확인하십시오.
        except IndexError:
             print("Error: No user IDs found in the dummy data. Check data generation.") # 오류: 더미 데이터에서 사용자 ID를 찾을 수 없습니다. 데이터 생성을 확인하십시오.

        # 혼합 유형 원본 ID에서 로드된 경우 Surprise가 내부적으로 저장하는 방식과 일치하도록 문자열로 변환합니다 (그렇지 않은 경우).
        # 그러나 Surprise는 일반적으로 원본 ID 유형을 유지합니다. 더미 데이터의 경우 일반적으로 정수입니다.
        # 이 더미 데이터의 CSV ID가 정수라고 가정합니다.

        if target_user_raw_id is not None:
            print(f"\nGenerating top-N recommendations for User ID (raw): {target_user_raw_id}...") # 사용자 ID (원본): {target_user_raw_id}에 대한 top-N 추천 생성 중...

            # 사용자가 이미 평가한 아이템 가져오기 (trainset에서)
            rated_item_raw_ids = set()
            try:
                # 원본 사용자 ID를 Surprise에서 사용하는 내부 ID로 변환
                target_user_inner_id = trainset.to_inner_uid(target_user_raw_id)
                # 이 사용자에 대한 (내부_아이템_ID, 평점) 튜플 가져오기
                user_ratings_inner = trainset.ur[target_user_inner_id]
                # 내부 아이템 ID를 원본 아이템 ID로 다시 변환
                rated_item_raw_ids = {trainset.to_raw_iid(inner_iid) for (inner_iid, _rating) in user_ratings_inner}
                print(f"User {target_user_raw_id} has rated {len(rated_item_raw_ids)} items. These will be excluded from recommendations.") # 사용자 {target_user_raw_id}는 {len(rated_item_raw_ids)}개 아이템을 평가했습니다. 추천에서 제외됩니다.
            except ValueError:
                # target_user_raw_id가 학습 세트에 없는 경우 발생 (예: 신규 사용자)
                print(f"Warning: User ID {target_user_raw_id} not found in the training set. " # 경고: 사용자 ID {target_user_raw_id}가 학습 세트에 없습니다.
                      "Assuming no items rated, so all items are candidates for recommendation.") # 평가한 아이템이 없다고 가정하므로 모든 아이템이 추천 후보입니다.

            # 학습 세트에 있는 모든 고유 아이템 ID 가져오기
            all_items_in_trainset_raw_ids = {trainset.to_raw_iid(inner_iid) for inner_iid in trainset.all_items()}

            # 예측할 아이템 식별: 학습 세트의 모든 아이템 - 사용자가 이미 평가한 아이템
            items_to_predict_raw_ids = [iid for iid in all_items_in_trainset_raw_ids if iid not in rated_item_raw_ids]

            if not items_to_predict_raw_ids:
                print(f"User {target_user_raw_id} has already rated all items in the training set, or no new items to recommend.") # 사용자 {target_user_raw_id}가 이미 학습 세트의 모든 아이템을 평가했거나 추천할 새 아이템이 없습니다.
            else:
                # 이러한 미평가 아이템에 대한 예측 생성
                # N개 예측에 대한 Time Complexity: O(N * N_factors)
                print(f"Predicting ratings for {len(items_to_predict_raw_ids)} unrated items for user {target_user_raw_id}...") # 사용자 {target_user_raw_id}에 대해 평가되지 않은 {len(items_to_predict_raw_ids)}개 아이템에 대한 평점 예측 중...
                user_predictions = []
                for item_raw_id in items_to_predict_raw_ids:
                    # algo.predict()는 원본 사용자 및 아이템 ID를 사용합니다.
                    prediction = algo.predict(uid=target_user_raw_id, iid=item_raw_id)
                    user_predictions.append(prediction)

                # 예측에서 top-N 추천 추출
                # 정렬 Time Complexity: O(N_predictions * log(N_predictions))
                top_recommendations = get_top_n_recommendations(user_predictions, n=5)

                if top_recommendations.get(target_user_raw_id):
                    print(f"\nTop 5 recommendations for User {target_user_raw_id} (SVD based):") # 사용자 {target_user_raw_id}를 위한 상위 5개 추천 (SVD 기반):
                    for item_raw_id, estimated_rating in top_recommendations[target_user_raw_id]:
                        print(f"- Item ID (raw): {item_raw_id}, Predicted Rating: {estimated_rating:.3f}")
                else:
                    print(f"No recommendations could be generated for User {target_user_raw_id}.") # 사용자 {target_user_raw_id}에 대한 추천을 생성할 수 없습니다.
        else:
            if data: # 데이터 로딩 자체가 성공한 경우에만 출력
                print("\nCould not determine a target user for generating recommendations.") # 추천 생성을 위한 대상 사용자를 결정할 수 없습니다.
    else:
        print("\nData loading failed. Cannot proceed with the SVD example.") # 데이터 로드 실패. SVD 예제를 진행할 수 없습니다.

    print("\n--- SVD (Singular Value Decomposition) example execution complete ---") # --- SVD (특이값 분해) 예제 실행 완료 ---
