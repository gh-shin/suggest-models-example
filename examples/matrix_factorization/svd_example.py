# examples/matrix_factorization/svd_example.py
import pandas as pd
import os
import sys
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from collections import defaultdict

# --- Singular Value Decomposition (SVD) for Recommendations: Basic Explanation ---
# Singular Value Decomposition (SVD) is a matrix factorization technique widely used in recommendation systems.
# The core idea is to decompose the sparse user-item interaction matrix (e.g., ratings) into
# lower-dimensional matrices representing latent factors of users and items.
#
# How it works (in the context of Surprise's SVD, which is often a variant like Funk SVD or SVD++):
# 1. User-Item Interaction Data: We start with data of (user, item, rating) triples.
# 2. Latent Factors: Assume there are 'k' latent factors (e.g., genres, user preferences for these genres).
#    - Each user is represented by a k-dimensional vector (user factors).
#    - Each item is represented by a k-dimensional vector (item factors).
# 3. Prediction: The predicted rating for a user 'u' and an item 'i' is typically calculated as the
#    dot product of their latent factor vectors: pred(u, i) = p_u^T * q_i.
#    More advanced SVD variants also incorporate biases: pred(u, i) = global_mean + bias_user + bias_item + p_u^T * q_i.
# 4. Training: The model learns these latent factor vectors and biases by minimizing the error
#    (e.g., Root Mean Squared Error - RMSE) between predicted ratings and actual ratings in the training data.
#    This is often done using stochastic gradient descent (SGD).
#
# Pros:
# - Handles Sparsity: SVD can often generalize better than neighborhood-based CF methods on sparse data
#   because it learns underlying latent features.
# - Compact Representation: User and item factors provide a lower-dimensional representation.
# - Can capture complex relationships: Latent factors might represent underlying characteristics
#   that are not explicitly available in metadata.
#
# Cons:
# - Interpretability: The latent factors learned by SVD are often not directly interpretable.
# - Cold-Start for New Users/Items: If a user or item has no ratings, their latent factors cannot be
#   learned directly. Some heuristics or content-based approaches might be needed as a fallback.
# - Training Complexity: Training can be computationally intensive for very large datasets, though
#   efficient algorithms exist.
# ---

# Dynamically add project root to sys.path for module imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root) # Insert at the beginning for priority

def load_data_for_surprise(base_filepath='data/dummy_interactions.csv'):
    """
    Loads data from a CSV file into a Surprise Dataset.
    The CSV file must be in the format: user, item, rating.
    Handles path relative to project root.
    """
    # Construct absolute path to data file if a relative path is given
    if not os.path.isabs(base_filepath) and base_filepath.startswith('data/'):
        filepath = os.path.join(project_root, base_filepath)
    else:
        filepath = base_filepath # Assume it's already an absolute path or a different relative one

    if not os.path.exists(filepath):
        print(f"오류: {filepath} 파일을 찾을 수 없습니다.")
        # Check if the missing file is the specific dummy data file we can generate
        if filepath.endswith('data/dummy_interactions.csv'): # Check against the absolute path
            print("더미 데이터 생성 시도 중... ('data/generate_dummy_data.py' 실행)")
            try:
                from data.generate_dummy_data import generate_dummy_data
                generate_dummy_data()
                print("더미 데이터 생성 완료.")
                if not os.path.exists(filepath): # Check again after generation
                    print(f"데이터 자동 생성 후에도 {filepath}를 찾을 수 없습니다.")
                    return None
            except ImportError as e_import:
                print(f"ImportError: 'generate_dummy_data' 임포트 실패. 오류: {e_import}")
                return None
            except Exception as e_general:
                print(f"더미 데이터 생성 실패: {e_general}")
                return None
        else:
            return None # Not the specific dummy file, or some other path issue

    # Surprise Reader an object that knows how to parse the file or dataframe
    # We need to specify the rating_scale
    reader = Reader(rating_scale=(1, 5)) # Assuming ratings are from 1 to 5

    # Load data from pandas DataFrame
    df = pd.read_csv(filepath)
    # Ensure columns are in the order: user, item, rating for Surprise
    if not ('user_id' in df.columns and 'item_id' in df.columns and 'rating' in df.columns):
        print("오류: CSV 파일에 'user_id', 'item_id', 'rating' 열이 필요합니다.")
        return None

    data = Dataset.load_from_df(df[['user_id', 'item_id', 'rating']], reader)
    return data

def get_top_n_recommendations(predictions, n=10):
    """
    Return the top-N recommendation for each user from a set of predictions.
    Args:
        predictions(list of Prediction objects): The list of predictions, as
            returned by the test method of an algorithm.
        n(int): The number of recommendation to output for each user. Default
            is 10.
    Returns:
    A dict where keys are user (raw) ids and values are lists of tuples:
        [(raw item id, rating estimation), ...] of size n.
    """
    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions: # uid, iid are raw ids
        top_n[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n

# 메인 실행 함수
if __name__ == "__main__":
    print("--- SVD (Singular Value Decomposition) 예제 시작 (Surprise 라이브러리 사용) ---")

    data_file_path = 'data/dummy_interactions.csv' # Relative to project root

    # 1. 데이터 로드
    print(f"\n데이터 로드 중 ({data_file_path})...")
    data = load_data_for_surprise(base_filepath=data_file_path)

    if data:
        # 2. 학습 데이터셋 구축 (Surprise는 전체 데이터를 사용하여 Trainset을 만듭니다)
        trainset = data.build_full_trainset()
        print("데이터 로드 및 학습 데이터셋 구축 완료.")

        # 3. SVD 모델 학습
        print("\nSVD 모델 학습 중...")
        # n_factors: 잠재 요인의 수, n_epochs: SGD 반복 횟수, random_state: 재현성을 위한 시드
        algo = SVD(n_factors=50, n_epochs=20, biased=True, random_state=42)
        algo.fit(trainset)
        print("모델 학습 완료.")

        # 4. 특정 사용자에 대한 추천 생성
        #   a. 해당 사용자가 아직 평가하지 않은 아이템 목록 가져오기
        #   b. 각 아이템에 대해 평점 예측
        #   c. 예측 평점이 높은 순으로 정렬하여 추천

        target_user_raw_id = None
        try:
            # Use pandas to quickly get a list of unique user IDs from the original data file
            # This ensures we pick a user that actually exists in our dataset
            df_interactions = pd.read_csv(os.path.join(project_root, data_file_path))
            if not df_interactions.empty:
                # Get the first user ID from the unique list of user_ids
                target_user_raw_id = df_interactions['user_id'].unique()[0]
            else:
                print("더미 데이터 파일이 비어있습니다. 추천을 생성할 사용자를 선택할 수 없습니다.")
        except FileNotFoundError:
             print(f"{data_file_path} 파일을 찾을 수 없습니다. 먼저 데이터를 생성해주세요.")
        except IndexError:
             print("더미 데이터에서 사용자 ID를 찾을 수 없습니다. 데이터가 올바르게 생성되었는지 확인해주세요.")

        if target_user_raw_id is not None:
            print(f"\n사용자 ID {target_user_raw_id} (원래 ID)에 대한 추천 아이템 생성 중...")

            # 사용자가 이미 평가한 아이템 가져오기 (학습 세트 기준)
            rated_items_raw_ids = set()
            try:
                target_user_inner_id = trainset.to_inner_uid(target_user_raw_id)
                rated_items_inner_ids = [item_inner_id for (item_inner_id, _) in trainset.ur[target_user_inner_id]]
                rated_items_raw_ids = {trainset.to_raw_iid(inner_id) for inner_id in rated_items_inner_ids}
            except ValueError:
                print(f"사용자 ID {target_user_raw_id}는 학습 데이터에 없어 평가한 아이템 목록을 가져올 수 없습니다 (이미 평가한 아이템은 추천에서 제외됩니다).")

            # 모든 아이템 ID 가져오기 (학습 세트 기준)
            all_items_raw_ids_in_trainset = {trainset.to_raw_iid(inner_id) for inner_id in trainset.all_items()}

            # 추천 대상 아이템: 학습 세트에 있는 전체 아이템 중 사용자가 아직 평가하지 않은 아이템
            items_to_predict = [iid for iid in all_items_raw_ids_in_trainset if iid not in rated_items_raw_ids]

            if not items_to_predict:
                print(f"사용자 {target_user_raw_id}는 이미 모든 (학습세트 내) 아이템을 평가했거나, 추천할 새로운 아이템이 없습니다.")
            else:
                # 예측 수행
                user_predictions = []
                for item_raw_id in items_to_predict:
                    # uid, iid는 raw id여야 합니다.
                    prediction = algo.predict(uid=target_user_raw_id, iid=item_raw_id)
                    user_predictions.append(prediction)

                # 상위 N개 추천 가져오기
                top_recommendations = get_top_n_recommendations(user_predictions, n=5)

                if top_recommendations.get(target_user_raw_id):
                    print(f"\n사용자 {target_user_raw_id}를 위한 추천 아이템 목록 (SVD):")
                    for item_raw_id, estimated_rating in top_recommendations[target_user_raw_id]:
                        print(f"- 아이템 {item_raw_id}: 예상 평점 {estimated_rating:.2f}")
                else:
                    print("추천할 아이템을 찾지 못했습니다.")
        else:
            if data: # Only print this if data loading itself was successful
                print("\n추천을 생성할 대상 사용자를 결정하지 못했습니다.")

    else:
        print("\n데이터 로드에 실패하여 예제를 실행할 수 없습니다.")

    print("\n--- SVD 예제 실행 완료 ---")
