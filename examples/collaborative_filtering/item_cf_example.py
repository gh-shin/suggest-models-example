# examples/collaborative_filtering/item_cf_example.py
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os

# --- Item-Based Collaborative Filtering: Basic Explanation ---
# Item-Based Collaborative Filtering (IBCF) is a recommendation algorithm that suggests items to a user
# based on the similarity between items. The core idea is: "Users who liked this item also liked...".
#
# How it works:
# 1. Build a User-Item Interaction Matrix: This matrix represents users' interactions with items
#    (e.g., ratings, purchases, views). Rows are typically users, and columns are items.
# 2. Calculate Item-Item Similarity: Compute how similar items are to each other.
#    A common method is to treat each item as a vector of ratings given by users and then
#    calculate the cosine similarity between these item vectors.
# 3. Generate Recommendations: To recommend items for a user:
#    a. Find items the user has positively interacted with in the past.
#    b. For each of these items, find the most similar items (based on the item-item similarity matrix).
#    c. Aggregate these similar items, possibly weighting them by similarity and the user's rating
#       for the source item, to produce a ranked list of recommendations.
#    d. Filter out items the user has already interacted with.
#
# Pros:
# - Stability: Item similarities tend to change less frequently than user preferences,
#   so the similarity matrix doesn't need to be recomputed as often as in User-Based CF.
# - Explainability: Recommendations can be explained (e.g., "Because you liked item X, you might like item Y").
# - Handles new users reasonably well if they rate a few items.
#
# Cons:
# - Data Sparsity: Performance can degrade if the user-item matrix is very sparse, as it becomes
#   hard to find overlapping user ratings to calculate item similarity.
# - Scalability of Similarity Calculation: Computing item-item similarity can be computationally
#   expensive (O(N^2 * U) where N is number of items, U is number of users) for very large item catalogs.
#   Efficient implementations (e.g., using sparse matrices or approximation techniques like LSH) are needed.
# - Popularity Bias: Tends to recommend popular items more frequently.
# - Cold-Start for New Items: New items with no interactions cannot be recommended as their similarity
#   to other items cannot be calculated.
# ---

# 1. 데이터 로드 및 전처리 (Big O: 로딩 O(N_interactions), 밀집 행렬 변환 O(U*I), 여기서 U=사용자 수, I=아이템 수)
def load_and_preprocess_data(filepath='data/dummy_interactions.csv'):
    """
    상호작용 데이터를 로드하고 사용자-아이템 행렬로 변환합니다.
    """
    if not os.path.exists(filepath):
        print(f"오류: {filepath} 파일을 찾을 수 없습니다. 'data/generate_dummy_data.py'를 먼저 실행해 주십시오.")
        # Try to generate dummy data if not found, specifically if the target is our known dummy file
        if filepath.endswith('data/dummy_interactions.csv'):
            print("더미 데이터 생성 시도 중...")
            try:
                import sys
                # os is already imported at the module level
                project_root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

                # Explicitly add project root to sys.path
                if project_root_dir not in sys.path:
                    sys.path.insert(0, project_root_dir)

                from data.generate_dummy_data import generate_dummy_data
                generate_dummy_data()
                print("더미 데이터 생성 완료.")
                # Ensure the file now exists
                if not os.path.exists(filepath):
                    print(f"자동 데이터 생성 후에도 {filepath}를 찾을 수 없습니다.")
                    return None
            except ImportError as e:
                print(f"ImportError: 'generate_dummy_data' 함수를 임포트할 수 없습니다. 경로 문제일 수 있습니다: {e}")
                print("PYTHONPATH를 확인하거나 프로젝트 루트에서 스크립트를 실행해 주십시오.")
                return None
            except Exception as e:
                print(f"더미 데이터 생성 실패: {e}")
                return None
        else:
            return None # Cannot generate other files

    df = pd.read_csv(filepath)
    # Pivot 테이블을 사용하여 사용자-아이템 행렬 생성
    # 상호작용이 없는 경우 0으로 채워 넣습니다. (실제로는 평균 평점 등으로 채울 수도 있음)
    user_item_matrix = df.pivot_table(index='user_id', columns='item_id', values='rating').fillna(0)
    return user_item_matrix

# 2. 아이템 간 유사도 계산 (Big O: O(I^2 * U) for dense matrix, 여기서 I=아이템 수, U=사용자 수)
# scikit-learn의 cosine_similarity는 효율적으로 구현되어 있지만, 아이템 수가 매우 많을 경우 여전히 병목이 될 수 있습니다.
# 대규모 시스템에서는 희소 행렬 라이브러리(scipy.sparse) 또는 근사 최근접 이웃(ANN) 알고리즘을 고려해야 합니다.
def calculate_item_similarity(user_item_matrix):
    """
    아이템 간의 코사인 유사도를 계산합니다. (아이템이 열에 있도록 전치하여 계산)
    """
    # 아이템을 행으로, 사용자를 열로 만들어 각 아이템 벡터 간 유사도 계산
    item_similarity_matrix = cosine_similarity(user_item_matrix.T)
    item_similarity_df = pd.DataFrame(item_similarity_matrix,
                                      index=user_item_matrix.columns,
                                      columns=user_item_matrix.columns)
    return item_similarity_df

# 3. 아이템 기반 협업 필터링 추천
# (Big O: 특정 사용자에 대해 O(I_rated * I_all), 여기서 I_rated는 사용자가 평가한 아이템 수, I_all은 전체 아이템 수)
# 최적화 가능: 모든 unrated item에 대해 계산하는 대신, 사용자가 평가한 아이템과 유사한 아이템들만 고려.
def get_item_based_recommendations(user_id, user_item_matrix, item_similarity_df, num_recommendations=5):
    """
    특정 사용자에게 아이템 기반 협업 필터링을 사용하여 아이템을 추천합니다.
    """
    if user_id not in user_item_matrix.index:
        print(f"사용자 ID {user_id}가 데이터에 없습니다.")
        return []

    # 사용자가 평가한 아이템 목록 (평점 > 0)
    user_ratings = user_item_matrix.loc[user_id]
    rated_items_with_ratings = user_ratings[user_ratings > 0]
    rated_items = rated_items_with_ratings.index

    # 아직 평가하지 않은 아이템 목록
    all_items = user_item_matrix.columns
    unrated_items = all_items.difference(rated_items)

    # 추천 점수를 저장할 딕셔너리
    recommendation_scores = {}

    # 평가하지 않은 각 아이템에 대해 예상 평점 계산
    for item_to_recommend in unrated_items:
        weighted_sum_of_ratings = 0
        sum_of_similarities = 0

        # 사용자가 평가한 각 아이템과 현재 추천 후보 아이템 간의 유사도 활용
        for rated_item in rated_items:
            similarity = item_similarity_df.loc[item_to_recommend, rated_item]
            rating = rated_items_with_ratings.loc[rated_item]

            if similarity > 0:
                weighted_sum_of_ratings += similarity * rating
                sum_of_similarities += similarity

        if sum_of_similarities > 0:
            predicted_rating = weighted_sum_of_ratings / sum_of_similarities
            recommendation_scores[item_to_recommend] = predicted_rating

    # 예측 평점이 높은 순으로 정렬하여 추천 목록 반환
    sorted_recommendations = sorted(recommendation_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_recommendations[:num_recommendations]

# 메인 실행 함수
if __name__ == "__main__":
    print("--- 아이템 기반 협업 필터링 예제 시작 ---")

    # 1. 데이터 로드 및 전처리
    # Adjust the path to dummy_interactions.csv relative to the project root
    # Assuming this script is in examples/collaborative_filtering
    # __file__ is examples/collaborative_filtering/item_cf_example.py
    # project_root is /app
    project_root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_file_path = os.path.join(project_root_path, 'data', 'dummy_interactions.csv') # Correctly /app/data/dummy_interactions.csv

    print(f"\n데이터 로드 및 전처리 중 (경로: {data_file_path})...")
    # Pass the absolute path to load_and_preprocess_data
    user_item_matrix = load_and_preprocess_data(filepath=data_file_path)

    if user_item_matrix is not None:
        print("사용자-아이템 행렬 (일부):\n", user_item_matrix.head())

        # 2. 아이템 유사도 계산
        print("\n아이템 유사도 계산 중... (아이템 수에 따라 시간이 소요될 수 있습니다.)")
        item_similarity_df = calculate_item_similarity(user_item_matrix)
        print("아이템 유사도 행렬 (일부):\n", item_similarity_df.iloc[:5, :5])

        # 3. 특정 사용자에 대한 추천 생성
        if not user_item_matrix.empty:
            # 예시로 첫 번째 사용자 ID를 선택합니다.
            target_user_id = user_item_matrix.index[0]
            print(f"\n사용자 ID {target_user_id}에 대한 추천 아이템 생성 중...")
            recommendations = get_item_based_recommendations(target_user_id, user_item_matrix, item_similarity_df, num_recommendations=5)

            if recommendations:
                print(f"\n사용자 {target_user_id}를 위한 추천 아이템 목록:")
                for item, score in recommendations:
                    print(f"- 아이템 {item}: 예상 평점 {score:.2f}")
            else:
                print("추천할 아이템이 없습니다. (모든 아이템을 이미 평가했거나 유사한 아이템이 존재하지 않을 수 있습니다.)")
        else:
            print("\n사용자-아이템 행렬이 비어있어 추천을 생성할 수 없습니다.")
    else:
        print("\n데이터 로드에 실패하여 예제를 실행할 수 없습니다.")

    print("\n--- 아이템 기반 협업 필터링 예제 실행 완료 ---")
