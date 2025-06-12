# examples/collaborative_filtering/user_cf_example.py
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
import sys

# --- User-Based Collaborative Filtering: Basic Explanation ---
# User-Based Collaborative Filtering (UBCF) is a recommendation algorithm that suggests items
# to a user based on the preferences of similar users. The core idea is: "Users similar to you also liked...".
#
# How it works:
# 1. Build a User-Item Interaction Matrix: This matrix represents users' interactions with items
#    (e.g., ratings, purchases, views). Rows are users, and columns are items.
# 2. Calculate User-User Similarity: Compute how similar users are to each other.
#    A common method is to treat each user as a vector of ratings they've given to items
#    and then calculate the cosine similarity between these user vectors.
# 3. Generate Recommendations: To recommend items for a target user:
#    a. Find users who are most similar to the target user (the "neighbors").
#    b. Identify items that these similar users have liked (and the target user hasn't interacted with yet).
#    c. Aggregate the preferences of these neighbors for those items, possibly weighting by
#       similarity scores, to produce a ranked list of recommendations.
#
# Pros:
# - Simplicity and Intuitiveness: The concept is easy to understand.
# - Effective for some datasets: Can uncover nuanced preferences if there are users with similar tastes.
# - Serendipity: Can sometimes recommend items that are not obviously similar to what the user
#   has consumed before, by leveraging tastes of similar users.
#
# Cons:
# - Data Sparsity: Performance degrades significantly if the user-item matrix is very sparse,
#   as it becomes hard to find users with overlapping ratings.
# - Scalability of Similarity Calculation: Computing user-user similarity can be computationally
#   expensive (O(U^2 * I) where U is number of users, I is number of items) for a large number of users.
# - New User (Cold-Start): Difficult to provide recommendations for new users with few or no interactions,
#   as their similarity to others cannot be reliably computed.
# - Popularity Bias: May over-recommend items popular among many users.
# - User preferences can change: The user similarity matrix might need frequent updates.
# ---

# Dynamically add project root to sys.path for module imports
# __file__ is examples/collaborative_filtering/user_cf_example.py
# os.path.dirname(__file__) is examples/collaborative_filtering
# os.path.join(os.path.dirname(__file__), '..', '..') is examples/collaborative_filtering/../../ which resolves to the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root) # Insert at the beginning for priority

# 1. 데이터 로드 및 전처리 (Big O: 로딩 O(N_interactions), 밀집 행렬 변환 O(U*I))
def load_and_preprocess_data(filepath='data/dummy_interactions.csv'):
    """
    상호작용 데이터를 로드하고 사용자-아이템 행렬로 변환합니다.
    """
    # Construct absolute path to data file if a relative path is given
    # This assumes 'data/dummy_interactions.csv' is relative to project root
    if not os.path.isabs(filepath) and filepath.startswith('data/'):
        abs_filepath = os.path.join(project_root, filepath)
    else:
        abs_filepath = filepath

    if not os.path.exists(abs_filepath):
        print(f"오류: {abs_filepath} 파일을 찾을 수 없습니다.")
        # Check if the missing file is the specific dummy data file we can generate
        if abs_filepath.endswith('data/dummy_interactions.csv'):
            print("더미 데이터 생성 시도 중... ('data/generate_dummy_data.py' 실행)")
            try:
                from data.generate_dummy_data import generate_dummy_data
                generate_dummy_data() # Call the function to create data
                print("더미 데이터 생성 완료.")
                # After generation, check again if the file exists
                if not os.path.exists(abs_filepath):
                    print(f"데이터 자동 생성 후에도 {abs_filepath}를 찾을 수 없습니다.")
                    return None
            except ImportError as e_import:
                print(f"ImportError: 'generate_dummy_data' 임포트 실패. 경로: {project_root} sys.path: {sys.path}. 오류: {e_import}")
                return None
            except Exception as e_general:
                print(f"더미 데이터 생성 실패: {e_general}")
                return None
        else:
            # Not the specific dummy file, or some other path issue
            return None

    df = pd.read_csv(abs_filepath)
    user_item_matrix = df.pivot_table(index='user_id', columns='item_id', values='rating').fillna(0)
    return user_item_matrix

# 2. 사용자 간 유사도 계산 (Big O: O(U^2 * I) for dense matrix, 여기서 U=사용자 수, I=아이템 수)
# 사용자 수가 매우 많을 경우 병목이 될 수 있습니다.
def calculate_user_similarity(user_item_matrix):
    """
    사용자 간의 코사인 유사도를 계산합니다. (사용자가 행에 있으므로 그대로 사용)
    """
    user_similarity_matrix = cosine_similarity(user_item_matrix)
    user_similarity_df = pd.DataFrame(user_similarity_matrix,
                                      index=user_item_matrix.index,
                                      columns=user_item_matrix.index)
    return user_similarity_df

# 3. 사용자 기반 협업 필터링 추천
# (Big O: 특정 사용자에 대해 O(U_similar * I_avg_rated), 여기서 U_similar는 유사 사용자 수, I_avg_rated는 유사 사용자들이 평가한 평균 아이템 수)
def get_user_based_recommendations(target_user_id, user_item_matrix, user_similarity_df, num_recommendations=5, top_n_similar_users=10):
    """
    특정 사용자에게 사용자 기반 협업 필터링을 사용하여 아이템을 추천합니다.
    """
    if target_user_id not in user_item_matrix.index:
        print(f"대상 사용자 ID {target_user_id}가 데이터에 없습니다.")
        return []

    # 대상 사용자가 평가한 아이템들
    target_user_ratings = user_item_matrix.loc[target_user_id]
    target_rated_items = target_user_ratings[target_user_ratings > 0].index

    # 대상 사용자와 다른 사용자들 간의 유사도
    similarities_to_target = user_similarity_df.loc[target_user_id].drop(target_user_id) # 자기 자신 제외

    # 유사도가 높은 상위 N명의 사용자 선택
    similar_users = similarities_to_target.nlargest(top_n_similar_users).index

    recommendation_scores = {}
    sum_similarity_scores = {} # For weighted average, if needed

    # 유사 사용자들이 평가한 아이템들을 기반으로 추천 점수 계산
    for similar_user_id in similar_users:
        similar_user_similarity_score = user_similarity_df.loc[target_user_id, similar_user_id]

        # 유사 사용자가 0 이하의 유사도를 가지면 건너뛰기 (선택 사항)
        if similar_user_similarity_score <= 0:
            continue

        # 유사 사용자가 평가한 아이템들
        similar_user_ratings = user_item_matrix.loc[similar_user_id]
        # similar_user_rated_items = similar_user_ratings[similar_user_ratings > 0].index # Not needed directly

        for item_id, rating in similar_user_ratings.items():
            # 대상 사용자가 아직 평가하지 않았고, 유사 사용자가 긍정적으로 평가한 아이템만
            if rating > 0 and item_id not in target_rated_items:
                if item_id not in recommendation_scores:
                    recommendation_scores[item_id] = 0
                    sum_similarity_scores[item_id] = 0 # For weighted average

                # 추천 점수에 (유사도 * 유사 사용자의 평점)을 더함
                recommendation_scores[item_id] += similar_user_similarity_score * rating
                sum_similarity_scores[item_id] += similar_user_similarity_score # 누적 유사도 (가중 평균 분모용)

    # 가중 평균으로 최종 점수 계산: sum(similarity * rating) / sum(similarity)
    # 이렇게 하면 평점 범위가 유지되고, 더 많은 유사 사용자가 지지하는 항목에 더 높은 신뢰를 줄 수 있음
    final_scores = {}
    for item_id, score in recommendation_scores.items():
        if sum_similarity_scores[item_id] > 0: # 0으로 나누는 것 방지
            final_scores[item_id] = score / sum_similarity_scores[item_id]
        # else: final_scores[item_id] = 0 # sum_similarity_scores가 0이면 score도 0이어야 함

    sorted_recommendations = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_recommendations[:num_recommendations]

# 메인 실행 함수
if __name__ == "__main__":
    print("--- 사용자 기반 협업 필터링 예제 시작 ---")

    # 데이터 파일 경로 설정 (프로젝트 루트 기준)
    data_file_path = 'data/dummy_interactions.csv'

    # 1. 데이터 로드 및 전처리
    print(f"\n데이터 로드 및 전처리 중 (경로: {data_file_path})...")
    user_item_matrix = load_and_preprocess_data(filepath=data_file_path)

    if user_item_matrix is not None:
        print(f"사용자-아이템 행렬 (상위 5개 표시):\n{user_item_matrix.head()}")

        # 2. 사용자 유사도 계산
        print("\n사용자 유사도 계산 중... (사용자 수에 따라 시간이 소요될 수 있습니다.)")
        user_similarity_df = calculate_user_similarity(user_item_matrix)
        print(f"사용자 유사도 행렬 (상위 5x5 표시):\n{user_similarity_df.iloc[:5, :5]}")

        # 3. 특정 사용자에 대한 추천 생성
        if not user_item_matrix.empty:
            # Test with a user who has rated some items, if possible
            # For robustness, pick the first user or handle case where user_item_matrix might be all zeros for a user
            target_user_id_example = user_item_matrix.index[0]

            # Check if the target user has rated any items to make recommendations more meaningful
            if user_item_matrix.loc[target_user_id_example].sum() == 0:
                print(f"\n주의: 사용자 ID {target_user_id_example}는 평가한 아이템이 없어 의미있는 추천이 어려울 수 있습니다.")
                # Try to find another user for a better demo
                for uid in user_item_matrix.index:
                    if user_item_matrix.loc[uid].sum() > 0:
                        target_user_id_example = uid
                        break

            print(f"\n사용자 ID {target_user_id_example}에 대한 추천 아이템 생성 중...")

            recommendations = get_user_based_recommendations(
                target_user_id_example,
                user_item_matrix,
                user_similarity_df,
                num_recommendations=5,
                top_n_similar_users=10 # 이웃으로 고려할 유사 사용자 수
            )

            if recommendations:
                print(f"\n사용자 {target_user_id_example}를 위한 추천 아이템 목록:")
                for item, score in recommendations:
                    print(f"- 아이템 {item}: 예상 점수 {score:.2f}")
            else:
                print(f"사용자 {target_user_id_example}에게 추천할 아이템이 없습니다. (유사 사용자가 없거나, 유사 사용자가 평가한 아이템을 이미 모두 평가했을 수 있습니다.)")
        else:
            print("\n사용자-아이템 행렬이 비어있어 추천을 생성할 수 없습니다.")
    else:
        print("\n데이터 로드에 실패하여 예제를 실행할 수 없습니다.")

    print("\n--- 사용자 기반 협업 필터링 예제 실행 완료 ---")
