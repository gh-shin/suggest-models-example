# examples/collaborative_filtering/user_cf_example.py
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
import sys

# --- 사용자 기반 협업 필터링: 기본 설명 ---
# 사용자 기반 협업 필터링(UBCF)은 비슷한 취향을 가진 다른 사용자의 선호도를 기반으로 사용자에게 아이템을 제안하는 추천 알고리즘입니다.
# 핵심 아이디어는 다음과 같습니다: "사용자 A와 B가 공통으로 평가한 아이템에 대해 유사한 선호도를 가진다면, 사용자 A는 사용자 B가 좋아했지만 사용자 A는 아직 보지 못한 아이템을 선호할 가능성이 높다."
#
# 작동 방식:
# 1. 사용자-아이템 상호작용 Matrix 구축: 이 Matrix(종종 희소)는 사용자의 아이템과의 상호작용을 포함합니다. 값은 명시적 평가(예: 별 1-5개) 또는 암시적 피드백(예: 조회수, 클릭수)이 될 수 있습니다.
#    행은 사용자를, 열은 아이템을 나타냅니다.
# 2. 사용자-사용자 유사성 계산: 상호작용 기록을 기반으로 사용자들이 서로 얼마나 유사한지 결정합니다. 일반적인 방법은 사용자 벡터(사용자-아이템 Matrix의 행)에 대해 계산되는 코사인 유사도(Cosine Similarity)입니다. 피어슨 상관계수(Pearson correlation)와 같은 다른 측정 항목도 사용할 수 있습니다.
# 3. 추천 생성: 대상 사용자를 위한 아이템 추천 방법:
#    a. "이웃 사용자" 식별: 대상 사용자와 가장 유사한 사용자 하위 집합을 찾습니다(예: 상위 N명의 유사한 사용자).
#    b. 보지 않은 아이템에 대한 점수 예측: 대상 사용자가 상호작용하지 않은 아이템에 대해 예상 평가를 예측합니다. 이는 종종 이웃 사용자가 해당 아이템에 부여한 평가의 가중 평균을 취하여 수행됩니다. 가중치는 일반적으로 이러한 이웃과 대상 사용자 간의 유사성 점수입니다.
#    c. 순위 지정 및 추천: 예측된 점수를 기준으로 아이템을 정렬하고 상위 K개의 아이템을 추천합니다.
#
# Pros (장점):
# - 단순성 및 직관성: 기본 개념이 간단하고 설명하기 쉽습니다.
# - 다양한 취향에 효과적: 유사하지만 명확하지 않은 취향 패턴을 가진 사용자가 있는 경우 미묘한 선호도를 발견할 수 있습니다.
# - Serendipity (우연한 발견): 유사한 사용자의 취향을 활용하여 사용자가 이전에 소비한 것과 (내용상) 직접적으로 유사하지 않은 아이템을 때때로 추천할 수 있습니다.
# - 아이템 특징 공학 불필요: 아이템 특성에 대한 지식이 필요하지 않습니다.
#
# Cons (단점):
# - Data Sparsity: 사용자-아이템 Matrix가 매우 희소하면 성능이 크게 저하됩니다.
#   사용자가 공통으로 평가한 아이템이 거의 없는 경우 유사성 계산이 신뢰할 수 없게 됩니다.
# - 유사성 계산 확장성: 사용자-사용자 유사성 계산은 계산 비용이 많이 듭니다. U명의 사용자와 I개의 아이템에 대해 일반적으로 밀집 데이터의 경우 O(U^2 * I) 또는
#   쌍별 코사인 유사도를 사용할 때 희소 데이터의 경우 O(U^2 * avg_items_rated_per_user)입니다.
#   수백만 명의 사용자에게는 이것이 엄청난 부담이 될 수 있습니다.
# - 신규 사용자 (Cold-Start): 상호작용이 거의 없거나 전혀 없는 신규 사용자에 대해서는 다른 사용자와의 유사성을 신뢰성 있게 계산할 수 없으므로 추천을 제공하기 어렵습니다.
# - 인기 편향: 많은 사용자에게 인기 있는 아이템을 과도하게 추천하는 경향이 있을 수 있으며, 적절히 처리하지 않으면 틈새 취향을 가진 사용자에 대한 개인화가 줄어들 수 있습니다.
# - 동적 사용자 선호도: 사용자 취향은 시간이 지남에 따라 변할 수 있습니다. 사용자 유사성 모델은 자주 재계산해야 할 수 있습니다.
# ---

# 모듈 임포트를 위해 프로젝트 루트를 sys.path에 동적으로 추가
# 이 스크립트를 모든 디렉토리에서 실행해도 프로젝트 모듈을 찾을 수 있도록 합니다.
# __file__은 examples/collaborative_filtering/user_cf_example.py 입니다.
# os.path.dirname(__file__)은 examples/collaborative_filtering 입니다.
# os.path.join(os.path.dirname(__file__), '..', '..')은 프로젝트 루트까지 두 단계 위로 이동합니다.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root) # 높은 우선순위를 위해 처음에 삽입

# 1. 데이터 로딩 및 전처리
# Time Complexity:
# - CSV 읽기: O(N_interactions) 여기서 N_interactions는 CSV의 행 수입니다.
# - Pivot table: 데이터를 반복하는 데 O(N_interactions), 그 다음 밀집 Matrix를 구성하는 데 O(U*I),
#   여기서 U는 고유 사용자 수이고 I는 고유 아이템 수입니다.
# - fillna(0): 밀집 Matrix의 경우 최악의 경우 O(U*I).
# Overall: 대략 O(N_interactions + U*I).
def load_and_preprocess_data(filepath='data/dummy_interactions.csv'):
    """
    한국어: CSV 파일에서 상호작용 데이터를 로드하고 사용자-아이템 Matrix로 변환합니다.
    지정된 CSV 파일을 찾을 수 없는 경우 더미 데이터를 생성하려고 시도합니다.

    Loads interaction data from a CSV file and transforms it into a user-item matrix.
    If the specified CSV file is not found, it attempts to generate dummy data.

    Args:
        filepath (str): Path to the CSV data file. Assumed to be relative to project root
                        if it starts with 'data/', otherwise an absolute path.

    Returns:
        pandas.DataFrame: A user-item matrix where rows are user_id, columns are item_id,
                          and values are ratings. Returns None if data loading fails.
    """
    # 데이터 파일에 대한 절대 경로 구성, 'data/' 경로는 프로젝트 루트에 상대적인 것으로 가정
    abs_filepath = filepath
    if not os.path.isabs(filepath) and filepath.startswith('data/'):
        abs_filepath = os.path.join(project_root, filepath)

    if not os.path.exists(abs_filepath):
        print(f"Error: Data file not found at {abs_filepath}.")
        # 기본 더미 데이터 경로인지 구체적으로 확인하여 생성 제안
        if abs_filepath.endswith('data/dummy_interactions.csv'):
            print("Attempting to generate dummy data by running 'data/generate_dummy_data.py'...")
            try:
                # 데이터 생성 스크립트를 동적으로 임포트하고 실행
                from data.generate_dummy_data import generate_dummy_data
                generate_dummy_data()
                print("Dummy data generated successfully.")
                # 파일이 생성되었는지 확인
                if not os.path.exists(abs_filepath):
                    print(f"Error: Dummy data generation completed, but file still not found at {abs_filepath}.")
                    return None
            except ImportError as e_import:
                print(f"ImportError: Failed to import 'generate_dummy_data'. Ensure it's in the 'data' directory. "
                      f"Project root: {project_root}, sys.path: {sys.path}. Error: {e_import}")
                return None
            except Exception as e_general:
                print(f"Error: Failed to generate dummy data. Exception: {e_general}")
                return None
        else:
            # 누락된 파일은 생성 방법을 아는 파일이 아님
            return None

    # CSV에서 데이터 로드
    df = pd.read_csv(abs_filepath)
    # 사용자-아이템 Matrix 생성:
    # - index='user_id': 사용자가 행이 됨
    # - columns='item_id': 아이템이 열이 됨
    # - values='rating': 평점이 Matrix 셀을 채움
    # - fillna(0): 누락된 평점 (사용자가 아이템을 평가하지 않은 경우)은 0으로 채워짐.
    #   이는 일반적인 관행이지만 유사성에 영향을 미칠 수 있음 (예: 중립성 또는 상호작용 없음을 의미).
    user_item_matrix = df.pivot_table(index='user_id', columns='item_id', values='rating').fillna(0)
    return user_item_matrix

# 2. 사용자 유사성 계산
# Time Complexity: O(U^2 * I) 여기서 U는 사용자 수이고 I는 아이템 수입니다.
# 이는 cosine_similarity가 U명의 모든 사용자 쌍 간의 유사성을 계산하고,
# 각 사용자의 선호도 벡터 길이가 I이기 때문입니다. 이 단계는 U가 클 경우 병목 현상이 발생할 수 있습니다.
def calculate_user_similarity(user_item_matrix):
    """
    한국어: 아이템 평점을 기반으로 사용자 간의 코사인 유사성을 계산합니다.

    Calculates cosine similarity between users based on their item ratings.

    Args:
        user_item_matrix (pandas.DataFrame): The user-item rating matrix.

    Returns:
        pandas.DataFrame: A square matrix where rows and columns are user_ids,
                          and values are the cosine similarity scores between users.
    """
    # cosine_similarity는 행에 샘플이 있을 것으로 예상하며, 이는 user_item_matrix가 구성된 방식입니다 (사용자가 행).
    user_similarity_matrix = cosine_similarity(user_item_matrix)
    # 결과 NumPy 배열을 다시 DataFrame으로 변환하여 쉽게 처리할 수 있도록 하고,
    # 원본 user_id를 인덱스와 열로 사용합니다.
    user_similarity_df = pd.DataFrame(user_similarity_matrix,
                                      index=user_item_matrix.index,
                                      columns=user_item_matrix.index)
    return user_similarity_df

# 3. 사용자 기반 협업 필터링 추천
# 단일 대상 사용자에 대한 Time Complexity:
# - 대상 사용자 평점 가져오기: O(I)
# - 대상 사용자에 대한 유사성 가져오기: O(U)
# - 상위 N명의 유사한 사용자 찾기: 모두 정렬하는 경우 O(U log U) 또는 힙(nlargest)을 사용하는 경우 O(U log N_similar).
# - top_n_similar_users 반복 (이를 K_sim이라 함):
#   - 각 유사한 사용자에 대해 해당 아이템 반복 (avg_items_rated_by_sim_user, 이를 I_avg_sim이라 함):
#     - 점수에 대한 딕셔너리 조회/업데이트: 평균 O(1).
# - 최종 추천 정렬: O(M log M) 여기서 M은 후보 아이템 수입니다.
# Overall: 대략 O(U + K_sim * I_avg_sim + M log M).
# K_sim이 작고 M이 너무 크지 않으면 사용자당 효율적일 수 있습니다.
def get_user_based_recommendations(target_user_id, user_item_matrix, user_similarity_df, num_recommendations=5, top_n_similar_users=10):
    """
    한국어: 사용자 기반 협업 필터링을 사용하여 대상 사용자에 대한 아이템 추천을 생성합니다.

    Generates item recommendations for a target user using User-Based Collaborative Filtering.

    Args:
        target_user_id (int or str): The ID of the user for whom to generate recommendations.
        user_item_matrix (pandas.DataFrame): The user-item rating matrix.
        user_similarity_df (pandas.DataFrame): The user-user similarity matrix.
        num_recommendations (int): The number of items to recommend.
        top_n_similar_users (int): The number of most similar users (neighbors) to consider.

    Returns:
        list: A list of tuples, where each tuple contains (item_id, predicted_score),
              sorted by predicted_score in descending order.
    """
    if target_user_id not in user_item_matrix.index:
        print(f"Error: Target user ID {target_user_id} not found in the user-item matrix.")
        return []

    # 추천에서 제외하기 위해 대상 사용자가 이미 평가한 아이템 가져오기
    target_user_ratings = user_item_matrix.loc[target_user_id]
    target_rated_items = target_user_ratings[target_user_ratings > 0].index

    # 대상 사용자에 대한 다른 모든 사용자의 유사성 점수 가져오기
    # 유사성 목록에서 대상 사용자 자체 제외
    similarities_to_target = user_similarity_df.loc[target_user_id].drop(target_user_id, errors='ignore')

    # 상위 N명의 가장 유사한 사용자(이웃) 선택
    # 유사성 점수가 높은 사용자는 더 영향력 있는 것으로 간주됨.
    similar_users = similarities_to_target.nlargest(top_n_similar_users).index

    if not len(similar_users):
        print(f"No similar users found for user {target_user_id} with the current criteria.")
        return []

    # 아이템에 대한 예상 점수 저장
    recommendation_scores = {}
    # 가중 평균 계산을 위한 유사성 점수 합계 저장
    sum_similarity_for_item = {}

    # 각 유사한 사용자 반복
    for similar_user_id in similar_users:
        # 대상 사용자와 현재 유사한 사용자 간의 유사성 점수 가져오기
        similarity_score = user_similarity_df.loc[target_user_id, similar_user_id]

        # 선택 사항: 양수가 아닌 유사성을 가진 사용자 건너뛰기 (유사성 측정 항목 및 데이터에 따라 다름)
        if similarity_score <= 0:
            continue

        # 현재 유사한 사용자로부터 평점 가져오기
        similar_user_ratings = user_item_matrix.loc[similar_user_id]

        # 유사한 사용자가 평가한 아이템 반복
        for item_id, rating in similar_user_ratings.items():
            # 유사한 사용자가 긍정적으로 평가하고
            # 대상 사용자가 아직 평가하지 않은 경우에만 추천
            if rating > 0 and item_id not in target_rated_items:
                # 점수 집계: (유사성_점수 * 평점) 추가
                recommendation_scores[item_id] = recommendation_scores.get(item_id, 0) + similarity_score * rating
                # 가중 평균의 분모에 대한 유사성 점수 집계
                sum_similarity_for_item[item_id] = sum_similarity_for_item.get(item_id, 0) + similarity_score

    # 각 후보 아이템에 대한 가중 평균 점수 계산
    # final_score = sum(similarity_score * rating_from_similar_user) / sum(similarity_scores_of_users_who_rated_the_item)
    final_scores = {}
    for item_id, total_weighted_score in recommendation_scores.items():
        if sum_similarity_for_item.get(item_id, 0) > 0: # 0으로 나누기 방지
            final_scores[item_id] = total_weighted_score / sum_similarity_for_item[item_id]
        # sum_similarity_for_item == 0인 아이템은 처리되지 않았거나 유사성이 0인 사용자에 의해서만 처리되었음을 의미.

    # 예상 점수를 기준으로 추천을 내림차순으로 정렬
    sorted_recommendations = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)

    return sorted_recommendations[:num_recommendations]

# 메인 실행 블록
if __name__ == "__main__":
    print("--- User-Based Collaborative Filtering Example ---")

    # 데이터 파일 경로 정의 (프로젝트 루트에 상대적)
    data_file_path = 'data/dummy_interactions.csv'
    print(f"\nAttempting to load and preprocess data from: {data_file_path}")
    user_item_matrix = load_and_preprocess_data(filepath=data_file_path)

    if user_item_matrix is not None and not user_item_matrix.empty:
        print("\nUser-Item Matrix (first 5 rows/users):")
        print(user_item_matrix.head())

        print("\nCalculating User Similarity Matrix... (this may take a while for many users)") # 사용자 유사도 Matrix 계산 중... (사용자가 많으면 시간이 걸릴 수 있음)
        user_similarity_df = calculate_user_similarity(user_item_matrix)
        print("\nUser Similarity Matrix (first 5x5 users):") # 사용자 유사도 Matrix (처음 5x5 사용자):
        # 5x5를 표시할 만큼 충분한 사용자와 아이템이 있는지 확인하고, 그렇지 않으면 조정
        display_limit = min(5, user_similarity_df.shape[0])
        print(user_similarity_df.iloc[:display_limit, :display_limit])

        # 추천을 위한 대상 사용자 선택
        # 안정성을 위해 첫 번째 사용자를 선택. Matrix가 작을 수 있는 경우 처리.
        if not user_item_matrix.empty:
            target_user_id_example = user_item_matrix.index[0]

            # 대상 사용자가 의미 있는 추천을 위해 아이템을 평가했는지 확인
            if user_item_matrix.loc[target_user_id_example].sum() == 0:
                print(f"\nWarning: Target user ID {target_user_id_example} has no rated items in the matrix. " # 경고: 대상 사용자 ID {target_user_id_example}는 Matrix에 평가된 아이템이 없습니다.
                      "Recommendations might be less meaningful or empty.") # 추천이 의미 없거나 비어 있을 수 있습니다.
                # 더 나은 시연을 위해 평점이 있는 사용자를 찾으려고 시도
                for uid in user_item_matrix.index:
                    if user_item_matrix.loc[uid].sum() > 0:
                        target_user_id_example = uid
                        print(f"Switching to user ID {target_user_id_example} for a better demo as they have rated items.") # 더 나은 데모를 위해 평점이 있는 사용자 ID {target_user_id_example}로 전환합니다.
                        break

            print(f"\nGenerating recommendations for User ID: {target_user_id_example}...") # 사용자 ID: {target_user_id_example}에 대한 추천 생성 중...
            # 추천 가져오기
            # top_n_similar_users: 고려할 이웃 수.
            # 숫자가 클수록 적용 범위가 넓어질 수 있지만 유사성이 낮은 사용자가 포함될 수 있음.
            recommendations = get_user_based_recommendations(
                target_user_id_example,
                user_item_matrix,
                user_similarity_df,
                num_recommendations=5, # 추천할 아이템 수
                top_n_similar_users=10 # 상위 10명의 유사한 사용자 고려
            )

            if recommendations:
                print(f"\nTop {len(recommendations)} recommendations for User {target_user_id_example}:") # 사용자 {target_user_id_example}를 위한 상위 {len(recommendations)}개 추천:
                for item_id, score in recommendations:
                    print(f"- Item ID: {item_id}, Predicted Score: {score:.4f}")
            else:
                print(f"No recommendations could be generated for User {target_user_id_example}. " # 사용자 {target_user_id_example}에 대한 추천을 생성할 수 없습니다.
                      "This could be due to no similar users found, or similar users " # 유사한 사용자를 찾을 수 없거나 유사한 사용자가
                      "not rating any new items not already seen by the target user.") # 대상 사용자가 아직 보지 않은 새 아이템을 평가하지 않았기 때문일 수 있습니다.
        else:
            print("\nUser-Item matrix is empty. Cannot generate recommendations.") # User-Item Matrix가 비어 있습니다. 추천을 생성할 수 없습니다.
    else:
        print("\nData loading failed or resulted in an empty matrix. Cannot proceed with the example.") # 데이터 로딩에 실패했거나 빈 Matrix가 생성되었습니다. 예제를 진행할 수 없습니다.

    print("\n--- User-Based Collaborative Filtering Example Finished ---") # --- 사용자 기반 협업 필터링 예제 완료 ---
