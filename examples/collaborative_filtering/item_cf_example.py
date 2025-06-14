# examples/collaborative_filtering/item_cf_example.py
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
import sys

# --- 아이템 기반 협업 필터링: 기본 설명 ---
# 아이템 기반 협업 필터링(IBCF)은 사용자가 이전에 상호작용한 아이템과 다른 아이템 간의 유사성을 기반으로 사용자에게 아이템을 추천합니다.
# 핵심 아이디어: "사용자가 아이템 X를 좋아했고, 아이템 X가 아이템 Y와 유사하다면, 사용자는 아이템 Y도 좋아할 수 있다."
#
# 작동 방식:
# 1. 사용자-아이템 상호작용 Matrix 생성: 사용자의 아이템에 대한 상호작용(예: 평점)을 나타냅니다.
#    행은 사용자, 열은 아이템입니다. 이는 사용자 기반 CF에서와 동일한 Matrix 입니다.
# 2. 아이템-아이템 유사성 계산: 모든 아이템 쌍 간의 유사성을 계산합니다.
#    이는 일반적으로 아이템 벡터(사용자-아이템 Matrix의 열, 경우에 따라 전치 후)를 생성하고
#    이 벡터들 간의 코사인 유사성을 계산하여 수행됩니다. 아이템의 벡터는 모든 사용자가 해당 아이템을 어떻게 평가했는지를 나타냅니다.
#    결과는 아이템-아이템 유사성 Matrix 입니다.
# 3. 사용자를 위한 추천 생성:
#    a. 대상 사용자가 긍정적으로 평가했거나 상호작용한 아이템을 식별합니다.
#    b. 평가되지 않은 각 아이템(추천 후보 아이템)에 대해:
#       i. 예상 점수를 계산합니다. 이는 종종 사용자가 상호작용한 아이템에 대한 사용자 평점의 가중 합입니다.
#       ii. 각 평가된 아이템의 가중치는 후보 아이템과의 유사성입니다.
#       iii. 사용자 'u'에 대한 아이템 'j'의 예상 점수 공식:
#           P(u,j) = sum( S(j,k) * R(u,k) ) / sum( |S(j,k)| )
#           여기서:
#             - S(j,k)는 아이템 'j'(후보)와 아이템 'k'(사용자가 평가한) 간의 유사성입니다.
#             - R(u,k)는 사용자 'u'가 아이템 'k'에 부여한 평점입니다.
#             - 합계는 사용자 'u'가 평가한 모든 아이템 'k'에 대한 것입니다.
#    c. 예상 점수를 기준으로 후보 아이템의 순위를 매기고 상위 N개를 추천합니다.
#    d. 사용자가 이미 상호작용한 아이템을 필터링합니다.
#
# Pros (장점):
# - 안정성: 아이템 유사성은 사용자-사용자 유사성보다 덜 자주 변경되는 경우가 많으므로,
#   아이템-아이템 유사성 Matrix를 미리 계산하고 덜 자주 업데이트할 수 있습니다.
#   이는 아이템 카탈로그가 사용자 기반이나 사용자 선호도보다 더 정적인 경우에 유용합니다.
# - 사용자 기반 확장성: 사용자 수가 아이템 수보다 훨씬 많을 때 확장성이 더 뛰어나며,
#   비용이 많이 드는 유사성 계산이 아이템에 대해 수행되기 때문입니다.
# - 설명 가능성: 추천을 쉽게 설명할 수 있습니다 (예: "당신이 아이템 X를 좋아했기 때문에 X와 유사한 아이템 Y를 좋아할 수도 있습니다").
# - 신규 사용자 처리: 몇 개의 아이템이라도 평가한 신규 사용자는 해당 아이템이 알려진 유사성을 가지고 있다면 괜찮은 추천을 받을 수 있습니다.
#
# Cons (단점):
# - Data Sparsity: 사용자-아이템 Matrix가 매우 희소하면 동일한 아이템 쌍을 평가한 사용자를 찾기 어려워
#   아이템 유사성 계산의 신뢰성이 떨어집니다.
# - 유사성 계산 확장성 (아이템의 경우): 아이템-아이템 유사성 계산은 계산 비용이 많이 들 수 있습니다.
#   아이템 수(I)가 매우 큰 경우. I개의 아이템과 U명의 사용자에 대해 일반적으로 O(I^2 * U)입니다.
#   밀집 데이터의 경우 또는 희소 데이터의 경우 O(I^2 * avg_users_rated_per_item)입니다.
# - 인기 편향: UBCF와 유사하게 인기 있는 아이템을 추천하는 경향이 있을 수 있습니다.
# - 신규 아이템 Cold-Start: 상호작용 데이터가 없는 신규 아이템은 다른 아이템과의 유사성을
#   판단할 수 없으므로 추천할 수 없습니다.
# - 제한된 Serendipity: 사용자가 이미 알고 있는 것과 너무 유사한 아이템을 추천하여 새로운 아이템 발견 가능성을 줄일 수 있습니다.
# ---

# 모듈 임포트를 위해 프로젝트 루트를 sys.path에 동적으로 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 1. 데이터 로딩 및 전처리
# Time Complexity:
# - CSV 읽기: O(N_interactions)
# - Pivot table & fillna: O(N_interactions + U*I)
# Overall: 대략 O(N_interactions + U*I).
def load_and_preprocess_data(filepath='data/dummy_interactions.csv'):
    """
    Loads interaction data and transforms it into a user-item matrix.
    Attempts to generate dummy data if the specified file is not found.
    """
    # 한국어: 상호작용 데이터를 로드하고 사용자-아이템 Matrix로 변환합니다.
    # 지정된 파일을 찾을 수 없는 경우 더미 데이터를 생성하려고 시도합니다.
    abs_filepath = filepath
    if not os.path.isabs(filepath) and filepath.startswith('data/'):
        abs_filepath = os.path.join(project_root, filepath)

    if not os.path.exists(abs_filepath):
        print(f"Error: Data file not found at {abs_filepath}.")
        if abs_filepath.endswith('data/dummy_interactions.csv'):
            print("Attempting to generate dummy data...")
            try:
                from data.generate_dummy_data import generate_dummy_data
                generate_dummy_data()
                print("Dummy data generated successfully.")
                if not os.path.exists(abs_filepath):
                    print(f"Error: Dummy data generation ran, but file still not found at {abs_filepath}.")
                    return None
            except ImportError as e_import:
                print(f"ImportError: Failed to import 'generate_dummy_data'. Project root: {project_root}, sys.path: {sys.path}. Error: {e_import}")
                return None
            except Exception as e_general:
                print(f"Error generating dummy data: {e_general}")
                return None
        else:
            return None

    df = pd.read_csv(abs_filepath)
    # Pivot table을 만들어 사용자-아이템 Matrix 생성 (사용자는 행, 아이템은 열, 평점은 값)
    # 누락된 상호작용은 0으로 채움 (중립 또는 상호작용 없음)
    user_item_matrix = df.pivot_table(index='user_id', columns='item_id', values='rating').fillna(0)
    return user_item_matrix

# 2. 아이템 유사성 계산
# Time Complexity: O(I^2 * U), 여기서 I는 아이템 수, U는 사용자 수입니다.
# user_item_matrix.T는 Matrix를 (I x U)로 전치합니다.
# 그 다음 cosine_similarity는 I개의 모든 아이템 쌍 간의 유사성을 계산하며,
# 각 아이템 벡터의 길이는 U입니다.
def calculate_item_similarity(user_item_matrix):
    """
    Calculates cosine similarity between items based on user ratings.
    """
    # 한국어: 사용자 평점을 기반으로 아이템 간의 코사인 유사성을 계산합니다.
    # Matrix를 전치하여 아이템이 행이고 사용자가 열이 되도록 합니다 (I x U)
    # 이렇게 하면 cosine_similarity가 아이템 벡터 간의 유사성을 계산할 수 있습니다.
    item_user_matrix = user_item_matrix.T
    item_similarity_matrix = cosine_similarity(item_user_matrix)
    # 사용하기 쉽도록 item_id를 인덱스와 열로 하는 DataFrame으로 변환
    item_similarity_df = pd.DataFrame(item_similarity_matrix,
                                      index=user_item_matrix.columns, # 원본 item_ids
                                      columns=user_item_matrix.columns) # 원본 item_ids
    return item_similarity_df

# 3. 아이템 기반 협업 필터링 추천
# 단일 사용자에 대한 Time Complexity:
# - 사용자 평가 아이템 가져오기: O(I)
# - 평가되지 않은 아이템 반복 (I_unrated, 최악의 경우 I):
#   - 평가되지 않은 각 아이템에 대해 사용자의 평가된 아이템 반복 (I_rated):
#     - 유사성 조회: O(1) (DataFrame 조회)
#     - 계산 수행.
# - 추천 정렬: O(M log M) 여기서 M은 후보 아이템 수 (I_unrated).
# Overall: 대략 O(I_unrated * I_rated + M log M).
# I_rated가 작으면 모든 I 아이템을 유사성에 대해 반복하는 것보다 효율적입니다.
def get_item_based_recommendations(user_id, user_item_matrix, item_similarity_df, num_recommendations=5):
    """
    Generates item recommendations for a target user using Item-Based Collaborative Filtering.
    """
    # 한국어: 아이템 기반 협업 필터링을 사용하여 대상 사용자에 대한 아이템 추천을 생성합니다.
    if user_id not in user_item_matrix.index:
        print(f"Error: User ID {user_id} not found in the data.")
        return []

    # 대상 사용자가 제공한 평점 가져오기
    user_ratings_series = user_item_matrix.loc[user_id]
    # 사용자가 긍정적으로 평가한 아이템 필터링 (평점 > 0)
    rated_items_with_scores = user_ratings_series[user_ratings_series > 0]
    rated_item_ids = rated_items_with_scores.index

    # 사용자가 아직 평가하지 않은 아이템 식별
    all_item_ids = user_item_matrix.columns
    unrated_item_ids = all_item_ids.difference(rated_item_ids)

    if not rated_item_ids.tolist(): # 사용자가 평가한 아이템이 없는 경우 처리
        print(f"User {user_id} has no rated items. Cannot generate item-based recommendations.")
        return []

    if not unrated_item_ids.tolist(): # 사용자가 모든 아이템을 평가한 경우 처리
        print(f"User {user_id} has rated all available items. No new recommendations to generate.")
        return []

    # 평가되지 않은 아이템의 예상 점수를 저장할 딕셔너리
    recommendation_scores = {}

    # 사용자가 평가하지 않은 각 아이템에 대해
    for item_to_predict_score_for in unrated_item_ids:
        weighted_sum_of_ratings = 0
        sum_of_similarity_scores = 0

        # 사용자가 이미 평가한 각 아이템에 대해
        for previously_rated_item_id in rated_item_ids:
            # 예측하려는 아이템과 사용자가 이미 평가한 아이템 간의 유사성 가져오기
            similarity = item_similarity_df.loc[item_to_predict_score_for, previously_rated_item_id]

            # 긍정적인 유사성만 고려
            if similarity > 0:
                # 사용자가 이미 평가한 아이템에 부여한 평점 가져오기
                rating_for_previously_rated_item = rated_items_with_scores.loc[previously_rated_item_id]

                # 가중 합에 추가: 유사성 * 유사한 아이템에 대한 사용자 평점
                weighted_sum_of_ratings += similarity * rating_for_previously_rated_item
                # 유사성 합계에 추가 (가중 평균의 분모용)
                sum_of_similarity_scores += similarity

        # 유사한 아이템이 있었으면 예상 점수 계산
        if sum_of_similarity_scores > 0:
            predicted_score = weighted_sum_of_ratings / sum_of_similarity_scores
            recommendation_scores[item_to_predict_score_for] = predicted_score

    # 예상 점수를 기준으로 평가되지 않은 아이템을 내림차순으로 정렬
    sorted_recommendations = sorted(recommendation_scores.items(), key=lambda x: x[1], reverse=True)

    return sorted_recommendations[:num_recommendations]

# 메인 실행 블록
if __name__ == "__main__":
    print("--- Item-Based Collaborative Filtering Example ---")

    data_file_path = 'data/dummy_interactions.csv' # 프로젝트 루트에 대한 상대 경로
    print(f"\nLoading and preprocessing data from: {data_file_path}...")
    user_item_matrix = load_and_preprocess_data(filepath=data_file_path)

    if user_item_matrix is not None and not user_item_matrix.empty:
        print("\nUser-Item Matrix (first 5 rows/users):")
        print(user_item_matrix.head())

        print("\nCalculating Item Similarity Matrix... (this may take a while for many items)")
        item_similarity_df = calculate_item_similarity(user_item_matrix)
        print("\nItem Similarity Matrix (first 5x5 items):")
        display_limit = min(5, item_similarity_df.shape[0])
        print(item_similarity_df.iloc[:display_limit, :display_limit])

        if not user_item_matrix.empty:
            target_user_id = user_item_matrix.index[0] # 예시: 첫 번째 사용자 사용

            # 대상 사용자가 의미 있는 추천을 위해 아이템을 평가했는지 확인
            if user_item_matrix.loc[target_user_id].sum() == 0:
                print(f"\nWarning: Target user ID {target_user_id} has no rated items in the matrix. "
                      "Recommendations might be less meaningful or empty.")
                # 더 나은 시연을 위해 평점이 있는 사용자를 찾으려고 시도
                for uid in user_item_matrix.index:
                    if user_item_matrix.loc[uid].sum() > 0:
                        target_user_id = uid
                        print(f"Switching to user ID {target_user_id} for a better demo as they have rated items.")
                        break

            print(f"\nGenerating recommendations for User ID: {target_user_id}...")
            recommendations = get_item_based_recommendations(
                target_user_id,
                user_item_matrix,
                item_similarity_df,
                num_recommendations=5
            )

            if recommendations:
                print(f"\nTop {len(recommendations)} recommendations for User {target_user_id}:")
                for item_id, score in recommendations:
                    print(f"- Item ID: {item_id}, Predicted Score: {score:.4f}")
            else:
                print(f"No recommendations could be generated for User {target_user_id}. "
                      "This could be because the user has rated all items, no rated items, "
                      "or no similar items were found for the unrated items.")
        else:
            print("\nUser-Item matrix is empty. Cannot generate recommendations.")
    else:
        print("\nData loading failed or resulted in an empty matrix. Cannot proceed with the example.")

    print("\n--- Item-Based Collaborative Filtering Example Finished ---")
