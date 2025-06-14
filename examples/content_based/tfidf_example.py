# examples/content_based/tfidf_example.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel # 정규화된 벡터에 대해 cosine_similarity보다 빠름
import os
import sys
import numpy as np

# --- TF-IDF를 사용한 콘텐츠 기반 필터링: 기본 설명 ---
# 콘텐츠 기반 필터링은 아이템 속성(콘텐츠)과 사용자가 이전에 좋아했거나 상호작용한 아이템과의 유사성을 기반으로 사용자에게 아이템을 추천합니다.
# 다른 사용자의 의견보다는 아이템의 속성에 중점을 둡니다.
#
# 작동 방식:
# 1. 아이템 표현 (콘텐츠 추출): 각 아이템은 특징 집합으로 설명되어야 합니다.
#    텍스트 콘텐츠가 있는 아이템(예: 영화 설명, 기사 텍스트, 제품 세부 정보)의 경우,
#    TF-IDF (Term Frequency-Inverse Document Frequency)는 텍스트를 숫자 벡터로 변환하는 일반적인 기술입니다.
#    - TF (Term Frequency): 특정 아이템의 설명에 용어가 얼마나 자주 나타나는지.
#      TF(t,d) = (문서 d에 용어 t가 나타난 횟수) / (문서 d의 총 용어 수)
#    - IDF (Inverse Document Frequency): 전체 아이템 설명 모음에서 용어의 중요성.
#      많은 아이템에 공통적인 용어(예: "the", "a")는 낮은 IDF 점수를 받고, 드문 용어는 높은 점수를 받습니다.
#      IDF(t,D) = log(총 문서 수 D / 용어 t를 포함하는 문서 수)
#    - TF-IDF 점수: TF와 IDF의 곱 (TF * IDF). 이 점수는 특정 아이템의 설명에서는 빈번하지만
#      모든 아이템 설명에서는 상대적으로 드문 용어에 대해 높게 나타나므로 좋은 판별자가 됩니다.
#    따라서 각 아이템은 각 차원이 용어의 TF-IDF 점수에 해당하는 벡터로 표현됩니다.
#
# 2. 사용자 프로필 생성 (선택 사항이지만 복잡한 시스템에서는 일반적):
#    사용자가 긍정적으로 상호작용한 아이템의 콘텐츠를 기반으로 사용자 프로필을 구축할 수 있습니다.
#    이는 사용자가 좋아한 아이템의 TF-IDF 벡터의 집계된 벡터(예: 평균 또는 가중 합)일 수 있습니다.
#    더 간단한 경우(이 예제와 같이), 사용자가 좋아한 특정 아이템의 콘텐츠와 새 아이템을 직접 비교하여 추천합니다.
#
# 3. 유사성 계산: 아이템 벡터 간의 유사성을 계산합니다.
#    코사인 유사도(Cosine similarity)는 TF-IDF 벡터에 일반적으로 사용되는데, 두 벡터 사이의 각도의 코사인을 측정하여
#    크기보다는 방향을 효과적으로 포착하기 때문입니다.
#    scikit-learn의 `linear_kernel`은 특히 TF-IDF 벡터가 L2 정규화된 경우(TfidfVectorizer가 기본적으로 수행하거나 명시적으로 수행 가능)
#    이를 효율적으로 계산할 수 있습니다.
#
# 4. 추천 생성:
#    사용자에게 아이템을 추천하려면:
#    a. 사용자가 좋아한 아이템을 식별합니다(예: 높은 평점을 받은 아이템).
#    b. 이러한 각 선호 아이템(또는 사용자의 집계된 프로필)에 대해 데이터셋에서 TF-IDF 벡터 유사성을 기반으로
#       가장 유사한 다른 아이템을 찾습니다.
#    c. 이러한 유사한 아이템을 유사성 점수로 순위를 매깁니다.
#    d. 사용자가 이미 상호작용한 아이템을 필터링하고 상위 N개의 아이템을 제시합니다.
#
# Pros (장점):
# - 사용자 독립성: 한 사용자에 대한 추천은 다른 사용자의 데이터에 의존하지 않고 자신의 상호작용에만 의존합니다.
#   즉, 새 사용자가 최소한 하나의 아이템과 상호작용하는 한 협업 필터링만큼 "신규 사용자 cold-start" 문제로 어려움을 겪지 않습니다.
# - 투명성 및 설명 가능성: 유사성을 유발한 콘텐츠 특징을 나열하여 추천을 쉽게 설명할 수 있습니다
#   (예: "이전에 즐겨찾던 아이템과 'SF 스릴러'와 같은 장르/키워드를 공유하므로 추천합니다").
# - 신규 아이템 처리 (아이템 Cold-Start): 새 아이템은 해당 새 아이템에 대한 사용자 상호작용 데이터 없이도
#   콘텐츠 특징을 사용할 수 있고 벡터화되는 즉시 추천될 수 있습니다.
#
# Cons (단점):
# - 제한된 Serendipity (필터 버블): 사용자가 이미 알고 있는 것과 내용이 매우 유사한 아이템을 추천하는 경향이 있어
#   현재 취향 프로필을 벗어난 아이템을 발견하기 어렵게 만듭니다.
#   사용자는 "과잉 전문화" 또는 "필터 버블"에 갇힐 수 있습니다.
# - 광범위한 특징 공학: 효과는 아이템 특징(콘텐츠 설명, 태그, 장르 등)의 품질, 완전성 및 표현에 크게 좌우됩니다.
#   도메인 지식과 신중한 전처리가 필요합니다.
# - 사용자 Cold-Start (어느 정도): 한 번의 상호작용 후 신규 사용자에게 추천할 수 있지만,
#   더 많은 상호작용이 사용자의 콘텐츠 선호도를 정의함에 따라 추천 품질이 향상됩니다.
# ---

# 모듈 임포트를 위해 프로젝트 루트를 sys.path에 동적으로 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root) # 우선순위를 위해 처음에 삽입

# --- 데이터 로딩 ---
# Time Complexity: CSV 읽기의 경우 O(N_interactions + N_items), 여기서 N_interactions는 상호작용 데이터의 행 수
# N_items는 메타데이터의 행 수입니다.
def load_data(interactions_filepath_rel='data/dummy_interactions.csv', metadata_filepath_rel='data/dummy_item_metadata.csv'):
    """
    한국어: 프로젝트 루트에서 지정된 상대 경로로부터 상호작용 및 아이템 메타데이터를 로드합니다.
    기본 파일이 누락된 경우 더미 데이터가 생성되도록 합니다.

    Loads interaction and item metadata from specified relative paths from project root.
    Ensures dummy data is generated if the default files are missing.

    Args:
        interactions_filepath_rel (str): Relative path to interactions CSV.
        metadata_filepath_rel (str): Relative path to item metadata CSV.

    Returns:
        tuple: (pandas.DataFrame, pandas.DataFrame) for interactions and items, or (None, None) on failure.
    """
    interactions_filepath_abs = os.path.join(project_root, interactions_filepath_rel)
    metadata_filepath_abs = os.path.join(project_root, metadata_filepath_rel)

    needs_generation = False
    # 생성 시도 전 두 파일의 존재 여부 확인
    if not os.path.exists(interactions_filepath_abs) and interactions_filepath_rel == 'data/dummy_interactions.csv':
        print(f"Warning: Interaction data file not found at {interactions_filepath_abs}.") # 경고: 상호작용 데이터 파일을 {interactions_filepath_abs}에서 찾을 수 없습니다.
        needs_generation = True
    if not os.path.exists(metadata_filepath_abs) and metadata_filepath_rel == 'data/dummy_item_metadata.csv':
        print(f"Warning: Item metadata file not found at {metadata_filepath_abs}.") # 경고: 아이템 메타데이터 파일을 {metadata_filepath_abs}에서 찾을 수 없습니다.
        needs_generation = True

    if needs_generation:
        print("Attempting to generate dummy data by running 'data/generate_dummy_data.py'...") # 'data/generate_dummy_data.py'를 실행하여 더미 데이터를 생성하려고 시도 중...
        try:
            from data.generate_dummy_data import generate_dummy_data
            generate_dummy_data() # 이것은 두 개의 더미 파일을 생성해야 합니다
            print("Dummy data generation script executed.") # 더미 데이터 생성 스크립트가 실행되었습니다.
            # 생성 시도 후 파일이 존재하는지 확인
            if not os.path.exists(interactions_filepath_abs):
                print(f"Error: Dummy interaction data still not found at {interactions_filepath_abs} after generation.") # 오류: 생성 후에도 {interactions_filepath_abs}에서 더미 상호작용 데이터를 찾을 수 없습니다.
                return None, None
            if not os.path.exists(metadata_filepath_abs):
                print(f"Error: Dummy item metadata still not found at {metadata_filepath_abs} after generation.") # 오류: 생성 후에도 {metadata_filepath_abs}에서 더미 아이템 메타데이터를 찾을 수 없습니다.
                return None, None
            print("Dummy data files should now be available.") # 이제 더미 데이터 파일을 사용할 수 있습니다.
        except ImportError as e_import:
            print(f"ImportError: Failed to import 'generate_dummy_data'. Error: {e_import}") # ImportError: 'generate_dummy_data'를 임포트하지 못했습니다. 오류: {e_import}
            return None, None
        except Exception as e_general:
            print(f"Error during dummy data generation: {e_general}") # 더미 데이터 생성 중 오류: {e_general}
            return None, None

    try:
        df_interactions = pd.read_csv(interactions_filepath_abs)
        df_items = pd.read_csv(metadata_filepath_abs)
        # 잠재적인 병합 또는 조회를 위해 item_id가 일관된 유형인지 확인합니다.
        # 더미 데이터에서 item_id는 정수형입니다. 문자열 ID와 혼합하는 경우 문자열로 변환하거나 일관성을 확인하십시오.
        if not df_items.empty and 'item_id' in df_items.columns:
             df_items['item_id'] = df_items['item_id'].astype(str) # 이 예제에서는 문자열로 표준화
        if not df_interactions.empty and 'item_id' in df_interactions.columns:
            df_interactions['item_id'] = df_interactions['item_id'].astype(str)

        return df_interactions, df_items
    except FileNotFoundError as e:
        print(f"Error loading data files: {e}. Please ensure paths are correct or dummy data exists.") # 데이터 파일 로드 오류: {e}. 경로가 올바른지 또는 더미 데이터가 있는지 확인하십시오.
        return None, None
    except Exception as e:
        print(f"An unexpected error occurred during data loading: {e}") # 데이터 로딩 중 예기치 않은 오류 발생: {e}
        return None, None

# --- 추천 로직 ---
# Time Complexity:
# - TF-IDF Vectorization: O(N_items * avg_doc_length + V_size * avg_doc_length) 여기서 N_items는 아이템 수,
#   avg_doc_length는 아이템 설명당 평균 용어 수, V_size는 어휘 크기입니다. 상당할 수 있습니다.
# - linear_kernel (유사도 Matrix): O(N_items^2 * N_features) 여기서 N_features는 TF-IDF 어휘 크기입니다.
#   N_items가 크면 이것이 가장 계산 집약적인 부분입니다.
# - 사용자별 추천 (Matrix 계산 후):
#   - 사용자 평가 아이템 식별: O(N_user_interactions).
#   - 한 아이템에 대한 유사성 점수 가져오기: O(N_items).
#   - 이러한 점수 정렬: O(N_items * log(N_items)).
#   - 상위 N개 필터링 및 선택: O(N_items).
# Overall, 전체 유사도 Matrix를 미리 계산하는 것은 비용이 많이 듭니다.
def get_content_based_recommendations(user_id, df_interactions, df_items, num_recommendations=5):
    """
    한국어: 아이템 설명을 사용하여 TF-IDF를 통해 사용자에 대한 콘텐츠 기반 추천을 생성합니다.
    이 예제는 사용자가 마지막으로 상호작용한 아이템을 기반으로 추천합니다.

    Generates content-based recommendations for a user using TF-IDF on item descriptions.
    This example bases recommendations on the last item the user interacted with.
    """
    if df_items is None or df_interactions is None or df_items.empty or df_interactions.empty:
        print("Error: DataFrames are invalid or empty. Cannot generate recommendations.") # 오류: DataFrame이 유효하지 않거나 비어 있습니다. 추천을 생성할 수 없습니다.
        return []

    if 'description' not in df_items.columns or 'item_id' not in df_items.columns:
        print("Error: Item metadata must contain 'item_id' and 'description' columns.") # 오류: 아이템 메타데이터에는 'item_id'와 'description' 열이 포함되어야 합니다.
        return []

    # 일관된 조회를 위해 user_id를 문자열로 표준화
    user_id = str(user_id)
    df_interactions['user_id'] = df_interactions['user_id'].astype(str)


    # 아이템 설명이 문자열인지 확인하고 누락된 값 처리
    df_items['description'] = df_items['description'].fillna('').astype(str)

    # 1. TF-IDF Vectorization
    # TfidfVectorizer는 원시 문서 모음을 TF-IDF 특징 Matrix로 변환합니다.
    # - stop_words='english': 일반적인 영어 불용어를 제거합니다.
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    # fit_transform은 'description'에서 어휘와 IDF를 학습한 다음 이를 문서-용어 Matrix로 변환합니다.
    tfidf_matrix = tfidf_vectorizer.fit_transform(df_items['description'])
    # tfidf_matrix는 (아이템_수, 특징_수/어휘_크기) 모양의 희소 Matrix입니다.

    # 2. 코사인 유사도 Matrix 계산
    # linear_kernel은 내적을 계산하며, 이는 L2 정규화된 벡터에 대한 코사인 유사도와 동일합니다.
    # TfidfVectorizer는 일반적으로 L2 정규화된 벡터를 생성합니다 (기본적으로 norm='l2').
    # 이것은 각 항목 (i, j)가 아이템 i와 아이템 j 간의 유사성인 (아이템_수 x 아이템_수) Matrix를 생성합니다.
    cosine_sim_matrix = linear_kernel(tfidf_matrix, tfidf_matrix)

    # cosine_sim_matrix에서 빠른 조회를 위해 item_id(문자열)에서 DataFrame 인덱스로의 매핑 생성
    item_id_to_idx = pd.Series(df_items.index, index=df_items['item_id'])

    try:
        # 대상 사용자가 평가/상호작용한 아이템 가져오기
        user_interacted_items_series = df_interactions[df_interactions['user_id'] == user_id]['item_id']

        if user_interacted_items_series.empty:
            print(f"User ID {user_id} has no interaction data. Cannot generate content-based recommendations on this basis.") # 사용자 ID {user_id}에 대한 상호작용 데이터가 없습니다. 이 기준으로 콘텐츠 기반 추천을 생성할 수 없습니다.
            return []

        # 단순화를 위해 이 예제에서는 사용자가 마지막으로 상호작용한 아이템을 추천 기준으로 사용합니다.
        # 더 강력한 접근 방식은 좋아한 모든 아이템 또는 가장 높은 평가를 받은 아이템의 집계를 사용할 수 있습니다.
        last_interacted_item_id = user_interacted_items_series.iloc[-1] # 이것은 이전의 astype(str)으로 인해 문자열입니다.

        if last_interacted_item_id not in item_id_to_idx:
            print(f"Error: The item ID {last_interacted_item_id} (last interacted by user {user_id}) " # 오류: 아이템 ID {last_interacted_item_id} (사용자 {user_id}가 마지막으로 상호작용한)
                  "is not found in the item metadata's index map. It might be missing from df_items.") # 이 아이템 메타데이터의 인덱스 맵에 없습니다. df_items에 누락되었을 수 있습니다.
            return []

        # 유사도 Matrix에서 이 마지막으로 상호작용한 아이템의 인덱스 가져오기
        last_interacted_item_idx = item_id_to_idx[last_interacted_item_id]

        # 이 마지막으로 상호작용한 아이템과 모든 아이템 간의 쌍별 유사성 점수 가져오기
        # cosine_sim_matrix[last_interacted_item_idx]는 유사성의 행을 제공합니다.
        item_similarity_scores = list(enumerate(cosine_sim_matrix[last_interacted_item_idx]))

        # 유사성 점수를 기준으로 아이템을 내림차순으로 정렬
        # x[0]은 아이템의 인덱스이고, x[1]은 유사성 점수입니다.
        sorted_similar_items = sorted(item_similarity_scores, key=lambda x: x[1], reverse=True)

        recommendations = []
        # 중복을 피하기 위해 사용자가 이미 추천했거나 상호작용한 아이템 추적
        # 효율적인 조회를 위해 세트로 변환
        seen_item_ids = set(user_interacted_items_series.values)
        # 아이템 자체(last_interacted_item_id)는 1.0의 유사성을 가지며 seen_item_ids에 이미 포함되어 있지 않다면 제외되어야 합니다.
        # 그러나 루프 구조는 `if recommended_item_id not in seen_item_ids`를 확인하여 자연스럽게 이를 처리합니다.

        print(f"\nGenerating recommendations based on similarity to item ID: {last_interacted_item_id} (Description: '{df_items.loc[item_id_to_idx[last_interacted_item_id], 'description'][:100]}...')") # 아이템 ID: {last_interacted_item_id}와의 유사성을 기반으로 추천 생성 중 (설명: '{df_items.loc[item_id_to_idx[last_interacted_item_id], 'description'][:100]}...')

        for item_matrix_idx, score in sorted_similar_items:
            if len(recommendations) >= num_recommendations:
                break # 충분한 추천을 찾으면 중지

            # DataFrame 인덱스(enumerate에서)를 실제 item_id로 다시 매핑
            recommended_item_id = df_items.iloc[item_matrix_idx]['item_id'] # 이것은 문자열입니다

            # 사용자가 아직 보거나 상호작용하지 않은 경우 추천에 추가
            if recommended_item_id not in seen_item_ids:
                recommendations.append({'item_id': recommended_item_id, 'similarity_score': score})
                seen_item_ids.add(recommended_item_id) # 다시 추천하는 것을 방지하기 위해 본 세트에 추가

        return recommendations

    except KeyError as e:
        print(f"KeyError during recommendation generation for user {user_id}. Possibly an ID mismatch. Error: {e}") # 사용자 {user_id}에 대한 추천 생성 중 KeyError. ID 불일치 가능성. 오류: {e}
        return []
    except Exception as e:
        print(f"An unexpected error occurred during recommendation generation: {e}") # 추천 생성 중 예기치 않은 오류 발생: {e}
        return []

# 메인 실행 블록
if __name__ == "__main__":
    print("--- Content-Based Filtering (TF-IDF) Example ---") # --- 콘텐츠 기반 필터링 (TF-IDF) 예제 ---

    print("\nLoading data...") # 데이터 로드 중...
    df_interactions, df_items = load_data(
        interactions_filepath_rel='data/dummy_interactions.csv',
        metadata_filepath_rel='data/dummy_item_metadata.csv'
    )

    if df_interactions is not None and df_items is not None:
        if not df_interactions.empty and not df_items.empty:
            if df_interactions['user_id'].nunique() > 0:
                # 예시: 상호작용 데이터에서 첫 번째 고유 user_id 사용
                target_user_id = df_interactions['user_id'].unique()[0]
                # get_content_based_recommendations 내부에서 사용되므로 target_user_id가 문자열인지 확인
                target_user_id = str(target_user_id)

                print(f"\nAttempting to generate content-based recommendations for User ID: {target_user_id}...") # 사용자 ID: {target_user_id}에 대한 콘텐츠 기반 추천 생성 시도 중...
                recommendations = get_content_based_recommendations(
                    target_user_id,
                    df_interactions,
                    df_items,
                    num_recommendations=5
                )

                if recommendations:
                    print(f"\nTop {len(recommendations)} recommendations for User {target_user_id} (based on TF-IDF item similarity):") # 사용자 {target_user_id}를 위한 상위 {len(recommendations)}개 추천 (TF-IDF 아이템 유사성 기반):
                    recs_df = pd.DataFrame(recommendations)
                    # 출력을 위해 df_items와 병합하여 자세한 내용(설명, 장르) 가져오기
                    # recs_df 및 df_items의 item_id가 동일한 유형(이 예제에서는 문자열)인지 확인
                    recs_df['item_id'] = recs_df['item_id'].astype(str)
                    df_items_for_merge = df_items[['item_id', 'description', 'genres']].copy()
                    df_items_for_merge['item_id'] = df_items_for_merge['item_id'].astype(str)

                    recs_df = pd.merge(recs_df, df_items_for_merge, on='item_id', how='left')

                    for index, row in recs_df.iterrows():
                        print(f"- Item ID: {row['item_id']} (Similarity: {row['similarity_score']:.4f}) "
                              f"| Genres: {row.get('genres', 'N/A')} "
                              f"| Description snippet: {row.get('description', 'N/A')[:70]}...")
                else:
                    print(f"No recommendations could be generated for User {target_user_id}.") # 사용자 {target_user_id}에 대한 추천을 생성할 수 없습니다.
            else:
                print("Error: No unique users found in the interaction data.") # 오류: 상호작용 데이터에서 고유한 사용자를 찾을 수 없습니다.
        else:
            print("Error: One or both data files (interactions, items) are empty.") # 오류: 하나 또는 두 개의 데이터 파일(상호작용, 아이템)이 비어 있습니다.
    else:
        print("\nData loading failed. Cannot proceed with the example.") # 데이터 로드에 실패했습니다. 예제를 진행할 수 없습니다.

    print("\n--- Content-Based Filtering (TF-IDF) Example Finished ---") # --- 콘텐츠 기반 필터링 (TF-IDF) 예제 완료 ---
