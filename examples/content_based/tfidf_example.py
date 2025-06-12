# examples/content_based/tfidf_example.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel # Faster than cosine_similarity for normalized vectors
import os
import sys
import numpy as np

# --- Content-Based Filtering with TF-IDF: Basic Explanation ---
# Content-Based Filtering recommends items to users based on the similarity of item attributes
# (content) to items the user has previously liked.
#
# How it works:
# 1. Item Representation: Each item is described by a set of features or attributes.
#    For text data (e.g., descriptions, genres, articles), TF-IDF is commonly used.
#    - TF (Term Frequency): Measures how frequently a term appears in a document (item description).
#    - IDF (Inverse Document Frequency): Measures how important a term is across all documents.
#      Rare terms get higher IDF scores.
#    - TF-IDF Score: The product of TF and IDF. It highlights terms that are important to a
#      specific document but not common across all documents.
#    Each item is then represented as a vector of TF-IDF scores.
# 2. User Profile: A user's profile is built based on the content of items they have positively
#    interacted with (e.g., rated highly, purchased). This can be as simple as the set of items
#    liked, or a more complex aggregation of the TF-IDF vectors of those items.
# 3. Similarity Calculation: The similarity between a new item and the user's profile (or items
#    the user has liked) is calculated. Cosine similarity is often used for TF-IDF vectors.
# 4. Recommendation: Items that are most similar to the user's profile (or liked items) and
#    haven't been interacted with yet are recommended.
#
# Pros:
# - User Independence: Recommendations for one user don't depend on other users' data.
# - Transparency: Recommendations can be easily explained by listing content features (e.g.,
#   "Recommended because it's also a 'sci-fi thriller' like items you've watched").
# - Handles New Items: New items can be recommended as soon as their content is available,
#   without needing interaction data (addresses item cold-start).
#
# Cons:
# - Limited Serendipity: Tends to recommend items similar to what the user already knows,
#   making it harder to discover items outside their current taste profile.
# - Feature Engineering: Effectiveness heavily depends on the quality and completeness of
#   the item features (content description). Requires domain knowledge.
# - Overspecialization: Users might get stuck in a "filter bubble."
# ---

# Dynamically add project root to sys.path for module imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root) # Insert at the beginning for priority

def load_data(interactions_filepath_rel='data/dummy_interactions.csv', metadata_filepath_rel='data/dummy_item_metadata.csv'):
    """
    Loads interaction and metadata.
    Ensures dummy data is generated if files are missing.
    Filepaths are relative to project root.
    """
    interactions_filepath_abs = os.path.join(project_root, interactions_filepath_rel)
    metadata_filepath_abs = os.path.join(project_root, metadata_filepath_rel)

    # Check and generate dummy data if necessary
    # This relies on generate_dummy_data creating both files if one is missing.
    # A more robust check might be needed if generate_dummy_data doesn't guarantee this.
    needs_generation = False
    for filepath_abs, filepath_rel in [(interactions_filepath_abs, interactions_filepath_rel),
                                       (metadata_filepath_abs, metadata_filepath_rel)]:
        if not os.path.exists(filepath_abs):
            print(f"오류: {filepath_abs} ({filepath_rel}) 파일을 찾을 수 없습니다.")
            if filepath_rel in ['data/dummy_interactions.csv', 'data/dummy_item_metadata.csv']:
                needs_generation = True
            else: # If it's not a standard dummy file path, can't auto-generate this specific file
                print(f"{filepath_rel}은 자동 생성 대상이 아닙니다.")
                return None, None

    if needs_generation:
        print("더미 데이터 생성 시도 중... ('data/generate_dummy_data.py' 실행)")
        try:
            from data.generate_dummy_data import generate_dummy_data
            generate_dummy_data() # This should create both dummy_interactions.csv and dummy_item_metadata.csv
            print("더미 데이터 생성 완료.")
            # Verify that files exist after generation attempt
            if not os.path.exists(interactions_filepath_abs):
                print(f"더미 데이터 생성 후에도 {interactions_filepath_abs}를 찾을 수 없습니다.")
                return None, None
            if not os.path.exists(metadata_filepath_abs):
                print(f"더미 데이터 생성 후에도 {metadata_filepath_abs}를 찾을 수 없습니다.")
                return None, None
        except ImportError as e_import:
            print(f"ImportError: 'generate_dummy_data' 임포트 실패. 오류: {e_import}")
            return None, None
        except Exception as e_general:
            print(f"더미 데이터 생성 실패: {e_general}")
            return None, None

    try:
        df_interactions = pd.read_csv(interactions_filepath_abs)
        df_items = pd.read_csv(metadata_filepath_abs)
        # Ensure item_id is of the same type for merging, if necessary
        if not df_items.empty and not df_interactions.empty:
            df_items['item_id'] = df_items['item_id'].astype(df_interactions['item_id'].dtype)
        return df_interactions, df_items
    except FileNotFoundError as e:
        print(f"데이터 로딩 오류: {e}")
        return None, None
    except Exception as e:
        print(f"데이터 로딩 중 예상치 못한 오류: {e}")
        return None, None


def get_content_based_recommendations(user_id, df_interactions, df_items, num_recommendations=5):
    """
    Generates content-based recommendations for a user using TF-IDF on item descriptions.
    """
    if df_items is None or df_interactions is None:
        print("데이터프레임이 유효하지 않아 추천을 생성할 수 없습니다.")
        return []

    if 'description' not in df_items.columns or 'item_id' not in df_items.columns:
        print("아이템 메타데이터에 'item_id' 및 'description' 열이 필요합니다.")
        return []

    # Ensure item descriptions are strings and handle missing values
    df_items['description'] = df_items['description'].fillna('').astype(str)

    # 1. TF-IDF Vectorization
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(df_items['description'])

    # 2. Compute Cosine Similarity Matrix using linear_kernel (efficient for TF-IDF)
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    # Create a mapping from item_id to DataFrame index for quick lookups
    # This is crucial because cosine_sim matrix uses DataFrame indices, not item_id directly
    item_id_to_idx = pd.Series(df_items.index, index=df_items['item_id'].astype(str))
    # Ensure all item_ids used for lookup are strings if that's what item_id_to_idx index uses
    # Or ensure item_id_to_idx index and lookup values are consistently typed (e.g. int)
    # Based on dummy data, item_id is int. So, let's keep it int.
    item_id_to_idx = pd.Series(df_items.index, index=df_items['item_id'])


    try:
        # Get items rated positively by the user
        user_rated_item_ids = df_interactions[df_interactions['user_id'] == user_id]['item_id']
        if user_rated_item_ids.empty:
            print(f"사용자 ID {user_id}는 평가한 아이템이 없습니다. 콘텐츠 기반 추천을 생성할 수 없습니다.")
            return []

        # Consider the last item rated by the user as the basis for similar item search
        last_rated_item_id = user_rated_item_ids.iloc[-1]

        if last_rated_item_id not in item_id_to_idx:
            print(f"사용자가 평가한 아이템 ID {last_rated_item_id}가 메타데이터의 인덱스에 없습니다. (item_id_to_idx 확인 필요)")
            # This can happen if an item_id in interactions is not in metadata's item_id list
            return []

        last_rated_item_idx = item_id_to_idx[last_rated_item_id]

        # Get pairwise similarity scores of all items with that item
        sim_scores = list(enumerate(cosine_sim[last_rated_item_idx]))

        # Sort the items based on the similarity scores
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        recommendations = []
        seen_item_ids = set(user_rated_item_ids.values) # Items user has already interacted with
        seen_item_ids.add(last_rated_item_id) # Also exclude the source item itself

        for item_idx, score in sim_scores:
            if len(recommendations) >= num_recommendations:
                break
            # Map DataFrame index back to item_id
            recommended_item_id = df_items.iloc[item_idx]['item_id']

            if recommended_item_id not in seen_item_ids:
                recommendations.append({'item_id': recommended_item_id, 'similarity_score': score})
                seen_item_ids.add(recommended_item_id) # Add to seen to avoid duplicates in this list

        return recommendations

    except KeyError as e:
        print(f"KeyError: 사용자 ID {user_id} 또는 아이템 ID 관련 조회 오류. 오류: {e}")
        return []
    except Exception as e:
        print(f"추천 생성 중 오류 발생: {e}")
        return []

# 메인 실행 함수
if __name__ == "__main__":
    print("--- 콘텐츠 기반 필터링 (TF-IDF) 예제 시작 ---")

    # 1. 데이터 로드
    print("\n데이터 로드 중...")
    # Using relative paths from project root for load_data function
    df_interactions, df_items = load_data(
        interactions_filepath_rel='data/dummy_interactions.csv',
        metadata_filepath_rel='data/dummy_item_metadata.csv'
    )

    if df_interactions is not None and df_items is not None:
        if not df_interactions.empty and not df_items.empty:
            # 예시 사용자 ID (dummy_interactions.csv에 있는 ID여야 함)
            if df_interactions['user_id'].nunique() > 0:
                target_user_id = df_interactions['user_id'].unique()[0]
                print(f"\n사용자 ID {target_user_id}에 대한 콘텐츠 기반 추천 아이템 생성 중...")

                recommendations = get_content_based_recommendations(target_user_id, df_interactions, df_items, num_recommendations=5)

                if recommendations:
                    print(f"\n사용자 {target_user_id}를 위한 추천 아이템 목록 (TF-IDF 기반):")
                    recs_df = pd.DataFrame(recommendations)
                    # Merge with df_items to get description or other details for printing
                    recs_df = pd.merge(recs_df, df_items[['item_id', 'description', 'genres']], on='item_id', how='left')
                    for index, row in recs_df.iterrows():
                        print(f"- 아이템 {row['item_id']} (유사도: {row['similarity_score']:.2f}) Genres: {row['genres']} | Desc: {row['description'][:60]}...")
                else:
                    print(f"사용자 {target_user_id}에게 추천할 아이템이 없습니다.")
            else:
                print("데이터에 유효한 사용자가 없습니다.")
        else:
            print("데이터 파일 중 하나 또는 둘 다 비어있습니다.")
    else:
        print("\n데이터 로드에 실패하여 예제를 실행할 수 없습니다.")

    print("\n--- 콘텐츠 기반 필터링 (TF-IDF) 예제 실행 완료 ---")
