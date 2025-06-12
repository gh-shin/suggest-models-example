# AI 추천 모델 샘플 저장소
### 프로젝트 개요
본 저장소는 다양한 인공지능(AI) 기반 추천 모델의 예시 코드와 활용 사례를 제공합니다. 본 프로젝트의 궁극적인 목표는 사용자가 추천 시스템의 기본 원리를 심층적으로 이해하고, 실제 프로젝트에 효율적으로 적용할 수 있도록 지원하는 것입니다. 특히, 파이썬 데이터 엔지니어는 모델 학습, 예측, 평가 등 일련의 과정을 명확하고 실용적인 예제를 통해 직관적으로 경험할 수 있도록 설계되었습니다.

### 파이썬 데이터 엔지니어를 위한 지침
본 프로젝트의 핵심 목표는 다음과 같습니다:
1. 추천 모델 예제 구현: 현재 저장소에 포함된 (또는 향후 추가될) 추천 모델들을 활용하여 사용자가 쉽게 따라 할 수 있는 파이썬 예제 코드를 개발해야 합니다.
2. 사용자 접근성 확보: 비전문가도 코드를 실행하고 그 결과를 명확하게 확인할 수 있도록 간결하고 명료한 인터페이스를 제공해야 합니다.
3. 상세한 주석 및 설명: 코드 내부에 충분한 주석을 포함하고, README.md 파일에 각 예제의 목적, 사용 방법, 예상 결과에 대한 구체적인 설명을 추가해야 합니다.
4. 재현 가능한 환경 구축: 예제 실행에 필요한 환경 설정(예: 라이브러리 종속성)에 대한 명확한 지침을 제공하여 재현성을 보장해야 합니다.

### 세부 작업 지침
#### 1. 개발 환경 설정
예제 코드 실행을 위한 파이썬 환경 설정 방법을 안내합니다.
- 파이썬 버전: 파이썬 3.9 이상 버전 사용을 권장합니다.
- 가상 환경 생성: `venv` 또는 `conda`를 활용하여 독립적인 가상 환경을 구축합니다.
```
python -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate   # Windows
```
- 필요 라이브러리 설치: `requirements.txt` 파일에 필요한 라이브러리 목록을 명시하고, 해당 파일을 통해 일괄 설치를 진행합니다.
```
pip install -r requirements.txt
```
(예시: pandas, numpy, scikit-learn, scipy, lightfm, implicit, tensorflow 또는 pytorch 등)

#### 2. 데이터 준비
예제 실행에 필요한 데이터셋을 준비합니다. 실제 데이터 활용이 어려운 경우, 모델의 기능 시연을 위한 더미 데이터를 생성할 수 있습니다.
- 데이터 형식: CSV, Parquet 또는 Pandas DataFrame으로 손쉽게 로드 가능한 형식을 선호합니다.
- 더미 데이터 생성 스크립트: 필요시 `data/generate_dummy_data.py`와 같은 스크립트를 제공하여 사용자가 더미 데이터를 직접 생성할 수 있도록 지원합니다. 데이터는 사용자-아이템 상호작용(예: 사용자 ID, 아이템 ID, 평점/시청 시간) 및 아이템 메타데이터 등을 포함해야 합니다.
```
# 예시 더미 데이터 생성 (data/generate_dummy_data.py)
import pandas as pd
import numpy as np

def generate_dummy_data(num_users=100, num_items=50, num_interactions=1000):
    user_ids = np.random.randint(1, num_users + 1, num_interactions)
    item_ids = np.random.randint(1, num_items + 1, num_interactions)
    ratings = np.random.randint(1, 6, num_interactions) # 1-5 평점을 의미합니다.

    df = pd.DataFrame({
        'user_id': user_ids,
        'item_id': item_ids,
        'rating': ratings
    })
    df.to_csv('data/dummy_interactions.csv', index=False)
    print(f"더미 상호작용 데이터 생성이 완료되었습니다: data/dummy_interactions.csv ({len(df)} 행)")

if __name__ == "__main__":
    generate_dummy_data()
```

#### 3. 추천 모델 통합 및 예제 개발
저장소에 포함되거나 향후 추가될 추천 모델들을 활용하여 예제 코드를 작성합니다. 각 모델별로 별도의 예제 파일을 생성하며, 다음 기능을 반드시 포함해야 합니다.
- 모델 선택: 현재 저장소 내의 모델(예: 협업 필터링, 콘텐츠 기반 필터링, 행렬 분해, 딥러닝 기반 모델 등) 중 하나를 선택하여 예제를 구성합니다. 사용자 지정 임베딩 모델 개발에 대한 관심이 높으므로, 임베딩을 활용하는 모델(예: Factorization Machines, Word2Vec/Doc2Vec 기반 아이템 임베딩) 예제를 우선적으로 고려할 수 있습니다.
- 데이터 로드 및 전처리: 데이터셋을 불러오고 모델 학습에 적합한 형태로 전처리하는 과정을 포함합니다.
- 모델 학습: 모델을 학습시키는 코드를 구현합니다.예측: 특정 사용자에게 아이템을 추천하거나, 특정 아이템에 대한 예측 평점을 생성하는 코드를 작성합니다.
- 평가 (선택 사항이나 강력히 권장): 모델의 성능을 평가하는 기본적인 지표(예: Precision@K, Recall@K, RMSE 등)를 포함합니다. 성능 평가 코드는 Big O 복잡도를 고려하여 효율적으로 구현되어야 합니다.
- 주석: 모든 코드 라인에 상세한 주석을 달아 각 작업의 목적을 명확히 설명해야 합니다.
- 클린 코드: 가독성이 높고 유지보수가 용이한 코드를 작성하는 것을 원칙으로 합니다.

#### 예제 코드 구조 (예시: `examples/collaborative_filtering/item_cf_example.py`)
```
# examples/collaborative_filtering/item_cf_example.py
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# 1. 데이터 로드 및 전처리 (Big O: 로딩 O(N), 밀집 행렬 변환 O(U*I), 여기서 U=사용자 수, I=아이템 수)
# 예시: 사용자-아이템 상호작용 데이터 (user_id, item_id, rating)
def load_and_preprocess_data(filepath='data/dummy_interactions.csv'):
    """
    더미 상호작용 데이터를 로드하고 사용자-아이템 행렬로 변환합니다.
    """
    df = pd.read_csv(filepath)
    # Pivot 테이블을 사용하여 사용자-아이템 행렬 생성
    # 상호작용이 없는 경우 0으로 채워 넣습니다.
    user_item_matrix = df.pivot_table(index='user_id', columns='item_id', values='rating').fillna(0)
    return user_item_matrix

# 2. 아이템 간 유사도 계산 (Big O: O(I^2 * M), 여기서 I=아이템 수, M=특징/사용자 수)
# 참고: 아이템 수가 많을수록 계산 성능 저하가 발생할 수 있습니다.
# 대규모 시스템에서는 희소 행렬 라이브러리(scipy.sparse) 또는 근사 최근접 이웃(ANN) 알고리즘을 고려해야 합니다.
def calculate_item_similarity(user_item_matrix):
    """
    아이템 간의 코사인 유사도를 계산합니다.
    """
    item_similarity_matrix = cosine_similarity(user_item_matrix.T) # .T는 전치(Transpose)하여 아이템별 유사도 계산
    item_similarity_df = pd.DataFrame(item_similarity_matrix,
                                      index=user_item_matrix.columns,
                                      columns=user_item_matrix.columns)
    return item_similarity_df

# 3. 아이템 기반 협업 필터링 추천 (Big O: O(U * I_k * I_s), 여기서 U=사용자 수, I_k=예측할 아이템 수, I_s=유사 아이템 수)
def get_item_based_recommendations(user_id, user_item_matrix, item_similarity_df, num_recommendations=5):
    """
    특정 사용자에게 아이템 기반 협업 필터링을 사용하여 아이템을 추천합니다.
    """
    # 사용자가 평가한 아이템 목록을 가져옵니다.
    user_ratings = user_item_matrix.loc[user_id]
    rated_items = user_ratings[user_ratings > 0].index

    # 아직 평가하지 않은 아이템 목록을 식별합니다.
    unrated_items = user_item_matrix.columns.difference(rated_items)

    # 추천 점수를 초기화합니다.
    recommendation_scores = {}

    # 평가하지 않은 각 아이템에 대해 예상 평점을 계산합니다.
    # 이 과정에서 O(I_k * I_s) 계산이 발생하며, 대규모 데이터셋에서는 최적화가 필요합니다.
    for item_to_recommend in unrated_items:
        total_score = 0
        sum_of_similarities = 0

        # 해당 아이템과 사용자가 평가한 아이템들의 유사도 및 평점을 활용합니다.
        for rated_item in rated_items:
            similarity = item_similarity_df.loc[item_to_recommend, rated_item]
            rating = user_ratings.loc[rated_item]

            # 유사도가 0이 아닌 경우에만 계산에 포함합니다.
            if similarity > 0:
                total_score += similarity * rating
                sum_of_similarities += similarity
        
        # 유사도 합이 0이 아닌 경우에만 예상 평점을 계산합니다.
        if sum_of_similarities > 0:
            predicted_rating = total_score / sum_of_similarities
            recommendation_scores[item_to_recommend] = predicted_rating

    # 예측 평점이 높은 순으로 정렬하여 추천 목록을 반환합니다.
    recommended_items = sorted(recommendation_scores.items(), key=lambda x: x[1], reverse=True)
    return recommended_items[:num_recommendations]

# 메인 실행 함수
if __name__ == "__main__":
    # 1. 더미 데이터 생성 (파일이 없는 경우)
    try:
        pd.read_csv('data/dummy_interactions.csv')
    except FileNotFoundError:
        print("더미 데이터 파일이 없습니다. 생성 중...")
        # 본 스크립트 실행 전에 'data/generate_dummy_data.py'를 먼저 실행하여 데이터를 생성해야 합니다.
        # 편의상, 이 예제에서는 해당 파일이 미리 실행되었다고 가정합니다.
        print("오류: data/dummy_interactions.csv 파일을 찾을 수 없습니다. 'data/generate_dummy_data.py'를 먼저 실행해 주십시오.")
        exit() # 데이터가 없는 경우 프로그램 종료

    print("데이터 로드 및 전처리 중...")
    user_item_matrix = load_and_preprocess_data()
    print("사용자-아이템 행렬 (일부):\n", user_item_matrix.head())

    print("\n아이템 유사도 계산 중... (대규모 데이터셋에서는 상당한 시간이 소요될 수 있습니다.)")
    item_similarity_df = calculate_item_similarity(user_item_matrix)
    print("아이템 유사도 행렬 (일부):\n", item_similarity_df.iloc[:5, :5])

    target_user_id = user_item_matrix.index[0] # 첫 번째 사용자 ID를 선택합니다.
    print(f"\n사용자 {target_user_id} 에 대한 추천 아이템을 생성 중입니다...")
    recommendations = get_item_based_recommendations(target_user_id, user_item_matrix, item_similarity_df)

    print(f"\n사용자 {target_user_id} 를 위한 추천 아이템 목록:")
    if recommendations:
        for item, score in recommendations:
            print(f"- 아이템 {item}: 예상 평점 {score:.2f}")
    else:
        print("추천할 아이템이 없습니다. (모든 아이템을 이미 평가했거나 유사한 아이템이 존재하지 않습니다.)")

    print("\n--- 예제 실행 완료 ---")
```

#### 4. 문서화 및 `README.md` 업데이트
각 예제 파일에 대한 설명과 사용 지침을 `README.md` 파일에 추가해야 합니다.
- 목차: 저장소의 구조와 내용을 한눈에 파악할 수 있도록 목차를 포함합니다.
- 각 예제 설명:
  - 예제 파일명
  - 예제에서 활용하는 모델/알고리즘 (예: 아이템 기반 협업 필터링)
  - 예제의 목적실행 방법 (예: python examples/collaborative_filtering/item_cf_example.py)
  - 예상 출력 결과
  - 모델의 장단점 및 한계점 (간략하게)
  - **성능 고려 사항**: 코드의 Big O 성능에 대한 간략한 설명과 대규모 데이터셋 적용 시 고려해야 할 사항을 명시합니다.
  
#### 5. 테스트
간단한 단위 테스트 또는 통합 테스트를 작성하여 코드의 정확성을 확보합니다.
- `pytes`t와 같은 테스트 프레임워크를 활용하여 테스트 코드를 작성합니다.
- 예시: `tests/test_model.py`

#### 6. 기여 가이드라인 (Contribution Guidelines)
향후 다른 개발자들이 본 저장소에 기여할 수 있도록 간략한 가이드라인을 제시합니다.
- 새로운 모델/예제 추가
- 버그 수정
- 문서 개선
- Pull Request (PR) 가이드라인 (예: 브랜치 전략, 커밋 메시지 규칙)

### 저장소 디렉토리 구조 (제안)
```
├── README.md                 # 현재 파일
├── requirements.txt          # Python 의존성 목록
├── data/                     # 데이터셋 저장 디렉토리
│   └── dummy_interactions.csv
│   └── generate_dummy_data.py # 더미 데이터 생성 스크립트
├── examples/                 # 각 추천 모델 예제 코드
│   ├── collaborative_filtering/
│   │   └── item_cf_example.py # 아이템 기반 협업 필터링 예제
│   │   └── user_cf_example.py # 사용자 기반 협업 필터링 예제
│   ├── matrix_factorization/
│   │   └── svd_example.py     # SVD (Singular Value Decomposition) 예제
│   ├── deep_learning/
│   │   └── dnn_recommender.py # 딥러닝 기반 추천 모델 예제 (사용자 임베딩 포함)
│   └── content_based/
│       └── tfidf_example.py   # TF-IDF 기반 콘텐츠 추천 예제
├── models/                   # 학습된 모델 저장 또는 모델 정의 코드 포함 디렉토리 (선택 사항)
├── tests/                    # 테스트 코드
│   └── test_example.py
└── .gitignore                # Git 무시 파일
```

### 윤리적 고려 사항
추천 시스템은 사용자 경험에 중대한 영향을 미칠 수 있으므로, 다음 사항들을 고려하여 개발해야 합니다.
- **다양성 및 공정성**: 추천 결과가 특정 아이템 또는 사용자에게 편향되지 않도록 노력해야 합니다.
- **투명성**: 가능하다면 특정 아이템이 추천된 이유에 대한 간략한 설명을 제공해야 합니다. (예시: "이 아이템은 사용자가 최근 시청한 'SF 영화'와 유사하기 때문에 추천되었습니다.")
- **개인 정보 보호**: 사용자 데이터 처리 시 관련 개인 정보 보호 규정을 엄격히 준수해야 합니다.

### 문의 사항
궁금한 점이나 제안 사항이 있으시면 언제든지 문의해 주시기 바랍니다.
