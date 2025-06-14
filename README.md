# AI 추천 모델 샘플 저장소

### 목차
1. [프로젝트 개요](#프로젝트-개요)
2. [파이썬 데이터 엔지니어를 위한 지침](#파이썬-데이터-엔지니어를-위한-지침)
3. [세부 작업 지침](#세부-작업-지침)
    1. [개발 환경 설정](#1-개발-환경-설정)
    2. [데이터 준비](#2-데이터-준비)
    3. [구현된 추천 모델 예제 상세](#3-구현된-추천-모델-예제-상세)
        * [아이템 기반 협업 필터링 (Item-Based Collaborative Filtering)](#아이템-기반-협업-필터링-item-based-collaborative-filtering-예제)
        * [사용자 기반 협업 필터링 (User-Based Collaborative Filtering)](#사용자-기반-협업-필터링-user-based-collaborative-filtering-예제)
        * [특이값 분해 (SVD - Singular Value Decomposition)](#특이값-분해-svd---singular-value-decomposition-예제)
        * [콘텐츠 기반 필터링 (TF-IDF)](#콘텐츠-기반-필터링-tf-idf-예제)
        * [딥러닝 기반 추천 (DNN with Embeddings)](#딥러닝-기반-추천-dnn-with-embeddings-예제)
        * [SASRec (Self-Attentive Sequential Recommendation)](#sasrec-self-attentive-sequential-recommendation-예제)
        * [투 타워 하이브리드 추천 (Two-Tower Hybrid Recommendation)](#투-타워-하이브리드-추천-two-tower-hybrid-recommendation-예제)
        * [LightGCN (Light Graph Convolution Network - TensorFlow/Keras)](#lightgcn-light-graph-convolution-network---tensorflowkeras-예제)
    4. [문서화 및 `README.md` 업데이트](#4-문서화-및-readmemd-업데이트) (본 문서)
    5. [테스트](#5-테스트)
4. [저장소 디렉토리 구조](#저장소-디렉토리-구조)
5. [윤리적 고려 사항](#윤리적-고려-사항)
6. [문의 사항](#문의-사항)

---

### 프로젝트 개요
본 저장소는 한국인을 위한 다양한 인공지능(AI) 기반 추천 모델의 예시 코드와 활용 사례를 제공합니다. 본 프로젝트의 궁극적인 목표는 사용자가 추천 시스템의 기본 원리를 심층적으로 이해하고, 실제 프로젝트에 효율적으로 적용할 수 있도록 지원하는 것입니다. 특히, 파이썬 데이터 엔지니어는 모델 학습, 예측, 평가 등 일련의 과정을 명확하고 실용적인 예제를 통해 직관적으로 경험할 수 있도록 설계되었습니다.

### 학습 경로 안내 (Learning Path Guide)
추천 모델 학습을 처음 시작하시는 분들을 위해 두 가지 학습 경로를 제공합니다:

1.  **단계별 학습 가이드:** 본 저장소의 핵심 가이드인 [학습 경로 안내 (LearningPath.md)](./LearningPath.md) 문서는 추천 시스템의 기본 개념부터 다양한 모델 예제들을 단계별로 학습할 수 있도록 구성되어 있으며, 이해를 돕기 위한 다이어그램도 포함되어 있습니다.
2.  **탐구 중심 학습 여정:** 좀 더 능동적이고 탐구적인 학습을 원하시는 분들께는 [추천 시스템 학습 여정 (ConstructivistLearningPath.md)](./ConstructivistLearningPath.md) 문서를 추천합니다. 이 가이드는 질문, 실험, 발견을 통해 추천 시스템에 대한 이해를 구축하도록 설계되었습니다.

어떤 경로를 선택하시든, 본 저장소의 예제 코드와 함께 학습하시면 추천 시스템 구현에 대한 실질적인 경험을 얻으실 수 있을 것입니다.

### 파이썬 데이터 엔지니어를 위한 지침
본 프로젝트의 핵심 목표는 다음과 같습니다:
1. 추천 모델 예제 구현: 현재 저장소에 포함된 추천 모델들을 활용하여 사용자가 쉽게 따라 할 수 있는 파이썬 예제 코드를 개발했습니다.
2. 사용자 접근성 확보: 비전문가도 코드를 실행하고 그 결과를 명확하게 확인할 수 있도록 간결하고 명료한 인터페이스를 제공합니다.
3. 상세한 주석 및 설명: 코드 내부에 충분한 주석을 포함하고, 본 `README.md` 파일에 각 예제의 목적, 사용 방법, 예상 결과에 대한 구체적인 설명을 추가했습니다.
4. 재현 가능한 환경 구축: 예제 실행에 필요한 환경 설정(예: 라이브러리 종속성)에 대한 명확한 지침을 제공하여 재현성을 보장합니다.

### 세부 작업 지침

#### 1. 개발 환경 설정
예제 코드 실행을 위한 파이썬 환경 설정 방법을 안내합니다.
- 파이썬 버전: 파이썬 3.10 이상 버전 사용을 권장합니다.
- 가상 환경 생성: `venv` 또는 `conda`를 활용하여 독립적인 가상 환경을 구축합니다.
```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate   # Windows
```
- 필요 라이브러리 설치: `requirements.txt` 파일에 필요한 라이브러리 목록이 명시되어 있으며, 해당 파일을 통해 일괄 설치를 진행합니다.
```bash
pip install -r requirements.txt
```
현재 `requirements.txt` 내용:
```
pandas
scikit-learn
numpy==1.26.4
surprise
tensorflow
pytest
scipy
```

#### 2. 데이터 준비
예제 실행에 필요한 데이터셋은 `data/generate_dummy_data.py` 스크립트를 통해 생성됩니다. 이 스크립트는 사용자-아이템 상호작용 데이터 (`dummy_interactions.csv`), 아이템 메타데이터 (`dummy_item_metadata.csv`), 그리고 순차적 추천 모델을 위한 사용자 아이템 시퀀스 데이터 (`dummy_sequences.csv`)를 생성할 수 있습니다. 각 예제 스크립트는 실행 시 필요에 따라 이 더미 데이터 생성 스크립트를 자동으로 호출하도록 구현되어 있습니다.
- `dummy_interactions.csv`: `user_id`, `item_id`, `rating` 열을 포함합니다.
- `dummy_item_metadata.csv`: `item_id`, `genres`, `description` 열을 포함합니다.
- `dummy_sequences.csv`: `user_id`, `item_ids_sequence` (공백으로 구분된 아이템 ID 시퀀스) 열을 포함합니다.

#### 3. 구현된 추천 모델 예제 상세

각 예제 스크립트의 상단에는 해당 모델에 대한 자세한 설명, 장단점, 성능 고려 사항 (Big O 표기법 또는 주요 복잡도 요인 포함)이 주석으로 기술되어 있습니다. 본 README에서는 간략한 요약을 제공합니다.

---
##### 아이템 기반 협업 필터링 (Item-Based Collaborative Filtering) 예제
*   **파일:** `examples/collaborative_filtering/item_cf_example.py`
*   **모델:** 아이템 기반 협업 필터링 (Item-Based Collaborative Filtering)
*   **목적:** 사용자가 특정 아이템을 좋아했을 때, 그 아이템과 유사한 다른 아이템들을 추천하는 방식을 시연합니다. 아이템 간 유사도는 사용자들의 평점 패턴을 기반으로 계산됩니다.
*   **실행 방법:** `python examples/collaborative_filtering/item_cf_example.py`
*   **예상 출력:** 샘플 사용자에 대한 상위 5개 추천 아이템 ID와 예상 평점을 출력합니다. 데이터 로딩 과정 및 아이템 유사도 행렬의 일부도 함께 표시됩니다.
*   **모델 개요:**
    *   *장점:* 안정적인 추천, 설명 가능성, 새로운 사용자에게도 일부 아이템 평가 시 추천 가능.
    *   *단점:* 데이터 희소성 문제, 유사도 계산 확장성, 인기 편향, 새로운 아이템 콜드 스타트.
*   **성능:** 아이템 유사도 계산은 밀집 행렬의 경우 O(I² * U) (I: 아이템 수, U: 사용자 수).
*   **참고:** 자세한 설명은 스크립트 내 주석을 참고하십시오.

---
##### 사용자 기반 협업 필터링 (User-Based Collaborative Filtering) 예제
*   **파일:** `examples/collaborative_filtering/user_cf_example.py`
*   **모델:** 사용자 기반 협업 필터링 (User-Based Collaborative Filtering)
*   **목적:** 특정 사용자와 유사한 취향을 가진 다른 사용자들이 좋아했던 아이템을 추천하는 방식을 시연합니다.
*   **실행 방법:** `python examples/collaborative_filtering/user_cf_example.py`
*   **예상 출력:** 샘플 사용자에 대한 상위 5개 추천 아이템 ID와 예상 점수를 출력합니다.
*   **모델 개요:**
    *   *장점:* 단순하고 직관적, 우연한 발견 가능성.
    *   *단점:* 데이터 희소성, 확장성 문제, 새로운 사용자 콜드 스타트, 인기 편향.
*   **성능:** 사용자 유사도 계산은 밀집 행렬의 경우 O(U² * I) (U: 사용자 수, I: 아이템 수).
*   **참고:** 자세한 설명은 스크립트 내 주석을 참고하십시오.

---
##### 특이값 분해 (SVD - Singular Value Decomposition) 예제
*   **파일:** `examples/matrix_factorization/svd_example.py`
*   **모델:** 특이값 분해 (SVD) - `surprise` 라이브러리 사용
*   **목적:** 사용자-아이템 평점 행렬을 저차원 잠재 요인 행렬로 분해하여 평점을 예측하고 아이템을 추천합니다.
*   **실행 방법:** `python examples/matrix_factorization/svd_example.py`
*   **예상 출력:** 샘플 사용자에 대한 상위 5개 추천 아이템 ID와 예측 평점을 출력합니다.
*   **모델 개요:**
    *   *장점:* 희소 데이터 처리 용이, 간결한 표현, 복잡한 관계 포착.
    *   *단점:* 해석 어려움, 콜드 스타트, 학습 비용.
*   **성능:** 학습 시간은 에포크 수, 잠재 요인 수, 데이터 크기에 따라 달라집니다.
*   **참고:** 자세한 설명은 스크립트 내 주석을 참고하십시오.

---
##### 콘텐츠 기반 필터링 (TF-IDF) 예제
*   **파일:** `examples/content_based/tfidf_example.py`
*   **모델:** TF-IDF를 활용한 콘텐츠 기반 필터링
*   **목적:** 아이템의 텍스트 설명을 TF-IDF 벡터로 변환하고, 사용자가 과거에 좋아했던 아이템의 콘텐츠와 유사한 아이템을 추천합니다.
*   **실행 방법:** `python examples/content_based/tfidf_example.py`
*   **예상 출력:** 샘플 사용자에 대한 상위 5개 추천 아이템 ID, 유사도 점수, 아이템 장르 및 설명 일부를 출력합니다.
*   **모델 개요:**
    *   *장점:* 사용자 독립성, 투명성, 새로운 아이템 처리 용이.
    *   *단점:* 제한된 우연성, 특징 공학의 중요성, 과적합 가능성.
*   **성능:** TF-IDF 벡터화 및 유사도 계산은 아이템 및 특징 수에 의존합니다.
*   **참고:** 자세한 설명은 스크립트 내 주석을 참고하십시오.

---
##### 딥러닝 기반 추천 (DNN with Embeddings) 예제
*   **파일:** `examples/deep_learning/dnn_recommender.py`
*   **모델:** 임베딩과 Dense 레이어를 사용한 심층 신경망 (DNN) - TensorFlow/Keras 사용
*   **목적:** 사용자 ID와 아이템 ID를 임베딩 벡터로 변환 후 DNN을 통과시켜 평점을 예측하고 추천합니다.
*   **실행 방법:** `python examples/deep_learning/dnn_recommender.py`
*   **예상 출력:** 모델 구조, 학습 과정, 평가 결과 (손실, MAE), 샘플 사용자에 대한 추천 아이템 및 예측 평점을 출력합니다.
*   **모델 개요:**
    *   *장점:* 풍부한 특징 표현 자동 학습, 비선형 관계 포착, 추가 특징 통합 용이.
    *   *단점:* 복잡성, 긴 학습 시간, 많은 데이터 요구, 해석 어려움, 콜드 스타트.
*   **성능:** 학습 시간은 데이터셋 크기, 네트워크 구조, 에포크 수에 크게 좌우됩니다.
*   **참고:** 자세한 설명은 스크립트 내 주석을 참고하십시오.

---
##### SASRec (Self-Attentive Sequential Recommendation) 예제
*   **파일:** `examples/sequential/transformer_sasrec_example.py`
*   **모델:** 자기 주의(Self-Attention) 메커니즘 기반 순차 추천 모델 (SASRec) - TensorFlow/Keras 사용
*   **목적:** 사용자의 아이템 상호작용 시퀀스를 입력으로 받아 다음 아이템을 예측하는 Transformer 기반 모델을 시연합니다.
*   **실행 방법:** `python examples/sequential/transformer_sasrec_example.py`
*   **예상 출력:** 모델 구조, 학습 과정(에포크별 손실 및 정확도), 그리고 샘플 사용자 시퀀스에 대한 다음 아이템 추천 목록 및 예측 점수를 출력합니다. `dummy_sequences.csv` 파일이 없을 경우 자동 생성합니다.
*   **모델 개요:**
    *   *장점:* 순차적 동적 특성 포착, 문맥적 이해 (Self-Attention), 병렬 처리 효율성.
    *   *단점:* 충분한 시퀀스 데이터 필요, 시퀀스 길이에 따른 계산 비용 증가 가능성, 새로운 아이템 콜드 스타트.
*   **성능:** 학습 시간은 시퀀스 수, 시퀀스 길이, 모델 파라미터(임베딩 차원, 헤드 수, 블록 수)에 따라 달라집니다. Self-Attention은 시퀀스 길이에 대해 제곱의 복잡도를 가질 수 있으나, 추천 시스템의 일반적인 시퀀스 길이는 감당 가능한 수준입니다.
*   **참고:** 자세한 설명은 스크립트 내 주석을 참고하십시오.

---
##### 투 타워 하이브리드 추천 (Two-Tower Hybrid Recommendation) 예제
*   **파일:** `examples/hybrid/two_tower_hybrid_example.py`
*   **모델:** 사용자 타워와 아이템 타워로 구성된 투-타워 하이브리드 추천 모델 - TensorFlow/Keras 사용
*   **목적:** 사용자 정보(ID)와 아이템 정보(ID 및 장르)를 각각의 타워에서 임베딩으로 처리한 후, 두 임베딩 간의 유사도(점곱)를 통해 사용자-아이템 간의 상호작용을 예측하고 추천하는 모델을 시연합니다. 아이템 타워는 아이템 ID와 장르 정보를 모두 활용합니다.
*   **실행 방법:** `python examples/hybrid/two_tower_hybrid_example.py`
*   **예상 출력:** 모델 구조, 학습 과정(에포크별 손실 및 정확도), 그리고 샘플 사용자에 대한 추천 아이템 목록과 아이템 설명 일부 및 유사도 점수를 출력합니다.
*   **모델 개요:**
    *   *장점:* 확장성 (사용자/아이템 임베딩 독립적 계산 및 사전 계산 가능), 다양한 특징 통합 용이, 대규모 추천 시스템의 후보 생성 단계에 효과적.
    *   *단점:* 상호작용 모델링 단순화 (주로 점곱 사용), 특징 공학의 중요성 지속.
*   **성능:** 학습 시간은 데이터셋 크기, 네거티브 샘플링 전략, 네트워크 구조에 따라 달라집니다. 서빙 시 사용자 임베딩과 사전 계산된 아이템 임베딩을 활용하여 빠른 후보 생성이 가능합니다.
*   **참고:** 자세한 설명은 스크립트 내 주석을 참고하십시오.

---
##### LightGCN (Light Graph Convolution Network - TensorFlow/Keras) 예제
*   **파일:** `examples/gnn/lightgcn_tf_example.py`
*   **모델:** Light Graph Convolution Network (LightGCN) - TensorFlow/Keras 사용
*   **목적:** 사용자-아이템 상호작용을 이분 그래프(bipartite graph)로 모델링하고, 그래프 컨볼루션을 통해 학습된 사용자 및 아이템 임베딩을 활용하여 추천을 생성하는 방식을 시연합니다. LightGCN은 GCN에서 특징 변환 및 비선형 활성화 함수를 제거하여 단순화한 모델입니다.
*   **실행 방법:** `python examples/gnn/lightgcn_tf_example.py`
*   **예상 출력:** 모델 구조 요약 (Keras 모델 서브클래싱 사용 시 학습 후 출력), BPR 손실을 사용한 학습 과정 (에포크별 손실 값), 그리고 샘플 사용자에 대한 상위 5개 추천 아이템 ID와 예측된 선호도 점수를 출력합니다.
*   **모델 개요:**
    *   *장점:* 단순성과 효율성, 많은 추천 벤치마크에서 강력한 성능, 고차원 연결성 포착 가능.
    *   *단점:* 너무 많은 레이어 사용 시 과도한 평탄화(over-smoothing) 문제 발생 가능, 그래프 구성 및 정규화에 주의 필요.
*   **성능:** 학습 시간은 에포크 수, 임베딩 차원, 레이어 수, 그리고 사용자-아이템 상호작용 수에 따라 달라집니다. 각 GCN 레이어의 전파는 희소 행렬 곱셈으로 효율적으로 수행될 수 있습니다.
*   **참고:** 자세한 설명은 스크립트 내 주석을 참고하십시오. PyTorch Geometric을 사용한 LightGCN 예제도 계획되었으나, 라이브러리 설치 공간 제약으로 인해 현재 버전에서는 보류되었습니다.

---
##### NGCF (Neural Graph Collaborative Filtering) 예제
*   **파일:** `examples/gnn/ngcf_example.py`
*   **모델:** NGCF (Neural Graph Collaborative Filtering) - 개념적 개요 및 구조
*   **목적:** NGCF 모델의 핵심 아이디어, 구성 요소(임베딩 레이어, 임베딩 전파 레이어, 예측 레이어), 장단점 및 학술 논문 정보를 제공합니다. 스크립트는 실제 실행 가능한 모델이 아닌, 개념 이해를 돕기 위한 고수준 코드 구조와 설명을 포함합니다.
*   **실행 방법:** `python examples/gnn/ngcf_example.py`
*   **예상 출력:** 스크립트가 NGCF의 개념적 개요 및 구조를 제공하는 플레이스홀더임을 알리는 메시지와 함께, 모델의 주요 구성 요소에 대한 설명을 출력합니다. 실제 모델 학습이나 추천 생성은 수행되지 않습니다.
*   **모델 개요:**
    *   *주요 특징:* 사용자-아이템 이분 그래프(Bipartite Graph) 활용, 임베딩 전파(Embedding Propagation) 레이어를 통해 고차원 연결성 학습 (이웃 노드 정보와 자체 임베딩을 결합하여 메시지 생성 및 집계).
    *   *장점:* 기존 협업 필터링 모델보다 풍부한 사용자-아이템 관계 학습 가능, 복잡한 상호작용 패턴 포착에 유리, 명시적인 고차원 정보 모델링.
    *   *단점:* LightGCN에 비해 모델 구조가 더 복잡하고 파라미터가 많을 수 있음, 특정 상황에서 LightGCN보다 성능이 낮거나 학습이 어려울 수 있음, 과적합 및 과도한 평탄화 가능성.
*   **성능:** 학습 시간은 그래프 크기, 임베딩 차원, 전파 레이어 수, 레이어 내 연산 복잡도에 따라 달라집니다.
*   **참고:** 스크립트(`examples/gnn/ngcf_example.py`)는 NGCF의 개념과 아키텍처를 이해하기 위한 상세한 설명과 고수준 코드 구조를 제공합니다. 실제 실행 가능한 모델이 아님을 유의하십시오.

---
##### PinSage 예제
*   **파일:** `examples/gnn/pinsage_example.py`
*   **모델:** PinSage (Graph Convolutional Neural Networks for Web-Scale Recommender Systems) - 개념적 개요 및 구조
*   **목적:** 대규모 그래프에서 아이템 임베딩을 효과적으로 학습하기 위해 Pinterest에서 개발한 PinSage 모델의 핵심 아이디어(랜덤 워크 기반 이웃 샘플링, 특징 집계 시 중요도 풀링, 멀티 레이어 구조 등), 장단점 및 학술 논문 정보를 제공합니다. 스크립트는 실제 실행 가능한 모델이 아닌, 개념 이해를 위한 고수준 코드 구조와 설명을 담고 있습니다.
*   **실행 방법:** `python examples/gnn/pinsage_example.py`
*   **예상 출력:** 스크립트가 PinSage의 개념적 개요 및 구조를 제공하는 플레이스홀더임을 알리는 메시지와 함께, 모델의 주요 구성 요소(샘플러, 컨볼루션 레이어) 및 작동 방식에 대한 설명을 출력합니다. 실제 모델 학습이나 임베딩 생성은 수행되지 않습니다.
*   **모델 개요:**
    *   *주요 특징:* 대규모 그래프 처리를 위한 랜덤 워크 기반 이웃 노드 샘플링, 콘텐츠 특징과 그래프 구조 정보 결합, 중요도 기반 풀링(Importance Pooling)을 통한 영향력 있는 이웃 가중치 부여, 여러 컨볼루션 레이어 스태킹.
    *   *장점:* 웹 스케일의 매우 큰 그래프에 확장 가능, 노드 콘텐츠 특징 효과적 통합, 귀납적 학습으로 새로운 아이템 임베딩 생성 가능.
    *   *단점:* 구현 및 전체 학습 파이프라인의 복잡성, 랜덤 워크 및 샘플링 파라미터 튜닝의 중요성, 잠재적인 샘플링 편향.
*   **성능:** 학습은 주로 오프라인 배치 단위로 수행. 임베딩 생성 후에는 ANN 검색을 통해 빠른 추천 가능. 계산 비용은 샘플링되는 이웃 수, 컨볼루션 레이어 깊이, 임베딩 차원 등에 따라 변동.
*   **참고:** 스크립트(`examples/gnn/pinsage_example.py`)는 PinSage의 개념과 아키텍처, 특히 샘플링과 집계 방식에 대한 상세한 설명과 고수준 코드 구조를 제공합니다. 실제 실행 가능한 모델이 아님을 유의하십시오.

---

현업에서 사용되는 대표적인 GNN 기반 추천 모델 외에도, GNN의 근간을 이루는 핵심 알고리즘들이 있습니다. 그래프 컨볼루션 네트워크(GCN), GraphSAGE, 그래프 어텐션 네트워크(GAT) 등은 이러한 기본 알고리즘에 해당하며, 추천 시스템 문제에 맞게 변형되거나 다른 모델과 결합되어 다양하게 활용됩니다. 아래에서는 이들 각 모델에 대한 간략한 소개와 예제 링크를 제공합니다.

---
##### GCN (Graph Convolutional Network) 추천 예제
*   **파일:** `examples/gnn/gcn_example.py`
*   **모델:** GCN (Graph Convolutional Network) - 개념적 개요 및 구조
*   **목적:** 표준 GCN 모델의 핵심 원리(이웃 노드 특징의 정규화된 합계 및 변환), GCN 레이어 구조, 추천 시스템에의 적용 방안, 장단점 및 학술 논문 정보를 제공합니다. 스크립트는 실제 실행 가능한 모델이 아닌, 개념 이해를 위한 고수준 코드 구조와 설명을 포함합니다.
*   **실행 방법:** `python examples/gnn/gcn_example.py`
*   **예상 출력:** 스크립트가 GCN의 개념적 개요 및 구조를 제공하는 플레이스홀더임을 알리는 메시지와 함께, 모델의 주요 구성 요소(GCN 레이어, 전체 모델 아키텍처) 및 작동 방식에 대한 설명을 출력합니다. 실제 모델 학습이나 추천 생성은 수행되지 않습니다.
*   **모델 개요:**
    *   *주요 특징:* 그래프 컨볼루션 연산을 통해 이웃 노드의 특징 정보를 집계하여 노드 표현 업데이트 (일반적으로 정규화된 인접 행렬 사용), 여러 GCN 레이어를 쌓아 고차원적 관계 학습, 각 레이어에 특징 변환(가중치 행렬) 및 비선형 활성화 함수 적용.
    *   *장점:* 노드 간의 관계 정보를 효과적으로 활용하여 임베딩 학습, 다양한 유형의 그래프 데이터 및 태스크에 적용 가능.
    *   *단점:* 원래 형태는 전이적(transductive) 학습, 깊은 레이어에서 과도한 평탄화(Over-smoothing) 문제 발생 가능, 대규모 그래프에서 계산 비용이 클 수 있음 (LightGCN은 이를 일부 완화).
*   **성능:** 학습 시간은 노드 수, 엣지 수, 초기 특징 차원, GCN 레이어 수 및 각 레이어의 연산 복잡도에 따라 달라집니다. 희소 행렬 연산을 통해 효율적 구현이 중요합니다.
*   **참고:** 스크립트(`examples/gnn/gcn_example.py`)는 표준 GCN의 개념, 레이어 수식 (`H^(l+1) = σ(D̃^(-0.5) Ã D̃^(-0.5) H^(l) W^(l))`), 아키텍처에 대한 상세한 설명과 고수준 코드 구조를 제공합니다. 실제 실행 가능한 모델이 아님을 유의하십시오.

---
##### GraphSAGE (Graph SAmple and aggreGatE) 추천 예제
*   **파일:** `examples/gnn/graphsage_example.py`
*   **모델:** GraphSAGE (Graph SAmple and aggreGatE) - 개념적 개요 및 구조
*   **목적:** 대규모 그래프에서 확장 가능하고 귀납적인 노드 임베딩 학습 방법인 GraphSAGE의 핵심 원리(이웃 노드 샘플링, 다양한 집계 함수 - Mean/LSTM/Pooling, 귀납적 학습)와 구조, 장단점 및 학술 논문 정보를 제공합니다. 스크립트는 실제 실행 가능한 모델이 아닌, 개념 이해를 위한 고수준 코드 구조와 설명을 담고 있습니다.
*   **실행 방법:** `python examples/gnn/graphsage_example.py`
*   **예상 출력:** 스크립트가 GraphSAGE의 개념적 개요 및 구조를 제공하는 플레이스홀더임을 알리는 메시지와 함께, 모델의 주요 구성 요소(샘플러, GraphSAGE 레이어) 및 작동 방식(샘플링 후 집계)에 대한 설명을 출력합니다. 실제 모델 학습이나 임베딩 생성은 수행되지 않습니다.
*   **모델 개요:**
    *   *주요 특징:* 각 노드에 대해 고정된 크기의 이웃을 샘플링하여 정보를 집계, 다양한 집계 함수(Mean, LSTM, Pooling 등) 사용, 여러 레이어를 쌓아 K-hop 이웃 정보 학습, 귀납적 학습으로 새로운 노드의 임베딩 생성 가능.
    *   *장점:* 대규모 그래프에 확장 용이 (전체 그래프 대신 샘플링된 부가그래프 사용), 새로운 노드(콜드 스타트)에 대한 임베딩 생성 가능 (노드 특징 활용 시), 다양한 집계 함수로 모델 표현력 조절.
    *   *단점:* 샘플링 전략 및 집계 함수 선택이 성능에 민감, 효과적인 귀납 학습을 위해 노드 특징이 중요할 수 있음, GCN보다 구현이 다소 복잡.
*   **성능:** 학습 시 각 노드마다 이웃을 샘플링하므로 전체 인접 행렬을 사용하지 않아도 됨. 집계 함수의 복잡도와 샘플링하는 이웃의 수가 계산 비용에 영향.
*   **참고:** 스크립트(`examples/gnn/graphsage_example.py`)는 GraphSAGE의 개념, 특히 샘플링과 집계 방식, 귀납적 특성에 대한 상세한 설명과 고수준 코드 구조를 제공합니다. 실제 실행 가능한 모델이 아님을 유의하십시오.

---
##### GAT (Graph Attention Network) 추천 예제
*   **파일:** `examples/gnn/gat_example.py`
*   **모델:** GAT (Graph Attention Network) - 개념적 개요 및 구조
*   **목적:** 그래프 컨볼루션 과정에서 어텐션 메커니즘을 활용하는 GAT 모델의 핵심 아이디어(이웃 노드에 대한 가중치 학습, 멀티 헤드 어텐션), 구조, 장단점 및 학술 논문 정보를 제공합니다. 스크립트는 실제 실행 가능한 모델이 아닌, 개념 이해를 위한 고수준 코드 구조와 설명을 담고 있습니다.
*   **실행 방법:** `python examples/gnn/gat_example.py`
*   **예상 출력:** 스크립트가 GAT의 개념적 개요 및 구조를 제공하는 플레이스홀더임을 알리는 메시지와 함께, 모델의 주요 구성 요소(어텐션 헤드, GAT 레이어) 및 작동 방식(어텐션 가중치를 사용한 이웃 정보 집계)에 대한 설명을 출력합니다. 실제 모델 학습이나 임베딩 생성은 수행되지 않습니다.
*   **모델 개요:**
    *   *주요 특징:* 어텐션 메커니즘을 사용하여 이웃 노드에 서로 다른 가중치(어텐션 스코어)를 할당하여 정보를 집계, 멀티 헤드 어텐션(Multi-head Attention)을 통해 여러 독립적인 어텐션 결과를 결합하여 표현력 강화, 귀납적 학습 가능.
    *   *장점:* 중요한 이웃 노드에 자동으로 높은 가중치를 부여하여 모델의 해석력과 성능 향상 가능, 암시적으로 이웃의 중요도를 학습하므로 그래프 구조에 대한 사전 지식 덜 요구.
    *   *단점:* 어텐션 계산으로 인해 GCN에 비해 계산 비용이 더 클 수 있음, 멀티 헤드 어텐션의 헤드 수 등 하이퍼파라미터 튜닝이 중요, 과적합(Overfitting)의 위험.
*   **성능:** 어텐션 가중치 계산은 각 노드와 그 이웃 간의 모든 쌍에 대해 수행됨. 멀티 헤드 어텐션은 계산량을 헤드 수만큼 증가시키지만 병렬 처리가 가능.
*   **참고:** 스크립트(`examples/gnn/gat_example.py`)는 GAT의 핵심 개념인 어텐션 메커니즘, 멀티 헤드 어텐션, 모델 아키텍처에 대한 상세한 설명과 고수준 코드 구조를 제공합니다. 실제 실행 가능한 모델이 아님을 유의하십시오.

---

#### 4. 문서화 및 `README.md` 업데이트
본 문서는 각 예제 파일에 대한 설명과 사용 지침을 포함하도록 업데이트되었습니다.

#### 5. 테스트
기본적인 단위 테스트 및 예제 스크립트 실행 테스트가 `tests` 디렉토리에 구현되어 있습니다. `pytest`를 사용하여 실행할 수 있습니다.
```bash
pytest -v
```
테스트는 다음을 포함합니다:
- `tests/test_data_generation.py`: 더미 데이터 생성 스크립트의 정상 동작 및 파일 생성 여부 검증 (기본 파일 및 시퀀스 파일 포함).
- `tests/test_collaborative_filtering.py`: 협업 필터링 예제 스크립트 테스트.
- `tests/test_content_based.py`: 콘텐츠 기반 필터링 예제 스크립트 테스트.
- `tests/test_deep_learning.py`: 딥러닝 추천 예제 스크립트 테스트.
- `tests/test_gnn_examples.py`: GNN 기반 추천 예제 스크립트 테스트.
- `tests/test_hybrid.py`: 하이브리드 추천 예제 스크립트 테스트.
- `tests/test_matrix_factorization.py`: 행렬 분해 예제 스크립트 테스트.
- `tests/test_sequential.py`: 순차 추천 예제 스크립트 테스트.
모든 예제 스크립트가 오류 없이 실행되고, 예상되는 형태의 추천 관련 출력을 생성하는지 각 해당 테스트 파일에서 검증합니다.

### 저장소 디렉토리 구조
```
├── README.md                 # 현재 파일
├── requirements.txt          # 파이썬 의존성 목록
├── data/                     # 데이터셋 저장 디렉토리
│   ├── dummy_interactions.csv  # (자동 생성됨) 사용자-아이템 상호작용 더미 데이터
│   ├── dummy_item_metadata.csv # (자동 생성됨) 아이템 메타데이터 더미 데이터
│   ├── dummy_sequences.csv     # (자동 생성됨) 사용자-아이템 시퀀스 더미 데이터
│   └── generate_dummy_data.py  # 더미 데이터 생성 스크립트
├── examples/                 # 각 추천 모델 예제 코드
│   ├── collaborative_filtering/
│   │   ├── item_cf_example.py    # 아이템 기반 협업 필터링 예제
│   │   └── user_cf_example.py    # 사용자 기반 협업 필터링 예제
│   ├── content_based/
│   │   └── tfidf_example.py      # TF-IDF 기반 콘텐츠 추천 예제
│   ├── deep_learning/
│   │   └── dnn_recommender.py    # DNN 기반 추천 모델 예제
│   ├── gnn/
│   │   ├── lightgcn_tf_example.py # LightGCN (TensorFlow/Keras) 예제
│   │   ├── ngcf_example.py       # NGCF (Neural Graph Collaborative Filtering) 예제 (플레이스홀더)
│   │   ├── pinsage_example.py    # PinSage 예제 (플레이스홀더)
│   │   ├── gcn_example.py        # GCN 추천 예제 (플레이스홀더)
│   │   ├── graphsage_example.py  # GraphSAGE 추천 예제 (플레이스홀더)
│   │   └── gat_example.py        # GAT 추천 예제 (플레이스홀더)
│   ├── hybrid/
│   │   └── two_tower_hybrid_example.py # 투-타워 하이브리드 추천 모델 예제
│   ├── matrix_factorization/
│   │   └── svd_example.py        # SVD (Singular Value Decomposition) 예제
│   └── sequential/
│       └── transformer_sasrec_example.py # SASRec 순차 추천 모델 예제
├── tests/                    # 테스트 코드
│   ├── __init__.py
│   ├── test_data_generation.py
│   ├── test_collaborative_filtering.py
│   ├── test_content_based.py
│   ├── test_deep_learning.py
│   ├── test_gnn_examples.py
│   ├── test_hybrid.py
│   ├── test_matrix_factorization.py
│   └── test_sequential.py
└── .gitignore                # Git 무시 파일
```

### 윤리적 고려 사항
추천 시스템은 사용자 경험에 중대한 영향을 미칠 수 있으므로, 다음 사항들을 고려하여 개발해야 합니다.
- **다양성 및 공정성**: 추천 결과가 특정 아이템 또는 사용자에게 편향되지 않도록 노력해야 합니다.
- **투명성**: 가능하다면 특정 아이템이 추천된 이유에 대한 간략한 설명을 제공해야 합니다. (예시: "이 아이템은 사용자가 최근 시청한 'SF 영화'와 유사하기 때문에 추천되었습니다.")
- **개인 정보 보호**: 사용자 데이터 처리 시 관련 개인 정보 보호 규정을 엄격히 준수해야 합니다.
