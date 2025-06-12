# examples/deep_learning/dnn_recommender.py
import pandas as pd
import numpy as np
import os
import sys
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Concatenate, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# --- Deep Learning (DNN) with Embeddings for Recommendations: Basic Explanation ---
# Deep Neural Networks (DNNs) combined with embedding layers can create powerful recommendation models.
# They can capture complex, non-linear relationships in user-item interaction data and learn
# rich representations (embeddings) for users and items.
#
# How it typically works (for a rating prediction task):
# 1. Input Data: User IDs, Item IDs, and their corresponding ratings.
# 2. Embedding Layers:
#    - User IDs are fed into a User Embedding layer, which learns a dense vector representation for each user.
#    - Item IDs are fed into an Item Embedding layer, which learns a dense vector representation for each item.
#    These embeddings capture latent features or characteristics.
# 3. Combining Embeddings: The learned user and item embeddings are combined. Common methods:
#    - Concatenation: The two vectors are joined end-to-end.
#    - Dot Product (or Element-wise Product): Captures interaction effects, similar to matrix factorization.
# 4. Deep Neural Network (DNN): The combined (or interacted) embedding vector is passed through
#    one or more dense (fully connected) layers. These layers can learn non-linear patterns.
#    Activation functions (e.g., ReLU) and dropout layers (for regularization) are often used.
# 5. Output Layer: A final dense layer outputs the predicted rating. For regression (rating prediction),
#    this layer usually has a single neuron and a linear activation function. For classification
#    (e.g., predict click/no-click), it might use a sigmoid activation.
# 6. Training: The model is trained by minimizing a loss function (e.g., Mean Squared Error for ratings)
#    using an optimizer like Adam.
#
# Pros:
# - Feature Representation: Learns rich, dense embeddings for users and items automatically.
# - Non-linear Relationships: DNNs can capture complex patterns that linear models (like basic SVD) might miss.
# - Flexibility: Easy to incorporate additional features (e.g., user demographics, item metadata)
#   by creating more embedding layers or inputting them directly into the DNN.
# - State-of-the-art Performance: Often achieves high accuracy on many recommendation tasks.
#
# Cons:
# - Complexity & Training Time: Can be computationally expensive to train, especially with large datasets
#   and deep/wide networks.
# - Data Requirements: Typically requires a large amount of interaction data to perform well and avoid overfitting.
# - Interpretability: Like many deep learning models, the learned relationships and embeddings can be hard to interpret directly.
# - Cold-Start: Still challenging for new users/items with no interaction history, though techniques like
#   content embeddings can be integrated to mitigate this.
# ---

# Dynamically add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root) # Ensure project root is at the start for priority

def load_and_preprocess_data(base_filepath='data/dummy_interactions.csv', test_size=0.2, random_state=42):
    """
    Loads interaction data, preprocesses it for the DNN model, and splits into train/test sets.
    """
    filepath = os.path.join(project_root, base_filepath)

    if not os.path.exists(filepath):
        print(f"오류: {filepath} 파일을 찾을 수 없습니다.")
        if base_filepath == 'data/dummy_interactions.csv': # Check against relative path
            print("더미 데이터 생성 시도 중... ('data/generate_dummy_data.py' 실행)")
            try:
                from data.generate_dummy_data import generate_dummy_data
                generate_dummy_data()
                print("더미 데이터 생성 완료.")
                if not os.path.exists(filepath): # Check again
                    print(f"데이터 자동 생성 후에도 {filepath}를 찾을 수 없습니다.")
                    return None, None, None, None, None, None, None, None, None
            except ImportError as e_import:
                print(f"ImportError: 'generate_dummy_data' 임포트 실패. 오류: {e_import}")
                return None, None, None, None, None, None, None, None, None
            except Exception as e_general:
                print(f"더미 데이터 생성 실패: {e_general}")
                return None, None, None, None, None, None, None, None, None
        else:
            return None, None, None, None, None, None, None, None, None

    df = pd.read_csv(filepath)
    if df.empty:
        print("데이터 파일이 비어있습니다.")
        return None, None, None, None, None, None, None, None, None

    # Encode User and Item IDs into integer indices
    user_encoder = LabelEncoder()
    df['user_idx'] = user_encoder.fit_transform(df['user_id'])

    item_encoder = LabelEncoder()
    df['item_idx'] = item_encoder.fit_transform(df['item_id'])

    num_users = df['user_idx'].nunique()
    num_items = df['item_idx'].nunique() # This is item_encoder.classes_.shape[0]

    # Split data
    X = df[['user_idx', 'item_idx']].values
    y = df['rating'].values.astype(np.float32) # Ensure ratings are float for TF
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    return X_train, X_test, y_train, y_test, num_users, num_items, user_encoder, item_encoder, df

def build_dnn_model(num_users, num_items, embedding_dim=32, dense_layers=[64, 32], dropout_rate=0.1):
    """
    Builds a DNN model with embedding layers for users and items.
    """
    # User Embedding Pathway
    user_input = Input(shape=(1,), name='user_input')
    user_embedding = Embedding(input_dim=num_users, output_dim=embedding_dim, name='user_embedding')(user_input)
    user_vec = Flatten(name='flatten_user_embedding')(user_embedding)
    user_vec = Dropout(dropout_rate)(user_vec)

    # Item Embedding Pathway
    item_input = Input(shape=(1,), name='item_input')
    item_embedding = Embedding(input_dim=num_items, output_dim=embedding_dim, name='item_embedding')(item_input)
    item_vec = Flatten(name='flatten_item_embedding')(item_embedding)
    item_vec = Dropout(dropout_rate)(item_vec)

    # Concatenate user and item embeddings
    concat = Concatenate()([user_vec, item_vec])
    concat = Dropout(dropout_rate)(concat)

    # Dense layers
    dense_layer = concat # Corrected variable name from 'dense' to 'dense_layer' to avoid conflict
    for units in dense_layers:
        dense_layer = Dense(units, activation='relu')(dense_layer)
        dense_layer = Dropout(dropout_rate)(dense_layer)

    # Output layer (predicting ratings)
    output = Dense(1, activation='linear', name='rating_output')(dense_layer)

    model = Model(inputs=[user_input, item_input], outputs=output)
    return model

def get_dnn_recommendations(model, user_id_original, user_encoder, item_encoder, df_all_interactions, num_total_items_encoded, num_recommendations=5):
    """
    Generates recommendations for a specific user using the trained DNN model.
    `num_total_items_encoded` is the total number of unique items the item_encoder knows (i.e., item_encoder.classes_.shape[0]).
    """
    try:
        # Transform the original user ID to its encoded index
        user_idx = user_encoder.transform([user_id_original])[0]
    except ValueError:
        print(f"사용자 ID {user_id_original}는 학습 데이터에 없어 인코딩할 수 없습니다. (새로운 사용자)")
        return []

    # Get items the user has already interacted with (original item IDs)
    rated_item_ids_original = set(df_all_interactions[df_all_interactions['user_id'] == user_id_original]['item_id'].unique())

    # Create a list of all possible item indices based on what item_encoder learned
    all_item_indices_encoded = np.arange(num_total_items_encoded)

    # Items to predict: all items not yet rated by the user
    # We need to find which of the `all_item_indices_encoded` correspond to items the user has NOT rated.
    items_to_predict_indices = []
    for item_idx_encoded in all_item_indices_encoded:
        original_item_id = item_encoder.inverse_transform([item_idx_encoded])[0]
        if original_item_id not in rated_item_ids_original:
            items_to_predict_indices.append(item_idx_encoded)

    items_to_predict_indices = np.array(items_to_predict_indices)

    if len(items_to_predict_indices) == 0:
        print(f"사용자 {user_id_original}는 이미 모든 아이템을 평가했거나 추천할 새로운 아이템이 없습니다.")
        return []

    # Prepare input for the model: (user_idx, item_idx_to_predict)
    user_input_array = np.full(len(items_to_predict_indices), user_idx, dtype=np.int32) # Ensure dtype
    item_input_array = items_to_predict_indices.astype(np.int32) # Ensure dtype

    predictions = model.predict([user_input_array, item_input_array], verbose=0).flatten()

    # Get top N recommendations
    # Combine item indices with their predicted ratings
    recommendation_results = list(zip(items_to_predict_indices, predictions))
    recommendation_results.sort(key=lambda x: x[1], reverse=True)

    top_n_recs = []
    for item_idx, score in recommendation_results[:num_recommendations]:
        original_item_id = item_encoder.inverse_transform([item_idx])[0]
        top_n_recs.append({'item_id': original_item_id, 'predicted_rating': score})

    return top_n_recs

# 메인 실행 함수
if __name__ == "__main__":
    print("--- DNN (Deep Neural Network) 기반 추천 예제 시작 (TensorFlow/Keras) ---")

    # 1. 데이터 로드 및 전처리
    print("\n데이터 로드 및 전처리 중...")
    # Using relative path for load_and_preprocess_data
    X_train, X_test, y_train, y_test, num_users, num_items, user_encoder, item_encoder, df_all_interactions = \
        load_and_preprocess_data(base_filepath='data/dummy_interactions.csv')

    if X_train is not None and df_all_interactions is not None:
        print(f"학습 데이터: {X_train.shape[0]} 샘플, 테스트 데이터: {X_test.shape[0]} 샘플")
        print(f"고유 사용자 수: {num_users}, 고유 아이템 수: {num_items}")

        # 2. DNN 모델 구축
        print("\nDNN 모델 구축 중...")
        embedding_dim = 50
        dense_layers_config = [128, 64, 32]
        dropout_rate_config = 0.2
        learning_rate_config = 0.001

        model = build_dnn_model(num_users, num_items, embedding_dim, dense_layers_config, dropout_rate_config)
        model.compile(optimizer=Adam(learning_rate=learning_rate_config), loss='mean_squared_error', metrics=['mae'])
        model.summary()

        # 3. 모델 학습
        print("\n모델 학습 중... (에포크 수에 따라 시간이 걸릴 수 있습니다)")
        epochs_config = 10
        batch_size_config = 64

        history = model.fit(
            [X_train[:, 0].astype(np.int32), X_train[:, 1].astype(np.int32)], y_train, # Ensure input data is int32
            epochs=epochs_config,
            batch_size=batch_size_config,
            validation_data=([X_test[:, 0].astype(np.int32), X_test[:, 1].astype(np.int32)], y_test), # Ensure validation data is int32
            verbose=1
        )
        print("모델 학습 완료.")

        loss, mae = model.evaluate([X_test[:, 0].astype(np.int32), X_test[:, 1].astype(np.int32)], y_test, verbose=0)
        print(f"\n테스트 데이터셋 평가: Loss = {loss:.4f}, MAE = {mae:.4f}")

        # 4. 특정 사용자에 대한 추천 생성
        if not df_all_interactions.empty:
            target_user_original_id = df_all_interactions['user_id'].unique()[0]
            print(f"\n사용자 ID {target_user_original_id} (원래 ID)에 대한 추천 아이템 생성 중...")

            # num_items here is item_encoder.classes_.shape[0] which is what num_total_items_encoded needs
            recommendations = get_dnn_recommendations(
                model,
                target_user_original_id,
                user_encoder,
                item_encoder,
                df_all_interactions,
                num_items,
                num_recommendations=5
            )

            if recommendations:
                print(f"\n사용자 {target_user_original_id}를 위한 추천 아이템 목록 (DNN 기반):")
                for rec in recommendations:
                    print(f"- 아이템 {rec['item_id']}: 예상 평점 {rec['predicted_rating']:.2f}")
            else:
                print("추천할 아이템이 없습니다.")
        else:
            print("\n상호작용 데이터가 없어 추천을 생성할 수 없습니다.")
    else:
        print("\n데이터 로드 및 전처리에 실패하여 예제를 실행할 수 없습니다.")

    print("\n--- DNN 기반 추천 예제 실행 완료 ---")
