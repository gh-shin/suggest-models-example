# examples/sequential/transformer_sasrec_example.py
import pandas as pd
import numpy as np
import os
import sys
import tensorflow as tf
from tensorflow.keras.layers import Layer, Embedding, Dense, Dropout, LayerNormalization, MultiHeadAttention, Input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

# --- SASRec (Self-Attentive Sequential Recommendation): Basic Explanation ---
# SASRec is a sequential recommendation model that uses a Transformer-like architecture (specifically, the self-attention mechanism)
# to capture the user's preferences based on their historical interaction sequence.
#
# How it works:
# 1. Input: A sequence of item IDs representing a user's interaction history (e.g., [item3, item1, item5]).
# 2. Item Embeddings: Each item in the sequence is mapped to a dense embedding vector.
# 3. Positional Embeddings: To incorporate the order of items, positional embeddings are added to item embeddings.
#    This tells the model where each item is in the sequence.
# 4. Self-Attention Block(s):
#    - The core of SASRec. Multiple self-attention "blocks" process the sequence.
#    - Each block typically consists of:
#        a. Multi-Head Self-Attention: Allows the model to jointly attend to information from different
#           representation subspaces at different positions. It weighs the importance of other items in the
#           sequence when predicting the next item. For example, to predict what comes after item5, it might
#           attend more to item1 than item3 if that pattern is learned.
#        b. Point-wise Feed-Forward Network (FFN): Applied to each position independently after attention.
#           Usually two dense layers with a non-linear activation in between.
#    - Residual connections and layer normalization are used around these components to help with training deeper models.
# 5. Output: After processing through the attention blocks, the model outputs a representation for each position
#    in the sequence. For next-item prediction, the representation of the *last* item in the input sequence
#    is typically used to predict the next item (e.g., by computing scores against all candidate item embeddings).
#
# Pros:
# - Captures Sequential Dynamics: Effectively models the order and dependencies within user interaction sequences.
# - Contextual Understanding: Self-attention allows it to understand which past items are more relevant for the next prediction.
# - Parallelizable: Attention computations can be parallelized, making it efficient on modern hardware.
#
# Cons:
# - Data Requirements: Like many deep learning models, it performs best with sufficient sequence data.
# - Computational Cost: Self-attention can be quadratic in sequence length, though typically sequence lengths in recsys are manageable.
# - Cold-Start for Items: New items not in training sequences cannot be directly recommended.
# ---

# Dynamically add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- Model Hyperparameters (small for example) ---
MAX_SEQ_LENGTH = 10  # Max length of user sequence
EMBEDDING_DIM = 32   # Dimension of item and positional embeddings
NUM_HEADS = 2        # Number of attention heads
NUM_BLOCKS = 1       # Number of Transformer blocks
FFN_UNITS = 64       # Units in the feed-forward network
DROPOUT_RATE = 0.2
LEARNING_RATE = 0.001
EPOCHS = 5           # Small number for quick example
BATCH_SIZE = 32

def load_sequences(base_filepath='data/dummy_sequences.csv', max_seq_len=MAX_SEQ_LENGTH):
    """Loads and preprocesses sequence data."""
    filepath = os.path.join(project_root, base_filepath)
    if not os.path.exists(filepath):
        print(f"오류: {filepath} 파일을 찾을 수 없습니다.")
        print("더미 데이터 생성 시도 중... ('data/generate_dummy_data.py' 실행, generate_sequences=True)")
        try:
            from data.generate_dummy_data import generate_dummy_data
            generate_dummy_data(num_users=200, num_items=100, num_interactions=2000, generate_sequences=True)
            print("더미 시퀀스 데이터 생성 완료.")
            if not os.path.exists(filepath): # Check again
                print(f"시퀀스 데이터 자동 생성 후에도 {filepath}를 찾을 수 없습니다.")
                return None, None, None, None
        except ImportError as e_import:
            print(f"ImportError: 'generate_dummy_data' 임포트 실패. 오류: {e_import}")
            return None, None, None, None
        except Exception as e_general:
            print(f"더미 데이터 생성 실패: {e_general}")
            return None, None, None, None


    df_sequences = pd.read_csv(filepath)
    if df_sequences.empty or 'item_ids_sequence' not in df_sequences.columns:
        print(f"오류: {filepath}가 비어있거나 'item_ids_sequence' 열이 없습니다.")
        return None, None, None, None

    # Convert space-separated item strings to lists of integers
    sequences = df_sequences['item_ids_sequence'].apply(lambda s: [int(i) for i in str(s).split(' ')]).tolist()

    # Create item vocabulary and encoder
    all_items = [item for seq in sequences for item in seq]
    if not all_items:
        print("오류: 시퀀스 데이터에서 아이템을 찾을 수 없습니다.")
        return None, None, None, None

    item_encoder = LabelEncoder()
    item_encoder.fit(all_items)
    # Vocabulary size for embedding layer: number of unique items + 1 for padding (0)
    num_unique_items_for_embedding = len(item_encoder.classes_) + 1

    # Transform sequences to encoded item indices (add 1 to all encoded indices to reserve 0 for padding)
    encoded_sequences = [item_encoder.transform(s) + 1 for s in sequences]

    X, y = [], []
    for seq in encoded_sequences:
        for i in range(1, len(seq)):
            input_seq = seq[:i] # Sequence up to item i-1
            target_item = seq[i] # Item i is the target
            X.append(input_seq)
            y.append(target_item)

    if not X: # No training samples could be generated
        print("오류: 학습 샘플(X,y 쌍)을 생성할 수 없습니다. 시퀀스 길이가 너무 짧을 수 있습니다.")
        return None, None, None, None

    X_padded = pad_sequences(X, maxlen=max_seq_len, padding='pre', truncating='pre', value=0)
    y_array = np.array(y) # These are already item indices (1 to N_items_actual)

    return X_padded, y_array, num_unique_items_for_embedding, item_encoder

# --- SASRec Model Components ---
class PositionalEmbedding(Layer):
    def __init__(self, max_seq_len, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim
        # input_dim is max_seq_len + 1 because positions can range from 1 to max_seq_len.
        # The 0th embedding could be for padding or a special token if needed,
        # but SASRec typically uses 1-based positions for actual items.
        self.pos_embeddings = Embedding(input_dim=max_seq_len + 1, output_dim=embed_dim)

    def call(self, x): # x is the item embedding sequence, shape (batch, seq_len, embed_dim)
        seq_len = tf.shape(x)[1]
        # Create position indices from 1 up to seq_len
        positions = tf.range(start=1, limit=seq_len + 1, delta=1)
        embedded_positions = self.pos_embeddings(positions)
        return embedded_positions # Shape: (seq_len, embed_dim) - will be broadcasted over batch

    def get_config(self):
        config = super().get_config()
        config.update({
            "max_seq_len": self.max_seq_len,
            "embed_dim": self.embed_dim,
        })
        return config

class TransformerBlock(Layer):
    def __init__(self, embed_dim, num_heads, ffn_units, dropout_rate, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ffn_units = ffn_units
        self.dropout_rate = dropout_rate

        if embed_dim % num_heads != 0:
            raise ValueError(f"임베딩 차원({embed_dim})은 헤드 수({num_heads})로 나누어 떨어져야 합니다.")

        self.mha = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim // num_heads, dropout=dropout_rate)
        self.ffn = tf.keras.Sequential([
            Dense(ffn_units, activation="relu"),
            Dropout(dropout_rate),
            Dense(embed_dim)
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(dropout_rate) # Dropout after MHA before Add&Norm
        # self.dropout2 = Dropout(dropout_rate) # Dropout is already in FFN's definition

    def call(self, inputs, training=False, attention_mask=None, causal_mask=None):
        # For SASRec, self-attention should be causal.
        # The MHA layer can create a causal mask if use_causal_mask=True (TF 2.8+).
        # Alternatively, a causal_mask can be passed.
        # Here, we assume the `attention_mask` might include padding mask and also causal properties.

        # If a combined mask is needed:
        # combined_mask = None
        # if attention_mask is not None and causal_mask is not None:
        #    combined_mask = tf.minimum(attention_mask, causal_mask) # Logical AND
        # elif attention_mask is not None:
        #    combined_mask = attention_mask
        # elif causal_mask is not None:
        #    combined_mask = causal_mask

        attn_output = self.mha(query=inputs, value=inputs, key=inputs, attention_mask=causal_mask, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)

        ffn_output = self.ffn(out1, training=training)
        # No explicit dropout here as it's inside self.ffn definition
        out2 = self.layernorm2(out1 + ffn_output)
        return out2

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ffn_units": self.ffn_units,
            "dropout_rate": self.dropout_rate,
        })
        return config

def build_sasrec_model(max_seq_len, num_items_for_embedding, embed_dim, num_blocks, num_heads, ffn_units, dropout_rate):
    input_seq = Input(shape=(max_seq_len,), name="input_sequence", dtype='int32')

    item_embedding_layer = Embedding(input_dim=num_items_for_embedding, output_dim=embed_dim, name="item_embedding", mask_zero=True)
    item_embs = item_embedding_layer(input_seq) # (batch, seq_len, embed_dim)

    pos_embedding_layer = PositionalEmbedding(max_seq_len, embed_dim)
    pos_embs = pos_embedding_layer(item_embs) # (seq_len, embed_dim)

    seq_embs = item_embs + pos_embs # Add positional encoding (broadcasting pos_embs over batch)
    seq_embs = Dropout(dropout_rate)(seq_embs)

    # Causal mask for self-attention
    # MHA expects mask shape (batch_size, Tq, Tv) or (batch_size, num_heads, Tq, Tv)
    # Tq = target sequence length (output), Tv = source sequence length (input)
    # For self-attention, Tq = Tv = max_seq_len. Use static max_seq_len for mask dimensions.
    seq_len = max_seq_len

    # Causal mask: True for positions that should be masked (future positions).
    # 1. Create a lower triangular matrix (diagonal and below are 1s, rest 0s).
    lower_triangular = tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
    # 2. Invert it so that upper triangular (future positions) are 1s (True) and others 0s (False).
    causal_attention_mask = tf.cast(1.0 - lower_triangular, dtype=tf.bool)
    # Reshape for MHA if needed, e.g. (1, seq_len, seq_len) or (1, 1, seq_len, seq_len)
    # Keras MHA can often broadcast from (Tq, Tv). Let's keep it (seq_len, seq_len) first.
    # No, MHA expects at least 3D for batch broadcasting: (batch_size, Tq, Tv)
    # So, (1, seq_len, seq_len) is fine for broadcasting across batch & heads.
    causal_attention_mask = causal_attention_mask[tf.newaxis, :, :]


    # Padding mask from the embedding layer
    padding_mask = item_embedding_layer.compute_mask(input_seq) # Shape (batch_size, max_seq_len)
    # Expand dims for MHA: (batch_size, 1, 1, max_seq_len) for from_storage_mask or (batch_size, 1, max_seq_len, 1) for to_storage_mask
    # Or let MHA use it directly if it supports boolean masks of shape (batch_size, seq_len) to mask tokens.
    # Keras MHA typically expects the attention_mask to be (batch_size, Tq, Tv).
    # If padding_mask is (batch_size, Tv), expand it to (batch_size, 1, Tv) or (batch_size, Tq, Tv) by repeating.
    # For this example, we'll pass the causal mask. MHA should also respect the mask from `mask_zero=True`.
    # Simpler: Keras MHA uses the mask from `mask_zero=True` automatically.
    # We only need to ensure the causal nature.
    # The causal mask should make future positions unable to be attended to.
    # So, for a query q_i, it can only attend to keys k_j where j <= i.
    # The `causal_mask` created (lower triangular) allows this.
    # MHA needs `True` for positions NOT to attend. So, `1 - causal_mask` for upper triangle.
    # Let's use tf.newaxis for broadcasting: causal_mask[tf.newaxis, tf.newaxis, :, :] for (batch, num_heads, Tq, Tv)
    # For now, let's pass the lower triangular. MHA layer with `use_causal_mask=True` in TF>2.8 is easier.
    # The provided MHA layer in TF versions might handle boolean masks (batch_size, T_q, T_v) where True means "masked out".
    # So, we want `mask[i, j] = True` if `j > i`. This is an upper triangular matrix (excluding diagonal).
    # `tf.linalg.band_part(tf.ones((L, L)), 0, -1)` gives upper triangular. `tf.linalg.band_part(tf.ones((L,L)), -1, 0)` gives lower.
    # `1.0 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)` -> upper triangle (excluding diag) has 1s.
    # `tf.cast(1.0 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0), dtype=tf.bool)`
    # This is getting complex. A simpler way for this example if full causal masking in MHA is tricky:
    # We will rely on the fact that we only use the *last output* for prediction.

    transformer_output = seq_embs
    for i in range(num_blocks):
        # In a real SASRec, a proper causal mask is crucial here.
        # Keras MHA layer uses the mask from `mask_zero=True` automatically for padding.
        # For causality, if not using MHA's built-in causal option, it's complex.
        # We'll proceed without explicit causal mask in MHA for this example's simplicity, (REMOVED THIS COMMENT)
        # The causal_attention_mask is now passed to the TransformerBlock.
        # The MHA layer in TransformerBlock will use this for causal attention.
        # The padding mask from `mask_zero=True` in Embedding layer is automatically handled by Keras MHA.
        transformer_output = TransformerBlock(embed_dim, num_heads, ffn_units, dropout_rate, name=f"transformer_block_{i}")(transformer_output, causal_mask=causal_attention_mask)

    last_item_representation = transformer_output[:, -1, :]

    logits = Dense(num_items_for_embedding, activation=None, name="output_logits")(last_item_representation)

    model = Model(inputs=input_seq, outputs=logits)
    return model

def get_sasrec_recommendations(model, input_sequence_encoded_1based, item_encoder, num_items_embedding_dim, num_recommendations=5):
    """Generates next-item recommendations for a given 1-based encoded input sequence."""
    if not input_sequence_encoded_1based: # Check if list is empty
        print("입력 시퀀스가 비어있어 추천을 생성할 수 없습니다.")
        return []

    padded_input_sequence = pad_sequences([input_sequence_encoded_1based], maxlen=MAX_SEQ_LENGTH, padding='pre', truncating='pre', value=0)

    predicted_scores = model.predict(padded_input_sequence, verbose=0)[0]

    # Mask items already in the input sequence and padding token 0
    predicted_scores[0] = -np.inf # Mask padding token
    for item_idx_1based in input_sequence_encoded_1based:
        if 0 < item_idx_1based < len(predicted_scores): # item_idx_1based is already the index for predicted_scores
             predicted_scores[item_idx_1based] = -np.inf

    top_n_item_indices_1based = np.argsort(predicted_scores)[-num_recommendations:][::-1]

    recommended_items = []
    for item_idx_1based in top_n_item_indices_1based:
        if item_idx_1based == 0: continue # Skip padding item
        # item_idx_1based is the value that was fed into embedding (1 to N_actual_items)
        # item_encoder maps original_id to 0 to N_actual_items-1
        original_item_id = item_encoder.inverse_transform([item_idx_1based -1])[0]
        recommended_items.append({'item_id': original_item_id, 'score': predicted_scores[item_idx_1based]})

    return recommended_items

# --- Main Execution ---
if __name__ == "__main__":
    print("--- SASRec Sequential Recommender 예제 시작 ---")

    print("\n데이터 로드 및 전처리 중...")
    X_padded, y_array, num_items_for_embedding, item_encoder = load_sequences()

    if X_padded is not None and item_encoder is not None:
        print(f"전처리된 입력 시퀀스 shape: {X_padded.shape}")
        print(f"타겟 아이템 배열 shape: {y_array.shape}")
        print(f"임베딩을 위한 고유 아이템 수 (패딩 포함): {num_items_for_embedding}")

        print("\nSASRec 모델 구축 중...")
        model = build_sasrec_model(MAX_SEQ_LENGTH, num_items_for_embedding, EMBEDDING_DIM, NUM_BLOCKS, NUM_HEADS, FFN_UNITS, DROPOUT_RATE)

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])
        model.summary()

        print("\n모델 학습 중...")
        history = model.fit(X_padded, y_array, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)
        print("모델 학습 완료.")

        if hasattr(item_encoder, 'classes_') and len(item_encoder.classes_) > 0:
            raw_sequences_df_path = os.path.join(project_root, 'data/dummy_sequences.csv')
            if os.path.exists(raw_sequences_df_path):
                raw_sequences_df = pd.read_csv(raw_sequences_df_path)
                if not raw_sequences_df.empty:
                    example_raw_sequence_str = raw_sequences_df['item_ids_sequence'].iloc[0]
                    example_raw_sequence_orig_ids = [int(i) for i in example_raw_sequence_str.split(' ')]

                    known_items_in_sequence_orig_ids = [item for item in example_raw_sequence_orig_ids if item in item_encoder.classes_]

                    if len(known_items_in_sequence_orig_ids) >= MAX_SEQ_LENGTH // 2 and len(known_items_in_sequence_orig_ids) > 0 :
                        # Use a portion of the sequence, e.g., last few items up to MAX_SEQ_LENGTH-1
                        input_for_recs_orig_ids = known_items_in_sequence_orig_ids[-(MAX_SEQ_LENGTH-1):]
                        input_for_recs_encoded_1based = item_encoder.transform(input_for_recs_orig_ids) + 1

                        print(f"\n입력 시퀀스 (원본 ID): {input_for_recs_orig_ids} -> (인코딩된 1-based ID): {input_for_recs_encoded_1based.tolist()}")

                        recommendations = get_sasrec_recommendations(model, input_for_recs_encoded_1based.tolist(), item_encoder, num_items_for_embedding)

                        print("\n추천된 다음 아이템:")
                        if recommendations:
                            for rec in recommendations:
                                print(f"- 아이템 {rec['item_id']}: 예측 점수 {rec['score']:.2f}")
                        else:
                            print("추천할 아이템이 없습니다.")
                    else:
                        print("\n예시 추천을 위한 충분한 길이의 사용자 시퀀스를 찾을 수 없거나, 시퀀스 내 아이템이 인코더에 없습니다.")
                else:
                    print("\n시퀀스 파일이 비어있어 예시 추천을 생성할 수 없습니다.")
            else:
                print(f"\n시퀀스 파일({raw_sequences_df_path})을 찾을 수 없어 예시 추천을 생성할 수 없습니다.")
        else:
            print("\n아이템 인코더가 준비되지 않아 예시 추천을 생성할 수 없습니다.")
    else:
        print("\n데이터 로드에 실패하여 예제를 실행할 수 없습니다.")

    print("\n--- SASRec 예제 실행 완료 ---")
