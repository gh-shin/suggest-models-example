import pandas as pd
import numpy as np
import os

# Dynamically add project root to sys.path if this script is run directly
# and needs to be importable or import other project modules (not strictly needed for this script itself).
# However, example scripts will add project_root to sys.path when they import this.
# So, this is mostly for completeness if running this script standalone AND it had complex local imports.
# current_dir = os.path.dirname(os.path.abspath(__file__))
# project_root_from_data = os.path.abspath(os.path.join(current_dir, '..'))
# if project_root_from_data not in sys.path:
#     sys.path.append(project_root_from_data)

def generate_dummy_data(num_users=100, num_items=50, num_interactions=1000, generate_sequences=False):
    """
    Generates dummy interaction data, item metadata, and optionally, user interaction sequences.
    All generated files are saved in the 'data/' directory relative to the project root.
    The 'data/' directory is created if it doesn't exist.
    """
    # Determine the absolute path for the 'data' directory
    # Assumes this script is in 'project_root/data/'
    # Or that project_root is added to sys.path and 'data' is a module path.
    # For robustness when called from other scripts (which add project_root to sys.path),
    # let's make paths relative to a known 'data' dir.

    # Base path for saving data. If script is in data/, this is '../data'.
    # If this script is called after project root is added to sys.path,
    # then 'data/' will resolve correctly from project root.
    data_folder = 'data'
    if not os.path.exists(data_folder):
        # This handles calls from tests or scripts in examples/ where CWD might be project root
        os.makedirs(data_folder, exist_ok=True)


    user_ids = np.random.randint(1, num_users + 1, num_interactions)
    item_ids = np.random.randint(1, num_items + 1, num_interactions)
    # Add a dummy timestamp to help with sorting for sequence generation
    # Timestamps are just integers from 1 to num_interactions, roughly sorted per user after grouping
    timestamps = np.arange(1, num_interactions + 1)


    df_interactions_raw = pd.DataFrame({
        'user_id': user_ids,
        'item_id': item_ids,
        'rating': np.random.randint(1, 6, num_interactions), # 1-5 평점
        'timestamp': timestamps # Add timestamp
    })

    # Sort by user and timestamp to make sequences more 'natural' if generated from this
    df_interactions = df_interactions_raw.sort_values(by=['user_id', 'timestamp']).reset_index(drop=True)

    interactions_path = os.path.join(data_folder, 'dummy_interactions.csv')
    df_interactions[['user_id', 'item_id', 'rating']].to_csv(interactions_path, index=False) # Save without timestamp for compatibility
    print(f"더미 상호작용 데이터 생성이 완료되었습니다: {interactions_path} ({len(df_interactions)} 행)")

    # Generate dummy item metadata
    item_meta_ids = np.arange(1, num_items + 1)
    genres = ['genreA', 'genreB', 'genreC', 'genreD', 'genreE']
    item_genres = [';'.join(np.random.choice(genres, size=np.random.randint(1, 4), replace=False)) for _ in range(num_items)]
    item_descriptions = [f"This is a description for item {i} belonging to genres {g.replace(';', ', ')}." for i, g in zip(item_meta_ids, item_genres)]

    df_items = pd.DataFrame({
        'item_id': item_meta_ids,
        'genres': item_genres,
        'description': item_descriptions
    })
    items_path = os.path.join(data_folder, 'dummy_item_metadata.csv')
    df_items.to_csv(items_path, index=False)
    print(f"더미 아이템 메타데이터 생성이 완료되었습니다: {items_path} ({len(df_items)} 행)")

    if generate_sequences:
        # df_interactions already sorted by user_id, timestamp
        # Group by user_id and aggregate item_ids into a space-separated string sequence
        # Using the already sorted df_interactions ensures sequences respect the timestamp order
        sequences_df = df_interactions.groupby('user_id')['item_id'].apply(
            lambda x: ' '.join(map(str, x))
        ).reset_index()
        sequences_df.rename(columns={'item_id': 'item_ids_sequence'}, inplace=True)

        sequences_path = os.path.join(data_folder, 'dummy_sequences.csv')
        sequences_df.to_csv(sequences_path, index=False)
        print(f"더미 시퀀스 데이터 생성이 완료되었습니다: {sequences_path} ({len(sequences_df)} 행)")


if __name__ == "__main__":
    # Example of how to call it, including sequence generation for testing that part
    print("기본 더미 데이터 생성 중 (시퀀스 파일 포함)...")
    generate_dummy_data(num_users=100, num_items=50, num_interactions=1000, generate_sequences=True)
    print("\n기본 더미 데이터 생성 완료.")
    # To generate only interactions and metadata (old behavior by default):
    # print("상호작용 및 메타데이터만 생성 (시퀀스 제외)...")
    # generate_dummy_data(num_users=50, num_items=30, num_interactions=200, generate_sequences=False)
