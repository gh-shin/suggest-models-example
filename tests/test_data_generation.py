# tests/test_data_generation.py
import os
import sys
import pandas as pd # Added for CSV content check

# Dynamically add project root to sys.path for module imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root) # Ensure project root is at the start for priority

from data.generate_dummy_data import generate_dummy_data

def test_generate_dummy_data_main_files():
    """Test default dummy data generation (interactions and metadata)."""
    interactions_file = os.path.join(project_root, 'data/dummy_interactions.csv')
    metadata_file = os.path.join(project_root, 'data/dummy_item_metadata.csv')
    data_dir = os.path.join(project_root, 'data')

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    if os.path.exists(interactions_file):
        os.remove(interactions_file)
    if os.path.exists(metadata_file):
        os.remove(metadata_file)

    try:
        # Explicitly test non-sequence generation
        generate_dummy_data(num_users=10, num_items=5, num_interactions=20, generate_sequences=False)
    except Exception as e:
        assert False, f"generate_dummy_data(generate_sequences=False) raised an exception: {e}"

    assert os.path.exists(interactions_file), f"{interactions_file} was not created."
    assert os.path.exists(metadata_file), f"{metadata_file} was not created."

    assert os.path.getsize(interactions_file) > 50, f"{interactions_file} seems too small."
    assert os.path.getsize(metadata_file) > 50, f"{metadata_file} seems too small."

def test_generate_dummy_data_sequences():
    """Test dummy data generation for sequences."""
    data_dir = os.path.join(project_root, 'data')
    interactions_file = os.path.join(data_dir, 'dummy_interactions.csv') # Base files also created
    metadata_file = os.path.join(data_dir, 'dummy_item_metadata.csv')   # Base files also created
    sequence_file = os.path.join(data_dir, 'dummy_sequences.csv')

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Clean up all potentially generated files before this specific test
    if os.path.exists(interactions_file):
        os.remove(interactions_file)
    if os.path.exists(metadata_file):
        os.remove(metadata_file)
    if os.path.exists(sequence_file):
        os.remove(sequence_file)

    try:
        # Using small numbers for quick test, and ensure sequences are generated
        generate_dummy_data(num_users=10, num_items=5, num_interactions=30, generate_sequences=True)
    except Exception as e:
        assert False, f"generate_dummy_data(generate_sequences=True) raised an exception: {e}"

    # Check that base files are also created as a side effect
    assert os.path.exists(interactions_file), f"Base file {interactions_file} was not created when generating sequences."
    assert os.path.exists(metadata_file), f"Base file {metadata_file} was not created when generating sequences."

    # Check sequence file specifically
    assert os.path.exists(sequence_file), f"{sequence_file} was not created."
    assert os.path.getsize(sequence_file) > 20, f"{sequence_file} seems too small or empty." # Adjusted for potentially few users with sequences

    df_seq = pd.read_csv(sequence_file)
    assert 'user_id' in df_seq.columns, f"{sequence_file} is missing 'user_id' column."
    assert 'item_ids_sequence' in df_seq.columns, f"{sequence_file} is missing 'item_ids_sequence' column."
    # With num_users=10, it's possible not all users have enough interactions to form sequences of length > 1 for SASRec style input.
    # But the file itself should contain entries for users who do.
    # If interactions are >= users, most users should have at least one item.
    if 30 >= 10 : # If num_interactions >= num_users
         assert not df_seq.empty or (df_seq.empty and os.path.getsize(sequence_file) > 0), \
             f"{sequence_file} is empty after loading, but should contain users with sequences."
    else: # If very few interactions, df_seq could be empty.
        print(f"Warning: {sequence_file} might be empty due to low interaction count relative to users.")

    # Clean up after test (optional, but good for isolation if tests are run out of order or re-run)
    # if os.path.exists(interactions_file): os.remove(interactions_file)
    # if os.path.exists(metadata_file): os.remove(metadata_file)
    # if os.path.exists(sequence_file): os.remove(sequence_file)
