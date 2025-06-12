# tests/test_deep_learning.py
import subprocess
import sys
import os
import pytest

# Dynamically add project root to sys.path to allow examples to import 'data' package
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def test_run_dnn_recommender_example():
    """
    Test that the dnn_recommender.py script runs successfully and produces output.
    """
    script_path_rel = "examples/deep_learning/dnn_recommender.py"
    script_path_abs = os.path.join(project_root, script_path_rel)
    assert os.path.exists(script_path_abs), f"Script not found: {script_path_abs}"

    data_files_to_clean = [
        os.path.join(project_root, 'data/dummy_interactions.csv'),
        os.path.join(project_root, 'data/dummy_item_metadata.csv'),
        os.path.join(project_root, 'data/dummy_sequences.csv')
    ]
    for f_path in data_files_to_clean:
        if os.path.exists(f_path):
            os.remove(f_path)
            print(f"INFO: Removed {f_path} before running {script_path_rel}")

    env = os.environ.copy()
    env["TF_CPP_MIN_LOG_LEVEL"] = "2" # Suppress TensorFlow INFO and WARNING logs
    env["KMP_DUPLICATE_LIB_OK"]="TRUE" # Allow duplicate OpenMP libraries, common in TF setups

    data_dir = os.path.join(project_root, 'data')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"INFO: Created data directory at {data_dir} for test run.")

    process = subprocess.run(
        [sys.executable, script_path_abs],
        capture_output=True,
        text=True,
        check=False,
        env=env,
        cwd=project_root
    )

    if process.returncode != 0:
        print(f"--- STDOUT for {script_path_rel} ---")
        print(process.stdout)
        print(f"--- STDERR for {script_path_rel} ---")
        print(process.stderr)

    assert process.returncode == 0, f"Script {script_path_rel} failed with error code {process.returncode}.\nStderr:\n{process.stderr}"
    assert len(process.stdout) > 0, f"Script {script_path_rel} produced no stdout output."

    assert "DNN Based Recommendation Example Finished" in process.stdout, \
        f"Script {script_path_rel} did not contain specific completion message in stdout.\nStdout:\n{process.stdout}"
    assert "MAE =" in process.stdout, \
        f"Script {script_path_rel} did not contain 'MAE =' in stdout.\nStdout:\n{process.stdout}"

    generic_keywords_present = (
        "추천 아이템 목록" in process.stdout or
        "recommendation list" in process.stdout.lower() or
        "예상 평점" in process.stdout or
        "predicted_rating" in process.stdout.lower()
    )
    assert generic_keywords_present, \
        f"Script {script_path_rel} did not contain expected recommendation keywords in stdout.\nStdout:\n{process.stdout}"
