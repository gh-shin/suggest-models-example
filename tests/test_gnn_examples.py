# tests/test_gnn_examples.py
import subprocess
import sys
import os
import pytest

# Dynamically add project root to sys.path to allow examples to import 'data' package
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Placeholder scripts, for which generic keyword checks might be skipped
placeholder_gnn_scripts = [
    "examples/gnn/ngcf_example.py",
    "examples/gnn/pinsage_example.py",
    "examples/gnn/gcn_example.py",
    "examples/gnn/graphsage_example.py",
    "examples/gnn/gat_example.py",
]

def _run_script_test(script_path_rel, specific_output_check, is_tf_script=False):
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
            # print(f"INFO: Removed {f_path} before running {script_path_rel}") # Optional: for verbosity

    env = os.environ.copy()
    if is_tf_script:
        env["TF_CPP_MIN_LOG_LEVEL"] = "2"
        env["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    data_dir = os.path.join(project_root, 'data')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        # print(f"INFO: Created data directory at {data_dir} for test run.") # Optional: for verbosity

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
    assert specific_output_check in process.stdout, \
        f"Script {script_path_rel} did not contain specific message: '{specific_output_check}'.\nStdout:\n{process.stdout}"

    if script_path_rel not in placeholder_gnn_scripts:
        generic_keywords_present = (
            "추천 아이템 목록" in process.stdout or
            "recommendation list" in process.stdout.lower() or
            "예상 평점" in process.stdout or
            "predicted_rating" in process.stdout.lower() or
            "추천된 아이템" in process.stdout # Common for LightGCN
        )
        assert generic_keywords_present, \
            f"Script {script_path_rel} did not contain expected recommendation keywords in stdout.\nStdout:\n{process.stdout}"


def test_run_lightgcn_tf_example():
    _run_script_test(
        "examples/gnn/lightgcn_tf_example.py",
        "LightGCN (TensorFlow/Keras) 예제 실행 완료",
        is_tf_script=True
    )

def test_run_ngcf_example():
    _run_script_test(
        "examples/gnn/ngcf_example.py",
        "NGCF (Neural Graph Collaborative Filtering) Example - Conceptual Outline" # Updated string
    )

def test_run_pinsage_example():
    _run_script_test(
        "examples/gnn/pinsage_example.py",
        "PinSage (Graph Convolutional Neural Networks for Web-Scale RecSys) - Conceptual Outline" # Updated string
    )

def test_run_gcn_example():
    _run_script_test(
        "examples/gnn/gcn_example.py",
        "GCN (Graph Convolutional Network) for Recommendations - Conceptual Outline" # Updated string
    )

def test_run_graphsage_example():
    _run_script_test(
        "examples/gnn/graphsage_example.py",
        "GraphSAGE (Inductive Representation Learning on Large Graphs) - Conceptual Outline" # Updated string
    )

def test_run_gat_example():
    _run_script_test(
        "examples/gnn/gat_example.py",
        "GAT (Graph Attention Network) for Recommendations - Conceptual Outline" # Updated string
    )
