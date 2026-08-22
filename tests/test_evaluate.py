from runpy import run_path

import pandas as pd

from gso.collect.execute import evaluate


def test_select_pid_commits_defaults_to_1_1_speedup():
    dataframe = pd.DataFrame(
        [
            {"pid": "repo-pass", "commit": "aaaaaaa", "speedup_factor": 1.10},
            {"pid": "repo-low", "commit": "bbbbbbb", "speedup_factor": 1.09},
        ]
    )

    assert evaluate.select_pid_commits(dataframe) == [("repo-pass", "aaaaaaa")]


def test_select_pid_commits_filters_deduplicates_and_sorts():
    dataframe = pd.DataFrame(
        [
            {"pid": "repo-z", "commit": "bbbbbbb", "speedup_factor": 1.30},
            {"pid": "repo-a", "commit": "aaaaaaa", "speedup_factor": 1.25},
            {"pid": "repo-a", "commit": "aaaaaaa", "speedup_factor": 1.40},
            {"pid": "repo-low", "commit": "ccccccc", "speedup_factor": 1.19},
        ]
    )

    assert evaluate.select_pid_commits(dataframe, 1.2) == [
        ("repo-a", "aaaaaaa"),
        ("repo-z", "bbbbbbb"),
    ]


def test_write_pids_config_creates_loadable_python(tmp_path):
    output_path = tmp_path / "demo_pids.py"

    evaluate.write_pids_config(
        "demo",
        [("demo-api", "abcdef1")],
        output_path,
    )

    namespace = run_path(str(output_path))
    assert namespace["TEST_PROBLEMS"] == {"demo": [("demo-api", "abcdef1")]}
    assert namespace["LONG_RUNNING_PROBLEMS"] == []


def test_build_evaluated_dataset_exports_pids_and_invokes_builder(
    tmp_path, monkeypatch
):
    from gso.collect import build_dataset

    experiments_dir = tmp_path / "experiments"
    datasets_dir = tmp_path / "datasets"
    monkeypatch.setattr(evaluate, "EXPS_DIR", experiments_dir)
    monkeypatch.setattr(evaluate, "DATASET_DIR", datasets_dir)
    calls = []

    def fake_build_dataset_main(**kwargs):
        calls.append(kwargs)
        datasets_dir.mkdir(parents=True)
        (datasets_dir / "gso_demo_dataset.jsonl").write_text(
            '{"instance_id":"demo"}\n', encoding="utf-8"
        )

    monkeypatch.setattr(build_dataset, "main", fake_build_dataset_main)
    dataframe = pd.DataFrame(
        [{"pid": "demo-api", "commit": "abcdef1", "speedup_factor": 1.3}]
    )

    result = evaluate.build_evaluated_dataset(
        exp_id="demo",
        dataframe=dataframe,
        backend="docker",
        results_file=None,
        pids_output=None,
        dataset_name=None,
        min_speedup_factor=1.2,
    )

    pids_path = experiments_dir / "demo" / "demo_pids.py"
    dataset_path = datasets_dir / "gso_demo_dataset.jsonl"
    assert result == (pids_path, dataset_path)
    assert run_path(str(pids_path))["TEST_PROBLEMS"] == {
        "demo": [("demo-api", "abcdef1")]
    }
    assert calls == [
        {
            "exp_id": "demo",
            "push_to_hf": False,
            "hf_username": None,
            "dataset_name": "gso_demo",
            "debug": False,
            "backend": "docker",
            "results_file": None,
            "pids_file": str(pids_path),
            "min_speedup_factor": 1.2,
        }
    ]
