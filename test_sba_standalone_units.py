"""Unit tests for the standalone SBA modules.

Ported from the CASCADES repo (tests/test_sba_validation_artifacts.py) when SBA
was split into this standalone repository. Run from the repo root: `pytest`.
"""
from pathlib import Path

from argparse import Namespace

from benchmark_sba_implementations import benchmark_arms, default_case_specs, summarize_results
from run_scratch_compare import ScratchConfig, checkpoint_path_for_step, should_save_checkpoint


def test_scratch_checkpoint_policy_saves_final_and_interval_steps():
    args = Namespace(save_checkpoints=True, checkpoint_every=100)
    assert should_save_checkpoint(step=1, final_step=250, args=args) is False
    assert should_save_checkpoint(step=100, final_step=250, args=args) is True
    assert should_save_checkpoint(step=250, final_step=250, args=args) is True

    disabled = Namespace(save_checkpoints=False, checkpoint_every=100)
    assert should_save_checkpoint(step=250, final_step=250, args=disabled) is False


def test_scratch_checkpoint_path_records_attention_and_step(tmp_path):
    config = ScratchConfig(block_size=128, n_layer=2, n_head=4, n_embd=128)
    path = checkpoint_path_for_step(Path(tmp_path), "sba", step=400, config=config)
    assert path == Path(tmp_path) / "sba" / "checkpoints" / "sba_step000400_2l_4h_128d.pt"


def test_sba_benchmark_arms_cover_softmax_eager_and_sdpa():
    assert benchmark_arms() == [
        {"name": "softmax", "attention": "softmax", "sba_impl": None},
        {"name": "sba_eager", "attention": "sba", "sba_impl": "eager"},
        {"name": "sba_sdpa", "attention": "sba", "sba_impl": "sdpa"},
    ]


def test_sba_benchmark_default_cases_are_bounded_and_include_realistic_local_probe():
    specs = default_case_specs()
    assert specs[0]["seq_len"] == 128
    assert max(int(spec["seq_len"]) for spec in specs) <= 1024
    assert max(int(spec["batch_size"]) for spec in specs) <= 4
    assert any(
        int(spec["n_layer"]) == 4
        and int(spec["n_head"]) == 8
        and int(spec["n_embd"]) == 256
        and int(spec["seq_len"]) >= 512
        for spec in specs
    )


def test_sba_benchmark_summary_ratios_only_use_successful_pairs():
    results = [
        {"case": "case_a", "arm": "softmax", "status": "success", "time_ms_mean": 10.0, "peak_allocated_mb": 100.0},
        {"case": "case_a", "arm": "sba_eager", "status": "success", "time_ms_mean": 20.0, "peak_allocated_mb": 200.0},
        {"case": "case_a", "arm": "sba_sdpa", "status": "success", "time_ms_mean": 12.0, "peak_allocated_mb": 120.0},
        {"case": "case_b", "arm": "softmax", "status": "oom", "time_ms_mean": None, "peak_allocated_mb": None},
        {"case": "case_b", "arm": "sba_eager", "status": "success", "time_ms_mean": 30.0, "peak_allocated_mb": 300.0},
        {"case": "case_b", "arm": "sba_sdpa", "status": "success", "time_ms_mean": 15.0, "peak_allocated_mb": 150.0},
    ]
    summary = summarize_results(results)
    assert summary["successful_cases_by_arm"] == {"softmax": 1, "sba_eager": 2, "sba_sdpa": 2}
    assert summary["ratios"]["case_a"]["sba_sdpa_vs_sba_eager_time"] == 0.6
    assert summary["ratios"]["case_a"]["sba_sdpa_vs_softmax_time"] == 1.2
    assert summary["ratios"]["case_b"]["sba_sdpa_vs_sba_eager_time"] == 0.5
    assert "sba_sdpa_vs_softmax_time" not in summary["ratios"]["case_b"]
