from __future__ import annotations

import argparse
import platform
import sys
import time
import traceback
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from artifacts import git_revision, utc_now, write_json
from run_scratch_compare import ScratchConfig, ScratchLM, count_parameters


CASE_SPECS: list[dict[str, int | str]] = [
    {
        "name": "tiny_b1_s2_1l_2h_16d",
        "batch_size": 1,
        "seq_len": 2,
        "block_size": 4,
        "n_layer": 1,
        "n_head": 2,
        "n_embd": 16,
    },
    {
        "name": "small_b2_s5_2l_4h_32d",
        "batch_size": 2,
        "seq_len": 5,
        "block_size": 8,
        "n_layer": 2,
        "n_head": 4,
        "n_embd": 32,
    },
    {
        "name": "longer_b3_s9_2l_4h_32d",
        "batch_size": 3,
        "seq_len": 9,
        "block_size": 12,
        "n_layer": 2,
        "n_head": 4,
        "n_embd": 32,
    },
]

TOLERANCES: dict[str, dict[str, float]] = {
    "cpu:float32": {
        "logits_max_abs": 2e-5,
        "loss_abs": 2e-6,
        "grad_max_abs": 2e-5,
    },
    "cuda:float32": {
        "logits_max_abs": 5e-4,
        "loss_abs": 5e-5,
        "grad_max_abs": 5e-4,
    },
    "cuda:bfloat16": {
        "logits_max_abs": 8e-2,
        "loss_abs": 8e-3,
        "grad_max_abs": 8e-2,
    },
    "cuda:float16": {
        "logits_max_abs": 5e-2,
        "loss_abs": 5e-3,
        "grad_max_abs": 5e-2,
    },
}


def _dtype_for_name(dtype_name: str) -> torch.dtype:
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "float32":
        return torch.float32
    raise ValueError(f"unsupported dtype_name {dtype_name!r}")


def _environment_info() -> dict[str, Any]:
    cuda_devices: list[dict[str, Any]] = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            cuda_devices.append(
                {
                    "index": index,
                    "name": props.name,
                    "capability": list(torch.cuda.get_device_capability(index)),
                    "total_memory_mb": props.total_memory / (1024 * 1024),
                }
            )
    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "cuda_devices": cuda_devices,
        "cuda_bf16_supported": torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        "cuda_allow_tf32_before_script": torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else None,
        "cudnn_allow_tf32_before_script": torch.backends.cudnn.allow_tf32 if torch.cuda.is_available() else None,
    }


def _build_run_matrix() -> tuple[list[dict[str, str | None]], list[dict[str, str]]]:
    matrix: list[dict[str, str | None]] = [{"device": "cpu", "dtype": "float32", "autocast_dtype": None}]
    skipped: list[dict[str, str]] = []
    if not torch.cuda.is_available():
        skipped.append({"device": "cuda", "dtype": "float32", "reason": "torch.cuda.is_available() is false"})
        skipped.append({"device": "cuda", "dtype": "float16", "reason": "torch.cuda.is_available() is false"})
        skipped.append({"device": "cuda", "dtype": "bfloat16", "reason": "torch.cuda.is_available() is false"})
        return matrix, skipped

    matrix.append({"device": "cuda", "dtype": "float32", "autocast_dtype": None})
    matrix.append({"device": "cuda", "dtype": "float16", "autocast_dtype": "float16"})
    if torch.cuda.is_bf16_supported():
        matrix.append({"device": "cuda", "dtype": "bfloat16", "autocast_dtype": "bfloat16"})
    else:
        skipped.append({"device": "cuda", "dtype": "bfloat16", "reason": "torch.cuda.is_bf16_supported() is false"})
    return matrix, skipped


def _max_abs_diff(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.detach().float() - right.detach().float()).abs().max().item())


def _all_finite_tensor(value: torch.Tensor) -> bool:
    return bool(torch.isfinite(value.detach()).all().item())


def _make_config(spec: dict[str, int | str]) -> ScratchConfig:
    return ScratchConfig(
        block_size=int(spec["block_size"]),
        n_layer=int(spec["n_layer"]),
        n_head=int(spec["n_head"]),
        n_embd=int(spec["n_embd"]),
        dropout=0.0,
    )


def _autocast_context(device_type: str, autocast_dtype_name: str | None):
    if device_type != "cuda" or autocast_dtype_name is None:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=_dtype_for_name(autocast_dtype_name))


def _case_inputs(config: ScratchConfig, batch_size: int, seq_len: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), generator=generator, dtype=torch.long)
    labels = torch.randint(0, config.vocab_size, (batch_size, seq_len), generator=generator, dtype=torch.long)
    return input_ids, labels


def run_equivalence_case(
    *,
    case_spec: dict[str, int | str],
    run_spec: dict[str, str | None],
    seed: int,
) -> dict[str, Any]:
    device_type = str(run_spec["device"])
    dtype_name = str(run_spec["dtype"])
    autocast_dtype_name = run_spec["autocast_dtype"]
    tolerance_key = f"{device_type}:{dtype_name}"
    tolerances = TOLERANCES[tolerance_key]
    device = torch.device(device_type)
    config = _make_config(case_spec)
    batch_size = int(case_spec["batch_size"])
    seq_len = int(case_spec["seq_len"])

    torch.manual_seed(seed)
    eager = ScratchLM(config, attention="sba", sba_impl="eager")
    sdpa = ScratchLM(config, attention="sba", sba_impl="sdpa")
    sdpa.load_state_dict(eager.state_dict())
    eager.to(device)
    sdpa.to(device)
    eager.train()
    sdpa.train()

    input_ids, labels = _case_inputs(config, batch_size=batch_size, seq_len=seq_len, seed=seed + 10_000)
    input_ids = input_ids.to(device)
    labels = labels.to(device)

    if device_type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    eager.zero_grad(set_to_none=True)
    sdpa.zero_grad(set_to_none=True)
    with _autocast_context(device_type, autocast_dtype_name):
        eager_logits, eager_loss = eager(input_ids, labels)
        sdpa_logits, sdpa_loss = sdpa(input_ids, labels)
    if eager_loss is None or sdpa_loss is None:
        raise RuntimeError("loss unexpectedly missing")
    eager_loss.backward()
    sdpa_loss.backward()
    if device_type == "cuda":
        torch.cuda.synchronize(device)

    grad_details: list[dict[str, Any]] = []
    grad_max_abs = 0.0
    finite_gradients = True
    missing_gradients: list[str] = []
    for (eager_name, eager_param), (sdpa_name, sdpa_param) in zip(eager.named_parameters(), sdpa.named_parameters(), strict=True):
        if eager_name != sdpa_name:
            raise RuntimeError(f"parameter order mismatch: {eager_name!r} != {sdpa_name!r}")
        if eager_param.grad is None or sdpa_param.grad is None:
            missing_gradients.append(eager_name)
            finite_gradients = False
            continue
        finite_gradients = finite_gradients and _all_finite_tensor(eager_param.grad) and _all_finite_tensor(sdpa_param.grad)
        diff = _max_abs_diff(eager_param.grad, sdpa_param.grad)
        grad_max_abs = max(grad_max_abs, diff)
        grad_details.append(
            {
                "name": eager_name,
                "max_abs_diff": diff,
                "eager_grad_max_abs": float(eager_param.grad.detach().float().abs().max().item()),
                "sdpa_grad_max_abs": float(sdpa_param.grad.detach().float().abs().max().item()),
            }
        )

    logits_max_abs = _max_abs_diff(eager_logits, sdpa_logits)
    loss_abs = abs(float(eager_loss.detach().float().item()) - float(sdpa_loss.detach().float().item()))
    finite_outputs = (
        _all_finite_tensor(eager_logits)
        and _all_finite_tensor(sdpa_logits)
        and _all_finite_tensor(eager_loss)
        and _all_finite_tensor(sdpa_loss)
    )
    passed = (
        finite_outputs
        and finite_gradients
        and not missing_gradients
        and logits_max_abs <= tolerances["logits_max_abs"]
        and loss_abs <= tolerances["loss_abs"]
        and grad_max_abs <= tolerances["grad_max_abs"]
    )
    result = {
        "case": case_spec["name"],
        "device": device_type,
        "dtype": dtype_name,
        "autocast_dtype": autocast_dtype_name,
        "passed": passed,
        "tolerances": tolerances,
        "scratch_config": asdict(config),
        "batch_size": batch_size,
        "seq_len": seq_len,
        "parameter_count": count_parameters(eager),
        "logits_max_abs_diff": logits_max_abs,
        "loss_abs_diff": loss_abs,
        "grad_max_abs_diff": grad_max_abs,
        "finite_outputs": finite_outputs,
        "finite_gradients": finite_gradients,
        "missing_gradients": missing_gradients,
        "eager_loss": float(eager_loss.detach().float().item()),
        "sdpa_loss": float(sdpa_loss.detach().float().item()),
        "gradient_parameter_count": len(grad_details),
        "worst_gradient_parameter": max(grad_details, key=lambda item: item["max_abs_diff"], default=None),
    }
    if device_type == "cuda":
        result["peak_vram_mb"] = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    else:
        result["peak_vram_mb"] = None
    del eager, sdpa, input_ids, labels, eager_logits, sdpa_logits, eager_loss, sdpa_loss
    if device_type == "cuda":
        torch.cuda.empty_cache()
    return result


def run_equivalence(seed: int) -> dict[str, Any]:
    started_at = utc_now()
    started = time.perf_counter()
    environment = _environment_info()

    previous_cuda_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else None
    previous_cudnn_tf32 = torch.backends.cudnn.allow_tf32 if torch.cuda.is_available() else None
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    run_matrix, skipped = _build_run_matrix()
    results: list[dict[str, Any]] = []
    failures: list[str] = []
    try:
        for run_spec in run_matrix:
            for case_index, case_spec in enumerate(CASE_SPECS):
                case_seed = seed + case_index * 1_000
                label = f"{run_spec['device']}:{run_spec['dtype']}:{case_spec['name']}"
                try:
                    result = run_equivalence_case(case_spec=case_spec, run_spec=run_spec, seed=case_seed)
                except Exception as exc:  # noqa: BLE001 - artifact must record CUDA/runtime failures explicitly.
                    result = {
                        "case": case_spec["name"],
                        "device": run_spec["device"],
                        "dtype": run_spec["dtype"],
                        "autocast_dtype": run_spec["autocast_dtype"],
                        "passed": False,
                        "error": repr(exc),
                        "traceback": traceback.format_exc(),
                    }
                if not result["passed"]:
                    failures.append(label)
                results.append(result)
    finally:
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = bool(previous_cuda_matmul_tf32)
            torch.backends.cudnn.allow_tf32 = bool(previous_cudnn_tf32)

    cpu_cases = [result for result in results if result.get("device") == "cpu"]
    valid = bool(cpu_cases) and not failures
    return {
        "script": "sba_standalone/check_sba_sdpa_equivalence.py",
        "valid": valid,
        "failure_reasons": failures,
        "started_at": started_at,
        "finished_at": utc_now(),
        "elapsed_seconds": time.perf_counter() - started,
        "git_revision": git_revision(),
        "seed": seed,
        "case_specs": CASE_SPECS,
        "run_matrix": run_matrix,
        "skipped": skipped,
        "tolerances": TOLERANCES,
        "environment": environment,
        "results": results,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check eager SBA and doubled-KV SDPA SBA equivalence for ScratchLM")
    parser.add_argument("--output-dir", default="sba_standalone/results/local_4060_equivalence")
    parser.add_argument("--seed", type=int, default=20260519)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    payload = run_equivalence(seed=args.seed)
    safe_timestamp = payload["finished_at"].replace(":", "").replace("+", "Z").replace(".", "_")
    artifact_path = output_dir / f"sba_sdpa_equivalence_{safe_timestamp}.json"
    latest_path = output_dir / "latest.json"
    payload["artifact_path"] = str(artifact_path)
    payload["latest_artifact_path"] = str(latest_path)
    write_json(artifact_path, payload)
    write_json(latest_path, payload)
    print(f"wrote {latest_path}")
    print(f"valid={payload['valid']}")
    if not payload["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
