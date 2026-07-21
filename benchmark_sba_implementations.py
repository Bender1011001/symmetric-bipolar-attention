from __future__ import annotations

import argparse
import gc
import platform
import statistics
import sys
import traceback
from contextlib import nullcontext
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from artifacts import git_revision, utc_now, write_json
from run_scratch_compare import ScratchConfig, ScratchLM, count_parameters


MiB = 1024 * 1024


def benchmark_arms() -> list[dict[str, str | None]]:
    return [
        {"name": "softmax", "attention": "softmax", "sba_impl": None},
        {"name": "sba_eager", "attention": "sba", "sba_impl": "eager"},
        {"name": "sba_sdpa", "attention": "sba", "sba_impl": "sdpa"},
    ]


def default_case_specs() -> list[dict[str, int | str]]:
    return [
        {
            "name": "compact_b1_s128_2l_4h_128d",
            "batch_size": 1,
            "seq_len": 128,
            "block_size": 128,
            "n_layer": 2,
            "n_head": 4,
            "n_embd": 128,
        },
        {
            "name": "compact_b2_s256_2l_4h_128d",
            "batch_size": 2,
            "seq_len": 256,
            "block_size": 256,
            "n_layer": 2,
            "n_head": 4,
            "n_embd": 128,
        },
        {
            "name": "local_probe_b2_s512_4l_8h_256d",
            "batch_size": 2,
            "seq_len": 512,
            "block_size": 512,
            "n_layer": 4,
            "n_head": 8,
            "n_embd": 256,
        },
        {
            "name": "local_probe_b4_s512_4l_8h_256d",
            "batch_size": 4,
            "seq_len": 512,
            "block_size": 512,
            "n_layer": 4,
            "n_head": 8,
            "n_embd": 256,
        },
        {
            "name": "long_probe_b2_s768_4l_8h_256d",
            "batch_size": 2,
            "seq_len": 768,
            "block_size": 768,
            "n_layer": 4,
            "n_head": 8,
            "n_embd": 256,
        },
        {
            "name": "long_probe_b1_s1024_4l_8h_256d",
            "batch_size": 1,
            "seq_len": 1024,
            "block_size": 1024,
            "n_layer": 4,
            "n_head": 8,
            "n_embd": 256,
        },
    ]


def _timestamp_for_filename() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def environment_info() -> dict[str, Any]:
    cuda_devices: list[dict[str, Any]] = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            cuda_devices.append(
                {
                    "index": index,
                    "name": props.name,
                    "capability": list(torch.cuda.get_device_capability(index)),
                    "total_memory_mb": props.total_memory / MiB,
                    "multi_processor_count": props.multi_processor_count,
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
        "cuda_allow_tf32": torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else None,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32 if torch.cuda.is_available() else None,
    }


def _make_config(case_spec: dict[str, int | str]) -> ScratchConfig:
    return ScratchConfig(
        block_size=int(case_spec["block_size"]),
        n_layer=int(case_spec["n_layer"]),
        n_head=int(case_spec["n_head"]),
        n_embd=int(case_spec["n_embd"]),
        dropout=0.0,
    )


def _autocast_dtype(amp_dtype: str) -> torch.dtype | None:
    if amp_dtype == "none":
        return None
    if amp_dtype == "bfloat16":
        return torch.bfloat16
    if amp_dtype == "float16":
        return torch.float16
    if amp_dtype != "auto":
        raise ValueError(f"unsupported amp dtype {amp_dtype!r}")
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16 if torch.cuda.is_available() else None


def _dtype_name(dtype: torch.dtype | None) -> str | None:
    if dtype is torch.bfloat16:
        return "bfloat16"
    if dtype is torch.float16:
        return "float16"
    if dtype is torch.float32:
        return "float32"
    return None


def _autocast_context(dtype: torch.dtype | None):
    if dtype is None:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=dtype)


def _make_inputs(
    *,
    config: ScratchConfig,
    batch_size: int,
    seq_len: int,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), generator=generator, dtype=torch.long)
    labels = torch.randint(0, config.vocab_size, (batch_size, seq_len), generator=generator, dtype=torch.long)
    return input_ids.to(device), labels.to(device)


def _run_forward_backward(
    *,
    model: ScratchLM,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    amp_dtype: torch.dtype | None,
) -> float:
    model.zero_grad(set_to_none=True)
    with _autocast_context(amp_dtype):
        _, loss = model(input_ids, labels)
    if loss is None:
        raise RuntimeError("loss unexpectedly missing")
    if not bool(torch.isfinite(loss.detach()).item()):
        raise RuntimeError(f"non-finite loss: {float(loss.detach().float().item())}")
    loss.backward()
    return float(loss.detach().float().item())


def _is_oom_error(error: BaseException) -> bool:
    text = str(error).lower()
    return "out of memory" in text or "cuda error: out of memory" in text


def benchmark_one_arm(
    *,
    case_spec: dict[str, int | str],
    arm: dict[str, str | None],
    device: torch.device,
    amp_dtype: torch.dtype | None,
    warmup_steps: int,
    measure_steps: int,
    seed: int,
) -> dict[str, Any]:
    config = _make_config(case_spec)
    batch_size = int(case_spec["batch_size"])
    seq_len = int(case_spec["seq_len"])
    result: dict[str, Any] = {
        "case": str(case_spec["name"]),
        "arm": str(arm["name"]),
        "attention": arm["attention"],
        "sba_impl": arm["sba_impl"],
        "status": "not_started",
        "scratch_config": asdict(config),
        "batch_size": batch_size,
        "seq_len": seq_len,
        "amp_dtype": _dtype_name(amp_dtype),
        "warmup_steps": warmup_steps,
        "measure_steps": measure_steps,
    }
    try:
        if device.type != "cuda":
            raise RuntimeError("benchmark_one_arm requires a CUDA device")
        torch.cuda.empty_cache()
        torch.manual_seed(seed)
        sba_impl = str(arm["sba_impl"] or "sdpa")
        model = ScratchLM(config, attention=str(arm["attention"]), sba_impl=sba_impl).to(device)
        model.train()
        input_ids, labels = _make_inputs(config=config, batch_size=batch_size, seq_len=seq_len, seed=seed + 10_000, device=device)
        result["parameter_count"] = count_parameters(model)

        for _ in range(warmup_steps):
            _run_forward_backward(model=model, input_ids=input_ids, labels=labels, amp_dtype=amp_dtype)
        torch.cuda.synchronize(device)

        timings_ms: list[float] = []
        losses: list[float] = []
        peak_allocated_mb = 0.0
        peak_reserved_mb = 0.0
        for _ in range(measure_steps):
            torch.cuda.reset_peak_memory_stats(device)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            losses.append(_run_forward_backward(model=model, input_ids=input_ids, labels=labels, amp_dtype=amp_dtype))
            end.record()
            torch.cuda.synchronize(device)
            timings_ms.append(float(start.elapsed_time(end)))
            peak_allocated_mb = max(peak_allocated_mb, torch.cuda.max_memory_allocated(device) / MiB)
            peak_reserved_mb = max(peak_reserved_mb, torch.cuda.max_memory_reserved(device) / MiB)

        result.update(
            {
                "status": "success",
                "time_ms_mean": statistics.fmean(timings_ms),
                "time_ms_median": statistics.median(timings_ms),
                "time_ms_min": min(timings_ms),
                "time_ms_max": max(timings_ms),
                "timings_ms": timings_ms,
                "loss_mean": statistics.fmean(losses),
                "peak_allocated_mb": peak_allocated_mb,
                "peak_reserved_mb": peak_reserved_mb,
            }
        )
    except RuntimeError as error:
        if _is_oom_error(error):
            result.update({"status": "oom", "error_type": type(error).__name__, "error": str(error).splitlines()[0][:500]})
        else:
            result.update(
                {
                    "status": "error",
                    "error_type": type(error).__name__,
                    "error": str(error)[:1000],
                    "traceback": traceback.format_exc(),
                }
            )
    finally:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()
        gc.collect()
    return result


def _successful_by_case(results: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    by_case: dict[str, dict[str, dict[str, Any]]] = {}
    for result in results:
        by_case.setdefault(str(result["case"]), {})[str(result["arm"])] = result
    return by_case


def _ratio(numerator: Any, denominator: Any) -> float | None:
    if numerator is None or denominator in {None, 0, 0.0}:
        return None
    return float(numerator) / float(denominator)


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    status_counts: dict[str, int] = {}
    successful_cases_by_arm: dict[str, int] = {str(arm["name"]): 0 for arm in benchmark_arms()}
    for result in results:
        status = str(result.get("status"))
        status_counts[status] = status_counts.get(status, 0) + 1
        arm = str(result.get("arm"))
        if status == "success":
            successful_cases_by_arm[arm] = successful_cases_by_arm.get(arm, 0) + 1

    ratios: dict[str, dict[str, float]] = {}
    ratio_lists: dict[str, list[float]] = {}
    for case, arms in _successful_by_case(results).items():
        case_ratios: dict[str, float] = {}
        sdpa = arms.get("sba_sdpa")
        if sdpa is None or sdpa.get("status") != "success":
            ratios[case] = case_ratios
            continue
        for base_arm, label in [("sba_eager", "sba_eager"), ("softmax", "softmax")]:
            base = arms.get(base_arm)
            if base is None or base.get("status") != "success":
                continue
            time_ratio = _ratio(sdpa.get("time_ms_mean"), base.get("time_ms_mean"))
            allocated_ratio = _ratio(sdpa.get("peak_allocated_mb"), base.get("peak_allocated_mb"))
            reserved_ratio = _ratio(sdpa.get("peak_reserved_mb"), base.get("peak_reserved_mb"))
            if time_ratio is not None:
                key = f"sba_sdpa_vs_{label}_time"
                case_ratios[key] = time_ratio
                ratio_lists.setdefault(key, []).append(time_ratio)
            if allocated_ratio is not None:
                key = f"sba_sdpa_vs_{label}_peak_allocated"
                case_ratios[key] = allocated_ratio
                ratio_lists.setdefault(key, []).append(allocated_ratio)
            if reserved_ratio is not None:
                key = f"sba_sdpa_vs_{label}_peak_reserved"
                case_ratios[key] = reserved_ratio
                ratio_lists.setdefault(key, []).append(reserved_ratio)
        ratios[case] = case_ratios

    mean_ratios = {key: statistics.fmean(values) for key, values in ratio_lists.items() if values}
    return {
        "status_counts": status_counts,
        "successful_cases_by_arm": successful_cases_by_arm,
        "ratios": ratios,
        "mean_ratios": mean_ratios,
    }


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    env = environment_info()
    amp_dtype = _autocast_dtype(args.amp_dtype)
    case_specs = default_case_specs()
    if args.max_cases is not None:
        case_specs = case_specs[: args.max_cases]

    payload: dict[str, Any] = {
        "created_at": utc_now(),
        "git_revision": git_revision(),
        "command": " ".join(sys.argv),
        "environment": env,
        "benchmark_policy": {
            "synthetic_random_tokens_only": True,
            "optimizer_step_included": False,
            "cuda_event_timing": True,
            "skip_arm_after_oom": True,
            "warmup_steps": args.warmup_steps,
            "measure_steps": args.measure_steps,
            "amp_dtype_requested": args.amp_dtype,
            "amp_dtype_used": _dtype_name(amp_dtype),
        },
        "arms": benchmark_arms(),
        "case_specs": case_specs,
        "results": [],
        "summary": {},
    }
    if not torch.cuda.is_available():
        payload["status"] = "cuda_unavailable"
        payload["summary"] = {"status_counts": {"skipped": len(case_specs) * len(benchmark_arms())}, "reason": "CUDA unavailable"}
        return payload

    device = torch.device(f"cuda:{args.device_index}")
    torch.cuda.set_device(device)
    oom_seen_by_arm: dict[str, str] = {}
    results: list[dict[str, Any]] = []
    for case_spec in case_specs:
        for arm in benchmark_arms():
            arm_name = str(arm["name"])
            if arm_name in oom_seen_by_arm:
                results.append(
                    {
                        "case": str(case_spec["name"]),
                        "arm": arm_name,
                        "attention": arm["attention"],
                        "sba_impl": arm["sba_impl"],
                        "status": "skipped_after_oom",
                        "reason": oom_seen_by_arm[arm_name],
                    }
                )
                continue
            print(f"Benchmarking {case_spec['name']} / {arm_name}...", flush=True)
            result = benchmark_one_arm(
                case_spec=case_spec,
                arm=arm,
                device=device,
                amp_dtype=amp_dtype,
                warmup_steps=args.warmup_steps,
                measure_steps=args.measure_steps,
                seed=args.seed,
            )
            results.append(result)
            print(
                f"  {result['status']}"
                + (f" mean={result.get('time_ms_mean'):.3f} ms peak={result.get('peak_allocated_mb'):.1f} MiB" if result.get("status") == "success" else ""),
                flush=True,
            )
            if result["status"] == "oom":
                oom_seen_by_arm[arm_name] = f"previous OOM at {case_spec['name']}"

    payload["status"] = "completed"
    payload["results"] = results
    payload["summary"] = summarize_results(results)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark softmax, eager SBA, and doubled-KV SDPA SBA on synthetic random-token ScratchLM steps.")
    parser.add_argument("--output-dir", default="sba_standalone/results/local_4060_benchmarks")
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--measure-steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--amp-dtype", choices=["auto", "bfloat16", "float16", "none"], default="auto")
    parser.add_argument("--max-cases", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = run_benchmark(args)
    output_dir = Path(args.output_dir)
    artifact_path = output_dir / f"benchmark_{_timestamp_for_filename()}.json"
    latest_path = output_dir / "latest.json"
    write_json(artifact_path, payload)
    write_json(latest_path, payload)
    print(f"Wrote {artifact_path}", flush=True)
    print(f"Wrote {latest_path}", flush=True)
    if any(result.get("status") == "error" for result in payload.get("results", [])):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
