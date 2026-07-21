from __future__ import annotations

import argparse
import ast
import json
import multiprocessing as mp
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

if __package__ is None or __package__ == "":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_scratch_tokenized_compare import ScratchConfig, ScratchLM

SAFE_BUILTINS: dict[str, Any] = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "enumerate": enumerate,
    "int": int,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "range": range,
    "reversed": reversed,
    "str": str,
    "sum": sum,
}


@dataclass(frozen=True)
class EvalTask:
    task_id: str
    kind: str
    prompt: str
    expected: Any
    function_name: str = ""
    tests: tuple[tuple[tuple[Any, ...], Any], ...] = ()


DEFAULT_TASKS: tuple[EvalTask, ...] = (
    EvalTask(
        task_id="linear_equation",
        kind="exact_number",
        prompt="Question:\nSolve for x: 2x + 5 = 17. Give only the value of x.\n\nAnswer:\n",
        expected=6,
    ),
    EvalTask(
        task_id="two_digit_addition",
        kind="exact_number",
        prompt="Question:\nWhat is 47 + 58? Give only the number.\n\nAnswer:\n",
        expected=105,
    ),
    EvalTask(
        task_id="small_multiplication",
        kind="exact_number",
        prompt="Question:\nWhat is 13 * 12? Give only the number.\n\nAnswer:\n",
        expected=156,
    ),
    EvalTask(
        task_id="factorial_function",
        kind="python_function",
        prompt="Question:\nWrite a Python function named factorial that returns n!.\n\nAnswer:\n",
        expected=None,
        function_name="factorial",
        tests=(((0,), 1), ((1,), 1), ((5,), 120)),
    ),
    EvalTask(
        task_id="binary_search_function",
        kind="python_function",
        prompt="Question:\nWrite a Python function named binary_search that returns the index of target in a sorted list, or -1 if missing.\n\nAnswer:\n",
        expected=None,
        function_name="binary_search",
        tests=((((1, 3, 5, 7), 5), 2), (((1, 3, 5, 7), 2), -1)),
    ),
    EvalTask(
        task_id="palindrome_function",
        kind="python_function",
        prompt="Question:\nWrite a Python function named is_palindrome that returns True if a string is a palindrome.\n\nAnswer:\n",
        expected=None,
        function_name="is_palindrome",
        tests=((('level',), True), (('robot',), False)),
    ),
)


def load_tasks(path: str) -> tuple[EvalTask, ...]:
    if not path:
        return DEFAULT_TASKS
    tasks: list[EvalTask] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            tests = tuple((tuple(test["args"]), test["expected"]) for test in item.get("tests", []))
            tasks.append(
                EvalTask(
                    task_id=item["task_id"],
                    kind=item["kind"],
                    prompt=item["prompt"],
                    expected=item.get("expected"),
                    function_name=item.get("function_name", ""),
                    tests=tests,
                )
            )
    return tuple(tasks)


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[ScratchLM, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if checkpoint.get("format") in {"sba-bridge-v3", "sba-standalone-scratch-lm-v3-tokenized"}:
        from run_softmax_bridge_conversion import load_centered_sba_model
        model, _ = load_centered_sba_model(checkpoint_path, device)
        return model, checkpoint
    config = ScratchConfig(**checkpoint["scratch_config"])
    sba_impl = checkpoint.get("sba_impl") or checkpoint.get("args", {}).get("sba_impl") or "sdpa"
    model = ScratchLM(config, checkpoint["attention"], sba_impl=sba_impl)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    return model, checkpoint


@torch.no_grad()
def generate_completion(
    model: ScratchLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
) -> str:
    model.eval()
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    ids_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    generated: list[int] = []

    # 1. Prefill step
    window = ids_tensor[:, -model.config.block_size :]
    logits, _, past_key_values = model(window, use_cache=True)

    next_logits = logits[:, -1, :].float()
    if temperature <= 0:
        next_id = int(torch.argmax(next_logits, dim=-1).item())
    else:
        # Mask EOS token if sampling
        next_logits[:, tokenizer.eos_token_id] = -float("inf")
        next_logits = next_logits / temperature
        if top_k > 0:
            values, indices = torch.topk(next_logits, k=min(top_k, next_logits.shape[-1]), dim=-1)
            probs = F.softmax(values, dim=-1)
            chosen = torch.multinomial(probs, num_samples=1)
            next_id = int(indices.gather(-1, chosen).item())
        else:
            probs = F.softmax(next_logits, dim=-1)
            next_id = int(torch.multinomial(probs, num_samples=1).item())

    if next_id == tokenizer.eos_token_id:
        return ""
    generated.append(next_id)
    ids_tensor = torch.cat([ids_tensor, torch.tensor([[next_id]], dtype=torch.long, device=device)], dim=1)

    current_pos = ids_tensor.shape[1] - 1

    # 2. Incremental generation steps
    for _ in range(max_new_tokens - 1):
        cache_len = past_key_values[0][0].shape[-2]

        if cache_len >= model.config.block_size - 1:
            new_past_key_values = []
            for k, v in past_key_values:
                # FIX SINK DELETION: Keep index 0 (BOS sink), drop index 1 (the oldest non-BOS token)
                k_kept = torch.cat([k[:, :, :1, :], k[:, :, 2:, :]], dim=-2)
                v_kept = torch.cat([v[:, :, :1, :], v[:, :, 2:, :]], dim=-2)
                new_past_key_values.append((k_kept, v_kept))
            past_key_values = new_past_key_values

        current_input = ids_tensor[:, -1:]
        pos_id = torch.tensor([[current_pos]], dtype=torch.long, device=device)
        logits, _, past_key_values = model(
            current_input, past_key_values=past_key_values, use_cache=True, position_ids=pos_id
        )

        next_logits = logits[:, -1, :].float()
        if temperature <= 0:
            next_id = int(torch.argmax(next_logits, dim=-1).item())
        else:
            # Mask EOS token if sampling
            next_logits[:, tokenizer.eos_token_id] = -float("inf")
            next_logits = next_logits / temperature
            if top_k > 0:
                values, indices = torch.topk(next_logits, k=min(top_k, next_logits.shape[-1]), dim=-1)
                probs = F.softmax(values, dim=-1)
                chosen = torch.multinomial(probs, num_samples=1)
                next_id = int(indices.gather(-1, chosen).item())
            else:
                probs = F.softmax(next_logits, dim=-1)
                next_id = int(torch.multinomial(probs, num_samples=1).item())

        if next_id == tokenizer.eos_token_id:
            break
        generated.append(next_id)
        ids_tensor = torch.cat([ids_tensor, torch.tensor([[next_id]], dtype=torch.long, device=device)], dim=1)
        current_pos += 1

    return tokenizer.decode(generated, skip_special_tokens=True)


def score_number(completion: str, expected: int | float) -> dict[str, Any]:
    text_to_search = completion
    if "</think>" in completion:
        text_to_search = completion.split("</think>")[-1]
    numbers = re.findall(r"[-+]?\d+(?:\.\d+)?", text_to_search)
    expected_float = float(expected)
    parsed = [float(number) for number in numbers]
    correct = False
    if parsed:
        correct = abs(parsed[-1] - expected_float) < 1e-9
    return {"correct": correct, "numbers_found": parsed, "searched_text": text_to_search}


def extract_python_source(completion: str, function_name: str) -> str:
    fence = re.search(r"```(?:python)?\s*(.*?)```", completion, flags=re.DOTALL | re.IGNORECASE)
    candidate = fence.group(1) if fence else completion
    match = re.search(rf"def\s+{re.escape(function_name)}\s*\(", candidate)
    if not match:
        return ""
    return candidate[match.start() :]


def reject_unsafe_python(source: str) -> str:
    if not source.strip():
        return "no function definition found"
    if "__" in source:
        return "dunder access rejected"
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal, ast.With, ast.AsyncWith, ast.Lambda)):
            return f"unsafe node rejected: {type(node).__name__}"
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in {"eval", "exec", "open", "compile", "input"}:
            return f"unsafe call rejected: {node.func.id}"
    return ""


def _run_code_worker(source: str, function_name: str, tests: tuple[tuple[tuple[Any, ...], Any], ...], queue: mp.Queue) -> None:
    try:
        namespace: dict[str, Any] = {"__builtins__": SAFE_BUILTINS}
        exec(compile(source, "<generated>", "exec"), namespace, namespace)
        fn = namespace.get(function_name)
        if not callable(fn):
            queue.put({"correct": False, "error": "function missing after exec", "test_results": []})
            return
        test_results = []
        all_correct = True
        for args, expected in tests:
            result = fn(*args)
            ok = result == expected
            all_correct = all_correct and ok
            test_results.append({"args": list(args), "expected": expected, "actual": result, "correct": ok})
        queue.put({"correct": all_correct, "error": "", "test_results": test_results})
    except Exception as exc:
        queue.put({"correct": False, "error": f"{type(exc).__name__}: {exc}", "test_results": []})


def score_python(completion: str, function_name: str, tests: tuple[tuple[tuple[Any, ...], Any], ...], timeout_seconds: float) -> dict[str, Any]:
    source = extract_python_source(completion, function_name)
    try:
        rejection = reject_unsafe_python(source)
    except SyntaxError as exc:
        return {"correct": False, "error": f"SyntaxError: {exc}", "source_excerpt": source[:500], "test_results": []}
    if rejection:
        return {"correct": False, "error": rejection, "source_excerpt": source[:500], "test_results": []}

    queue: mp.Queue = mp.Queue()
    process = mp.Process(target=_run_code_worker, args=(source, function_name, tests, queue))
    process.start()
    process.join(timeout_seconds)
    if process.is_alive():
        process.terminate()
        process.join(1)
        return {"correct": False, "error": "timeout", "source_excerpt": source[:500], "test_results": []}
    if queue.empty():
        return {"correct": False, "error": "worker failed to return a result", "source_excerpt": source[:500], "test_results": []}
    result = queue.get()
    result["source_excerpt"] = source[:500]
    return result


def evaluate_task(
    model: ScratchLM,
    tokenizer: AutoTokenizer,
    task: EvalTask,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, Any]:
    completion = generate_completion(
        model=model,
        tokenizer=tokenizer,
        prompt=task.prompt,
        device=device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    if task.kind == "exact_number":
        score = score_number(completion, task.expected)
    elif task.kind == "python_function":
        score = score_python(completion, task.function_name, task.tests, args.code_timeout_seconds)
    else:
        raise ValueError(f"unsupported task kind: {task.kind}")
    return {
        "task_id": task.task_id,
        "kind": task.kind,
        "prompt": task.prompt,
        "expected": task.expected,
        "function_name": task.function_name,
        "completion": completion,
        "score": score,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate scratch tokenized SBA/softmax LMs")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tasks", default="")
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--code-timeout-seconds", type=float, default=2.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    model, checkpoint = load_model(Path(args.checkpoint), device)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tasks = load_tasks(args.tasks)
    results = [evaluate_task(model, tokenizer, task, device, args) for task in tasks]
    correct = sum(1 for result in results if result["score"].get("correct"))

    payload = {
        "checkpoint": args.checkpoint,
        "checkpoint_step": checkpoint.get("step"),
        "attention": checkpoint.get("attention"),
        "scratch_config": checkpoint.get("scratch_config"),
        "generation": {
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_k": args.top_k,
        },
        "summary": {
            "task_count": len(results),
            "correct": correct,
            "accuracy": correct / len(results) if results else 0.0,
        },
        "results": results,
    }
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
