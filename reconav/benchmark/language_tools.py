"""Generate benchmark instructions or audit their fidelity to symbolic tasks."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path

from reconav.benchmark.language import (
    LanguageResponseError,
    _instruction_payload,
    _instruction_schema,
    _instruction_system_prompt,
    FIDELITY_SYSTEM_PROMPT,
    generate_instructions,
    language_intent,
    with_canonical_intent,
)


def _audit_schema() -> dict:
    return {
        "type": "object",
        "properties": {"items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "task_id": {"type": "string"},
                    "equivalent": {"type": "boolean"},
                    "issues": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["task_id", "equivalent", "issues"],
                "additionalProperties": False,
            },
        }},
        "required": ["items"],
        "additionalProperties": False,
    }


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def _select_tasks(args) -> list[dict]:
    files = args.task if args.task else sorted(args.tasks_root.rglob("*.json"))
    tasks = [json.loads(path.read_text()) for path in files]
    if args.task_ids:
        requested = set(args.task_ids)
        missing = requested - {task["task_id"] for task in tasks}
        if missing:
            raise ValueError(f"task IDs not found: {sorted(missing)}")
        tasks = [task for task in tasks if task["task_id"] in requested]
    if not tasks:
        raise ValueError("no tasks selected")
    if len({task["task_id"] for task in tasks}) != len(tasks):
        raise ValueError("duplicate task IDs")
    return tasks


def _request(mode: str, tasks: list[dict]) -> dict:
    if mode == "generate":
        return {
            "system_prompt": _instruction_system_prompt(),
            "user_prompt": json.dumps(_instruction_payload(tasks), ensure_ascii=False),
            "schema": _instruction_schema(len(tasks)),
        }
    payload = [{"task_id": task["task_id"], "instruction": task["instruction"],
                "canonical_intent": language_intent(task["canonical_intent"])}
               for task in tasks]
    return {
        "system_prompt": FIDELITY_SYSTEM_PROMPT,
        "user_prompt": json.dumps(payload, ensure_ascii=False),
        "schema": _audit_schema(),
    }


def _audit_batch(tasks: list[dict], request: dict, calls_file: Path) -> list[dict]:
    from reconav.llm.shared import SharedLLMClient
    from reconav.method.runtime.model_calls import RecordedClient

    client = RecordedClient(SharedLLMClient("openai"), max_input_characters=512000)
    try:
        with client.at("dataset_instruction_fidelity_audit"):
            raw = client.generate(
                request["system_prompt"], request["user_prompt"], max_tokens=4096,
                response_format={"type": "json_schema", "json_schema": {
                    "name": "instruction_fidelity", "strict": True,
                    "schema": request["schema"],
                }},
            )
    finally:
        _write_json(calls_file, {"model_calls": client.calls})
    items = json.loads(raw)["items"]
    expected = {task["task_id"] for task in tasks}
    if len(items) != len(expected) or {item["task_id"] for item in items} != expected:
        raise ValueError("auditor omitted or duplicated task IDs")
    return items


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    for mode in ("generate", "audit"):
        command = subparsers.add_parser(mode)
        source = command.add_mutually_exclusive_group(required=True)
        source.add_argument("--task", type=Path, nargs="+", help="Task JSON files")
        source.add_argument("--tasks-root", type=Path, help="Directory containing task JSON files")
        command.add_argument("--task-ids", nargs="+", help="Select task IDs from the input")
        command.add_argument("--output-dir", type=Path, required=True, help="New working directory")
        command.add_argument("--batch-size", type=int, default=6)
        command.add_argument("--workers", type=int, default=4)
        command.add_argument("--prepare-only", action="store_true", help="Save requests without API calls")
        if mode == "generate":
            command.add_argument("--provider", choices=("gemini", "openai"), required=True)
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.workers < 1:
        parser.error("batch-size and workers must be positive")
    originals = _select_tasks(args)
    tasks = [with_canonical_intent(task) for task in originals]
    provider = args.provider if args.mode == "generate" else "openai"
    key_name = "GEMINI_API_KEY" if provider == "gemini" else "OPENAI_API_KEY"
    if not args.prepare_only and not os.environ.get(key_name):
        parser.error(f"set {key_name}")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    batches = [tasks[i:i + args.batch_size] for i in range(0, len(tasks), args.batch_size)]
    requests = [_request(args.mode, batch) for batch in batches]
    for index, request in enumerate(requests):
        _write_json(args.output_dir / "requests" / f"{index:03d}.json", {
            "provider": provider, **request,
        })
    if args.prepare_only:
        print(json.dumps({"tasks": len(tasks), "batches": len(batches), "api_calls": 0}))
        return 0

    def one(index):
        batch = batches[index]
        calls_file = args.output_dir / "calls" / f"{index:03d}.json"
        if args.mode == "audit":
            return _audit_batch(batch, requests[index], calls_file)
        try:
            generated, record = generate_instructions(
                batch, provider=provider, api_key=os.environ[key_name],
            )
        except LanguageResponseError as error:
            _write_json(calls_file, error.call_record)
            raise
        _write_json(calls_file, record)
        return [{"task_id": task["task_id"], "instruction": task["instruction"]}
                for task in generated]

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        items = [item for batch in pool.map(one, range(len(batches))) for item in batch]
    if args.mode == "generate":
        instructions = {item["task_id"]: item["instruction"] for item in items}
        for task in originals:
            updated = {**task, "instruction": instructions[task["task_id"]]}
            relative = Path("tasks/hm3dsem") / task["scene_id"] / (task["task_id"] + ".json")
            _write_json(args.output_dir / relative, updated)
        report = {"tasks": len(items), "provider": provider, "items": items}
    else:
        report = {"tasks": len(items), "all_equivalent": all(item["equivalent"] for item in items),
                  "items": items}
    _write_json(args.output_dir / "report.json", report)
    print(json.dumps({key: value for key, value in report.items() if key != "items"}))
    return int(args.mode == "audit" and not report["all_equivalent"])


if __name__ == "__main__":
    raise SystemExit(main())
