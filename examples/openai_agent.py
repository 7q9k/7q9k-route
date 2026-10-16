"""Minimal OpenAI-compatible coding agent using the oc-free-api proxy.

Demonstrates the full agent loop: model calls tools -> agent executes them
locally -> results go back -> model finishes. Uses ONLY the standard
OpenAI-compatible interface (no Ollama bridge).

Run:
    python examples/openai_agent.py "Schreibe eine Funktion mul(a,b) nach agent_demo.py und teste sie"
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from openai import OpenAI

HERE = Path(__file__).resolve().parent
KEY_FILE = HERE.parent / "api-key.txt"
WORKSPACE = HERE / "demo-workspace"
WORKSPACE.mkdir(exist_ok=True)

BASE_URL = "http://127.0.0.1:8787/v1"
MODEL = "big-pickle"
MAX_STEPS = 12

client = OpenAI(base_url=BASE_URL, api_key=KEY_FILE.read_text().strip())

TOOLS = [
    {"type": "function", "function": {
        "name": "write_file",
        "description": "Write text content to a file inside the demo workspace.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"},
            "content": {"type": "string"},
        }, "required": ["path", "content"]},
    }},
    {"type": "function", "function": {
        "name": "read_file",
        "description": "Read a file from the demo workspace.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"},
        }, "required": ["path"]},
    }},
    {"type": "function", "function": {
        "name": "run_python",
        "description": "Run a python file from the demo workspace with its interpreter"
                       " and return stdout+stderr.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"},
        }, "required": ["path"]},
    }},
]

SYSTEM = (
    "You are a careful coding agent. You write and verify code using ONLY the"
    " provided tools. Always verify your code by executing it before you say"
    " the task is done. Workspace-relative paths only."
)


def write_file(path: str, content: str) -> str:
    target = (WORKSPACE / path).resolve()
    if not str(target).startswith(str(WORKSPACE)):
        return "ERROR: outside workspace"
    target.write_text(content, encoding="utf-8")
    return f"wrote {len(content)} chars to {path}"


def read_file(path: str) -> str:
    target = (WORKSPACE / path).resolve()
    if not str(target).startswith(str(WORKSPACE)):
        return "ERROR: outside workspace"
    try:
        return target.read_text(encoding="utf-8")
    except OSError as exc:
        return f"ERROR: {exc}"


def run_python(path: str) -> str:
    target = (WORKSPACE / path).resolve()
    if not str(target).startswith(str(WORKSPACE)):
        return "ERROR: outside workspace"
    result = subprocess.run(
        [sys.executable, str(target)],
        capture_output=True,
        text=True,
        timeout=30,
        cwd=str(WORKSPACE),
    )
    out = (result.stdout + result.stderr).strip() or "(no output)"
    return f"exit={result.returncode}\n{out}"


def execute(name: str, arguments: str) -> str:
    args = json.loads(arguments or "{}")
    if name == "write_file":
        return write_file(str(args.get("path", "")), str(args.get("content", "")))
    if name == "read_file":
        return read_file(str(args.get("path", "")))
    if name == "run_python":
        return run_python(str(args.get("path", "")))
    return f"ERROR: unknown tool {name}"


def run(task: str) -> None:
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": task},
    ]
    for _ in range(MAX_STEPS):
        response = client.chat.completions.create(
            model=MODEL, messages=messages, tools=TOOLS
        )
        message = response.choices[0].message
        calls = message.tool_calls or []

        if not calls:
            print("\n== agent done ==")
            print(message.content)
            return

        messages.append(message.model_dump(exclude_none=True))
        for call in calls:
            print(f"[tool] {call.function.name}({call.function.arguments})")
            result = execute(call.function.name, call.function.arguments)
            print(f"  -> {result[:200]}")
            messages.append(
                {"role": "tool", "tool_call_id": call.id, "content": result}
            )

    print("MAX_STEPS reached without final answer")


if __name__ == "__main__":
    task = " ".join(sys.argv[1:]) or (
        "Write a python function mul(a,b) into agent_demo.py and then verify"
        " it by running the file so it prints 21 for mul(3,7)."
    )
    print(f"task: {task}\n")
    run(task)
