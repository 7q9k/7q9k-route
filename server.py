"""OpenCode Zen free models as a local OpenAI-compatible API.

Exposes an OpenAI-style endpoint with its own API key so any client
(coding agent, IDE plugin, chat tool) can use the free models like a
normal API:

    base_url = http://127.0.0.1:8787/v1
    api_key  = sk-oc-free-...

Endpoints:
    GET  /health                -> simple liveness check (no auth)
    GET  /v1/models             -> discovered free models
    POST /v1/chat/completions   -> chat completions (stream + non-stream)

Standard library only, personal use only. The upstream envelope follows
OpenCode's own client requests; availability can change at any time.
"""

from __future__ import annotations

import argparse
import ctypes
import itertools
import json
import os
import secrets
import string
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator

DASHBOARD_FILE = Path(__file__).resolve().parent / "dashboard.html"
REQUEST_LOG: deque = deque(maxlen=200)

BASE_DIR = Path(__file__).resolve().parent
MODEL_CACHE = BASE_DIR / "models.json"
KEY_FILE = BASE_DIR / "api-key.txt"

DIRECT_URL = "https://opencode.ai/inference/v1/chat/completions"
RESPONSES_URL = "https://opencode.ai/inference/v1/responses"
ZEN_MODELS_URL = "https://opencode.ai/zen/v1/models"
MODELS_DEV_URL = "https://models.dev/api.json"
OPENCODE_VERSION = "1.19.0"
OPENCODE_UA_CHAT = f"opencode/{OPENCODE_VERSION} ai-sdk/provider-utils/4.0.23 runtime/bun/1.3.14"
OPENCODE_UA_RESPONSES = (
    f"opencode/{OPENCODE_VERSION} ai-sdk/provider-utils/4.0.40 runtime/bun/1.3.14"
)
OPENCODE_CLIENT_HEADER = "cli"
OPENCODE_PROJECT_HEADER = "global"

DEFAULT_MODEL = "big-pickle"
FALLBACK_MODELS = (
    DEFAULT_MODEL,
    "ling-3.0-flash-fin-free",
    "mimo-v2.5-free",
    "muse-spark-1.2-contributor-free",
    "muse-spark-1.3-contributor-free",
    "nemotron-3-ultra-free",
    "nemotron-3.5-lightning-free",
)
RESPONSES_MODELS = frozenset(
    {"muse-spark-1.2-contributor-free", "muse-spark-1.3-contributor-free"}
)

UPSTREAM_TIMEOUT = 900.0

_MODEL_LOCK = threading.Lock()
_MODEL_SNAPSHOT: tuple[str, ...] | None = None

_BASE62 = string.digits + string.ascii_letters
_ID_LOCK = threading.Lock()
_LAST_ID_TS = 0
_ID_COUNTER = 0


class ProxyError(Exception):
    status_code: int = 502

    def __init__(self, message: str, status_code: int = 502):
        super().__init__(message)
        self.status_code = status_code


# ---------------------------------------------------------------- session ids


def _session_id() -> str:
    return _oc_id("ses", descending=True)


def _request_id() -> str:
    return _oc_id("msg", descending=False)


def _oc_id(prefix: str, *, descending: bool) -> str:
    """Replica of opencode's ID generator (packages/opencode/src/id/id.ts).

    Sessions are descending, requests ascending; the free tier validates
    this shape and may reject unknown clients with 403.
    """
    global _LAST_ID_TS, _ID_COUNTER
    with _ID_LOCK:
        now_ts = int(time.time() * 1000)
        if now_ts != _LAST_ID_TS:
            _LAST_ID_TS = now_ts
            _ID_COUNTER = 0
        _ID_COUNTER += 1
        value = (now_ts << 12) | _ID_COUNTER
        if descending:
            value = ~value & ((1 << 56) - 1)
        time_bytes = value.to_bytes(7, "big")[-6:]
        return (
            prefix
            + "_"
            + time_bytes.hex()
            + "".join(secrets.choice(_BASE62) for _ in range(14))
        )


# ---------------------------------------------------------------- model discovery


def _json_get(url: str, timeout: float) -> Any:
    request = urllib.request.Request(
        url, headers={"Accept": "application/json", "User-Agent": OPENCODE_UA_CHAT}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def _free_live_models(zen: Any, catalog: Any) -> list[str]:
    zen_models = {
        str(item.get("id") or "")
        for item in (zen.get("data") if isinstance(zen, dict) else []) or []
        if isinstance(item, dict) and item.get("id")
    }
    provider = catalog.get("opencode") if isinstance(catalog, dict) else None
    metadata = provider.get("models") if isinstance(provider, dict) else None
    if not isinstance(metadata, dict):
        return []

    free = []
    for model, details in metadata.items():
        if not isinstance(details, dict) or details.get("status") == "deprecated":
            continue
        costs = details.get("cost")
        if (
            model in zen_models
            and details.get("tool_call") is True
            and isinstance(costs, dict)
            and costs
            and all(
                isinstance(cost, (int, float))
                and not isinstance(cost, bool)
                and cost == 0
                for cost in costs.values()
            )
        ):
            free.append(str(model))
    return sorted(free)


def _load_model_cache() -> list[str]:
    try:
        models = json.loads(MODEL_CACHE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    if not isinstance(models, list) or any(
        not isinstance(model, str) or not model for model in models
    ):
        return []
    return sorted(dict.fromkeys(models))


def _save_model_cache(models: list[str]) -> None:
    try:
        MODEL_CACHE.write_text(json.dumps(models) + "\n", encoding="utf-8")
    except OSError:
        pass


def available_models(timeout: float = 15.0, refresh: bool = False) -> list[str]:
    global _MODEL_SNAPSHOT
    with _MODEL_LOCK:
        if _MODEL_SNAPSHOT is None or refresh:
            try:
                models = _free_live_models(
                    _json_get(ZEN_MODELS_URL, timeout),
                    _json_get(MODELS_DEV_URL, timeout),
                )
            except (OSError, ValueError, urllib.error.URLError):
                models = []
            if models:
                _save_model_cache(models)
            else:
                models = _load_model_cache() or list(FALLBACK_MODELS)
            _MODEL_SNAPSHOT = tuple(models)
        return list(_MODEL_SNAPSHOT)


def uses_responses_api(model: str) -> bool:
    return model in RESPONSES_MODELS


# ---------------------------------------------------------------- SSE plumbing


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req: Any, fp: Any, code: int, msg: str, headers: Any, newurl: Any) -> None:
        return None


_OPENER = urllib.request.build_opener(_NoRedirect)


def _sse_lines_to_payloads(response: Any) -> Iterator[dict[str, Any]]:
    for raw_line in response:
        line = raw_line.decode("utf-8", "replace").strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            return
        if payload:
            yield json.loads(payload)


def upstream_stream(
    url: str, body: dict[str, Any], timeout: float = UPSTREAM_TIMEOUT
) -> Iterator[dict[str, Any]]:
    session = _session_id()
    ua = OPENCODE_UA_RESPONSES if "responses" in url else OPENCODE_UA_CHAT
    request = urllib.request.Request(
        url,
        data=json.dumps(body, separators=(",", ":")).encode(),
        method="POST",
        headers={
            "Authorization": "Bearer public",
            "Content-Type": "application/json",
            "Accept": "*/*",
            "User-Agent": ua,
            "x-opencode-session": session,
            "x-opencode-session-id": session,
            "x-opencode-client": OPENCODE_CLIENT_HEADER,
            "x-opencode-project": OPENCODE_PROJECT_HEADER,
            "x-opencode-request": _request_id(),
        },
    )
    try:
        response = _OPENER.open(request, timeout=timeout)
    except urllib.error.HTTPError as exc:
        raise ProxyError(
            f" upstream rejected request (HTTP {exc.code}): "
            f"{exc.read().decode('utf-8', 'replace')[:500]}",
            502 if exc.code >= 500 else exc.code,
        ) from exc
    except urllib.error.URLError as exc:
        raise ProxyError(f"upstream connection failed: {exc.reason}") from exc

    return _sse_lines_to_payloads(response)


# ---------------------------------------------------------------- chat payload


# Real schemas for opencode's native tools. The free tier validates that
# request tool schemas are non-empty; emitting tool markers without
# parameters is rejected. Marker descriptions stay "unavailable" so the
# model should not call them unless the client actually supplied a
# matching tool.
COMPAT_TOOL_PARAMETERS: dict[str, dict[str, Any]] = {
    "bash": {
        "type": "object",
        "properties": {
            "command": {"type": "string"},
            "timeout": {"type": "integer", "minimum": 1},
            "workdir": {"type": "string"},
        },
        "required": ["command"],
    },
    "edit": {
        "type": "object",
        "properties": {
            "filePath": {"type": "string"},
            "oldString": {"type": "string"},
            "newString": {"type": "string"},
            "replaceAll": {"type": "boolean"},
        },
        "required": ["filePath", "oldString", "newString"],
    },
    "glob": {
        "type": "object",
        "properties": {"pattern": {"type": "string"}, "path": {"type": "string"}},
        "required": ["pattern"],
    },
    "grep": {
        "type": "object",
        "properties": {
            "pattern": {"type": "string"},
            "path": {"type": "string"},
            "include": {"type": "string"},
        },
        "required": ["pattern"],
    },
    "read": {
        "type": "object",
        "properties": {
            "filePath": {"type": "string"},
            "offset": {"type": "integer", "minimum": 1},
            "limit": {"type": "integer", "minimum": 1},
        },
        "required": ["filePath"],
    },
    "skill": {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
    },
    "task": {
        "type": "object",
        "properties": {"goal": {"type": "string"}, "context": {"type": "string"}},
        "required": ["goal"],
    },
    "todowrite": {
        "type": "object",
        "properties": {
            "todos": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string"},
                        "status": {
                            "type": "string",
                            "enum": ["pending", "in_progress", "completed", "cancelled"],
                        },
                    },
                    "required": ["content", "status"],
                },
            }
        },
        "required": ["todos"],
    },
    "webfetch": {
        "type": "object",
        "properties": {"url": {"type": "string"}},
        "required": ["url"],
    },
    "websearch": {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    },
    "write": {
        "type": "object",
        "properties": {"filePath": {"type": "string"}, "content": {"type": "string"}},
        "required": ["filePath", "content"],
    },
}


def _mesh_tool_names(tools: list[dict[str, Any]]) -> set[str]:
    names: set[str] = set()
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        if tool.get("type") == "function" and isinstance(tool.get("function"), dict):
            names.add(str(tool["function"].get("name") or ""))
        elif tool.get("name"):
            names.add(str(tool["name"]))
    return names


def ensure_required_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The free tier requires bash and read tools in EVERY request.

    Injects native-tool markers (with real, non-empty schemas) when the
    client did not supply them. Injected tools precede client tools.
    """
    tools = [tool for tool in tools if isinstance(tool, dict)]
    names = _mesh_tool_names(tools)
    result = list(tools)
    for required in ("bash", "read"):
        if required not in names:
            result.insert(
                0,
                {
                    "type": "function",
                    "function": {
                        "name": required,
                        "description": (
                            "Compatibility marker; unavailable. Never call."
                        ),
                        "parameters": COMPAT_TOOL_PARAMETERS[required],
                    },
                },
            )
    return result


def ensure_responses_required_tools(
    tools: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    tools = [tool for tool in tools if isinstance(tool, dict)]
    names = {str(tool.get("name") or "") for tool in tools}
    for required in ("bash", "read"):
        if required not in names:
            tools.insert(
                0,
                {
                    "type": "function",
                    "name": required,
                    "description": "Compatibility marker; unavailable. Never call.",
                    "parameters": COMPAT_TOOL_PARAMETERS[required],
                    "strict": False,
                },
            )
    return tools




_PASSTHROUGH_KEYS = (
    "temperature",
    "top_p",
    "max_tokens",
    "stop",
    "presence_penalty",
    "frequency_penalty",
    "seed",
    "response_format",
    "parallel_tool_calls",
    "logit_bias",
    "user",
)


def build_chat_request(req: dict[str, Any]) -> dict[str, Any]:
    messages = list(req.get("messages") or [])
    if not any(
        isinstance(m, dict) and m.get("role") == "system" for m in messages
    ):
        messages.insert(0, {"role": "system", "content": "You are opencode"})

    body: dict[str, Any] = {
        "model": req.get("model") or DEFAULT_MODEL,
        "messages": messages,
        "tools": ensure_required_tools(req.get("tools") if isinstance(req.get("tools"), list) else []),
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    for key in _PASSTHROUGH_KEYS:
        if req.get(key) is not None:
            body[key] = req[key]
    if isinstance(req.get("tools"), list) and req["tools"]:
        if req.get("tool_choice") is not None:
            body["tool_choice"] = req["tool_choice"]
    return body


# ---------------------------------------------------------------- responses payload


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(part.get("text") or "")
            for part in content
            if isinstance(part, dict) and part.get("text") is not None
        )
    return "" if content is None else str(content)


def _responses_content(content: Any, *, output: bool = False) -> list[dict[str, Any]]:
    text_type = "output_text" if output else "input_text"
    if not isinstance(content, list):
        text = _message_text(content)
        return [{"type": text_type, "text": text}] if text else []
    result = []
    for part in content:
        if not isinstance(part, dict):
            result.append({"type": text_type, "text": str(part)})
            continue
        kind = str(part.get("type") or "")
        if kind in {"text", "input_text", "output_text"}:
            result.append({"type": text_type, "text": str(part.get("text") or "")})
    return result


def _responses_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for tool in tools:
        function = tool.get("function") or {}
        result.append(
            {
                "type": "function",
                "name": str(function.get("name") or ""),
                "description": str(function.get("description") or ""),
                "parameters": function.get("parameters") or {"type": "object"},
                "strict": bool(function.get("strict", False)),
            }
        )
    return result


def _responses_input(
    messages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], str]:
    instructions = ["You are opencode"]
    items: list[dict[str, Any]] = []
    for message in messages:
        role = str(message.get("role") or "user")
        if role == "system":
            if text := _message_text(message.get("content")):
                instructions.append(text)
            continue
        if role == "tool":
            call_id = str(message.get("tool_call_id") or "")
            if call_id:
                items.append(
                    {
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": _message_text(message.get("content")),
                    }
                )
            continue
        if role == "assistant":
            for detail in message.get("reasoning_details") or []:
                if isinstance(detail, dict) and detail.get("type") == "reasoning":
                    items.append(dict(detail))
            if content := _responses_content(message.get("content"), output=True):
                items.append({"role": "assistant", "content": content})
            for call in message.get("tool_calls") or []:
                if not isinstance(call, dict):
                    continue
                function = call.get("function") or {}
                arguments = function.get("arguments") or "{}"
                if not isinstance(arguments, str):
                    arguments = json.dumps(arguments, separators=(",", ":"))
                call_id = str(call.get("id") or call.get("call_id") or "")
                if call_id:
                    items.append(
                        {
                            "type": "function_call",
                            "call_id": call_id,
                            "name": str(function.get("name") or ""),
                            "arguments": arguments,
                        }
                    )
            continue
        content = _responses_content(message.get("content"))
        if content:
            items.append(
                {"role": "user" if role == "user" else role, "content": content}
            )
    return items, "\n\n".join(instructions)


def build_responses_request(req: dict[str, Any]) -> dict[str, Any]:
    tools = req.get("tools") if isinstance(req.get("tools"), list) else []
    input_items, instructions = _responses_input(list(req.get("messages") or []))
    body: dict[str, Any] = {
        "model": req.get("model") or DEFAULT_MODEL,
        "input": input_items,
        "instructions": instructions,
        "tools": ensure_responses_required_tools(
            _responses_tools(tools) if tools else []
        ),
        "stream": True,
        "store": False,
        "include": ["reasoning.encrypted_content"],
        "prompt_cache_key": _session_id(),
    }
    for key, target in (
        ("temperature", "temperature"),
        ("top_p", "top_p"),
        ("max_tokens", "max_output_tokens"),
    ):
        if req.get(key) is not None:
            body[target] = req[key]
    if tools and req.get("tool_choice") is not None:
        choice = req["tool_choice"]
        if isinstance(choice, dict) and isinstance(choice.get("function"), dict):
            choice = {
                "type": "function",
                "name": str(choice["function"].get("name") or ""),
            }
        body["tool_choice"] = choice
    return body


def responses_events_to_deltas(events: Iterator[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    """Convert /responses SSE into OpenAI chat-completion style delta chunks."""
    seen = False
    saw_tool_call = False
    call_indexes: dict[int, int] = {}
    selected_model = ""
    for event in events:
        seen = True
        kind = str(event.get("type") or "")
        response = event.get("response") or {}
        selected_model = str(response.get("model") or selected_model)
        if kind == "error" or event.get("error"):
            raise ProxyError(f"upstream inference error: {json.dumps(event)[:500]}")
        if kind == "response.output_text.delta":
            yield {
                "model": selected_model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": str(event.get("delta") or "")},
                        "finish_reason": None,
                    }
                ],
            }
        elif kind == "response.reasoning_summary_text.delta":
            yield {
                "model": selected_model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"reasoning_content": str(event.get("delta") or "")},
                        "finish_reason": None,
                    }
                ],
            }
        elif kind == "response.output_item.done":
            item = event.get("item") or {}
            if item.get("type") == "reasoning":
                yield {
                    "model": selected_model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"reasoning_details": [item]},
                            "finish_reason": None,
                        }
                    ],
                }
            elif item.get("type") == "function_call":
                saw_tool_call = True
                output_index = int(event.get("output_index", 0) or 0)
                index = call_indexes.setdefault(output_index, len(call_indexes))
                yield {
                    "model": selected_model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": index,
                                        "id": str(item.get("call_id") or ""),
                                        "type": "function",
                                        "function": {
                                            "name": str(item.get("name") or ""),
                                            "arguments": str(
                                                item.get("arguments") or "{}"
                                            ),
                                        },
                                    }
                                ]
                            },
                            "finish_reason": None,
                        }
                    ],
                }
        elif kind in {"response.failed", "response.incomplete"}:
            raise ProxyError(
                f"upstream inference did not complete: {json.dumps(response)[:500]}"
            )
        elif kind == "response.completed":
            if response.get("status") not in {None, "completed"}:
                raise ProxyError(
                    f"upstream inference did not complete: {json.dumps(response)[:500]}"
                )
            usage = response.get("usage") or {}
            yield {
                "model": selected_model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "tool_calls" if saw_tool_call else "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": usage.get("input_tokens", 0),
                    "completion_tokens": usage.get("output_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                    "prompt_tokens_details": usage.get("input_tokens_details", {}),
                },
            }
    if not seen:
        raise ProxyError("upstream returned no events.")


# ---------------------------------------------------------------- aggregation


def _usage(value: Any = None) -> dict[str, Any]:
    raw = value if isinstance(value, dict) else {}
    prompt = int(raw.get("prompt_tokens", raw.get("input", 0)) or 0)
    completion = int(raw.get("completion_tokens", raw.get("output", 0)) or 0)
    total = int(raw.get("total_tokens", prompt + completion) or prompt + completion)
    details = raw.get("prompt_tokens_details")
    details = details if isinstance(details, dict) else {}
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
        "prompt_tokens_details": {"cached_tokens": int(details.get("cached_tokens", 0) or 0)},
    }


def merge_deltas(events: Iterator[dict[str, Any]], model: str) -> dict[str, Any]:
    """Aggregate chat-style delta chunks into one non-streaming response."""
    content: list[str] = []
    reasoning: list[str] = []
    reasoning_details: list[Any] = []
    calls: dict[int, dict[str, Any]] = {}
    finish_reason = "stop"
    usage: dict[str, Any] | None = None
    selected = model
    seen = False

    for event in events:
        seen = True
        if event.get("error"):
            raise ProxyError(f"upstream inference error: {event['error']}")
        selected = str(event.get("model") or selected)
        if isinstance(event.get("usage"), dict):
            usage = event["usage"]
        for choice in event.get("choices") or []:
            finish_reason = choice.get("finish_reason") or finish_reason
            delta = choice.get("delta") or {}
            if delta.get("content") is not None:
                content.append(str(delta["content"]))
            for key in ("reasoning_content", "reasoning"):
                if delta.get(key) is not None:
                    reasoning.append(str(delta[key]))
                    break
            details = delta.get("reasoning_details")
            if isinstance(details, list):
                reasoning_details.extend(details)
            elif details is not None:
                reasoning_details.append(details)
            for raw_call in delta.get("tool_calls") or []:
                index = int(raw_call.get("index", 0) or 0)
                if raw_call.get("type") == "custom":
                    raise ProxyError(
                        "custom tool calls are not supported; the client "
                        f"requested a custom tool: {raw_call.get('name')}"
                    )
                item = calls.setdefault(
                    index, {"id": "", "type": "function", "name": "", "arguments": ""}
                )
                if raw_call.get("id"):
                    item["id"] = str(raw_call["id"])
                item["type"] = str(raw_call.get("type") or item["type"])
                function = raw_call.get("function") or {}
                if function.get("name"):
                    item["name"] = str(function["name"])
                item["arguments"] += str(function.get("arguments") or "")
    if not seen:
        raise ProxyError("upstream returned no events.")
    if finish_reason == "content_filter":
        raise ProxyError("upstream blocked the response by content filter.", 400)

    tool_calls = [
        {
            "id": calls[index].get("id") or f"call_{index}",
            "type": "function",
            "function": {
                "name": calls[index].get("name", ""),
                "arguments": calls[index].get("arguments", "{}"),
            },
        }
        for index in sorted(calls)
    ]
    text = "".join(content) or None
    thought = "".join(reasoning) or None

    message: dict[str, Any] = {"role": "assistant", "content": text}
    if thought:
        message["reasoning"] = thought
        message["reasoning_content"] = thought
    if reasoning_details:
        message["reasoning_details"] = reasoning_details
    if tool_calls:
        message["tool_calls"] = tool_calls

    return {
        "id": "chatcmpl-" + secrets.token_hex(12),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": selected,
        "choices": [
            {
                "index": 0,
                "finish_reason": "tool_calls" if tool_calls else finish_reason,
                "message": message,
            }
        ],
        "usage": usage if usage is not None else _usage(),
    }


# ---------------------------------------------------------------- ollama bridge


_CALL_ID_SEQ = itertools.count(1)


def _json_obj(text: Any) -> dict[str, Any]:
    try:
        value = json.loads(text if isinstance(text, str) and text.strip() else "{}")
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _ollama_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + "Z"


def _ollama_usage(usage: dict[str, Any] | None) -> tuple[int, int]:
    usage = usage if isinstance(usage, dict) else {}
    return int(usage.get("prompt_tokens", 0) or 0), int(
        usage.get("completion_tokens", 0) or 0
    )


def _ollama_details() -> dict[str, Any]:
    return {
        "format": "api",
        "family": "opencode-free",
        "families": ["opencode-free"],
        "parameter_size": "",
        "parent_model": "",
        "quantization_level": "",
    }


def _ollama_model_entries() -> list[dict[str, Any]]:
    return [
        {
            "name": model,
            "model": model,
            "modified_at": _ollama_now(),
            "size": 0,
            "digest": "oc-free-" + model,
            "details": _ollama_details(),
            "capabilities": ["completion", "streaming", "tools"],
        }
        for model in available_models()
    ]


def normalize_model_name(name: Any) -> str:
    text = str(name or "").strip()
    if "/" in text:
        provider, text = text.split("/", 1)
        if provider != "opencode":
            raise ProxyError(
                "only models from the 'opencode' provider are supported "
                "(e.g. 'opencode/big-pickle').",
                400,
            )
    if text.endswith(":latest"):
        text = text[: -len(":latest")]
    return text or DEFAULT_MODEL


def ollama_to_openai_messages(messages: list[Any]) -> list[dict[str, Any]]:
    """Convert Ollama chat messages (incl. tool_calls / tool results)."""
    result: list[dict[str, Any]] = []
    ids_by_name: dict[str, list[str]] = {}
    fifo_ids: list[str] = []
    for message in messages or []:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "")
        if role == "assistant" and isinstance(message.get("tool_calls"), list):
            calls = []
            for call in message["tool_calls"]:
                function = call.get("function") or {}
                name = str(function.get("name") or "")
                arguments = function.get("arguments") or "{}"
                if not isinstance(arguments, str):
                    arguments = json.dumps(arguments, separators=(",", ":"))
                call_id = f"call_{next(_CALL_ID_SEQ)}"
                calls.append(
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {"name": name, "arguments": arguments},
                    }
                )
                ids_by_name.setdefault(name, []).append(call_id)
                fifo_ids.append(call_id)
            item: dict[str, Any] = {"role": "assistant", "tool_calls": calls}
            content = message.get("content")
            if content:
                item["content"] = content
            result.append(item)
            continue
        if role == "tool":
            call_id = message.get("tool_call_id")
            if not call_id:
                name = str(message.get("tool_name") or message.get("name") or "")
                if name and ids_by_name.get(name):
                    call_id = ids_by_name[name].pop(0)
                elif fifo_ids:
                    call_id = fifo_ids.pop(0)
                else:
                    call_id = f"call_{next(_CALL_ID_SEQ)}"
            result.append(
                {"role": "tool", "tool_call_id": call_id, "content": message.get("content") or ""}
            )
            continue
        result.append({"role": role or "user", "content": message.get("content") or ""})
    return result


def ollama_options(req: dict[str, Any]) -> dict[str, Any]:
    """Build an OpenAI chat request out of an Ollama /api/chat|generate body."""
    options = req.get("options") if isinstance(req.get("options"), dict) else {}
    openai_req: dict[str, Any] = {
        "model": req.get("model"),
        "stream": bool(req.get("stream")),
    }
    for key in ("temperature", "top_p", "seed"):
        if options.get(key) is not None:
            openai_req[key] = options[key]
    if options.get("num_predict") is not None:
        openai_req["max_tokens"] = options["num_predict"]
    if options.get("stop") is not None:
        openai_req["stop"] = (
            options["stop"] if isinstance(options["stop"], list) else [options["stop"]]
        )
    if req.get("format") == "json":
        openai_req["response_format"] = {"type": "json_object"}
    if isinstance(req.get("tools"), list) and req["tools"]:
        tools = []
        for tool in req["tools"]:
            if not isinstance(tool, dict):
                continue
            function = dict(tool.get("function") or {})
            if "parameters" not in function and "arguments" in function:
                function["parameters"] = function["arguments"]
            tools.append({"type": "function", "function": function})
        if tools:
            openai_req["tools"] = tools
            if req.get("tool_choice") is not None:
                openai_req["tool_choice"] = req["tool_choice"]
    return openai_req


def ollama_chat_response(completion: dict[str, Any], include_thinking: bool = False) -> dict[str, Any]:
    message = completion["choices"][0]["message"]
    content = message.get("content") or ""
    out_message: dict[str, Any] = {"role": "assistant", "content": content}
    if include_thinking and message.get("reasoning"):
        out_message["thinking"] = str(message["reasoning"])
    raw_calls = message.get("tool_calls") or []
    if raw_calls:
        out_message["tool_calls"] = [
            {
                "function": {
                    "name": call["function"]["name"],
                    "arguments": _json_obj(call["function"].get("arguments")),
                }
            }
            for call in raw_calls
        ]
    prompt_count, eval_count = _ollama_usage(completion.get("usage"))
    return {
        "model": completion.get("model"),
        "created_at": _ollama_now(),
        "message": out_message,
        "done_reason": completion["choices"][0].get("finish_reason") or "stop",
        "done": True,
        "total_duration": 0,
        "load_duration": 0,
        "prompt_eval_count": prompt_count,
        "eval_count": eval_count,
    }


def ollama_generate_response(completion: dict[str, Any]) -> dict[str, Any]:
    message = completion["choices"][0]["message"]
    prompt_count, eval_count = _ollama_usage(completion.get("usage"))
    return {
        "model": completion.get("model"),
        "created_at": _ollama_now(),
        "response": message.get("content") or "",
        "done_reason": completion["choices"][0].get("finish_reason") or "stop",
        "done": True,
        "total_duration": 0,
        "load_duration": 0,
        "prompt_eval_count": prompt_count,
        "eval_count": eval_count,
    }


# ---------------------------------------------------------------- dashboard API

_TEST_PROMPT = "Reply with exactly the token: OK-DASH"


def open_upstream_events(req: dict[str, Any], model: str) -> Iterator[dict[str, Any]]:
    """Shared upstream opener for the chat handler and the dashboard test."""
    if uses_responses_api(model):
        events = upstream_stream(RESPONSES_URL, build_responses_request(req))
        return responses_events_to_deltas(events)
    return upstream_stream(DIRECT_URL, build_chat_request({**req, "model": model}))


def run_model_test(model: str, prompt: str | None = None) -> dict[str, Any]:
    start = time.perf_counter()
    req: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt or _TEST_PROMPT}],
        "stream": False,
    }
    deltas = open_upstream_events(req, model)
    completion = merge_deltas(deltas, model)
    latency_ms = (time.perf_counter() - start) * 1000
    message = completion["choices"][0]["message"]
    usage = completion.get("usage") or {}
    return {
        "ok": True,
        "text": str(message.get("content") or ""),
        "latency_ms": round(latency_ms),
        "tokens": int(usage.get("total_tokens", 0) or 0),
    }


def regenerate_api_key() -> str:
    key = "sk-oc-free-" + secrets.token_hex(20)
    KEY_FILE.write_text(key + "\n", encoding="utf-8")
    try:
        KEY_FILE.chmod(0o600)
    except OSError:
        pass
    return key


SERVER: Any = None


def dashboard_state() -> dict[str, Any]:
    try:
        models = available_models()
    except Exception:  # noqa: BLE001
        models = list(FALLBACK_MODELS)
    return {
        "ok": True,
        "api_key": api_key_cache() or "(benoetigt Neustart)",
        "models": models,
        "log": list(REQUEST_LOG),
    }


def api_key_cache() -> str | None:
    try:
        value = KEY_FILE.read_text(encoding="utf-8").strip()
        return value or None
    except OSError:
        return None


# ---------------------------------------------------------------- API key


def ensure_api_key() -> str:
    if KEY_FILE.exists():
        key = KEY_FILE.read_text(encoding="utf-8").strip()
        if key:
            return key
    key = "sk-oc-free-" + secrets.token_hex(20)
    KEY_FILE.write_text(key + "\n", encoding="utf-8")
    try:
        KEY_FILE.chmod(0o600)
    except OSError:
        pass
    return key


# ---------------------------------------------------------------- HTTP server

# --- console cosmetics -----------------------------------------------


def _enable_windows_vt() -> bool:
    """Enable ANSI colors in the Windows console; False if not supported."""
    if os.name != "nt":
        return True
    try:
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        for handle_id in (-11, -12):  # stdout, stderr
            handle = kernel32.GetStdHandle(handle_id)
            mode = ctypes.c_uint32()
            if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
                return False
            if not kernel32.SetConsoleMode(handle, mode.value | 0x0004):
                return False
        kernel32.SetConsoleTitleW("oc-free-api - Port 8787")
        return True
    except Exception:  # noqa: BLE001
        return False


VT: bool = None  # set in _console_init()

_RESET = "\x1b[0m"
_COLORS = {
    "green": "\x1b[92m",
    "bright": "\x1b[1m",
    "dim": "\x1b[90m",
    "cyan": "\x1b[96m",
    "yellow": "\x1b[93m",
    "red": "\x1b[91m",
    "magenta": "\x1b[95m",
}


def _paint(text: str, color: str) -> str:
    if not VT:
        return text
    return _COLORS[color] + text + _RESET


def _emit(line: str) -> None:
    try:
        print(line, flush=True)
    except UnicodeEncodeError:
        print(line.encode("ascii", "replace").decode("ascii", "replace"), flush=True)


def _console_init() -> None:
    global VT
    VT = _enable_windows_vt()


def _print_banner(port: int, host: str, key: str, models: list[str]) -> None:
    dash_url = f"http://{host}:{port}/"
    base_url = f"http://{host}:{port}/v1"
    shown_models = ", ".join(models[:5]) + (" …" if len(models) > 5 else "")

    _emit("")
    _emit(_paint("┌" + "─" * 62 + "┐", "dim"))
    _emit("│ " + _paint("oc-free-api", "bright") + "  " + _paint("OpenCode Zen free models — lokal", "dim"))
    _emit(_paint("├" + "─" * 62 + "┤", "dim"))
    _emit("│ " + _paint(f"Dashboard  ", "cyan") + _paint(dash_url, "green"))
    _emit("│ " + _paint(f"Base URL   ", "cyan") + _paint(base_url, "bright"))
    _emit("│ " + _paint(f"API-Key    ", "cyan") + _paint(key, "yellow"))
    _emit("│ " + _paint(f"Modelle    ", "cyan") + _paint(f"{len(models)} verfügbar: {shown_models}", "bright"))
    _emit("│ " + _paint(f"Beenden    ", "cyan") + _paint("Ctrl+C hier oder Fenster schließen", "dim"))
    _emit(_paint("├" + "─" * 62 + "┤", "dim"))
    _emit("│ " + _paint("Live-Log der Anfragen …", "dim"))
    _emit(_paint("└" + "─" * 62 + "┘", "dim"))
    _emit("")


_STATUS_WORD = {
    200: "OK", 201: "OK", 204: "OK",
    400: "BAD REQUEST", 401: "UNAUTHORIZED", 403: "FORBIDDEN", 404: "NOT FOUND",
    429: "RATE LIMITED", 451: "BLOCKED",
    500: "SERVER ERROR", 502: "UPSTREAM ERROR", 503: "SERVICE UNAVAILABLE",
}


def _status_color(code: int) -> str:
    if 200 <= code < 300:
        return "green"
    if code == 429:
        return "magenta"
    if 400 <= code < 500:
        return "yellow"
    return "red"


class OpenAICompatibleHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "oc-free-api/1.0"

    api_key: str = ""

    # ---- console logging

    def _console_line(self, code: int | str) -> None:
        requestline = getattr(self, "requestline", "") or ""
        parts = requestline.split(" ")
        method = parts[0] if parts else "?"
        path = parts[1] if len(parts) > 1 else "/"
        try:
            code_num = int(code)
        except (TypeError, ValueError):
            code_num = 0
        word = _STATUS_WORD.get(code_num, "")
        elapsed = self._started_at() or ""
        line = (
            _paint(time.strftime("%H:%M:%S"), "dim")
            + "  "
            + _paint(f"{method:<6}", "bright")
            + f" {path:<38} "
            + _paint(str(code_num or code), _status_color(code_num) if code_num else "yellow")
        )
        del word
        if word:
            line += " " + _paint(word, "dim")
        if elapsed:
            line += " " + _paint(elapsed, "dim")
        REQUEST_LOG.append({"t": time.strftime("%H:%M:%S"), "msg": f"{method} {path} -> {code_num or code}"})
        sys.stderr.write(_paint("[oc-free-api] ", "dim") + line + "\n")

    def _started_at(self) -> str:
        start = getattr(self, "_t0", None)
        if not start:
            return ""
        return f"{(time.time() - start) * 1000:.0f}ms"

    def log_request(self, code: Any = "-", size: Any = "-") -> None:
        try:
            self._console_line(code)
        except Exception:  # noqa: BLE001
            pass

    def log_error(self, format: str, *args: Any) -> None:  # noqa: A002
        return

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        return

    # ---- helpers

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        # All request logging is done by log_request with colors.
        return

    def _authorized(self) -> bool:
        auth = self.headers.get("Authorization", "")
        if auth.startswith("Bearer ") and auth[7:].strip() == self.api_key:
            return True
        if self.headers.get("x-api-key", "").strip() == self.api_key:
            return True
        return False

    def _send_json(self, code: int, payload: dict[str, Any]) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(data)

    def _send_error_json(self, code: int, message: str) -> None:
        self._send_json(
            code,
            {"error": {"message": message, "type": "oc_free_api_error", "code": code}},
        )

    def _unauthorized(self) -> None:
        self._send_error_json(
            401,
            "Missing or invalid API key. Pass it as "
            "'Authorization: Bearer <key>' (key file: api-key.txt).",
        )

    # ---- GET

    def do_GET(self) -> None:  # noqa: N802
        self._t0 = time.time()
        path = self.path.split("?", 1)[0]
        if path == "/" or path == "/dashboard":
            try:
                payload = DASHBOARD_FILE.read_bytes()
            except OSError:
                self._send_error_json(500, "dashboard.html not found next to server.py")
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(payload)
            return
        if path == "/health":
            self._send_json(200, {"ok": True})
            return
        if path in {"/v1/models", "/api/v1/models"}:
            try:
                models = available_models()
            except Exception as exc:  # noqa: BLE001
                self._send_error_json(502, f"model discovery failed: {exc}")
                return
            self._send_json(
                200,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": model,
                            "object": "model",
                            "created": 0,
                            "owned_by": "opencode",
                        }
                        for model in models
                    ],
                },
            )
            return
        if path == "/api/dashboard/log":
            self._send_json(200, {"log": list(REQUEST_LOG)})
            return
        if path == "/api/dashboard":
            self._send_json(200, dashboard_state())
            return
        if path == "/api/tags":
            try:
                self._send_json(200, {"models": _ollama_model_entries()})
            except Exception as exc:  # noqa: BLE001
                self._send_error_json(502, f"model discovery failed: {exc}")
            return
        if path == "/api/ps":
            self._send_json(
                200,
                {
                    "models": [
                        {
                            "name": entry["name"],
                            "model": entry["model"],
                            "size": 0,
                            "size_vram": 0,
                            "digest": entry["digest"],
                            "details": entry["details"],
                            "expires_at": "0001-01-01T00:00:00Z",
                        }
                        for entry in _ollama_model_entries()
                    ]
                },
            )
            return
        if path in {"/api/version", "/version"}:
            self._send_json(200, {"version": "oc-free-api/1.0.0"})
            return
        if path in {"/v1/props", "/props"}:
            try:
                models = available_models()
            except Exception:  # noqa: BLE001
                models = list(FALLBACK_MODELS)
            model = models[0] if models else DEFAULT_MODEL
            self._send_json(
                200,
                {
                    "model": model,
                    "family": "opencode-free",
                    "prompt": "",
                    "n_keep": 0,
                    "n_discard": 0,
                    "n_predict": -1,
                    "n_ctx": 131072,
                    "temperature": 1.0,
                    "top_k": 40,
                    "top_p": 0.95,
                    "total_slots": 1,
                    "modalities": {"vision": False, "audio_in": False, "audio_out": False},
                    "default_generation_settings": {
                        "id": "slot-0",
                        "port": 0,
                        "model": model,
                        "n_predict": -1,
                        "n_keep": 0,
                        "truncation": 50,
                        "n_ctx": 131072,
                    },
                },
            )
            return
        self._send_error_json(404, f"unknown path: {path}")

    # ---- POST

    def do_POST(self) -> None:  # noqa: N802
        self._t0 = time.time()
        path = self.path.split("?", 1)[0]
        if path in {"/v1/chat/completions", "/chat/completions"}:
            self._post_chat_completions()
        elif path == "/api/chat":
            self._post_ollama_chat()
        elif path == "/api/generate":
            self._post_ollama_generate()
        elif path == "/api/show":
            self._post_ollama_show()
        elif path == "/api/dashboard/refresh":
            self._post_refresh_models()
        elif path == "/api/dashboard/test":
            self._post_dashboard_test()
        elif path == "/api/dashboard/key":
            self._post_new_key()
        elif path == "/api/dashboard/stop":
            self._post_stop_server()
        else:
            self._send_error_json(404, f"unknown path: {path}")

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        try:
            req = json.loads(raw.decode("utf-8")) if raw else {}
        except json.JSONDecodeError as exc:
            raise ProxyError(f"invalid JSON body: {exc}", 400) from exc
        if not isinstance(req, dict):
            raise ProxyError("request body must be a JSON object", 400)
        return req

    def _resolve_model(self, name: Any) -> str:
        model = normalize_model_name(name)
        try:
            models = available_models()
        except Exception:  # noqa: BLE001
            models = list(FALLBACK_MODELS)
        if models and model not in models:
            raise ProxyError(
                f"model '{model}' is not an available free model. "
                f"Call GET /v1/models or /api/tags for the current list "
                f"({', '.join(models)})",
                404,
            )
        return model

    def _open_upstream(self, req: dict[str, Any], model: str) -> Iterator[dict[str, Any]]:
        return open_upstream_events(req, model)

    # ---- dashboard API

    def _post_refresh_models(self) -> None:
        try:
            available_models(refresh=True)
            self._send_json(200, {"ok": True})
        except Exception as exc:  # noqa: BLE001
            self._send_error_json(502, f"model discovery failed: {exc}")

    def _post_dashboard_test(self) -> None:
        try:
            req = self._read_json()
            model = self._resolve_model(req.get("model"))
            self._send_json(200, run_model_test(model, req.get("prompt")))
        except ProxyError as exc:
            self._send_error_json(exc.status_code, str(exc))

    def _post_new_key(self) -> None:
        new_key = regenerate_api_key()
        OpenAICompatibleHandler.api_key = new_key
        self._send_json(200, {"ok": True, "api_key": new_key})

    def _post_stop_server(self) -> None:
        self._send_json(200, {"ok": True, "stopping": True})
        if SERVER is not None:
            threading.Thread(target=SERVER.shutdown, daemon=True).start()

    def _post_chat_completions(self) -> None:
        if not self._authorized():
            self._unauthorized()
            return
        try:
            req = self._read_json()
            model = self._resolve_model(req.get("model"))
            deltas = self._open_upstream(req, model)
            if bool(req.get("stream")):
                self._handle_streaming(deltas, model)
            else:
                self._send_json(200, merge_deltas(deltas, model))
        except ProxyError as exc:
            self._send_error_json(exc.status_code, str(exc))

    # ---- Ollama API

    def _post_ollama_chat(self) -> None:
        try:
            req = self._read_json()
            model = self._resolve_model(req.get("model"))
            openai_req = ollama_options(req)
            openai_req["messages"] = ollama_to_openai_messages(
                req.get("messages") or []
            )
            deltas = self._open_upstream(openai_req, model)
            include_thinking = bool(req.get("think"))
            if bool(req.get("stream")):
                self._write_header_stream("application/x-ndjson")
                self._ollama_chat_stream(deltas, model, include_thinking)
            else:
                completion = merge_deltas(deltas, model)
                self._send_json(200, ollama_chat_response(completion, include_thinking))
        except ProxyError as exc:
            self._send_json(
                exc.status_code,
                {"error": {"message": str(exc), "type": "oc_free_api_error"}},
            )

    def _post_ollama_generate(self) -> None:
        try:
            req = self._read_json()
            model = self._resolve_model(req.get("model"))
            if req.get("raw"):
                prompt = str(req.get("prompt") or "")
                messages = [{"role": "user", "content": prompt}]
            else:
                messages = [
                    m
                    for m in (
                        (
                            {"role": "system", "content": str(req.get("system") or "")}
                            if req.get("system")
                            else None
                        ),
                        {"role": "user", "content": str(req.get("prompt") or "")},
                    )
                    if m
                ]
            openai_req = ollama_options(req)
            openai_req["messages"] = messages
            deltas = self._open_upstream(openai_req, model)
            if bool(req.get("stream")):
                self._write_header_stream("application/x-ndjson")
                self._ollama_generate_stream(deltas, model, bool(req.get("think")))
            else:
                completion = merge_deltas(deltas, model)
                self._send_json(200, ollama_generate_response(completion))
        except ProxyError as exc:
            self._send_json(
                exc.status_code,
                {"error": {"message": str(exc), "type": "oc_free_api_error"}},
            )

    def _post_ollama_show(self) -> None:
        try:
            req = self._read_json()
        except ProxyError as exc:
            self._send_json(
                exc.status_code,
                {"error": {"message": str(exc), "type": "oc_free_api_error"}},
            )
            return
        name = normalize_model_name(req.get("name") or req.get("model"))
        response: dict[str, Any] = {
            "name": name,
            "model": name,
            "modified_at": _ollama_now(),
            "license": "personal-use (see repository)",
            "modelfile": f"FROM {name}",
            "parameters": "",
            "template": "{{ .Prompt }}",
            "details": _ollama_details(),
            "capabilities": ["completion", "streaming", "tools"],
            "model_info": {
                "general.architecture": "opencode-free",
                "general.name": name,
                "general.parameter_count": 0,
                "general.quantization_level": "",
            },
        }
        self._send_json(200, response)

    def _write_header_stream(self, content_type: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

    def _write_ndjson(self, payload: dict[str, Any]) -> bool:
        try:
            data = (json.dumps(payload, separators=(",", ":")) + "\n").encode("utf-8")
            self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
            self.wfile.flush()
            return True
        except OSError:
            return False

    def _ollama_chat_stream(
        self, deltas: Iterator[dict[str, Any]], model: str, include_thinking: bool
    ) -> None:
        created = _ollama_now()
        calls: dict[int, dict[str, Any]] = {}
        finish = "stop"
        usage: dict[str, Any] | None = None
        try:
            saw = False
            for event in deltas:
                saw = True
                if isinstance(event.get("usage"), dict):
                    usage = event["usage"]
                for choice in event.get("choices") or []:
                    if choice.get("finish_reason"):
                        finish = choice["finish_reason"]
                    delta = choice.get("delta") or {}
                    if delta.get("content"):
                        if not self._write_ndjson(
                            {
                                "model": model,
                                "created_at": created,
                                "message": {"role": "assistant", "content": str(delta["content"])},
                                "done": False,
                            }
                        ):
                            return
                    if include_thinking and delta.get("reasoning_content"):
                        if not self._write_ndjson(
                            {
                                "model": model,
                                "created_at": created,
                                "message": {
                                    "role": "assistant",
                                    "thinking": str(delta["reasoning_content"]),
                                },
                                "done": False,
                            }
                        ):
                            return
                    for raw_call in delta.get("tool_calls") or []:
                        index = int(raw_call.get("index", 0) or 0)
                        item = calls.setdefault(index, {"name": "", "arguments": ""})
                        function = raw_call.get("function") or {}
                        if function.get("name"):
                            item["name"] = str(function["name"])
                        item["arguments"] += str(function.get("arguments") or "")
            if not saw:
                raise ProxyError("upstream returned no events.")
            if finish == "content_filter":
                raise ProxyError("upstream blocked the response by content filter.", 400)
            ollama_calls = [
                {
                    "function": {
                        "name": item["name"],
                        "arguments": _json_obj(item["arguments"]),
                    }
                }
                for _, item in sorted(calls.items())
                if item["name"]
            ]
            message: dict[str, Any] = {"role": "assistant", "content": ""}
            if ollama_calls:
                message["tool_calls"] = ollama_calls
            chunk: dict[str, Any] = {
                "model": model,
                "created_at": created,
                "message": message,
                "done": True,
                "done_reason": "tool_calls" if ollama_calls else finish,
            }
            prompt_count, eval_count = _ollama_usage(usage)
            chunk["prompt_eval_count"] = prompt_count
            chunk["eval_count"] = eval_count
            self._write_ndjson(chunk)
        except (ProxyError, BrokenPipeError, ConnectionResetError):
            pass
        finally:
            try:
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
            except OSError:
                pass
            self.close_connection = True

    def _ollama_generate_stream(
        self, deltas: Iterator[dict[str, Any]], model: str, include_thinking: bool
    ) -> None:
        created = _ollama_now()
        finish = "stop"
        usage: dict[str, Any] | None = None
        try:
            saw = False
            for event in deltas:
                saw = True
                if isinstance(event.get("usage"), dict):
                    usage = event["usage"]
                for choice in event.get("choices") or []:
                    if choice.get("finish_reason"):
                        finish = choice["finish_reason"]
                    delta = choice.get("delta") or {}
                    if delta.get("content"):
                        if not self._write_ndjson(
                            {
                                "model": model,
                                "created_at": created,
                                "response": str(delta["content"]),
                                "done": False,
                            }
                        ):
                            return
                    if include_thinking and delta.get("reasoning_content"):
                        if not self._write_ndjson(
                            {
                                "model": model,
                                "created_at": created,
                                "thinking": str(delta["reasoning_content"]),
                                "done": False,
                            }
                        ):
                            return
            if not saw:
                raise ProxyError("upstream returned no events.")
            if finish == "content_filter":
                raise ProxyError("upstream blocked the response by content filter.", 400)
            prompt_count, eval_count = _ollama_usage(usage)
            self._write_ndjson(
                {
                    "model": model,
                    "created_at": created,
                    "response": "",
                    "done": True,
                    "done_reason": finish,
                    "prompt_eval_count": prompt_count,
                    "eval_count": eval_count,
                }
            )
        except (ProxyError, BrokenPipeError, ConnectionResetError):
            pass
        finally:
            try:
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
            except OSError:
                pass
            self.close_connection = True

    # ---- streaming response

    def _handle_streaming(self, deltas: Iterator[dict[str, Any]], model: str) -> None:
        completion_id = "chatcmpl-" + secrets.token_hex(12)
        created = int(time.time())
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        def write_chunk(payload: bytes) -> bool:
            self.wfile.write(f"{len(payload):x}\r\n".encode() + payload + b"\r\n")
            self.wfile.flush()
            return True

        def sse(event: dict[str, Any]) -> bytes:
            chunk: dict[str, Any] = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": str(event.get("model") or model),
                "choices": [
                    {
                        **choice,
                        "index": int(choice.get("index", 0) or 0),
                    }
                    for choice in (event.get("choices") or [])
                ],
            }
            if event.get("usage"):
                chunk["usage"] = event["usage"]
            return b"data: " + json.dumps(chunk, separators=(",", ":")).encode() + b"\n\n"

        try:
            first_done = False
            for event in deltas:
                if not first_done:
                    first_done = True
                write_chunk(sse(event))
            if not first_done:
                raise ProxyError("upstream returned no events.")
            write_chunk(b"data: [DONE]\n\n")
        except (ProxyError, BrokenPipeError, ConnectionResetError):
            pass
        finally:
            try:
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
            except OSError:
                pass
            self.close_connection = True


# ---------------------------------------------------------------- main


def main() -> None:
    parser = argparse.ArgumentParser(
        description="OpenAI-compatible local API for OpenCode Zen free models."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--key", help="use a specific API key instead of the generated one")
    parser.add_argument(
        "--refresh-models",
        action="store_true",
        help="refresh the model list once before starting",
    )
    args = parser.parse_args()

    if args.key:
        KEY_FILE.write_text(args.key + "\n", encoding="utf-8")

    api_key = ensure_api_key()
    models = available_models(refresh=args.refresh_models)

    OpenAICompatibleHandler.api_key = api_key

    server = ThreadingHTTPServer((args.host, args.port), OpenAICompatibleHandler)
    server.daemon_threads = True
    global SERVER
    SERVER = server
    base = f"http://{args.host}:{args.port}/v1"

    _console_init()
    _print_banner(args.port, args.host, api_key, models)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
