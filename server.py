from __future__ import annotations

import argparse
import ctypes
import difflib
import http.client
import ipaddress
import itertools
import json
import os
import re
import secrets
import shutil
import socket
import string
import subprocess
import sys
import threading
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterator

import integrations

APP_VERSION = "1.1.0"

FROZEN = bool(getattr(sys, "frozen", False))
BASE_DIR = Path(sys.executable).resolve().parent if FROZEN else Path(__file__).resolve().parent
RES_DIR = Path(getattr(sys, "_MEIPASS", BASE_DIR))
DASHBOARD_FILE = RES_DIR / "dashboard.html"
ASSETS_DIR = RES_DIR / "assets"
def default_data_dir() -> Path:
    if os.environ.get("ROUTE7Q9K_DATA_DIR"):
        return Path(os.environ["ROUTE7Q9K_DATA_DIR"])
    if FROZEN and os.name == "nt":
        return Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local") / "7q9k-route"
    return BASE_DIR


DATA_DIR = default_data_dir().resolve()
DATA_FILES = ("api-key.txt", "stats.json", "config.json", "pricing.json", "models.json", "server.log", "server.log.1")
STATS_FILE = DATA_DIR / "stats.json"
PRICING_FILE = DATA_DIR / "pricing.json"
CONFIG_FILE = DATA_DIR / "config.json"
LOG_FILE = DATA_DIR / "server.log"
ASSET_TYPES = {".woff2": "font/woff2", ".svg": "image/svg+xml", ".css": "text/css; charset=utf-8",
               ".ico": "image/x-icon", ".png": "image/png"}
MODEL_CACHE = DATA_DIR / "models.json"
KEY_FILE = DATA_DIR / "api-key.txt"

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
MAX_BODY_BYTES = 32 * 1024 * 1024

_MODEL_LOCK = threading.Lock()
_MODEL_SNAPSHOT: tuple[str, ...] | None = None
_MODEL_SOURCE = "fallback"
_MODEL_UPDATED = 0.0

_BASE62 = string.digits + string.ascii_letters
_ID_LOCK = threading.Lock()
_LAST_ID_TS = 0
_ID_COUNTER = 0


class ProxyError(Exception):
    status_code: int = 502
    # Set for upstream failures worth retrying (5xx, connection errors).
    retryable: bool = False

    def __init__(self, message: str, status_code: int = 502):
        super().__init__(message)
        self.status_code = status_code


def _session_id() -> str:
    return _oc_id("ses", descending=True)


def _request_id() -> str:
    return _oc_id("msg", descending=False)


def _oc_id(prefix: str, *, descending: bool) -> str:
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
    global _MODEL_SNAPSHOT, _MODEL_SOURCE, _MODEL_UPDATED
    with _MODEL_LOCK:
        if _MODEL_SNAPSHOT is not None and not refresh:
            return list(_MODEL_SNAPSHOT)
    try:
        models = _free_live_models(
            _json_get(ZEN_MODELS_URL, timeout),
            _json_get(MODELS_DEV_URL, timeout),
        )
    except (OSError, ValueError):
        models = []
    with _MODEL_LOCK:
        if models:
            _save_model_cache(models)
            _MODEL_SNAPSHOT, _MODEL_SOURCE = tuple(models), "live"
            _MODEL_UPDATED = time.time()
        elif _MODEL_SNAPSHOT is None:
            _preload_models_locked()
        return list(_MODEL_SNAPSHOT)


def _preload_models_locked() -> None:
    global _MODEL_SNAPSHOT, _MODEL_SOURCE, _MODEL_UPDATED
    cached = _load_model_cache()
    _MODEL_SNAPSHOT = tuple(cached or FALLBACK_MODELS)
    _MODEL_SOURCE = "cache" if cached else "fallback"
    _MODEL_UPDATED = time.time()


def preload_models() -> None:
    with _MODEL_LOCK:
        if _MODEL_SNAPSHOT is None:
            _preload_models_locked()


def model_source() -> dict[str, Any]:
    with _MODEL_LOCK:
        return {"source": _MODEL_SOURCE, "updated": _MODEL_UPDATED}


def uses_responses_api(model: str) -> bool:
    return model in RESPONSES_MODELS


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req: Any, fp: Any, code: int, msg: str, headers: Any, newurl: Any) -> None:
        return None


_OPENER = urllib.request.build_opener(_NoRedirect)


def _sse_lines_to_payloads(response: Any) -> Iterator[dict[str, Any]]:
    try:
        for raw_line in response:
            line = raw_line.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                return
            if payload:
                try:
                    yield json.loads(payload)
                except json.JSONDecodeError as exc:
                    raise ProxyError(f"upstream sent invalid JSON: {payload[:200]}") from exc
    except TimeoutError as exc:
        raise ProxyError("upstream timed out while streaming.", 504) from exc
    except (OSError, http.client.HTTPException) as exc:
        raise ProxyError(f"upstream connection lost: {type(exc).__name__}") from exc
    finally:
        try:
            response.close()
        except Exception:
            pass


# OpenCode answers HTTP 5xx ("Endpoint is unavailable") while a free model's
# backend briefly drops out; it usually works again within a few seconds.
# Retried before anything is streamed to the client, so no output repeats.
# Rate limits (429) are never retried.
UPSTREAM_RETRY_DELAYS = (0.5, 1.0, 2.0, 4.0)
RETRYABLE_UPSTREAM_STATUS = frozenset({500, 502, 503, 504})


def upstream_stream(
    url: str, body: dict[str, Any], timeout: float = UPSTREAM_TIMEOUT
) -> Iterator[dict[str, Any]]:
    delays = iter(UPSTREAM_RETRY_DELAYS)
    while True:
        try:
            return _upstream_stream_once(url, body, timeout)
        except ProxyError as exc:
            delay = next(delays, None) if getattr(exc, "retryable", False) else None
            if delay is None:
                raise
            _emit(_paint("[7q9k route] ", "dim")
                  + f"{body.get('model', '?')}: {str(exc)[:160]} - retrying in {delay:g}s", sys.stderr)
            time.sleep(delay)


def _upstream_stream_once(
    url: str, body: dict[str, Any], timeout: float
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
        try:
            detail = exc.read().decode("utf-8", "replace")[:500]
        except Exception:
            detail = ""
        status = exc.code if 400 <= exc.code < 500 else 502
        error = ProxyError(f"upstream rejected request (HTTP {exc.code}): {detail}", status)
        error.retryable = exc.code in RETRYABLE_UPSTREAM_STATUS
        raise error from exc
    except urllib.error.URLError as exc:
        error = ProxyError(f"upstream connection failed: {exc.reason}")
        error.retryable = True
        raise error from exc
    except TimeoutError as exc:
        raise ProxyError("upstream timed out.", 504) from exc
    except (OSError, http.client.HTTPException) as exc:
        raise ProxyError(f"upstream connection failed: {exc}") from exc

    return _sse_lines_to_payloads(response)


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


def validate_messages(messages: Any) -> list[dict[str, Any]]:
    if messages is None:
        return []
    if not isinstance(messages, list) or not all(isinstance(m, dict) for m in messages):
        raise ProxyError("'messages' must be a list of message objects.", 400)
    return messages


def build_chat_request(req: dict[str, Any]) -> dict[str, Any]:
    messages = list(validate_messages(req.get("messages")))
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
        elif kind in {"image_url", "input_image"} and not output:
            url = part.get("image_url") or ""
            if isinstance(url, dict):
                url = url.get("url") or ""
            if url:
                result.append({"type": "input_image", "image_url": str(url)})
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
    tools = [t for t in req.get("tools") or [] if isinstance(t, dict)] if isinstance(req.get("tools"), list) else []
    input_items, instructions = _responses_input(list(validate_messages(req.get("messages"))))
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


def _responses_parts_to_chat(content: Any) -> Any:
    if not isinstance(content, list):
        return _message_text(content)
    parts: list[dict[str, Any]] = []
    for part in content:
        if isinstance(part, str):
            parts.append({"type": "text", "text": part})
            continue
        if not isinstance(part, dict):
            continue
        kind = str(part.get("type") or "")
        if kind in {"input_text", "output_text", "text", "summary_text"}:
            parts.append({"type": "text", "text": str(part.get("text") or "")})
        elif kind == "refusal":
            parts.append({"type": "text", "text": str(part.get("refusal") or "")})
        elif kind == "input_image":
            url = part.get("image_url") or ""
            if isinstance(url, dict):
                url = url.get("url") or ""
            if url:
                parts.append({"type": "image_url", "image_url": {"url": str(url)}})
    if all(p["type"] == "text" for p in parts):
        return "".join(p["text"] for p in parts)
    return parts


def _tool_output_text(output: Any) -> str:
    if isinstance(output, str):
        return output
    if isinstance(output, list):
        return _message_text(output)
    return json.dumps(output) if output is not None else ""


def responses_to_chat_request(req: dict[str, Any]) -> dict[str, Any]:
    if req.get("previous_response_id"):
        raise ProxyError(
            "previous_response_id is not supported (stateless proxy); "
            "send the full conversation in 'input' and set store=false.",
            400,
        )
    messages: list[dict[str, Any]] = []
    if req.get("instructions"):
        messages.append({"role": "system", "content": str(req["instructions"])})

    raw_input = req.get("input")
    items = [{"role": "user", "content": raw_input}] if isinstance(raw_input, str) else raw_input
    for item in items or []:
        if not isinstance(item, dict):
            continue
        kind = str(item.get("type") or "message")
        if kind == "message":
            role = str(item.get("role") or "user")
            role = "system" if role == "developer" else role
            messages.append({"role": role, "content": _responses_parts_to_chat(item.get("content"))})
        elif kind == "function_call":
            call = {
                "id": str(item.get("call_id") or item.get("id") or ""),
                "type": "function",
                "function": {
                    "name": str(item.get("name") or ""),
                    "arguments": str(item.get("arguments") or "{}"),
                },
            }
            last = messages[-1] if messages else None
            if last and last.get("role") == "assistant" and "tool_calls" in last:
                last["tool_calls"].append(call)
            elif last and last.get("role") == "assistant" and not last.get("tool_calls"):
                last["tool_calls"] = [call]
            else:
                messages.append({"role": "assistant", "content": None, "tool_calls": [call]})
        elif kind == "function_call_output":
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": str(item.get("call_id") or ""),
                    "content": _tool_output_text(item.get("output")),
                }
            )

    chat: dict[str, Any] = {"model": req.get("model"), "messages": messages}
    tools = []
    for tool in req.get("tools") or []:
        if isinstance(tool, dict) and tool.get("type") == "function" and tool.get("name"):
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": str(tool["name"]),
                        "description": str(tool.get("description") or ""),
                        "parameters": tool.get("parameters") or {"type": "object", "properties": {}},
                    },
                }
            )
    if tools:
        chat["tools"] = tools
        choice = req.get("tool_choice")
        if isinstance(choice, dict) and choice.get("type") == "function":
            chat["tool_choice"] = {"type": "function", "function": {"name": str(choice.get("name") or "")}}
        elif choice in {"auto", "none", "required"}:
            chat["tool_choice"] = choice
        if req.get("parallel_tool_calls") is not None:
            chat["parallel_tool_calls"] = bool(req["parallel_tool_calls"])
    for key in ("temperature", "top_p", "user"):
        if req.get(key) is not None:
            chat[key] = req[key]
    if req.get("max_output_tokens") is not None:
        chat["max_tokens"] = req["max_output_tokens"]
    fmt = (req.get("text") or {}).get("format") if isinstance(req.get("text"), dict) else None
    if isinstance(fmt, dict):
        if fmt.get("type") == "json_object":
            chat["response_format"] = {"type": "json_object"}
        elif fmt.get("type") == "json_schema":
            chat["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": str(fmt.get("name") or "output"),
                    "schema": fmt.get("schema") or {},
                    "strict": bool(fmt.get("strict", False)),
                },
            }
    return chat


def _responses_usage(usage: dict[str, Any] | None) -> dict[str, Any]:
    u = _usage(usage)
    return {
        "input_tokens": u["prompt_tokens"],
        "input_tokens_details": {"cached_tokens": u["prompt_tokens_details"]["cached_tokens"]},
        "output_tokens": u["completion_tokens"],
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": u["total_tokens"],
    }


class ResponsesBuilder:

    def __init__(self, model: str, req: dict[str, Any]) -> None:
        self.id = "resp_" + secrets.token_hex(12)
        self.created = int(time.time())
        self.model = model
        self.req = req
        self.output: list[dict[str, Any]] = []
        self.usage: dict[str, Any] | None = None
        self.status = "in_progress"
        self.error: dict[str, Any] | None = None
        self.seq = 0
        self._open: dict[str, Any] | None = None
        self._tool_items: dict[int, dict[str, Any]] = {}


    def _event(self, kind: str, **fields: Any) -> dict[str, Any]:
        event = {"type": kind, "sequence_number": self.seq, **fields}
        self.seq += 1
        return event

    def response(self) -> dict[str, Any]:
        body: dict[str, Any] = {
            "id": self.id,
            "object": "response",
            "created_at": self.created,
            "status": self.status,
            "error": self.error,
            "incomplete_details": None,
            "instructions": self.req.get("instructions"),
            "model": self.model,
            "output": self.output,
            "parallel_tool_calls": bool(self.req.get("parallel_tool_calls", True)),
            "previous_response_id": None,
            "store": False,
            "temperature": self.req.get("temperature"),
            "top_p": self.req.get("top_p"),
            "max_output_tokens": self.req.get("max_output_tokens"),
            "tool_choice": self.req.get("tool_choice", "auto"),
            "tools": self.req.get("tools") or [],
            "text": self.req.get("text") or {"format": {"type": "text"}},
            "usage": _responses_usage(self.usage) if self.usage is not None else None,
            "metadata": self.req.get("metadata") or {},
        }
        if self.status == "completed":
            body["output_text"] = "".join(
                part.get("text", "")
                for item in self.output
                if item.get("type") == "message"
                for part in item.get("content", [])
            )
        return body

    def start(self) -> list[dict[str, Any]]:
        return [
            self._event("response.created", response=self.response()),
            self._event("response.in_progress", response=self.response()),
        ]


    def _close_open(self) -> list[dict[str, Any]]:
        item, self._open = self._open, None
        if item is None:
            return []
        index = self.output.index(item)
        events: list[dict[str, Any]] = []
        if item["type"] == "message":
            part = item["content"][0]
            events.append(self._event("response.output_text.done", item_id=item["id"],
                                      output_index=index, content_index=0, text=part["text"]))
            events.append(self._event("response.content_part.done", item_id=item["id"],
                                      output_index=index, content_index=0, part=part))
        elif item["type"] == "reasoning":
            summary = item["summary"][0]
            events.append(self._event("response.reasoning_summary_text.done", item_id=item["id"],
                                      output_index=index, summary_index=0, text=summary["text"]))
            events.append(self._event("response.reasoning_summary_part.done", item_id=item["id"],
                                      output_index=index, summary_index=0, part=summary))
        elif item["type"] == "function_call":
            events.append(self._event("response.function_call_arguments.done", item_id=item["id"],
                                      output_index=index, arguments=item["arguments"]))
        item["status"] = "completed"
        events.append(self._event("response.output_item.done", output_index=index, item=item))
        return events

    def _open_item(self, item: dict[str, Any]) -> list[dict[str, Any]]:
        events = self._close_open()
        self.output.append(item)
        self._open = item
        index = len(self.output) - 1
        events.append(self._event("response.output_item.added", output_index=index,
                                  item={**item, "content": []} if item["type"] == "message" else
                                  {**item, "summary": []} if item["type"] == "reasoning" else dict(item)))
        if item["type"] == "message":
            events.append(self._event("response.content_part.added", item_id=item["id"],
                                      output_index=index, content_index=0,
                                      part={"type": "output_text", "text": "", "annotations": []}))
        elif item["type"] == "reasoning":
            events.append(self._event("response.reasoning_summary_part.added", item_id=item["id"],
                                      output_index=index, summary_index=0,
                                      part={"type": "summary_text", "text": ""}))
        return events

    def feed(self, event: dict[str, Any]) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        if event.get("error"):
            raise ProxyError(f"upstream inference error: {event['error']}")
        if isinstance(event.get("usage"), dict):
            self.usage = event["usage"]
        for choice in event.get("choices") or []:
            delta = choice.get("delta") or {}
            thought = delta.get("reasoning_content") or delta.get("reasoning")
            if thought:
                if not self._open or self._open["type"] != "reasoning":
                    events += self._open_item({"id": "rs_" + secrets.token_hex(8), "type": "reasoning",
                                               "summary": [{"type": "summary_text", "text": ""}]})
                item = self._open
                item["summary"][0]["text"] += str(thought)
                events.append(self._event("response.reasoning_summary_text.delta", item_id=item["id"],
                                          output_index=self.output.index(item), summary_index=0,
                                          delta=str(thought)))
            if delta.get("content"):
                if not self._open or self._open["type"] != "message":
                    events += self._open_item({"id": "msg_" + secrets.token_hex(8), "type": "message",
                                               "status": "in_progress", "role": "assistant",
                                               "content": [{"type": "output_text", "text": "",
                                                            "annotations": []}]})
                item = self._open
                item["content"][0]["text"] += str(delta["content"])
                events.append(self._event("response.output_text.delta", item_id=item["id"],
                                          output_index=self.output.index(item), content_index=0,
                                          delta=str(delta["content"])))
            # Tool calls are buffered and sent in finish(): upstream may interleave
            # argument chunks with text, reasoning or other calls.
            for raw in delta.get("tool_calls") or []:
                index = int(raw.get("index", 0) or 0)
                function = raw.get("function") or {}
                item = self._tool_items.setdefault(
                    index, {"id": "fc_" + secrets.token_hex(8), "type": "function_call",
                            "status": "in_progress", "call_id": "", "name": "", "arguments": ""})
                if raw.get("id") and not item["call_id"]:
                    item["call_id"] = str(raw["id"])
                if function.get("name") and not item["name"]:
                    item["name"] = str(function["name"])
                item["arguments"] += str(function.get("arguments") or "")
            if choice.get("finish_reason") == "content_filter":
                raise ProxyError("upstream blocked the response by content filter.", 400)
        return events

    def finish(self) -> list[dict[str, Any]]:
        events = self._close_open()
        for index in sorted(self._tool_items):
            item = self._tool_items[index]
            item["call_id"] = item["call_id"] or f"call_{secrets.token_hex(6)}"
            item["arguments"] = item["arguments"] or "{}"
            arguments, item["arguments"] = item["arguments"], ""
            events += self._open_item(item)
            item["arguments"] = arguments
            events.append(self._event("response.function_call_arguments.delta", item_id=item["id"],
                                      output_index=self.output.index(item), delta=arguments))
            events += self._close_open()
        self._tool_items = {}
        self.status = "completed"
        events.append(self._event("response.completed", response=self.response()))
        return events

    def fail(self, message: str, code: str = "upstream_error") -> list[dict[str, Any]]:
        self.status = "failed"
        self.error = {"code": code, "message": message}
        return [self._event("response.failed", response=self.response())]


def build_responses_object(deltas: Iterator[dict[str, Any]], model: str, req: dict[str, Any]) -> dict[str, Any]:
    builder = ResponsesBuilder(model, req)
    seen = False
    for event in deltas:
        seen = True
        builder.feed(event)
    if not seen:
        raise ProxyError("upstream returned no events.")
    builder.finish()
    return builder.response()


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
            "digest": "7q9k-route-" + model,
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
                if not isinstance(call, dict):
                    continue
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


# Anthropic Messages API, used by Claude Code. Requests are turned into Chat
# Completions for OpenCode and the answer is turned back into Anthropic blocks.

ANTHROPIC_STOP_REASONS = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "function_call": "tool_use",
    "content_filter": "refusal",
}
ANTHROPIC_ERROR_TYPES = {
    400: "invalid_request_error",
    401: "authentication_error",
    403: "permission_error",
    404: "not_found_error",
    413: "request_too_large",
    429: "rate_limit_error",
}
# Thinking blocks need a signature; Claude Code sends it back but nothing checks it here.
ANTHROPIC_THINKING_SIGNATURE = "7q9k-route"
# Model names Claude Code uses on its own (aliases, background tasks).
CLAUDE_MODEL_ALIASES = frozenset({"opus", "sonnet", "haiku", "default", "best", "opusplan"})


def _anthropic_blocks(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if not isinstance(content, list):
        return []
    return [
        {"type": "text", "text": block} if isinstance(block, str) else block
        for block in content
        if isinstance(block, (str, dict))
    ]


def _anthropic_text(content: Any) -> str:
    return "\n\n".join(
        str(block.get("text") or "")
        for block in _anthropic_blocks(content)
        if block.get("type") == "text" and block.get("text")
    )


def _anthropic_image(block: dict[str, Any]) -> dict[str, Any] | None:
    source = block.get("source") if isinstance(block.get("source"), dict) else {}
    if source.get("type") == "base64" and source.get("data"):
        url = f"data:{source.get('media_type') or 'image/png'};base64,{source['data']}"
    elif source.get("type") == "url" and source.get("url"):
        url = str(source["url"])
    else:
        return None
    return {"type": "image_url", "image_url": {"url": url}}


def _anthropic_tool_result(block: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    """Text for the tool message and images, which tool messages cannot carry."""
    content = block.get("content")
    if isinstance(content, list):
        texts, images = [], []
        for part in _anthropic_blocks(content):
            if part.get("type") == "text":
                texts.append(str(part.get("text") or ""))
            elif part.get("type") == "image":
                image = _anthropic_image(part)
                if image:
                    images.append(image)
                    texts.append("[image attached below]")
        text = "\n".join(texts)
    else:
        text, images = ("" if content is None else str(content)), []
    if block.get("is_error"):
        text = "Error: " + text
    return text or "(no output)", images


def _anthropic_user_content(parts: list[dict[str, Any]]) -> Any:
    if all(part["type"] == "text" for part in parts):
        return "\n\n".join(part["text"] for part in parts)
    return parts


def anthropic_to_chat_messages(req: dict[str, Any]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    system = _anthropic_text(req.get("system"))
    if system.strip():
        messages.append({"role": "system", "content": system})
    for message in validate_messages(req.get("messages")):
        blocks = _anthropic_blocks(message.get("content"))
        if message.get("role") == "assistant":
            texts: list[str] = []
            calls: list[dict[str, Any]] = []
            for block in blocks:
                kind = block.get("type")
                if kind == "text":
                    texts.append(str(block.get("text") or ""))
                elif kind in ("tool_use", "server_tool_use"):
                    arguments = block.get("input") if block.get("input") is not None else {}
                    calls.append(
                        {
                            "id": str(block.get("id") or f"toolu_{len(calls)}"),
                            "type": "function",
                            "function": {
                                "name": str(block.get("name") or ""),
                                "arguments": json.dumps(arguments, ensure_ascii=False),
                            },
                        }
                    )
            entry: dict[str, Any] = {"role": "assistant", "content": "".join(texts) or None}
            if calls:
                entry["tool_calls"] = calls
            if entry["content"] is not None or calls:
                messages.append(entry)
            continue
        parts: list[dict[str, Any]] = []
        images: list[dict[str, Any]] = []
        for block in blocks:
            kind = block.get("type")
            if kind == "tool_result":
                text, found = _anthropic_tool_result(block)
                messages.append(
                    {"role": "tool", "tool_call_id": str(block.get("tool_use_id") or ""), "content": text}
                )
                images.extend(found)
            elif kind == "text" and block.get("text"):
                parts.append({"type": "text", "text": str(block["text"])})
            elif kind == "image":
                image = _anthropic_image(block)
                if image:
                    parts.append(image)
            elif kind == "document":
                source = block.get("source") if isinstance(block.get("source"), dict) else {}
                if source.get("type") == "text" and source.get("data"):
                    parts.append({"type": "text", "text": str(source["data"])})
        parts = images + parts
        if parts:
            messages.append({"role": "user", "content": _anthropic_user_content(parts)})
    return messages


def anthropic_tools(tools: Any) -> list[dict[str, Any]]:
    result = []
    for tool in tools if isinstance(tools, list) else []:
        # Typed tools (web_search_..., text_editor_...) run on Anthropic's side.
        if not isinstance(tool, dict) or not tool.get("name") or tool.get("type") not in (None, "custom"):
            continue
        schema = tool.get("input_schema")
        result.append(
            {
                "type": "function",
                "function": {
                    "name": str(tool["name"]),
                    "description": str(tool.get("description") or ""),
                    "parameters": schema if isinstance(schema, dict) else {"type": "object", "properties": {}},
                },
            }
        )
    return result


def _anthropic_tool_choice(choice: Any) -> Any:
    if not isinstance(choice, dict):
        return None
    kind = choice.get("type")
    if kind == "tool" and choice.get("name"):
        return {"type": "function", "function": {"name": str(choice["name"])}}
    return {"auto": "auto", "any": "required", "none": "none"}.get(str(kind))


def anthropic_to_chat_request(req: dict[str, Any]) -> dict[str, Any]:
    # max_tokens is not passed on: Claude Code asks for more than most free models allow.
    chat: dict[str, Any] = {"model": req.get("model"), "messages": anthropic_to_chat_messages(req)}
    tools = anthropic_tools(req.get("tools"))
    if tools:
        chat["tools"] = tools
        choice = _anthropic_tool_choice(req.get("tool_choice"))
        if choice is not None:
            chat["tool_choice"] = choice
        if isinstance(req.get("tool_choice"), dict) and req["tool_choice"].get("disable_parallel_tool_use"):
            chat["parallel_tool_calls"] = False
    for key in ("temperature", "top_p"):
        if isinstance(req.get(key), (int, float)) and not isinstance(req.get(key), bool):
            chat[key] = req[key]
    stops = req.get("stop_sequences")
    if isinstance(stops, list) and stops:
        chat["stop"] = [str(s) for s in stops][:4]
    return chat


def anthropic_wants_thinking(req: dict[str, Any]) -> bool:
    thinking = req.get("thinking")
    return isinstance(thinking, dict) and thinking.get("type") in ("enabled", "adaptive")


def anthropic_estimate_tokens(req: dict[str, Any]) -> int:
    # Rough count (about 4 characters per token); OpenCode has no counting endpoint.
    text = json.dumps(
        [req.get("system"), req.get("messages"), req.get("tools")], ensure_ascii=False, separators=(",", ":")
    )
    return max(1, len(text) // 4)


def _anthropic_usage(usage: Any) -> dict[str, int]:
    usage = _usage(usage)
    cached = usage["prompt_tokens_details"]["cached_tokens"]
    return {
        "input_tokens": max(0, usage["prompt_tokens"] - cached),
        "output_tokens": usage["completion_tokens"],
        "cache_read_input_tokens": cached,
        "cache_creation_input_tokens": 0,
    }


def _anthropic_tool_id(value: Any) -> str:
    text = re.sub(r"[^A-Za-z0-9_-]", "_", str(value or ""))
    return text or "toolu_" + secrets.token_hex(12)


def anthropic_message(completion: dict[str, Any], *, thinking: bool) -> dict[str, Any]:
    choice = completion["choices"][0]
    message = choice["message"]
    content: list[dict[str, Any]] = []
    reasoning = message.get("reasoning_content") or message.get("reasoning")
    if thinking and reasoning:
        content.append({"type": "thinking", "thinking": str(reasoning), "signature": ANTHROPIC_THINKING_SIGNATURE})
    if message.get("content"):
        content.append({"type": "text", "text": str(message["content"])})
    for call in message.get("tool_calls") or []:
        function = call.get("function") or {}
        content.append(
            {
                "type": "tool_use",
                "id": _anthropic_tool_id(call.get("id")),
                "name": str(function.get("name") or ""),
                "input": _json_obj(function.get("arguments")),
            }
        )
    reason = "tool_use" if message.get("tool_calls") else ANTHROPIC_STOP_REASONS.get(
        str(choice.get("finish_reason") or "stop"), "end_turn"
    )
    return {
        "id": "msg_" + secrets.token_hex(12),
        "type": "message",
        "role": "assistant",
        "model": completion.get("model"),
        "content": content,
        "stop_reason": reason,
        "stop_sequence": None,
        "usage": _anthropic_usage(completion.get("usage")),
    }


class AnthropicStream:
    """Turns Chat Completions deltas into Anthropic stream events.

    Text and thinking stream as they arrive; tool calls are sent whole at the
    end, because OpenAI-style deltas may interleave several calls.
    """

    def __init__(self, model: str, *, thinking: bool, input_tokens: int) -> None:
        self.model = model
        self.thinking = thinking
        self.input_tokens = input_tokens
        self.id = "msg_" + secrets.token_hex(12)
        self.index = -1
        self.open: str | None = None
        self.calls: dict[int, dict[str, str]] = {}
        self.finish_reason = "stop"
        self.usage: Any = None

    @staticmethod
    def _event(kind: str, **fields: Any) -> tuple[str, dict[str, Any]]:
        return kind, {"type": kind, **fields}

    def start(self) -> list[tuple[str, dict[str, Any]]]:
        message = {
            "id": self.id,
            "type": "message",
            "role": "assistant",
            "model": self.model,
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": self.input_tokens, "output_tokens": 0},
        }
        return [self._event("message_start", message=message), self._event("ping")]

    def _close(self) -> list[tuple[str, dict[str, Any]]]:
        if self.open is None:
            return []
        events = []
        if self.open == "thinking":
            events.append(self._event(
                "content_block_delta", index=self.index,
                delta={"type": "signature_delta", "signature": ANTHROPIC_THINKING_SIGNATURE},
            ))
        events.append(self._event("content_block_stop", index=self.index))
        self.open = None
        return events

    def _open(self, kind: str, block: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
        events = self._close()
        self.index += 1
        self.open = kind
        events.append(self._event("content_block_start", index=self.index, content_block=block))
        return events

    def _delta(self, kind: str, delta: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
        events: list[tuple[str, dict[str, Any]]] = []
        if self.open != kind:
            block = {"type": "thinking", "thinking": "", "signature": ""} if kind == "thinking" \
                else {"type": "text", "text": ""}
            events = self._open(kind, block)
        events.append(self._event("content_block_delta", index=self.index, delta=delta))
        return events

    def feed(self, event: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
        if event.get("error"):
            raise ProxyError(f"upstream inference error: {event['error']}")
        self.model = str(event.get("model") or self.model)
        if isinstance(event.get("usage"), dict):
            self.usage = event["usage"]
        events: list[tuple[str, dict[str, Any]]] = []
        for choice in event.get("choices") or []:
            if choice.get("finish_reason"):
                self.finish_reason = str(choice["finish_reason"])
            delta = choice.get("delta") or {}
            thought = delta.get("reasoning_content") or delta.get("reasoning")
            if self.thinking and thought:
                events += self._delta("thinking", {"type": "thinking_delta", "thinking": str(thought)})
            if delta.get("content"):
                events += self._delta("text", {"type": "text_delta", "text": str(delta["content"])})
            for raw_call in delta.get("tool_calls") or []:
                index = int(raw_call.get("index", 0) or 0)
                item = self.calls.setdefault(index, {"id": "", "name": "", "arguments": ""})
                if raw_call.get("id"):
                    item["id"] = str(raw_call["id"])
                function = raw_call.get("function") or {}
                if function.get("name"):
                    item["name"] = str(function["name"])
                item["arguments"] += str(function.get("arguments") or "")
        return events

    def finish(self) -> list[tuple[str, dict[str, Any]]]:
        if self.finish_reason == "content_filter" and not self.calls and self.open is None:
            raise ProxyError("upstream blocked the response by content filter.", 400)
        events = self._close()
        calls = [self.calls[i] for i in sorted(self.calls) if self.calls[i]["name"]]
        for call in calls:
            events += self._open("tool_use", {
                "type": "tool_use", "id": _anthropic_tool_id(call["id"]), "name": call["name"], "input": {},
            })
            partial = json.dumps(_json_obj(call["arguments"]), ensure_ascii=False)
            events.append(self._event(
                "content_block_delta", index=self.index, delta={"type": "input_json_delta", "partial_json": partial},
            ))
            events += self._close()
        reason = "tool_use" if calls else ANTHROPIC_STOP_REASONS.get(self.finish_reason, "end_turn")
        usage = _anthropic_usage(self.usage) if self.usage is not None else {
            "input_tokens": self.input_tokens, "output_tokens": 0,
        }
        events.append(self._event("message_delta", delta={"stop_reason": reason, "stop_sequence": None}, usage=usage))
        events.append(self._event("message_stop"))
        return events

    @staticmethod
    def fail(message: str, status: int = 502) -> list[tuple[str, dict[str, Any]]]:
        kind = ANTHROPIC_ERROR_TYPES.get(status, "api_error")
        return [("error", {"type": "error", "error": {"type": kind, "message": message}})]


_TEST_PROMPT = "Reply with exactly the token: OK-DASH"


TEST_TIMEOUT = 120.0


def open_upstream_events(
    req: dict[str, Any], model: str, timeout: float = UPSTREAM_TIMEOUT
) -> Iterator[dict[str, Any]]:
    if uses_responses_api(model):
        events = upstream_stream(RESPONSES_URL, build_responses_request({**req, "model": model}), timeout)
        return responses_events_to_deltas(events)
    return upstream_stream(DIRECT_URL, build_chat_request({**req, "model": model}), timeout)


def run_model_test(model: str, prompt: str | None = None) -> dict[str, Any]:
    start = time.perf_counter()
    req: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt or _TEST_PROMPT}],
        "stream": False,
    }
    completion = merge_deltas(open_upstream_events(req, model, timeout=TEST_TIMEOUT), model)
    latency_ms = (time.perf_counter() - start) * 1000
    message = completion["choices"][0]["message"]
    usage = _usage(completion.get("usage"))
    return {
        "ok": True,
        "text": str(message.get("content") or ""),
        "latency_ms": round(latency_ms),
        "tokens": usage["total_tokens"],
        "usage": usage,
    }


class Metrics:

    TIMELINE_MINUTES = 30
    SAVE_VERSION = 1

    def __init__(self, path: Path | None = None) -> None:
        self.lock = threading.Lock()
        self.save_lock = threading.Lock()
        self.path = path
        self.started = time.time()
        self.saved_at: float | None = None
        self.dirty = False
        self._reset_counters()

    def _reset_counters(self) -> None:
        self.since = time.time()
        self.requests = 0
        self.errors = 0
        self.tokens_in = 0
        self.tokens_out = 0
        self.tokens_cached = 0
        self.latency_total = 0.0
        self.models: dict[str, dict[str, float]] = {}
        self.minutes: dict[int, list[int]] = {}
        self.hours: dict[int, list[int]] = {}
        self.limit_hits: list[float] = []
        self.log: deque = deque(maxlen=200)

    @staticmethod
    def is_limit_error(entry: dict[str, Any]) -> bool:
        error = str(entry.get("error") or "")
        return "FreeUsageLimit" in error or "Free usage exceeded" in error

    def record(self, entry: dict[str, Any], *, inference: bool) -> None:
        failed = entry["status"] >= 400 or bool(entry.get("error"))
        tokens_in = int(entry.get("tokens_in", 0))
        tokens_out = int(entry.get("tokens_out", 0))
        tokens_cached = int(entry.get("tokens_cached", 0))
        with self.lock:
            self.log.append(entry)
            self.dirty = True
            if not inference:
                return
            self.requests += 1
            self.errors += failed
            self.tokens_in += tokens_in
            self.tokens_out += tokens_out
            self.tokens_cached += tokens_cached
            self.latency_total += entry["ms"]
            stats = self.models.setdefault(entry["model"], self._new_model_stats())
            stats["requests"] += 1
            stats["errors"] += failed
            stats["tokens_in"] += tokens_in
            stats["tokens_out"] += tokens_out
            stats["tokens_cached"] += tokens_cached
            stats["latency_total"] += entry["ms"]
            stats["last_ms"] = entry["ms"]
            minute = int(time.time() // 60)
            hour = int(time.time() // 3600)
            for buckets, key in ((self.minutes, minute), (self.hours, hour)):
                bucket = buckets.setdefault(key, [0, 0, 0, 0])
                bucket[0] += 1
                bucket[1] += failed
                bucket[2] += tokens_in
                bucket[3] += tokens_out
            for old in [m for m in self.minutes if m <= minute - self.TIMELINE_MINUTES]:
                del self.minutes[old]
            for old in [h for h in self.hours if h <= hour - 48]:
                del self.hours[old]
            if self.is_limit_error(entry):
                self.limit_hits = (self.limit_hits + [time.time()])[-50:]

    @staticmethod
    def _new_model_stats() -> dict[str, float]:
        return {
            "requests": 0, "errors": 0, "tokens_in": 0, "tokens_out": 0,
            "tokens_cached": 0, "latency_total": 0.0, "last_ms": 0,
        }

    def snapshot(self) -> dict[str, Any]:
        now_minute = int(time.time() // 60)
        empty = [0, 0, 0, 0]

        def point(bucket: list[int]) -> dict[str, int]:
            return {"requests": bucket[0], "errors": bucket[1], "tokens_in": bucket[2], "tokens_out": bucket[3]}

        with self.lock:
            now = time.time()
            hour = int(now // 3600)
            timeline = [
                point(self.minutes.get(m, empty))
                for m in range(now_minute - self.TIMELINE_MINUTES + 1, now_minute + 1)
            ]
            hourly = [point(self.hours.get(h, empty)) for h in range(hour - 23, hour + 1)]
            return {
                "limits": {
                    "requests_5h": sum(b[0] for h, b in self.hours.items() if h > hour - 5),
                    "requests_24h": sum(b[0] for h, b in self.hours.items() if h > hour - 24),
                    "last_limit_hit": self.limit_hits[-1] if self.limit_hits else None,
                    "limit_hits_24h": sum(1 for t in self.limit_hits if t > now - 86400),
                },
                "uptime_s": round(time.time() - self.started),
                "since": self.since,
                "saved_at": self.saved_at,
                "persistent": self.path is not None,
                "requests": self.requests,
                "errors": self.errors,
                "tokens_in": self.tokens_in,
                "tokens_out": self.tokens_out,
                "tokens_cached": self.tokens_cached,
                "avg_ms": round(self.latency_total / self.requests) if self.requests else None,
                "models": {
                    name: {
                        "requests": int(s["requests"]),
                        "errors": int(s["errors"]),
                        "tokens_in": int(s["tokens_in"]),
                        "tokens_out": int(s["tokens_out"]),
                        "tokens_cached": int(s["tokens_cached"]),
                        "tokens": int(s["tokens_in"] + s["tokens_out"]),
                        "avg_ms": round(s["latency_total"] / s["requests"]) if s["requests"] else None,
                        "last_ms": round(s["last_ms"]),
                    }
                    for name, s in self.models.items()
                },
                "timeline": timeline,
                "hourly": hourly,
                "log": [dict(e) for e in self.log],
            }

    def _state(self) -> dict[str, Any]:
        return {
            "version": self.SAVE_VERSION,
            "since": self.since,
            "requests": self.requests,
            "errors": self.errors,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "tokens_cached": self.tokens_cached,
            "latency_total": self.latency_total,
            "models": self.models,
            "minutes": {str(k): v for k, v in self.minutes.items()},
            "hours": {str(k): v for k, v in self.hours.items()},
            "limit_hits": self.limit_hits,
            "log": list(self.log),
        }

    def save(self) -> float | None:
        if self.path is None:
            return None
        with self.save_lock:
            with self.lock:
                data = json.dumps(self._state(), separators=(",", ":"))
                self.dirty = False
            tmp = self.path.with_name(self.path.name + ".tmp")
            try:
                tmp.write_text(data, encoding="utf-8")
                os.replace(tmp, self.path)
            except OSError:
                with self.lock:
                    self.dirty = True
                raise
            with self.lock:
                self.saved_at = time.time()
                return self.saved_at

    def load(self) -> bool:
        if self.path is None or not self.path.exists():
            return False
        try:
            state = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        if not isinstance(state, dict) or state.get("version") != self.SAVE_VERSION:
            return False
        try:
            self._apply_state(state)
        except (TypeError, ValueError, AttributeError, KeyError):
            with self.lock:
                self._reset_counters()
            return False
        return True

    def _apply_state(self, state: dict[str, Any]) -> None:
        with self.lock:
            self.since = float(state.get("since") or time.time())
            self.requests = int(state.get("requests", 0))
            self.errors = int(state.get("errors", 0))
            self.tokens_in = int(state.get("tokens_in", 0))
            self.tokens_out = int(state.get("tokens_out", 0))
            self.tokens_cached = int(state.get("tokens_cached", 0))
            self.latency_total = float(state.get("latency_total", 0.0))
            self.models = {}
            for name, raw in (state.get("models") or {}).items():
                stats = self._new_model_stats()
                if isinstance(raw, dict):
                    stats.update({k: raw[k] for k in stats if isinstance(raw.get(k), (int, float))})
                self.models[str(name)] = stats
            self.minutes = {
                int(k): ([int(x) for x in v] + [0, 0, 0, 0])[:4]
                for k, v in (state.get("minutes") or {}).items()
                if isinstance(v, list) and len(v) in (2, 4)
            }
            self.hours = {
                int(k): [int(v), 0, 0, 0] if isinstance(v, (int, float)) else ([int(x) for x in v] + [0, 0, 0, 0])[:4]
                for k, v in (state.get("hours") or {}).items()
                if isinstance(v, (int, float)) or (isinstance(v, list) and len(v) == 4)
            }
            self.limit_hits = [
                float(t) for t in state.get("limit_hits") or [] if isinstance(t, (int, float))
            ][-50:]
            self.log = deque(
                (e for e in state.get("log") or [] if isinstance(e, dict)), maxlen=200
            )
            self.saved_at = self.path.stat().st_mtime
            self.dirty = False

    def reset(self) -> None:
        with self.lock:
            self._reset_counters()
            self.dirty = True

    def autosave_loop(self, interval: float = 2.0) -> None:
        while True:
            time.sleep(interval)
            if self.dirty:
                try:
                    self.save()
                except OSError:
                    pass


OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
# Daily ECB reference rate, used to show the OpenRouter value in euros.
FX_URL = "https://api.frankfurter.dev/v1/latest?base=USD&symbols=EUR"
PRICING_TTL = 6 * 3600


class Pricing:

    def __init__(self, path: Path | None = None) -> None:
        self.lock = threading.Lock()
        self.path = path
        self.fetched = 0.0
        self.prices: dict[str, dict[str, float]] = {}
        self.error: str | None = None
        self.usd_eur: float | None = None
        self.fx_date: str | None = None
        self._refreshing = False
        self._last_attempt = 0.0
        # model -> (OpenRouter id, exact match?)
        self._matches: dict[str, tuple[str | None, bool]] = {}
        if path is not None:
            self._load()

    def _load(self) -> None:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self.fetched = float(data["fetched"])
            self.prices = {str(k): dict(v) for k, v in data["prices"].items()}
            if data.get("usd_eur"):
                self.usd_eur = float(data["usd_eur"])
                self.fx_date = data.get("fx_date")
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            pass

    def _refresh_rate(self, timeout: float) -> None:
        """Fetch USD -> EUR. A failure keeps the last known rate."""
        try:
            data = _json_get(FX_URL, timeout)
            rate = float(data["rates"]["EUR"])
            if rate <= 0:
                raise ValueError("bad rate")
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            return
        with self.lock:
            self.usd_eur = rate
            self.fx_date = str(data.get("date") or "") or None

    @staticmethod
    def _price(value: Any) -> float | None:
        try:
            price = float(value)
        except (TypeError, ValueError):
            return None
        return price if price >= 0 else None

    def refresh(self, timeout: float = 20.0) -> bool:
        self._refresh_rate(timeout)
        try:
            data = _json_get(OPENROUTER_MODELS_URL, timeout)
            prices: dict[str, dict[str, float]] = {}
            for item in data.get("data") or []:
                model_id = str(item.get("id") or "")
                pricing = item.get("pricing") or {}
                prompt = self._price(pricing.get("prompt"))
                completion = self._price(pricing.get("completion"))
                if not model_id or prompt is None or completion is None:
                    continue
                entry = {"prompt": prompt, "completion": completion}
                cache_read = self._price(pricing.get("input_cache_read"))
                if cache_read is not None:
                    entry["cache_read"] = cache_read
                prices[model_id] = entry
            if not prices:
                raise ValueError("no prices in response")
        except (OSError, ValueError, AttributeError) as exc:
            with self.lock:
                self.error = str(exc)
            return False
        with self.lock:
            self.prices = prices
            self.fetched = time.time()
            self.error = None
            self._matches = {}
        if self.path is not None:
            try:
                self.path.write_text(
                    json.dumps({
                        "fetched": self.fetched, "prices": prices,
                        "usd_eur": self.usd_eur, "fx_date": self.fx_date,
                    }),
                    encoding="utf-8",
                )
            except OSError:
                pass
        return True

    def refresh_async(self, *, force: bool = False) -> None:
        now = time.time()
        with self.lock:
            if self._refreshing:
                return
            # A missing euro rate (older pricing.json) is fetched without waiting for the TTL.
            fresh = now - self.fetched < PRICING_TTL and self.usd_eur is not None
            if not force and (fresh or now - self._last_attempt < 300):
                return
            self._refreshing = True
            self._last_attempt = now

        def run() -> None:
            try:
                self.refresh()
            finally:
                with self.lock:
                    self._refreshing = False

        threading.Thread(target=run, daemon=True).start()

    def match(self, model: str) -> str | None:
        return self._lookup(model)[0]

    def _lookup(self, model: str) -> tuple[str | None, bool]:
        with self.lock:
            if model in self._matches:
                return self._matches[model]
        found = self._find(model)
        with self.lock:
            self._matches[model] = found
        return found

    def _find(self, model: str) -> tuple[str | None, bool]:
        """The paid OpenRouter model for an OpenCode model id.

        Exact names win. Otherwise the closest paid model of the same family is
        used (e.g. minimax-m3.1-flash -> minimax/minimax-m3), so new free models
        get a value without adding them by hand. Free (`:free`), batch and
        `~latest` aliases are skipped because they do not reflect the real price.
        """
        base = model[: -len("-free")] if model.endswith("-free") else model
        candidates = [base]
        if base.endswith("-preview"):
            candidates.append(base[: -len("-preview")])
        with self.lock:
            ids = sorted(
                i for i in self.prices
                if ":" not in i and not i.startswith("~") and "/" in i
            )
        names = {model_id: model_id.split("/", 1)[1] for model_id in ids}
        for candidate in candidates:
            for model_id in ids:
                if names[model_id] == candidate:
                    return model_id, True
        family = base.split("-", 1)[0]
        similar = [model_id for model_id in ids if names[model_id].split("-", 1)[0] == family]
        if not similar:
            return None, False
        # Longest shared start first (keeps the version: m3.1 -> m3, not m2.1),
        # then overall similarity, then the shorter, plainer name.
        best = max(
            similar,
            key=lambda model_id: (
                len(os.path.commonprefix([base, names[model_id]])),
                difflib.SequenceMatcher(None, base, names[model_id]).ratio(),
                -len(names[model_id]),
            ),
        )
        return best, False

    def for_model(self, model: str) -> dict[str, Any] | None:
        model_id, exact = self._lookup(model)
        if model_id is None:
            return None
        with self.lock:
            return {"openrouter_id": model_id, "similar": not exact, **self.prices[model_id]}

    def info(self) -> dict[str, Any]:
        with self.lock:
            return {
                "source": "openrouter",
                "fetched": self.fetched or None,
                "error": self.error,
                "models": len(self.prices),
                "usd_eur": self.usd_eur,
                "fx_date": self.fx_date,
            }


def usage_cost(stats: dict[str, Any], price: dict[str, Any] | None) -> float | None:
    if price is None:
        return None
    cached = min(int(stats.get("tokens_cached", 0)), int(stats.get("tokens_in", 0)))
    uncached = int(stats.get("tokens_in", 0)) - cached
    cache_rate = price.get("cache_read", price["prompt"])
    return (
        uncached * price["prompt"]
        + cached * cache_rate
        + int(stats.get("tokens_out", 0)) * price["completion"]
    )


def usage_report(snapshot: dict[str, Any], pricing: "Pricing") -> dict[str, Any]:
    names = sorted(set(snapshot["models"]) | set(available_models()))
    rows = []
    total = 0.0
    unpriced_tokens = 0
    for name in names:
        stats = snapshot["models"].get(name) or {
            "requests": 0, "tokens_in": 0, "tokens_out": 0, "tokens_cached": 0, "tokens": 0,
        }
        price = pricing.for_model(name)
        cost = usage_cost(stats, price)
        if cost is not None:
            total += cost
        else:
            unpriced_tokens += int(stats.get("tokens", 0))
        rows.append(
            {
                "model": name,
                "requests": int(stats.get("requests", 0)),
                "tokens_in": int(stats.get("tokens_in", 0)),
                "tokens_out": int(stats.get("tokens_out", 0)),
                "tokens_cached": int(stats.get("tokens_cached", 0)),
                "cost_usd": cost,
                "price": None if price is None else {
                    "openrouter_id": price["openrouter_id"],
                    "similar": price["similar"],
                    "prompt_per_m": price["prompt"] * 1_000_000,
                    "completion_per_m": price["completion"] * 1_000_000,
                    "cache_read_per_m": price["cache_read"] * 1_000_000 if "cache_read" in price else None,
                },
            }
        )
    return {
        "rows": rows,
        "total_cost_usd": total,
        "unpriced_tokens": unpriced_tokens,
        "pricing": pricing.info(),
    }


METRICS = Metrics()
PRICING = Pricing()

SERVER: Any = None
BOUND_OPEN: bool = False
BOUND_PORT: int = 8787
BOUND_HOST: str = "0.0.0.0"
RESTART_REQUESTED = False
DEFAULT_PORT = 8787
CLI_HOST: str | None = None
CLI_PORT: int | None = None

DEFAULT_CONFIG: dict[str, Any] = {
    "access": "network",
    "port": DEFAULT_PORT,
    "open_browser": True,
    "integrations": [],
}
ACCESS_HOSTS = {"local": "127.0.0.1", "network": "0.0.0.0"}
CONFIG: dict[str, Any] = dict(DEFAULT_CONFIG)
CONFIG_LOCK = threading.Lock()


def load_config() -> dict[str, Any]:
    config = dict(DEFAULT_CONFIG)
    try:
        stored = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
        if isinstance(stored, dict):
            config.update({k: v for k, v in stored.items() if k in DEFAULT_CONFIG})
    except (OSError, ValueError):
        pass
    if config.get("access") not in ACCESS_HOSTS:
        config["access"] = DEFAULT_CONFIG["access"]
    try:
        config["port"] = int(config["port"])
    except (TypeError, ValueError):
        config["port"] = DEFAULT_PORT
    if not 1 <= config["port"] <= 65535:
        config["port"] = DEFAULT_PORT
    if not isinstance(config.get("integrations"), list):
        config["integrations"] = []
    config["integrations"] = [str(i) for i in config["integrations"] if isinstance(i, str)]
    return config


def env_overrides() -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    if os.environ.get("ROUTE7Q9K_ACCESS") in ACCESS_HOSTS:
        overrides["access"] = os.environ["ROUTE7Q9K_ACCESS"]
    if os.environ.get("ROUTE7Q9K_PORT", "").isdigit() and 1 <= int(os.environ["ROUTE7Q9K_PORT"]) <= 65535:
        overrides["port"] = int(os.environ["ROUTE7Q9K_PORT"])
    return overrides


def save_config() -> None:
    # CONFIG holds what the user set; environment overrides are applied at
    # start only and never written to config.json.
    with CONFIG_LOCK:
        data = json.dumps(CONFIG, indent=2)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tmp = CONFIG_FILE.with_name(CONFIG_FILE.name + ".tmp")
    tmp.write_text(data, encoding="utf-8")
    os.replace(tmp, CONFIG_FILE)


AUTOSTART_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"
AUTOSTART_NAME = "7q9k-route"
# Name used before the rename to 7q9k route; replaced on start.
LEGACY_AUTOSTART_NAME = "oc-free-api"


def autostart_supported() -> bool:
    return os.name == "nt"


def autostart_command() -> str:
    if FROZEN:
        return f'"{sys.executable}" --background'
    pythonw = Path(sys.executable).with_name("pythonw.exe")
    interpreter = pythonw if pythonw.exists() else Path(sys.executable)
    return f'"{interpreter}" "{Path(__file__).resolve()}" --background'


def _autostart_value() -> str | None:
    import winreg
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, AUTOSTART_KEY) as key:
            return str(winreg.QueryValueEx(key, AUTOSTART_NAME)[0])
    except OSError:
        return None


def _take_legacy_autostart() -> bool:
    """Removes the pre-rename `oc-free-api` entry; returns whether it existed."""
    import winreg
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, AUTOSTART_KEY, 0,
                            winreg.KEY_QUERY_VALUE | winreg.KEY_SET_VALUE) as key:
            winreg.QueryValueEx(key, LEGACY_AUTOSTART_NAME)
            winreg.DeleteValue(key, LEGACY_AUTOSTART_NAME)
            return True
    except OSError:
        return False


def refresh_autostart() -> None:
    if not autostart_supported():
        return
    try:
        # Keep "start with Windows" on across the rename to 7q9k route.
        if _take_legacy_autostart():
            set_autostart(True)
            return
        value = _autostart_value()
        if value is not None and value != autostart_command():
            set_autostart(True)
    except OSError:
        pass


def autostart_enabled() -> bool | None:
    if not autostart_supported():
        return None
    import winreg
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, AUTOSTART_KEY) as key:
            winreg.QueryValueEx(key, AUTOSTART_NAME)
            return True
    except OSError:
        return False


def set_autostart(enabled: bool) -> None:
    if not autostart_supported():
        raise ProxyError("start with Windows is only available on Windows.", 400)
    import winreg
    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, AUTOSTART_KEY, 0, winreg.KEY_SET_VALUE) as key:
        if enabled:
            winreg.SetValueEx(key, AUTOSTART_NAME, 0, winreg.REG_SZ, autostart_command())
        else:
            try:
                winreg.DeleteValue(key, AUTOSTART_NAME)
            except FileNotFoundError:
                pass


def _family(host: str) -> int:
    return socket.AF_INET6 if ":" in host else socket.AF_INET


def port_available(host: str, port: int) -> bool:
    sock = socket.socket(_family(host), socket.SOCK_STREAM)
    try:
        if os.name == "nt" and hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        sock.bind((host, port))
        return True
    except OSError:
        return False
    finally:
        sock.close()


def public_urls(port: int) -> dict[str, str | None]:
    lan = _lan_ip() if BOUND_OPEN else None
    return {
        "local": f"http://127.0.0.1:{port}",
        "network": f"http://{lan}:{port}" if lan and lan != "127.0.0.1" else None,
    }


def settings_payload() -> dict[str, Any]:
    with CONFIG_LOCK:
        config = dict(CONFIG)
    wanted_host = CLI_HOST or ACCESS_HOSTS[config["access"]]
    wanted_port = CLI_PORT or config["port"]
    return {
        "ok": True,
        "access": config["access"],
        "port": config["port"],
        "open_browser": bool(config["open_browser"]),
        "autostart": autostart_enabled(),
        "data_dir": str(DATA_DIR),
        "can_open_folder": hasattr(os, "startfile"),
        "running": {"host": BOUND_HOST, "port": BOUND_PORT},
        "restart_required": wanted_host != BOUND_HOST or wanted_port != BOUND_PORT,
        "overridden": bool(CLI_HOST or CLI_PORT),
    }


def dashboard_state(*, local: bool) -> dict[str, Any]:
    models = available_models()
    return {
        "ok": True,
        "version": APP_VERSION,
        "local": local,
        "urls": public_urls(BOUND_PORT),
        "api_key": (OpenAICompatibleHandler.api_key or api_key_cache()) if local else None,
        "models": [
            {"id": m, "api": "responses" if uses_responses_api(m) else "chat"}
            for m in models
        ],
        "model_list": model_source(),
        "lan_ip": _lan_ip() if BOUND_OPEN else None,
        "port": BOUND_PORT,
        "default_model": route_default_model(models),
    }


def api_key_cache() -> str | None:
    try:
        value = KEY_FILE.read_text(encoding="utf-8").strip()
        return value or None
    except OSError:
        return None


def _write_key(key: str) -> str:
    KEY_FILE.write_text(key + "\n", encoding="utf-8")
    try:
        KEY_FILE.chmod(0o600)
    except OSError:
        pass
    return key


def regenerate_api_key() -> str:
    return _write_key("sk-7q9k-" + secrets.token_hex(20))


KEY_PATTERN = re.compile(r"^[A-Za-z0-9._~+/=-]{8,200}$")


def route_default_model(models: list[str] | None = None) -> str:
    models = available_models() if models is None else models
    return DEFAULT_MODEL if DEFAULT_MODEL in models else (models[0] if models else DEFAULT_MODEL)


def integration_target() -> integrations.Target:
    models = available_models()
    default = route_default_model(models)
    return integrations.Target(
        base_url=f"http://127.0.0.1:{BOUND_PORT}/v1",
        api_key=OpenAICompatibleHandler.api_key,
        models=list(models),
        default_model=default,
    )


def backups_dir() -> Path:
    return DATA_DIR / "backups"


def sync_integrations() -> list[str]:
    with CONFIG_LOCK:
        enabled = list(CONFIG.get("integrations") or [])
    if not enabled:
        return []
    updated = integrations.sync(enabled, integration_target(), backups_dir(), log=lambda m: _emit(f"[7q9k route] {m}"))
    for item in updated:
        _emit(f"[7q9k route] updated {integrations.get(item).name}")
    return updated


def sync_integrations_async() -> None:
    threading.Thread(target=sync_integrations, daemon=True).start()


def ensure_api_key() -> str:
    return api_key_cache() or regenerate_api_key()


def _enable_windows_vt(port: int) -> bool:
    if os.name != "nt":
        return sys.stderr.isatty()
    try:
        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleTitleW(f"7q9k route - Port {port}")
        for handle_id in (-11, -12):
            handle = kernel32.GetStdHandle(handle_id)
            mode = ctypes.c_uint32()
            if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
                return False
            if not kernel32.SetConsoleMode(handle, mode.value | 0x0004):
                return False
        return True
    except Exception:
        return False


VT: bool = False

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


LOG_STREAM: Any = None


LOG_MAX_BYTES = 5_000_000


def _rotate_log_if_needed() -> None:
    global LOG_STREAM
    try:
        if LOG_STREAM is None or LOG_STREAM.tell() < LOG_MAX_BYTES:
            return
        LOG_STREAM.close()
        os.replace(LOG_FILE, LOG_FILE.with_name(LOG_FILE.name + ".1"))
        LOG_STREAM = open(LOG_FILE, "a", encoding="utf-8", buffering=1)
    except (OSError, ValueError):
        pass


def _emit(line: str, stream: Any = None) -> None:
    _rotate_log_if_needed()
    targets = [LOG_STREAM] if LOG_STREAM is not None else [stream or sys.stdout]
    for target in targets:
        if target is None:
            continue
        try:
            try:
                target.write(line + "\n")
            except UnicodeEncodeError:
                target.write(line.encode("ascii", "replace").decode("ascii") + "\n")
        except (OSError, ValueError):
            continue
        try:
            target.flush()
        except (OSError, ValueError):
            pass


def _console_init(port: int) -> None:
    global VT
    for stream in (sys.stdout, sys.stderr):
        if stream is not None and not stream.isatty() and hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    VT = sys.stderr is not None and _enable_windows_vt(port)


def open_log_file() -> None:
    global LOG_STREAM, VT
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    try:
        if LOG_FILE.exists() and LOG_FILE.stat().st_size > 1_000_000:
            os.replace(LOG_FILE, LOG_FILE.with_name(LOG_FILE.name + ".1"))
        LOG_STREAM = open(LOG_FILE, "a", encoding="utf-8", buffering=1)
    except OSError:
        LOG_STREAM = None
    VT = False


def _set_console_icon() -> None:
    try:
        from ctypes import wintypes
        user32 = ctypes.windll.user32
        kernel32 = ctypes.windll.kernel32
        kernel32.GetConsoleWindow.restype = wintypes.HWND
        user32.LoadImageW.restype = wintypes.HANDLE
        user32.LoadImageW.argtypes = [wintypes.HINSTANCE, wintypes.LPCWSTR, wintypes.UINT,
                                      ctypes.c_int, ctypes.c_int, wintypes.UINT]
        user32.SendMessageW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
        hwnd = kernel32.GetConsoleWindow()
        icon = ASSETS_DIR / "icon.ico"
        if not hwnd or not icon.exists():
            return
        for which, size in ((0, 16), (1, 32)):
            handle = user32.LoadImageW(None, str(icon), 1, size, size, 0x10)
            if handle:
                user32.SendMessageW(hwnd, 0x80, which, handle)
    except Exception:
        pass


def _windows_console_open() -> bool:
    if os.name != "nt":
        return False
    try:
        kernel32 = ctypes.windll.kernel32
        if not kernel32.AllocConsole():
            return False
        kernel32.SetConsoleTitleW("7q9k route")
        _set_console_icon()
        global CONSOLE_OUT
        CONSOLE_OUT = open("CONOUT$", "w", encoding="utf-8", buffering=1)
        sys.stdout = CONSOLE_OUT
        sys.stderr = CONSOLE_OUT
        return True
    except Exception:
        return False


CONSOLE_OUT: Any = None


def _windows_console_close() -> None:
    global CONSOLE_OUT
    sys.stdout = None
    sys.stderr = None
    if CONSOLE_OUT is not None:
        try:
            CONSOLE_OUT.close()
        except Exception:
            pass
        CONSOLE_OUT = None
    try:
        ctypes.windll.kernel32.FreeConsole()
    except Exception:
        pass


def _lan_ip() -> str:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("9.9.9.9", 53))
        return str(sock.getsockname()[0])
    except OSError:
        return "127.0.0.1"
    finally:
        sock.close()


def _print_banner(port: int, host: str, key: str, models: list[str]) -> None:
    shown_models = ", ".join(models[:5]) + (" …" if len(models) > 5 else "")
    _emit("")
    _emit(
        _paint("7q9k route", "bright")
        + "   "
        + _paint(f"v{APP_VERSION} · OpenCode free models · OpenAI-compatible", "dim")
    )
    _emit("  " + _paint("Dashboard ", "cyan") + _paint(f"http://localhost:{port}/", "green"))
    if host in ("0.0.0.0", "::"):
        _emit(
            "  " + _paint("Network   ", "cyan")
            + _paint(f"http://{_lan_ip()}:{port}/", "green")
            + _paint("  (LAN access requires the API key)", "dim")
        )
    _emit("  " + _paint("Base URL  ", "cyan") + _paint(f"http://127.0.0.1:{port}/v1", "bright"))
    _emit("  " + _paint("API-Key   ", "cyan") + _paint(key, "yellow"))
    _emit(
        "  " + _paint("Models    ", "cyan")
        + _paint(f"{len(models)} available: {shown_models}", "bright")
    )
    stop_hint = "dashboard > Settings > Stop server" if LOG_STREAM is not None else "Ctrl+C or close this window"
    _emit("  " + _paint("Stop      ", "cyan") + _paint(stop_hint, "dim"))
    _emit("")


_STATUS_WORD = {
    200: "OK", 201: "OK", 204: "OK",
    400: "BAD REQUEST", 401: "UNAUTHORIZED", 403: "FORBIDDEN", 404: "NOT FOUND",
    411: "LENGTH REQUIRED", 413: "TOO LARGE", 429: "RATE LIMITED", 451: "BLOCKED",
    500: "SERVER ERROR", 502: "UPSTREAM ERROR", 503: "SERVICE UNAVAILABLE",
    504: "UPSTREAM TIMEOUT",
}


def _status_color(code: int) -> str:
    if 200 <= code < 300:
        return "green"
    if code == 429:
        return "magenta"
    if 400 <= code < 500:
        return "yellow"
    return "red"


ROUTES: dict[tuple[str, str], tuple[str, str]] = {
    ("GET", "/"): ("open", "_get_dashboard_page"),
    ("GET", "/dashboard"): ("open", "_get_dashboard_page"),
    ("GET", "/health"): ("open", "_get_health"),
    ("GET", "/favicon.ico"): ("open", "_get_favicon"),
    ("GET", "/v1/models"): ("open", "_get_models"),
    ("GET", "/models"): ("open", "_get_models"),
    ("GET", "/api/v1/models"): ("open", "_get_models"),
    ("GET", "/api/tags"): ("open", "_get_ollama_tags"),
    ("GET", "/api/ps"): ("open", "_get_ollama_ps"),
    ("GET", "/api/version"): ("open", "_get_version"),
    ("GET", "/version"): ("open", "_get_version"),
    ("GET", "/v1/props"): ("open", "_get_props"),
    ("GET", "/props"): ("open", "_get_props"),
    ("POST", "/api/show"): ("open", "_post_ollama_show"),
    ("POST", "/v1/chat/completions"): ("client", "_post_chat_completions"),
    ("POST", "/chat/completions"): ("client", "_post_chat_completions"),
    ("POST", "/v1/responses"): ("client", "_post_responses"),
    ("POST", "/responses"): ("client", "_post_responses"),
    ("POST", "/api/chat"): ("client", "_post_ollama_chat"),
    ("POST", "/api/generate"): ("client", "_post_ollama_generate"),
    ("POST", "/v1/messages"): ("client", "_post_anthropic_messages"),
    ("POST", "/v1/messages/count_tokens"): ("client", "_post_anthropic_count_tokens"),
    ("GET", "/api/dashboard"): ("dashboard", "_get_dashboard_state"),
    ("GET", "/api/dashboard/stats"): ("dashboard", "_get_dashboard_stats"),
    ("GET", "/api/dashboard/log"): ("dashboard", "_get_dashboard_log"),
    ("POST", "/api/dashboard/test"): ("dashboard", "_post_dashboard_test"),
    ("POST", "/api/dashboard/refresh"): ("dashboard", "_post_refresh_models"),
    ("POST", "/api/dashboard/save"): ("dashboard", "_post_save_stats"),
    ("POST", "/api/dashboard/prices"): ("dashboard", "_post_refresh_prices"),
    ("GET", "/api/dashboard/settings"): ("admin", "_get_settings"),
    ("POST", "/api/dashboard/settings"): ("admin", "_post_settings"),
    ("POST", "/api/dashboard/restart"): ("admin", "_post_restart"),
    ("POST", "/api/dashboard/reset"): ("admin", "_post_reset_stats"),
    ("POST", "/api/dashboard/key"): ("admin", "_post_new_key"),
    ("POST", "/api/dashboard/stop"): ("admin", "_post_stop_server"),
    ("POST", "/api/dashboard/open-folder"): ("admin", "_post_open_folder"),
    ("GET", "/api/dashboard/integrations"): ("admin", "_get_integrations"),
    ("POST", "/api/dashboard/integrations/connect"): ("admin", "_post_integration_connect"),
    ("POST", "/api/dashboard/integrations/disconnect"): ("admin", "_post_integration_disconnect"),
}

INFERENCE_PATHS = frozenset(
    {
        "/v1/chat/completions", "/chat/completions", "/v1/responses", "/responses",
        "/api/chat", "/api/generate", "/api/dashboard/test", "/v1/messages",
    }
)
OLLAMA_PATHS = frozenset({"/api/chat", "/api/generate", "/api/show"})
ANTHROPIC_PATHS = frozenset({"/v1/messages", "/v1/messages/count_tokens"})
QUIET_PATHS = frozenset(
    {"/", "/dashboard", "/favicon.ico", "/health", "/api/dashboard", "/api/dashboard/stats",
     "/api/dashboard/log", "/api/dashboard/settings", "/api/dashboard/integrations",
     "/assets/fonts/geist.woff2", "/assets/fonts/geist-mono.woff2"}
)
CLIENT_GONE = (BrokenPipeError, ConnectionResetError, ConnectionAbortedError)


class OpenAICompatibleHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = f"7q9k-route/{APP_VERSION}"
    timeout = 300

    api_key: str = ""


    def do_GET(self) -> None:
        self._dispatch("GET")

    def do_POST(self) -> None:
        self._dispatch("POST")

    def do_OPTIONS(self) -> None:
        self._dispatch("OPTIONS")

    def _dispatch(self, method: str) -> None:
        self._t0 = time.time()
        self._status = 0
        self._meta: dict[str, Any] = {}
        self._body_read = False
        self._path = urllib.parse.urlsplit(self.path).path
        try:
            if method == "OPTIONS":
                self.send_response(204)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            route = ROUTES.get((method, self._path))
            if route is None and method == "GET" and self._path.startswith("/assets/"):
                route = ("open", "_get_asset")
            if route is None:
                if any(path == self._path for _, path in ROUTES):
                    raise ProxyError(f"method {method} not allowed for {self._path}", 405)
                raise ProxyError(f"unknown path: {self._path}", 404)
            access, handler_name = route
            self._check_access(access)
            getattr(self, handler_name)()
        except ProxyError as exc:
            self._fail(exc.status_code, str(exc))
        except CLIENT_GONE:
            self.close_connection = True
        except Exception as exc:
            self._fail(500, f"internal error: {type(exc).__name__}: {exc}")
        finally:
            self._finish()

    def _fail(self, code: int, message: str) -> None:
        self._meta.setdefault("error", message)
        if self._status:
            self.close_connection = True
            return
        self._drain_body()
        try:
            if self._path in OLLAMA_PATHS:
                self._send_json(code, {"error": message})
            elif self._path in ANTHROPIC_PATHS:
                self._send_json(code, AnthropicStream.fail(message, code)[0][1])
            else:
                self._send_json(
                    code, {"error": {"message": message, "type": "route_error", "code": code}}
                )
        except CLIENT_GONE:
            self.close_connection = True

    def _drain_body(self) -> None:
        if self._body_read:
            return
        self._body_read = True
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            return
        if 0 < length <= 1024 * 1024:
            try:
                self.rfile.read(length)
            except OSError:
                pass


    def log_request(self, code: Any = "-", size: Any = "-") -> None:
        try:
            self._status = int(code)
        except (TypeError, ValueError):
            self._status = 0

    def log_error(self, format: str, *args: Any) -> None:
        return

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _finish(self) -> None:
        method = self.command or "?"
        status = self._status or 0
        if method == "OPTIONS" or (
            method == "GET" and status < 400 and self._path in QUIET_PATHS
        ):
            return
        usage = self._meta.get("usage") or {}
        entry: dict[str, Any] = {
            "t": time.strftime("%H:%M:%S"),
            "ts": time.time(),
            "method": method,
            "path": "".join(c if c.isprintable() else "?" for c in self._path)[:200],
            "status": status,
            "ms": round((time.time() - self._t0) * 1000),
            "client": "local" if self._is_loopback() else self._client_ip(),
        }
        model = self._meta.get("model")
        if model:
            entry["model"] = model
            entry["tokens_in"] = int(usage.get("prompt_tokens", 0) or 0)
            entry["tokens_out"] = int(usage.get("completion_tokens", 0) or 0)
            details = usage.get("prompt_tokens_details") or {}
            entry["tokens_cached"] = int(details.get("cached_tokens", 0) or 0)
        if self._meta.get("error"):
            entry["error"] = str(self._meta["error"])[:300]
        try:
            METRICS.record(entry, inference=bool(model) and self._path in INFERENCE_PATHS)
            self._console_line(entry)
        except Exception:
            pass

    def _console_line(self, entry: dict[str, Any]) -> None:
        code = entry["status"]
        word = _STATUS_WORD.get(code, "")
        line = (
            _paint(entry["t"], "dim")
            + "  "
            + _paint(f"{entry['method']:<6}", "bright")
            + f" {entry['path']:<26} "
            + _paint(str(code or "---"), _status_color(code) if code else "red")
        )
        if word:
            line += " " + _paint(f"{word:<14}", "dim")
        line += " " + _paint(f"{entry['ms']:>6}ms", "dim")
        if entry.get("model"):
            line += "  " + _paint(entry["model"], "cyan")
            tokens = entry["tokens_in"] + entry["tokens_out"]
            if tokens:
                line += _paint(f" {tokens} tok", "dim")
        if entry["client"] != "local":
            line += _paint(f"  [{entry['client']}]", "magenta")
        if entry.get("error"):
            line += "  " + _paint(entry["error"][:120], "red")
        _emit(_paint("[7q9k route] ", "dim") + line, sys.stderr)


    def _via_proxy(self) -> bool:
        return any(self.headers.get(h) for h in ("X-Forwarded-For", "Forwarded", "X-Real-IP"))

    def _is_loopback(self) -> bool:
        if self._via_proxy():
            return False
        try:
            ip = ipaddress.ip_address(str(self.client_address[0]).split("%", 1)[0])
        except ValueError:
            return False
        mapped = getattr(ip, "ipv4_mapped", None)
        return bool((mapped or ip).is_loopback)

    def _host_ok(self) -> bool:
        host = (self.headers.get("Host") or "").strip().lower()
        if not host:
            return True
        hostname = urllib.parse.urlsplit("//" + host).hostname or ""
        if hostname == "localhost" or hostname.endswith(".localhost"):
            return True
        try:
            ipaddress.ip_address(hostname)
            return True
        except ValueError:
            pass
        machine = socket.gethostname().lower().split(".", 1)[0]
        return hostname in (machine, f"{machine}.local", f"{machine}.lan", f"{machine}.home")

    def _origin_ok(self) -> bool:
        if self.headers.get("Sec-Fetch-Site", "").lower() == "cross-site":
            return False
        origin = self.headers.get("Origin")
        if origin is None:
            return True
        netloc = urllib.parse.urlsplit(origin).netloc.lower()
        return bool(netloc) and netloc == (self.headers.get("Host") or "").strip().lower()

    def _trusted(self) -> bool:
        return self._is_loopback() and self._host_ok() and self._origin_ok()

    def _presented_key(self) -> str | None:
        auth = self.headers.get("Authorization", "")
        if auth.lower().startswith("bearer "):
            return auth[7:].strip()
        return self.headers.get("x-api-key", "").strip() or None

    def _authorized(self) -> bool:
        key = self._presented_key()
        return key is not None and secrets.compare_digest(
            key.encode("utf-8"), self.api_key.encode("utf-8")
        )

    def _client_ip(self) -> str:
        forwarded = (self.headers.get("X-Forwarded-For") or "").split(",")[0].strip()
        return forwarded or str(self.client_address[0])

    def _check_access(self, access: str) -> None:
        if access == "open":
            return
        if access == "admin":
            if not self._trusted():
                raise ProxyError("only allowed from the server machine itself.", 403)
            return
        if self._trusted() or self._authorized():
            return
        raise ProxyError(
            "Missing or invalid API key. Pass it as 'Authorization: Bearer <key>' "
            "(key file: api-key.txt). Requests from other machines always need it.",
            401,
        )


    def _send_json(self, code: int, payload: dict[str, Any]) -> None:
        self._send_bytes(code, json.dumps(payload).encode("utf-8"), "application/json")

    def _send_bytes(self, code: int, data: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(data)

    def _start_chunked(self, content_type: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Accel-Buffering", "no")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

    def _chunk(self, data: bytes) -> None:
        self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
        self.wfile.flush()

    def _end_chunked(self) -> None:
        try:
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        except OSError:
            pass
        self.close_connection = True

    def _read_json(self) -> dict[str, Any]:
        if "chunked" in self.headers.get("Transfer-Encoding", "").lower():
            raise ProxyError("chunked request bodies are not supported; send Content-Length.", 411)
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError as exc:
            raise ProxyError("invalid Content-Length header.", 400) from exc
        if length < 0:
            raise ProxyError("invalid Content-Length header.", 400)
        if length > MAX_BODY_BYTES:
            raise ProxyError(
                f"request body too large ({length} bytes, max {MAX_BODY_BYTES}).", 413
            )
        raw = self.rfile.read(length) if length else b""
        self._body_read = True
        try:
            req = json.loads(raw.decode("utf-8")) if raw else {}
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ProxyError(f"invalid JSON body: {exc}", 400) from exc
        if not isinstance(req, dict):
            raise ProxyError("request body must be a JSON object", 400)
        return req

    def _resolve_model(self, name: Any) -> str:
        model = normalize_model_name(name)
        models = available_models()
        if models and model not in models:
            raise ProxyError(
                f"model '{model}' is not an available free model. "
                f"Call GET /v1/models or /api/tags for the current list "
                f"({', '.join(models)})",
                404,
            )
        self._meta["model"] = model
        return model

    def _open_upstream(self, req: dict[str, Any], model: str) -> Iterator[dict[str, Any]]:
        return self._tap(open_upstream_events(req, model))

    def _tap(self, deltas: Iterator[dict[str, Any]]) -> Iterator[dict[str, Any]]:
        try:
            for event in deltas:
                if isinstance(event.get("usage"), dict):
                    self._meta["usage"] = _usage(event["usage"])
                yield event
        finally:
            close = getattr(deltas, "close", None)
            if close:
                close()


    def _get_dashboard_page(self) -> None:
        try:
            payload = DASHBOARD_FILE.read_bytes()
        except OSError as exc:
            raise ProxyError("dashboard.html not found next to server.py", 500) from exc
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(payload)

    def _get_asset(self) -> None:
        root = ASSETS_DIR.resolve()
        target = (root / urllib.parse.unquote(self._path[len("/assets/"):])).resolve()
        content_type = ASSET_TYPES.get(target.suffix.lower())
        if root not in target.parents or not content_type or not target.is_file():
            raise ProxyError(f"unknown path: {self._path}", 404)
        data = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "public, max-age=86400")
        self.end_headers()
        self.wfile.write(data)

    def _get_favicon(self) -> None:
        self._path = "/assets/icon.ico"
        self._get_asset()

    def _get_health(self) -> None:
        self._send_json(200, {"ok": True, "version": APP_VERSION})

    def _get_models(self) -> None:
        models = available_models()
        self._send_json(
            200,
            {
                "object": "list",
                "data": [
                    {"id": model, "object": "model", "created": 0, "owned_by": "opencode",
                     "type": "model", "display_name": model, "created_at": "1970-01-01T00:00:00Z"}
                    for model in models
                ],
                "has_more": False,
                "first_id": models[0] if models else None,
                "last_id": models[-1] if models else None,
            },
        )

    def _get_ollama_tags(self) -> None:
        self._send_json(200, {"models": _ollama_model_entries()})

    def _get_ollama_ps(self) -> None:
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

    def _get_version(self) -> None:
        self._send_json(200, {"version": f"7q9k-route/{APP_VERSION}"})

    def _get_props(self) -> None:
        models = available_models()
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


    def _get_dashboard_state(self) -> None:
        self._send_json(200, dashboard_state(local=self._trusted()))

    def _get_settings(self) -> None:
        self._send_json(200, settings_payload())

    def _post_settings(self) -> None:
        req = self._read_json()
        changes: dict[str, Any] = {}
        if "access" in req:
            if req["access"] not in ACCESS_HOSTS:
                raise ProxyError("access must be local or network.", 400)
            changes["access"] = req["access"]
        if "port" in req:
            try:
                port = int(str(req["port"]).strip())
            except (TypeError, ValueError) as exc:
                raise ProxyError("port must be a number.", 400) from exc
            if not 1024 <= port <= 65535:
                raise ProxyError("port must be between 1024 and 65535.", 400)
            changes["port"] = port
        for flag in ("open_browser", "autostart"):
            if flag in req and req[flag] is not None and not isinstance(req[flag], bool):
                raise ProxyError(f"{flag} must be true or false.", 400)
        if "open_browser" in req:
            changes["open_browser"] = bool(req["open_browser"])
        if "autostart" in req and req["autostart"] is not None:
            try:
                set_autostart(bool(req["autostart"]))
            except OSError as exc:
                raise ProxyError(f"could not change autostart: {exc}", 500) from exc
        with CONFIG_LOCK:
            CONFIG.update(changes)
        try:
            save_config()
        except OSError as exc:
            raise ProxyError(f"could not write {CONFIG_FILE.name}: {exc}", 500) from exc
        self._send_json(200, settings_payload())

    def _post_restart(self) -> None:
        global RESTART_REQUESTED
        with CONFIG_LOCK:
            host = CLI_HOST or ACCESS_HOSTS[CONFIG["access"]]
            port = CLI_PORT or CONFIG["port"]
        # Only a new port can be taken by someone else; the current one is ours
        # and is released before the new process binds it.
        if port != BOUND_PORT and not port_available(host, port):
            raise ProxyError(f"port {port} is already used by another program. Pick another port.", 409)
        RESTART_REQUESTED = True
        self._send_json(200, {"ok": True, "restarting": True, "port": port})
        if SERVER is not None:
            threading.Thread(target=SERVER.shutdown, daemon=True).start()

    def _get_dashboard_stats(self) -> None:
        PRICING.refresh_async()
        snapshot = METRICS.snapshot()
        for entry in snapshot["log"]:
            if entry.get("model"):
                entry["cost_usd"] = usage_cost(entry, PRICING.for_model(entry["model"]))
        self._send_json(200, {"ok": True, **snapshot, "usage": usage_report(snapshot, PRICING)})

    def _post_save_stats(self) -> None:
        if METRICS.path is None:
            raise ProxyError("saving is disabled for this server instance.", 409)
        try:
            saved_at = METRICS.save()
        except OSError as exc:
            raise ProxyError(f"could not write {METRICS.path.name}: {exc}", 500) from exc
        self._send_json(200, {"ok": True, "saved_at": saved_at, "file": str(METRICS.path)})

    def _post_reset_stats(self) -> None:
        METRICS.reset()
        if METRICS.path is not None:
            try:
                METRICS.save()
            except OSError as exc:
                raise ProxyError(f"could not write {METRICS.path.name}: {exc}", 500) from exc
        self._send_json(200, {"ok": True})

    def _post_refresh_prices(self) -> None:
        if not PRICING.refresh():
            raise ProxyError(f"OpenRouter prices could not be loaded: {PRICING.info()['error']}", 502)
        self._send_json(200, {"ok": True, **PRICING.info()})

    def _get_dashboard_log(self) -> None:
        self._send_json(200, {"ok": True, "log": METRICS.snapshot()["log"]})

    def _post_refresh_models(self) -> None:
        models = available_models(refresh=True)
        sync_integrations_async()
        self._send_json(200, {"ok": True, "models": models, **model_source()})

    def _post_dashboard_test(self) -> None:
        req = self._read_json()
        model = self._resolve_model(req.get("model"))
        result = run_model_test(model, req.get("prompt"))
        self._meta["usage"] = result["usage"]
        self._send_json(200, result)

    def _post_new_key(self) -> None:
        req = self._read_json()
        custom = str(req.get("key") or "").strip()
        if custom:
            if not KEY_PATTERN.match(custom):
                raise ProxyError("the key needs 8 to 200 characters: letters, digits and . _ ~ + / = -", 400)
            new_key = _write_key(custom)
        else:
            new_key = regenerate_api_key()
        OpenAICompatibleHandler.api_key = new_key
        sync_integrations_async()
        self._send_json(200, {"ok": True, "api_key": new_key})

    def _get_integrations(self) -> None:
        with CONFIG_LOCK:
            enabled = list(CONFIG.get("integrations") or [])
        apps = integrations.describe_all(integration_target())
        for app in apps:
            app["auto_update"] = app["id"] in enabled
        self._send_json(200, {"ok": True, "apps": apps, "backups": str(backups_dir())})

    def _integration_id(self) -> str:
        req = self._read_json()
        integration_id = str(req.get("id") or "")
        try:
            integrations.get(integration_id)
        except integrations.IntegrationError as exc:
            raise ProxyError(str(exc), 404) from exc
        return integration_id

    def _set_enabled(self, integration_id: str, enabled: bool) -> None:
        with CONFIG_LOCK:
            current = [i for i in CONFIG.get("integrations") or [] if i != integration_id]
            if enabled:
                current.append(integration_id)
            CONFIG["integrations"] = current
        try:
            save_config()
        except OSError:
            pass

    def _post_integration_connect(self) -> None:
        integration_id = self._integration_id()
        try:
            app = integrations.connect(integration_id, integration_target(), backups_dir())
        except integrations.IntegrationError as exc:
            raise ProxyError(str(exc), 409) from exc
        except UnicodeDecodeError as exc:
            raise ProxyError(f"the app config is not valid UTF-8: {exc}", 409) from exc
        except OSError as exc:
            raise ProxyError(f"could not write the app config: {exc}", 500) from exc
        self._set_enabled(integration_id, True)
        app["auto_update"] = True
        self._send_json(200, {"ok": True, "app": app})

    def _post_integration_disconnect(self) -> None:
        integration_id = self._integration_id()
        try:
            app = integrations.disconnect(integration_id, integration_target(), backups_dir())
        except integrations.IntegrationError as exc:
            raise ProxyError(str(exc), 409) from exc
        except UnicodeDecodeError as exc:
            raise ProxyError(f"the app config is not valid UTF-8: {exc}", 409) from exc
        except OSError as exc:
            raise ProxyError(f"could not write the app config: {exc}", 500) from exc
        self._set_enabled(integration_id, False)
        app["auto_update"] = False
        self._send_json(200, {"ok": True, "app": app})

    def _post_open_folder(self) -> None:
        if not hasattr(os, "startfile"):
            raise ProxyError("opening folders is only supported on Windows.", 400)
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        os.startfile(str(DATA_DIR))
        self._send_json(200, {"ok": True, "path": str(DATA_DIR)})

    def _post_stop_server(self) -> None:
        self._send_json(200, {"ok": True, "stopping": True})
        if SERVER is not None:
            threading.Thread(target=SERVER.shutdown, daemon=True).start()


    def _post_chat_completions(self) -> None:
        req = self._read_json()
        validate_messages(req.get("messages"))
        model = self._resolve_model(req.get("model"))
        deltas = self._open_upstream(req, model)
        if bool(req.get("stream")):
            self._openai_stream(deltas, model)
        else:
            self._send_json(200, merge_deltas(deltas, model))

    def _openai_stream(self, deltas: Iterator[dict[str, Any]], model: str) -> None:
        completion_id = "chatcmpl-" + secrets.token_hex(12)
        created = int(time.time())
        self._start_chunked("text/event-stream; charset=utf-8")

        def sse(payload: dict[str, Any]) -> bytes:
            return b"data: " + json.dumps(payload, separators=(",", ":")).encode() + b"\n\n"

        try:
            seen = False
            for event in deltas:
                seen = True
                chunk: dict[str, Any] = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": str(event.get("model") or model),
                    "choices": [
                        {**choice, "index": int(choice.get("index", 0) or 0)}
                        for choice in (event.get("choices") or [])
                    ],
                }
                if event.get("usage"):
                    chunk["usage"] = event["usage"]
                self._chunk(sse(chunk))
            if not seen:
                raise ProxyError("upstream returned no events.")
        except CLIENT_GONE:
            self._meta["error"] = "client disconnected"
            self.close_connection = True
            return
        except Exception as exc:
            code = exc.status_code if isinstance(exc, ProxyError) else 502
            self._meta["error"] = str(exc)
            try:
                self._chunk(
                    sse({"error": {"message": str(exc), "type": "route_error", "code": code}})
                )
            except OSError:
                pass
        try:
            self._chunk(b"data: [DONE]\n\n")
        except OSError:
            pass
        self._end_chunked()


    def _post_responses(self) -> None:
        req = self._read_json()
        model = self._resolve_model(req.get("model"))
        deltas = self._open_upstream(responses_to_chat_request(req), model)
        if bool(req.get("stream")):
            self._responses_stream(deltas, model, req)
        else:
            self._send_json(200, build_responses_object(deltas, model, req))

    def _responses_stream(
        self, deltas: Iterator[dict[str, Any]], model: str, req: dict[str, Any]
    ) -> None:
        builder = ResponsesBuilder(model, req)
        self._start_chunked("text/event-stream; charset=utf-8")

        def send(events: list[dict[str, Any]]) -> None:
            for event in events:
                payload = json.dumps(event, separators=(",", ":")).encode()
                self._chunk(f"event: {event['type']}\ndata: ".encode() + payload + b"\n\n")

        try:
            send(builder.start())
            seen = False
            for event in deltas:
                seen = True
                send(builder.feed(event))
            if not seen:
                raise ProxyError("upstream returned no events.")
            send(builder.finish())
        except CLIENT_GONE:
            self._meta["error"] = "client disconnected"
            self.close_connection = True
            return
        except Exception as exc:
            self._meta["error"] = str(exc)
            try:
                send(builder.fail(str(exc)))
            except OSError:
                pass
        self._end_chunked()


    def _resolve_anthropic_model(self, name: Any) -> str:
        text = re.sub(r"\[[^\]]*\]$", "", str(name or "").strip())  # "[1m]" context suffix
        bare = text.split("/", 1)[1] if text.startswith("opencode/") else text
        models = available_models()
        lowered = bare.lower()
        if models and bare not in models and (
            not bare or lowered.startswith("claude") or lowered in CLAUDE_MODEL_ALIASES
        ):
            # Claude Code asks for Claude models unless every model setting points here.
            text = route_default_model(models)
        return self._resolve_model(text)

    def _post_anthropic_messages(self) -> None:
        req = self._read_json()
        validate_messages(req.get("messages"))
        model = self._resolve_anthropic_model(req.get("model"))
        thinking = anthropic_wants_thinking(req)
        estimate = anthropic_estimate_tokens(req)
        deltas = self._open_upstream(anthropic_to_chat_request(req), model)
        if bool(req.get("stream")):
            self._anthropic_stream(deltas, model, thinking, estimate)
        else:
            self._send_json(200, anthropic_message(merge_deltas(deltas, model), thinking=thinking))

    def _post_anthropic_count_tokens(self) -> None:
        req = self._read_json()
        validate_messages(req.get("messages"))
        self._send_json(200, {"input_tokens": anthropic_estimate_tokens(req)})

    def _anthropic_stream(
        self, deltas: Iterator[dict[str, Any]], model: str, thinking: bool, estimate: int
    ) -> None:
        builder = AnthropicStream(model, thinking=thinking, input_tokens=estimate)
        self._start_chunked("text/event-stream; charset=utf-8")

        def send(events: list[tuple[str, dict[str, Any]]]) -> None:
            for name, payload in events:
                data = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
                self._chunk(f"event: {name}\ndata: ".encode() + data + b"\n\n")

        try:
            send(builder.start())
            seen = False
            for event in deltas:
                seen = True
                send(builder.feed(event))
            if not seen:
                raise ProxyError("upstream returned no events.")
            send(builder.finish())
        except CLIENT_GONE:
            self._meta["error"] = "client disconnected"
            self.close_connection = True
            return
        except Exception as exc:
            code = exc.status_code if isinstance(exc, ProxyError) else 502
            self._meta["error"] = str(exc)
            try:
                send(AnthropicStream.fail(str(exc), code))
            except OSError:
                pass
        self._end_chunked()


    def _post_ollama_chat(self) -> None:
        req = self._read_json()
        model = self._resolve_model(req.get("model"))
        openai_req = ollama_options(req)
        openai_req["messages"] = ollama_to_openai_messages(req.get("messages") or [])
        deltas = self._open_upstream(openai_req, model)
        include_thinking = bool(req.get("think"))
        if req.get("stream", True) is not False:
            self._ollama_stream(deltas, model, include_thinking, chat=True)
        else:
            completion = merge_deltas(deltas, model)
            self._send_json(200, ollama_chat_response(completion, include_thinking))

    def _post_ollama_generate(self) -> None:
        req = self._read_json()
        model = self._resolve_model(req.get("model"))
        messages: list[dict[str, Any]] = []
        if req.get("system") and not req.get("raw"):
            messages.append({"role": "system", "content": str(req["system"])})
        messages.append({"role": "user", "content": str(req.get("prompt") or "")})
        openai_req = ollama_options(req)
        openai_req["messages"] = messages
        deltas = self._open_upstream(openai_req, model)
        if req.get("stream", True) is not False:
            self._ollama_stream(deltas, model, bool(req.get("think")), chat=False)
        else:
            completion = merge_deltas(deltas, model)
            self._send_json(200, ollama_generate_response(completion))

    def _post_ollama_show(self) -> None:
        req = self._read_json()
        name = normalize_model_name(req.get("name") or req.get("model"))
        self._send_json(
            200,
            {
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
            },
        )

    def _ollama_stream(
        self,
        deltas: Iterator[dict[str, Any]],
        model: str,
        include_thinking: bool,
        *,
        chat: bool,
    ) -> None:
        created = _ollama_now()
        calls: dict[int, dict[str, Any]] = {}
        finish = "stop"
        usage: dict[str, Any] | None = None
        self._start_chunked("application/x-ndjson")

        def line(payload: dict[str, Any]) -> None:
            self._chunk((json.dumps(payload, separators=(",", ":")) + "\n").encode("utf-8"))

        def piece(field: str, text: str) -> dict[str, Any]:
            base: dict[str, Any] = {"model": model, "created_at": created, "done": False}
            if chat:
                base["message"] = {"role": "assistant", field: text}
            else:
                base["response" if field == "content" else field] = text
            return base

        try:
            seen = False
            for event in deltas:
                seen = True
                if isinstance(event.get("usage"), dict):
                    usage = event["usage"]
                for choice in event.get("choices") or []:
                    if choice.get("finish_reason"):
                        finish = choice["finish_reason"]
                    delta = choice.get("delta") or {}
                    if delta.get("content"):
                        line(piece("content", str(delta["content"])))
                    thought = delta.get("reasoning_content") or delta.get("reasoning")
                    if include_thinking and thought:
                        line(piece("thinking", str(thought)))
                    for raw_call in delta.get("tool_calls") or []:
                        index = int(raw_call.get("index", 0) or 0)
                        item = calls.setdefault(index, {"name": "", "arguments": ""})
                        function = raw_call.get("function") or {}
                        if function.get("name"):
                            item["name"] = str(function["name"])
                        item["arguments"] += str(function.get("arguments") or "")
            if not seen:
                raise ProxyError("upstream returned no events.")
            if finish == "content_filter":
                raise ProxyError("upstream blocked the response by content filter.", 400)
            final: dict[str, Any] = {"model": model, "created_at": created, "done": True}
            tool_calls = [
                {"function": {"name": item["name"], "arguments": _json_obj(item["arguments"])}}
                for _, item in sorted(calls.items())
                if item["name"]
            ]
            if chat:
                final["message"] = {"role": "assistant", "content": ""}
                if tool_calls:
                    final["message"]["tool_calls"] = tool_calls
            else:
                final["response"] = ""
            final["done_reason"] = "tool_calls" if tool_calls else finish
            final["prompt_eval_count"], final["eval_count"] = _ollama_usage(usage)
            line(final)
        except CLIENT_GONE:
            self._meta["error"] = "client disconnected"
            self.close_connection = True
            return
        except Exception as exc:
            self._meta["error"] = str(exc)
            try:
                line({"error": str(exc)})
            except OSError:
                pass
        self._end_chunked()


def _probe_running(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1.5) as response:
            data = json.load(response)
    except (OSError, ValueError):
        return False
    return isinstance(data, dict) and data.get("ok") is True and "version" in data


class AppHTTPServer(ThreadingHTTPServer):
    # On Windows SO_REUSEADDR lets a second socket bind a port that is in use,
    # so a busy port would not be detected. Use exclusive binding there instead.
    allow_reuse_address = os.name != "nt"

    def __init__(self, address: tuple[str, int], handler: Any) -> None:
        self.address_family = _family(address[0])
        super().__init__(address, handler)

    def server_bind(self) -> None:
        if os.name == "nt" and hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        if self.address_family == socket.AF_INET6 and self.server_address[0] == "::":
            self.socket.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
        super().server_bind()


def _bind(host: str, port: int, wait: float) -> ThreadingHTTPServer:
    deadline = time.time() + wait
    while True:
        try:
            return AppHTTPServer((host, port), OpenAICompatibleHandler)
        except OSError:
            if time.time() >= deadline:
                raise
            time.sleep(0.3)


def _print_window_banner(port: int) -> None:
    urls = public_urls(port)
    _emit("")
    _emit("  7q9k route is running")
    _emit("")
    _emit(f"  Dashboard        http://localhost:{port}")
    _emit(f"  Base URL         {urls['local']}/v1")
    if urls["network"]:
        _emit(f"  On your network  {urls['network']}")
    _emit("")
    _emit("  This window closes in a few seconds. The server keeps running")
    _emit("  in the background. Stop it in the dashboard under Settings.")
    _emit("")


def _restart_process() -> None:
    env = {k: v for k, v in os.environ.items() if not k.startswith("_PYI") and k != "_MEIPASS2"}
    env["ROUTE7Q9K_RESTARTED"] = "1"
    env["PYINSTALLER_RESET_ENVIRONMENT"] = "1"
    argv = [sys.executable] + (sys.argv[1:] if FROZEN else sys.argv)
    if os.name == "nt":
        flags = 0x00000008 | 0x00000200 | 0x08000000
        subprocess.Popen(argv, env=env, creationflags=flags, close_fds=True)
    else:
        os.execve(sys.executable, argv, env)


def _use_data_dir(path: Path) -> None:
    global DATA_DIR, STATS_FILE, PRICING_FILE, CONFIG_FILE, LOG_FILE, MODEL_CACHE, KEY_FILE
    DATA_DIR = path.resolve()
    STATS_FILE = DATA_DIR / "stats.json"
    PRICING_FILE = DATA_DIR / "pricing.json"
    CONFIG_FILE = DATA_DIR / "config.json"
    LOG_FILE = DATA_DIR / "server.log"
    MODEL_CACHE = DATA_DIR / "models.json"
    KEY_FILE = DATA_DIR / "api-key.txt"


def _writable(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".write-test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return True
    except OSError:
        return False


def migrate_legacy_files(source: Path, target: Path) -> list[str]:
    moved: list[str] = []
    try:
        if source.resolve() == target.resolve():
            return moved
    except OSError:
        return moved
    for name in DATA_FILES:
        old, new = source / name, target / name
        if old.is_file() and not new.exists():
            try:
                target.mkdir(parents=True, exist_ok=True)
                shutil.move(str(old), str(new))
                moved.append(name)
            except OSError:
                pass
    return moved


def legacy_data_dir() -> Path:
    """Data folder used before the rename from oc-free-api."""
    return Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local") / "oc-free-api"


def migrate_legacy_data_dir(source: Path, target: Path) -> list[str]:
    """Moves key, stats, settings and app backups from the old data folder."""
    moved = migrate_legacy_files(source, target)
    old_backups, new_backups = source / "backups", target / "backups"
    if old_backups.is_dir() and not new_backups.exists():
        try:
            shutil.move(str(old_backups), str(new_backups))
            moved.append("backups")
        except OSError:
            pass
    return moved


def _ensure_data_dir() -> None:
    if _writable(DATA_DIR):
        return
    fallback = Path(os.environ.get("LOCALAPPDATA") or Path.home()) / "7q9k-route"
    _use_data_dir(fallback)
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    try:
        _main()
    except SystemExit:
        raise
    except Exception:
        details = traceback.format_exc()
        _emit("")
        _emit("  7q9k route could not start:")
        _emit("  " + details.strip().splitlines()[-1])
        try:
            with open(LOG_FILE, "a", encoding="utf-8") as log:
                log.write(details + "\n")
        except OSError:
            pass
        if CONSOLE_OUT is not None:
            _emit(f"  Details are in {LOG_FILE}. This window closes in 30 seconds.")
            time.sleep(30)
            _windows_console_close()
        raise SystemExit(1)


def _main() -> None:
    global SERVER, BOUND_OPEN, BOUND_PORT, BOUND_HOST, CONFIG, METRICS, PRICING, CLI_HOST, CLI_PORT
    parser = argparse.ArgumentParser(
        description="OpenAI-compatible local API for OpenCode Zen free models."
    )
    parser.add_argument("--host", help="bind address; overrides the dashboard setting")
    parser.add_argument("--port", type=int, help="port; overrides the dashboard setting")
    parser.add_argument("--key", help="use a specific API key instead of the generated one")
    parser.add_argument("--no-save", action="store_true", help="do not load or save usage statistics")
    parser.add_argument("--refresh-models", action="store_true", help="refresh the model list on start")
    parser.add_argument("--no-browser", action="store_true", help="do not open the dashboard on start")
    parser.add_argument("--background", action="store_true", help="start without a window (Windows)")
    args = parser.parse_args()
    if args.port is not None and not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")

    restarted = os.environ.pop("ROUTE7Q9K_RESTARTED", "") == "1"
    windows_app = FROZEN and os.name == "nt"
    show_window = windows_app and not args.background and not restarted
    if show_window and not _windows_console_open():
        show_window = False
    _ensure_data_dir()
    if FROZEN and not os.environ.get("ROUTE7Q9K_DATA_DIR"):
        migrate_legacy_files(BASE_DIR, DATA_DIR)
        migrate_legacy_data_dir(legacy_data_dir(), DATA_DIR)
    if not show_window and (sys.stdout is None or (os.name == "nt" and (restarted or args.background))):
        open_log_file()

    CONFIG = load_config()
    env = env_overrides()
    CLI_HOST = args.host or (ACCESS_HOSTS[env["access"]] if "access" in env else None)
    CLI_PORT = args.port or env.get("port")
    host = CLI_HOST or ACCESS_HOSTS[CONFIG["access"]]
    port = CLI_PORT or CONFIG["port"]
    key = args.key or os.environ.get("ROUTE7Q9K_API_KEY")
    if key:
        _write_key(key.strip())
    open_browser = (
        windows_app and bool(CONFIG["open_browser"])
        and not (args.no_browser or args.background or restarted)
    )
    dashboard_url = f"http://localhost:{port}/"

    if not restarted and _probe_running(port):
        _emit("")
        _emit("  7q9k route is already running.")
        _emit(f"  Dashboard  {dashboard_url}")
        _emit("")
        if open_browser:
            webbrowser.open(dashboard_url)
        if show_window:
            time.sleep(4)
            _windows_console_close()
        return

    api_key = ensure_api_key()
    if args.refresh_models:
        models = available_models(refresh=True)
    else:
        preload_models()
        models = available_models()
    refresh_autostart()
    METRICS = Metrics(None if args.no_save else STATS_FILE)
    restored = METRICS.load()
    PRICING = Pricing(PRICING_FILE)
    PRICING.refresh_async()
    OpenAICompatibleHandler.api_key = api_key

    try:
        server = _bind(host, port, wait=15 if restarted else 0)
    except OSError as exc:
        _emit(f"  Cannot listen on {host}:{port}: {exc}")
        if show_window:
            _emit("  Another program may use this port. This window closes in 15 seconds.")
            time.sleep(15)
            _windows_console_close()
        sys.exit(1)
    server.daemon_threads = True
    SERVER = server
    BOUND_OPEN = host in ("0.0.0.0", "::")
    BOUND_PORT = port
    BOUND_HOST = host

    if LOG_STREAM is None:
        _console_init(port)
    if show_window:
        _print_window_banner(port)
    else:
        _print_banner(port, host, api_key, models)
        if restored:
            tokens = f"{METRICS.tokens_in + METRICS.tokens_out:,}"
            _emit(
                "  " + _paint("Stats     ", "cyan")
                + _paint(f"loaded from {STATS_FILE.name}: {METRICS.requests} requests, {tokens} tokens", "dim")
            )
            _emit("")
    if METRICS.path is not None:
        threading.Thread(target=METRICS.autosave_loop, daemon=True).start()

    def refresh_and_sync() -> None:
        if not args.refresh_models:
            available_models(refresh=True)
        sync_integrations()

    threading.Thread(target=refresh_and_sync, daemon=True).start()
    if open_browser:
        opener = threading.Timer(0.3, webbrowser.open, args=(dashboard_url,))
        opener.daemon = True
        opener.start()
    if show_window:
        def detach() -> None:
            open_log_file()
            _windows_console_close()
        closer = threading.Timer(6.0, detach)
        closer.daemon = True
        closer.start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        _emit("\nShutting down.")
    finally:
        server.server_close()
        if METRICS.path is not None:
            try:
                METRICS.save()
            except OSError:
                pass
    if RESTART_REQUESTED:
        _restart_process()


if __name__ == "__main__":
    main()
