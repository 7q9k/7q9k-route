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
import json
import secrets
import string
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator

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


class OpenAICompatibleHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "oc-free-api/1.0"

    api_key: str = ""

    # ---- helpers

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        sys.stderr.write(
            "[oc-free-api] %s - %s\n" % (self.address_string(), format % args)
        )

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
        path = self.path.split("?", 1)[0]
        if path == "/health":
            self._send_json(200, {"ok": True})
            return
        if path == "/v1/models":
            if not self._authorized():
                self._unauthorized()
                return
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
        self._send_error_json(404, f"unknown path: {path}")

    # ---- POST

    def do_POST(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path not in {"/v1/chat/completions", "/chat/completions"}:
            self._send_error_json(404, f"unknown path: {path}")
            return
        if not self._authorized():
            self._unauthorized()
            return

        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        try:
            req = json.loads(raw.decode("utf-8")) if raw else {}
        except json.JSONDecodeError as exc:
            self._send_error_json(400, f"invalid JSON body: {exc}")
            return
        if not isinstance(req, dict):
            self._send_error_json(400, "request body must be a JSON object")
            return

        model = str(req.get("model") or DEFAULT_MODEL)
        if "/" in model:
            provider, model = model.split("/", 1)
            if provider != "opencode":
                self._send_error_json(
                    400,
                    "only models from the 'opencode' provider are supported "
                    "(e.g. 'opencode/big-pickle').",
                )
                return
        try:
            models = available_models() if model else []
        except Exception:  # noqa: BLE001
            models = list(FALLBACK_MODELS)
        if models and model not in models:
            self._send_error_json(
                404,
                f"model '{model}' is not an available free model. "
                f"Call GET /v1/models for the current list ({', '.join(models)})",
            )
            return

        want_stream = bool(req.get("stream"))

        try:
            if uses_responses_api(model):
                upstream_body = build_responses_request(req)
                events = upstream_stream(RESPONSES_URL, upstream_body)
                deltas = responses_events_to_deltas(events)
            else:
                upstream_body = build_chat_request({**req, "model": model})
                events = upstream_stream(DIRECT_URL, upstream_body)
                deltas = events
        except ProxyError as exc:
            self._send_error_json(exc.status_code, str(exc))
            return

        if want_stream:
            self._handle_streaming(deltas, model)
        else:
            try:
                completion = merge_deltas(deltas, model)
            except ProxyError as exc:
                self._send_error_json(exc.status_code, str(exc))
                return
            self._send_json(200, completion)

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
    base = f"http://{args.host}:{args.port}/v1"

    print("oc-free-api: OpenCode Zen free models via OpenAI-compatible API")
    print(f"  base_url : {base}")
    print(f"  api_key  : {api_key}")
    print(f"  models   : {', '.join(models)}")
    print()
    print("Client config:")
    print(f'  base_url = "{base}"')
    print(f'  api_key  = "{api_key}"')
    print()
    print(f"Key stored in: {KEY_FILE}")
    if args.host != "127.0.0.1":
        print("WARNING: bound to a non-local address. Keep it private, share no key.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
