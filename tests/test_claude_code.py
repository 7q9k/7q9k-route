"""Tests for the Claude Code app and the Anthropic Messages API it uses."""

from __future__ import annotations

import json
import re
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import integrations  # noqa: E402
import server  # noqa: E402
from test_integrations import Base, target  # noqa: E402
from test_server import KEY, ServerTests, chunk, fake_upstream  # noqa: E402


class ClaudeCodeAppTests(Base):
    def setUp(self):
        super().setUp()
        self.app = integrations.ClaudeCode()
        self.path = self.home / ".claude" / "settings.json"

    def test_not_installed_without_claude(self):
        self.assertFalse(self.app.installed())

    def test_connect_update_disconnect_keeps_other_settings(self):
        self.path.parent.mkdir()
        original = {"model": "opus", "env": {"MY_VAR": "1"}, "permissions": {"allow": ["Bash(ls:*)"]}}
        self.path.write_bytes(json.dumps(original, indent=2).replace("\n", "\r\n").encode("utf-8"))
        self.assertTrue(self.app.installed())
        self.assertEqual(self.app.status(target()), "not_connected")

        self.app.connect(target(), self.backups)
        raw = self.path.read_bytes()
        self.assertIn(b"\r\n", raw)
        data = json.loads(raw)
        env = data["env"]
        self.assertEqual(env["ANTHROPIC_BASE_URL"], "http://127.0.0.1:8787")
        self.assertEqual(env["ANTHROPIC_AUTH_TOKEN"], "sk-test-1")
        for key in self.app.model_keys:
            self.assertEqual(env[key], "big-pickle")
        self.assertEqual(env["MY_VAR"], "1")
        self.assertEqual(data["permissions"], original["permissions"])
        self.assertEqual(self.app.status(target()), "connected")
        self.assertEqual(self.app.status(target(port=9000)), "outdated")
        self.assertEqual(self.app.status(target(key="other-key-1")), "outdated")

        self.app.connect(target(port=9000), self.backups)
        self.assertEqual(json.loads(self.path.read_bytes())["env"]["ANTHROPIC_BASE_URL"], "http://127.0.0.1:9000")
        self.assertTrue(list(self.backups.glob("*.bak")))

        self.app.disconnect(self.backups)
        self.assertEqual(json.loads(self.path.read_bytes()), original)
        self.assertEqual(self.app.status(target()), "not_connected")

    def test_creates_settings_and_removes_empty_env(self):
        self.path.parent.mkdir()
        self.app.connect(target(), self.backups)
        self.assertEqual(self.app.status(target()), "connected")
        self.app.disconnect(self.backups)
        self.assertEqual(json.loads(self.path.read_text(encoding="utf-8")), {})

    def test_leaves_a_hand_made_proxy_alone(self):
        self.path.parent.mkdir()
        other = {"env": {"ANTHROPIC_BASE_URL": "http://127.0.0.1:3456", "ANTHROPIC_AUTH_TOKEN": "x"}}
        self.path.write_text(json.dumps(other), encoding="utf-8")
        self.assertEqual(self.app.status(target()), "not_connected")
        self.app.disconnect(self.backups)
        self.assertEqual(json.loads(self.path.read_text(encoding="utf-8")), other)

    def test_invalid_json_is_reported_not_overwritten(self):
        self.path.parent.mkdir()
        self.path.write_text("{ // comment\n}", encoding="utf-8")
        with self.assertRaises(integrations.IntegrationError):
            self.app.connect(target(), self.backups)
        self.assertEqual(self.path.read_text(encoding="utf-8"), "{ // comment\n}")
        self.assertEqual(self.app.describe(target())["status"], "error")

    def test_respects_claude_config_dir(self):
        other = Path(self.tmp.name) / "claude-cfg"
        with mock.patch.dict(integrations.os.environ, {"CLAUDE_CONFIG_DIR": str(other)}):
            self.assertEqual(self.app.config_paths(), [other / "settings.json"])

    def test_listed_instead_of_the_old_agent(self):
        ids = [item.id for item in integrations.REGISTRY]
        self.assertIn("claude-code", ids)
        self.assertNotIn("7q9k-agent", ids)


class ConversionTests(unittest.TestCase):
    def test_messages_tools_and_results(self):
        req = {
            "model": "claude-sonnet-4-5",
            "system": [{"type": "text", "text": "Be brief.", "cache_control": {"type": "ephemeral"}}],
            "max_tokens": 32000,
            "temperature": 1,
            "tool_choice": {"type": "any"},
            "tools": [
                {"name": "Read", "description": "read a file", "input_schema": {"type": "object"}},
                {"type": "web_search_20250305", "name": "web_search"},
            ],
            "messages": [
                {"role": "user", "content": "read a.txt"},
                {"role": "assistant", "content": [
                    {"type": "thinking", "thinking": "hmm", "signature": "s"},
                    {"type": "text", "text": "Reading."},
                    {"type": "tool_use", "id": "toolu_1", "name": "Read", "input": {"file_path": "a.txt"}},
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_1", "content": [{"type": "text", "text": "hello"}]},
                    {"type": "text", "text": "and now?"},
                ]},
            ],
        }
        chat = server.anthropic_to_chat_request(req)
        self.assertNotIn("max_tokens", chat)
        self.assertEqual(chat["tool_choice"], "required")
        self.assertEqual([t["function"]["name"] for t in chat["tools"]], ["Read"])
        roles = [m["role"] for m in chat["messages"]]
        self.assertEqual(roles, ["system", "user", "assistant", "tool", "user"])
        self.assertEqual(chat["messages"][0]["content"], "Be brief.")
        call = chat["messages"][2]["tool_calls"][0]
        self.assertEqual(call["id"], "toolu_1")
        self.assertEqual(json.loads(call["function"]["arguments"]), {"file_path": "a.txt"})
        self.assertEqual(chat["messages"][3], {"role": "tool", "tool_call_id": "toolu_1", "content": "hello"})
        self.assertEqual(chat["messages"][4]["content"], "and now?")

    def test_error_results_and_images(self):
        req = {"messages": [{"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t", "is_error": True, "content": "boom"},
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"}},
        ]}]}
        messages = server.anthropic_to_chat_messages(req)
        self.assertEqual(messages[0]["content"], "Error: boom")
        self.assertEqual(messages[1]["content"][0]["image_url"]["url"], "data:image/png;base64,AAAA")

    def test_non_stream_message(self):
        completion = server.merge_deltas(iter([
            chunk("Hi"),
            chunk(tool={"index": 0, "id": "call_9", "function": {"name": "Bash", "arguments": '{"command":'}}),
            chunk(tool={"index": 0, "function": {"arguments": '"ls"}'}}),
            chunk(finish="tool_calls", usage={"prompt_tokens": 10, "completion_tokens": 3,
                                             "prompt_tokens_details": {"cached_tokens": 4}}),
        ]), "big-pickle")
        message = server.anthropic_message(completion, thinking=False)
        self.assertEqual(message["type"], "message")
        self.assertEqual(message["stop_reason"], "tool_use")
        self.assertEqual(message["content"][0], {"type": "text", "text": "Hi"})
        self.assertEqual(message["content"][1]["input"], {"command": "ls"})
        self.assertEqual(message["usage"]["input_tokens"], 6)
        self.assertEqual(message["usage"]["cache_read_input_tokens"], 4)


def sse_events(body: bytes) -> list[tuple[str, dict]]:
    return [(name.decode(), json.loads(data)) for name, data in re.findall(rb"event: (\S+)\ndata: (.*)\n\n", body)]


class AnthropicServerTests(ServerTests):
    def test_messages_non_stream(self):
        status, body = self.request("POST", "/v1/messages?beta=true", {
            "model": "claude-sonnet-4-5", "max_tokens": 100, "messages": [{"role": "user", "content": "hi"}],
        })
        self.assertEqual(status, 200)
        data = json.loads(body)
        self.assertEqual(data["content"], [{"type": "text", "text": "Hello"}])
        self.assertEqual(data["stop_reason"], "end_turn")

    def test_messages_stream_with_thinking_and_tool(self):
        events = [
            {"choices": [{"index": 0, "delta": {"reasoning_content": "think"}, "finish_reason": None}]},
            chunk("ok"),
            chunk(tool={"index": 0, "id": "call_1", "function": {"name": "Bash", "arguments": '{"command":"ls"}'}}),
            chunk(finish="tool_calls", usage={"prompt_tokens": 5, "completion_tokens": 2}),
        ]
        with mock.patch.object(server, "open_upstream_events", fake_upstream(events)):
            status, body = self.request("POST", "/v1/messages", {
                "model": "big-pickle", "stream": True, "thinking": {"type": "enabled", "budget_tokens": 1024},
                "messages": [{"role": "user", "content": "hi"}],
            })
        self.assertEqual(status, 200)
        parsed = sse_events(body)
        names = [n for n, _ in parsed]
        self.assertEqual(names[0], "message_start")
        self.assertEqual(names[-2:], ["message_delta", "message_stop"])
        starts = [p["content_block"]["type"] for n, p in parsed if n == "content_block_start"]
        self.assertEqual(starts, ["thinking", "text", "tool_use"])
        self.assertEqual(names.count("content_block_start"), names.count("content_block_stop"))
        partial = "".join(p["delta"].get("partial_json", "") for n, p in parsed if n == "content_block_delta")
        self.assertEqual(json.loads(partial), {"command": "ls"})
        self.assertEqual(parsed[-2][1]["delta"]["stop_reason"], "tool_use")
        self.assertEqual(parsed[-2][1]["usage"]["output_tokens"], 2)

    def test_stream_error_is_an_anthropic_error_event(self):
        with mock.patch.object(server, "open_upstream_events", fake_upstream([chunk("a"), chunk("b")], fail_after=1)):
            status, body = self.request("POST", "/v1/messages", {
                "model": "big-pickle", "stream": True, "messages": [{"role": "user", "content": "hi"}],
            })
        parsed = sse_events(body)
        self.assertEqual(parsed[-1][0], "error")
        self.assertEqual(parsed[-1][1]["error"]["type"], "api_error")

    def test_unknown_model_gives_anthropic_error(self):
        status, body = self.request("POST", "/v1/messages", {
            "model": "gpt-9", "messages": [{"role": "user", "content": "hi"}],
        })
        self.assertEqual(status, 404)
        self.assertEqual(json.loads(body)["type"], "error")
        self.assertEqual(json.loads(body)["error"]["type"], "not_found_error")

    def test_aliases_and_context_suffix_map_to_route_models(self):
        for name in ("opus", "claude-haiku-4-5-20251001", "muse-spark-1.3-contributor-free[1m]"):
            status, _ = self.request("POST", "/v1/messages", {
                "model": name, "messages": [{"role": "user", "content": "hi"}],
            })
            self.assertEqual(status, 200, name)

    def test_count_tokens(self):
        status, body = self.request("POST", "/v1/messages/count_tokens", {
            "model": "big-pickle", "messages": [{"role": "user", "content": "x" * 400}],
        })
        self.assertEqual(status, 200)
        self.assertGreater(json.loads(body)["input_tokens"], 90)

    def test_remote_needs_key_via_bearer_or_x_api_key(self):
        self.as_remote()
        req = {"model": "big-pickle", "messages": [{"role": "user", "content": "hi"}]}
        self.assertEqual(self.request("POST", "/v1/messages", req)[0], 401)
        self.assertEqual(self.request("POST", "/v1/messages", req, {"x-api-key": KEY})[0], 200)
        self.assertEqual(self.request("POST", "/v1/messages", req, {"Authorization": f"Bearer {KEY}"})[0], 200)

    def test_models_list_has_anthropic_fields(self):
        data = json.loads(self.request("GET", "/v1/models", headers={"anthropic-version": "2023-06-01"})[1])
        self.assertFalse(data["has_more"])
        self.assertEqual(data["data"][0]["type"], "model")
        self.assertEqual(data["first_id"], "big-pickle")


# Run only the tests above here; the inherited ServerTests run in test_server.
for _name in [n for n in dir(ServerTests) if n.startswith("test_")]:
    if _name not in AnthropicServerTests.__dict__:
        setattr(AnthropicServerTests, _name, None)


if __name__ == "__main__":
    unittest.main()
