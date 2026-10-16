"""Tests for oc-free-api. No network: the upstream is replaced by fakes.

Run:  python -m unittest discover -s tests -v
"""

from __future__ import annotations

import http.client
import json
import re
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import server  # noqa: E402

KEY = "sk-oc-free-test"
MODELS = ["big-pickle", "muse-spark-1.3-contributor-free"]


def chunk(content=None, *, finish=None, usage=None, tool=None):
    delta = {}
    if content is not None:
        delta["content"] = content
    if tool:
        delta["tool_calls"] = [tool]
    event = {"choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
    if usage:
        event["usage"] = usage
    return event


def fake_upstream(events, fail_after=None):
    def opener(req, model):
        def gen():
            for i, event in enumerate(events):
                if fail_after is not None and i == fail_after:
                    raise server.ProxyError("upstream exploded", 502)
                yield event
        return gen()
    return opener


OK_EVENTS = [
    chunk("Hel"),
    chunk("lo"),
    chunk(finish="stop", usage={"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}),
]


class PureFunctionTests(unittest.TestCase):
    def test_oc_ids_have_opencode_shape_and_order(self):
        a, b = server._request_id(), server._request_id()
        self.assertRegex(a, r"^msg_[0-9a-f]{12}[0-9A-Za-z]{14}$")
        self.assertLess(a[:16], b[:16])  # ascending
        s1, s2 = server._session_id(), server._session_id()
        self.assertGreater(s1[:16], s2[:16])  # descending

    def test_required_tools_injected_once(self):
        tools = server.ensure_required_tools([])
        self.assertEqual([t["function"]["name"] for t in tools], ["read", "bash"])
        mine = [{"type": "function", "function": {"name": "bash", "parameters": {}}}]
        self.assertEqual(
            [t["function"]["name"] for t in server.ensure_required_tools(mine)], ["read", "bash"]
        )

    def test_build_chat_request_adds_system_and_passthrough(self):
        body = server.build_chat_request(
            {"model": "big-pickle", "messages": [{"role": "user", "content": "x"}],
             "temperature": 0.2, "bogus": 1}
        )
        self.assertEqual(body["messages"][0], {"role": "system", "content": "You are opencode"})
        self.assertEqual(body["temperature"], 0.2)
        self.assertNotIn("bogus", body)
        self.assertTrue(body["stream"])

    def test_merge_deltas_text_and_tool_calls(self):
        events = [
            chunk("a"),
            chunk(tool={"index": 0, "id": "c1", "function": {"name": "f", "arguments": '{"x"'}}),
            chunk(tool={"index": 0, "function": {"arguments": ":1}"}}),
            chunk(finish="tool_calls"),
        ]
        result = server.merge_deltas(iter(events), "m")
        msg = result["choices"][0]["message"]
        self.assertEqual(msg["content"], "a")
        self.assertEqual(msg["tool_calls"][0]["function"], {"name": "f", "arguments": '{"x":1}'})
        self.assertEqual(result["choices"][0]["finish_reason"], "tool_calls")

    def test_merge_deltas_errors(self):
        with self.assertRaises(server.ProxyError):
            server.merge_deltas(iter([]), "m")
        with self.assertRaises(server.ProxyError):
            server.merge_deltas(iter([chunk(finish="content_filter")]), "m")

    def test_responses_events_to_deltas(self):
        events = [
            {"type": "response.output_text.delta", "delta": "hi", "response": {"model": "x"}},
            {"type": "response.output_item.done", "output_index": 1,
             "item": {"type": "function_call", "call_id": "c", "name": "f", "arguments": "{}"}},
            {"type": "response.completed", "response": {"status": "completed",
             "usage": {"input_tokens": 3, "output_tokens": 4, "total_tokens": 7}}},
        ]
        merged = server.merge_deltas(server.responses_events_to_deltas(iter(events)), "x")
        self.assertEqual(merged["choices"][0]["message"]["content"], "hi")
        self.assertEqual(merged["choices"][0]["finish_reason"], "tool_calls")
        self.assertEqual(merged["usage"]["total_tokens"], 7)

    def test_normalize_model_name(self):
        self.assertEqual(server.normalize_model_name("opencode/big-pickle:latest"), "big-pickle")
        self.assertEqual(server.normalize_model_name(None), server.DEFAULT_MODEL)
        with self.assertRaises(server.ProxyError):
            server.normalize_model_name("openai/gpt")

    def test_ollama_tool_roundtrip_ids(self):
        msgs = server.ollama_to_openai_messages([
            {"role": "assistant", "tool_calls": [{"function": {"name": "f", "arguments": {"a": 1}}}]},
            {"role": "tool", "tool_name": "f", "content": "ok"},
        ])
        self.assertEqual(msgs[0]["tool_calls"][0]["id"], msgs[1]["tool_call_id"])
        self.assertEqual(msgs[0]["tool_calls"][0]["function"]["arguments"], '{"a":1}')

    def test_ollama_options(self):
        req = server.ollama_options(
            {"model": "m", "format": "json", "options": {"num_predict": 9, "stop": "x"}}
        )
        self.assertEqual(req["max_tokens"], 9)
        self.assertEqual(req["stop"], ["x"])
        self.assertEqual(req["response_format"], {"type": "json_object"})

    def test_metrics_timeline_and_models(self):
        m = server.Metrics()
        m.record({"status": 200, "ms": 100, "model": "a", "tokens_in": 1, "tokens_out": 2}, inference=True)
        m.record({"status": 502, "ms": 300, "model": "a"}, inference=True)
        m.record({"status": 404, "ms": 1}, inference=False)
        snap = m.snapshot()
        self.assertEqual((snap["requests"], snap["errors"]), (2, 1))
        self.assertEqual(snap["models"]["a"]["avg_ms"], 200)
        self.assertEqual(len(snap["timeline"]), m.TIMELINE_MINUTES)
        self.assertEqual(snap["timeline"][-1], {"requests": 2, "errors": 1})
        self.assertEqual(len(snap["log"]), 3)


class ServerTests(unittest.TestCase):
    """Runs the real HTTP server on a random port with a fake upstream."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.patches = [
            mock.patch.object(server, "KEY_FILE", Path(cls.tmp.name) / "api-key.txt"),
            mock.patch.object(server, "available_models", lambda *a, **k: list(MODELS)),
            mock.patch.object(server, "_lan_ip", lambda: "192.168.1.20"),
        ]
        for p in cls.patches:
            p.start()
        server.KEY_FILE.write_text(KEY + "\n", encoding="utf-8")
        server.OpenAICompatibleHandler.api_key = KEY
        cls.httpd = server.ThreadingHTTPServer(("127.0.0.1", 0), server.OpenAICompatibleHandler)
        cls.httpd.daemon_threads = True
        cls.port = cls.httpd.server_address[1]
        threading.Thread(target=cls.httpd.serve_forever, daemon=True).start()
        cls.baseline_threads = threading.active_count()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        for p in cls.patches:
            p.stop()
        cls.tmp.cleanup()

    def settle(self):
        """Wait until handler threads are done (metrics are recorded after the
        response bytes reach the client)."""
        deadline = time.time() + 5
        while threading.active_count() > self.baseline_threads and time.time() < deadline:
            time.sleep(0.01)

    def setUp(self):
        self.settle()
        server.METRICS = server.Metrics()
        self.upstream = mock.patch.object(server, "open_upstream_events", fake_upstream(OK_EVENTS))
        self.upstream.start()
        self.addCleanup(self.upstream.stop)

    def request(self, method, path, body=None, headers=None, raw=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
        conn.request(method, path, body=data, headers=headers or {})
        resp = conn.getresponse()
        payload = resp.read()
        conn.close()
        return resp.status, payload

    def as_remote(self):
        patcher = mock.patch.object(server.OpenAICompatibleHandler, "_is_loopback", lambda self: False)
        patcher.start()
        self.addCleanup(patcher.stop)

    # ---- basics

    def test_health_and_models_open(self):
        self.assertEqual(self.request("GET", "/health")[0], 200)
        status, body = self.request("GET", "/v1/models")
        self.assertEqual([m["id"] for m in json.loads(body)["data"]], MODELS)

    def test_dashboard_page_served_as_utf8(self):
        status, body = self.request("GET", "/")
        self.assertEqual(status, 200)
        text = body.decode("utf-8")
        self.assertIn("oc-free-api", text)
        self.assertNotRegex(text, "[┬╖ΓÇ]")  # no mojibake

    def test_assets_served_without_path_escape(self):
        status, body = self.request("GET", "/assets/fonts/geist.woff2")
        self.assertEqual(status, 200)
        self.assertTrue(body.startswith(b"wOF2"))
        for path in ("/assets/../server.py", "/assets/%2e%2e/api-key.txt", "/assets/fonts/nope.woff2"):
            self.assertEqual(self.request("GET", path)[0], 404, path)

    def test_unknown_path_and_method(self):
        self.assertEqual(self.request("GET", "/nope")[0], 404)
        self.assertEqual(self.request("GET", "/v1/chat/completions")[0], 405)

    # ---- chat

    def test_chat_completion_non_stream(self):
        status, body = self.request(
            "POST", "/v1/chat/completions",
            {"model": "big-pickle", "messages": [{"role": "user", "content": "hi"}]},
        )
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["choices"][0]["message"]["content"], "Hello")

    def test_chat_completion_stream(self):
        status, body = self.request(
            "POST", "/v1/chat/completions",
            {"model": "big-pickle", "stream": True, "messages": []},
        )
        events = re.findall(rb"data: (.*)\n\n", body)
        self.assertEqual(events[-1], b"[DONE]")
        text = "".join(
            c["delta"].get("content", "") for e in events[:-1] for c in json.loads(e)["choices"]
        )
        self.assertEqual(text, "Hello")

    def test_stream_error_is_reported_in_band(self):
        with mock.patch.object(server, "open_upstream_events", fake_upstream(OK_EVENTS, fail_after=1)):
            status, body = self.request(
                "POST", "/v1/chat/completions", {"model": "big-pickle", "stream": True}
            )
        self.assertEqual(status, 200)
        self.assertIn(b'"error"', body)
        self.assertIn(b"upstream exploded", body)
        self.assertTrue(body.rstrip().endswith(b"data: [DONE]"))
        self.settle()
        self.assertEqual(server.METRICS.snapshot()["errors"], 1)

    def test_unknown_model_404(self):
        status, body = self.request("POST", "/v1/chat/completions", {"model": "gpt-9"})
        self.assertEqual(status, 404)
        self.assertIn("not an available free model", json.loads(body)["error"]["message"])

    def test_upstream_failure_non_stream(self):
        def boom(req, model):
            raise server.ProxyError("upstream rejected request (HTTP 429): slow down", 429)
        with mock.patch.object(server, "open_upstream_events", boom):
            status, _ = self.request("POST", "/v1/chat/completions", {"model": "big-pickle"})
        self.assertEqual(status, 429)

    def test_unexpected_exception_becomes_500(self):
        def boom(req, model):
            raise RuntimeError("kaputt")
        with mock.patch.object(server, "open_upstream_events", boom):
            status, body = self.request("POST", "/v1/chat/completions", {"model": "big-pickle"})
        self.assertEqual(status, 500)
        self.assertIn("kaputt", json.loads(body)["error"]["message"])

    def test_invalid_json_and_body_limit(self):
        status, _ = self.request("POST", "/v1/chat/completions", raw=b"{nope",
                                 headers={"Content-Type": "application/json"})
        self.assertEqual(status, 400)
        with mock.patch.object(server, "MAX_BODY_BYTES", 10):
            status, _ = self.request("POST", "/v1/chat/completions", {"model": "big-pickle"})
        self.assertEqual(status, 413)

    # ---- ollama

    def test_ollama_chat_streams_by_default(self):
        status, body = self.request("POST", "/api/chat", {"model": "big-pickle:latest", "messages": []})
        lines = [json.loads(l) for l in body.decode().splitlines() if l.strip()]
        self.assertEqual("".join(l["message"]["content"] for l in lines), "Hello")
        self.assertTrue(lines[-1]["done"])
        self.assertEqual(lines[-1]["eval_count"], 2)

    def test_ollama_generate_non_stream(self):
        status, body = self.request(
            "POST", "/api/generate", {"model": "big-pickle", "prompt": "x", "stream": False}
        )
        self.assertEqual(json.loads(body)["response"], "Hello")

    def test_ollama_stream_error_line(self):
        with mock.patch.object(server, "open_upstream_events", fake_upstream(OK_EVENTS, fail_after=1)):
            _, body = self.request("POST", "/api/chat", {"model": "big-pickle"})
        self.assertEqual(json.loads(body.decode().splitlines()[-1]), {"error": "upstream exploded"})

    # ---- access control

    def test_local_requests_need_no_key(self):
        self.assertEqual(self.request("POST", "/api/chat", {"model": "big-pickle", "stream": False})[0], 200)
        # placeholder keys from local clients are fine too
        status, _ = self.request("POST", "/v1/chat/completions", {"model": "big-pickle"},
                                 headers={"Authorization": "Bearer ollama"})
        self.assertEqual(status, 200)

    def test_remote_requests_need_key(self):
        self.as_remote()
        for path in ("/v1/chat/completions", "/api/chat", "/api/generate", "/api/dashboard/test"):
            self.assertEqual(self.request("POST", path, {"model": "big-pickle"})[0], 401, path)
        self.assertEqual(self.request("GET", "/api/dashboard/stats")[0], 401)
        ok = {"Authorization": f"Bearer {KEY}"}
        self.assertEqual(self.request("POST", "/v1/chat/completions", {"model": "big-pickle"}, ok)[0], 200)
        self.assertEqual(self.request("POST", "/api/chat", {"model": "big-pickle", "stream": False},
                                      {"x-api-key": KEY})[0], 200)
        # open endpoints stay open
        self.assertEqual(self.request("GET", "/api/tags")[0], 200)

    def test_remote_dashboard_hides_key_and_cannot_control(self):
        self.as_remote()
        ok = {"Authorization": f"Bearer {KEY}"}
        state = json.loads(self.request("GET", "/api/dashboard", headers=ok)[1])
        self.assertIsNone(state["api_key"])
        self.assertFalse(state["local"])
        self.assertEqual(self.request("POST", "/api/dashboard/key", {}, ok)[0], 403)
        self.assertEqual(self.request("POST", "/api/dashboard/stop", {}, ok)[0], 403)

    def test_local_dashboard_shows_key(self):
        state = json.loads(self.request("GET", "/api/dashboard")[1])
        self.assertEqual(state["api_key"], KEY)
        self.assertTrue(state["local"])
        self.assertEqual(state["models"][1], {"id": MODELS[1], "api": "responses"})

    def test_cross_site_browser_request_blocked(self):
        evil = {"Origin": "https://evil.example", "Content-Type": "text/plain"}
        self.assertEqual(self.request("POST", "/api/dashboard/stop", {}, evil)[0], 403)
        self.assertEqual(self.request("POST", "/api/chat", {"model": "big-pickle"}, evil)[0], 401)
        sec = {"Sec-Fetch-Site": "cross-site"}
        self.assertEqual(self.request("POST", "/api/dashboard/key", {}, sec)[0], 403)

    def test_same_origin_browser_request_allowed(self):
        headers = {"Origin": f"http://127.0.0.1:{self.port}", "Sec-Fetch-Site": "same-origin"}
        self.assertEqual(self.request("GET", "/api/dashboard", headers=headers)[0], 200)

    def test_dns_rebinding_host_rejected(self):
        status, body = self.request("GET", "/api/dashboard", headers={"Host": f"evil.example:{self.port}"})
        self.assertEqual(status, 401)
        self.assertNotIn(KEY.encode(), body)

    def test_key_rotation_from_local(self):
        status, body = self.request("POST", "/api/dashboard/key", {})
        new_key = json.loads(body)["api_key"]
        self.addCleanup(server.KEY_FILE.write_text, KEY, encoding="utf-8")
        self.addCleanup(setattr, server.OpenAICompatibleHandler, "api_key", KEY)
        self.assertNotEqual(new_key, KEY)
        self.assertEqual(server.KEY_FILE.read_text(encoding="utf-8").strip(), new_key)
        self.assertEqual(server.OpenAICompatibleHandler.api_key, new_key)

    # ---- metrics / log

    def test_requests_are_logged_and_counted(self):
        self.request("POST", "/v1/chat/completions", {"model": "big-pickle"})
        self.request("GET", "/api/dashboard/stats")  # quiet path, not logged
        self.settle()
        stats = json.loads(self.request("GET", "/api/dashboard/stats")[1])
        self.assertEqual(stats["requests"], 1)
        self.assertEqual(stats["tokens_in"] + stats["tokens_out"], 7)
        self.assertEqual(len(stats["log"]), 1)
        entry = stats["log"][0]
        self.assertEqual((entry["path"], entry["status"], entry["model"]),
                         ("/v1/chat/completions", 200, "big-pickle"))

    def test_dashboard_test_endpoint(self):
        status, body = self.request("POST", "/api/dashboard/test", {"model": "big-pickle"})
        result = json.loads(body)
        self.assertEqual((status, result["text"], result["tokens"]), (200, "Hello", 7))
        self.settle()
        self.assertEqual(server.METRICS.snapshot()["models"]["big-pickle"]["requests"], 1)


if __name__ == "__main__":
    unittest.main()
