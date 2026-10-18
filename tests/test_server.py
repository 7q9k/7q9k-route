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

import server

KEY = "sk-oc-free-test"
MODELS = ["big-pickle", "muse-spark-1.3-contributor-free"]

FAKE_PRICES = {
    "meta/muse-spark-1.3-contributor": {"prompt": 1e-07, "completion": 2e-07, "cache_read": 2e-09},
    "meta/muse-spark-1.3": {"prompt": 1.25e-06, "completion": 4.25e-06},
    "z-ai/glm-5.3-flash": {"prompt": 2e-08, "completion": 7e-07},
    "z-ai/glm-5.3-flash:free": {"prompt": 0.0, "completion": 0.0},
    "inclusionai/longcat-2.5": {"prompt": 3e-07, "completion": 1.2e-06},
}


def fake_pricing():
    pricing = server.Pricing()
    pricing.prices = dict(FAKE_PRICES)
    pricing.fetched = 1_700_000_000.0
    pricing.refresh_async = lambda **k: None
    return pricing


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
    def opener(req, model, timeout=None):
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
        self.assertLess(a[:16], b[:16])
        s1, s2 = server._session_id(), server._session_id()
        self.assertGreater(s1[:16], s2[:16])

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

    def test_responses_request_to_chat(self):
        chat = server.responses_to_chat_request({
            "model": "big-pickle",
            "instructions": "be brief",
            "input": [
                {"role": "developer", "content": [{"type": "input_text", "text": "rules"}]},
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]},
                {"type": "reasoning", "summary": []},
                {"type": "function_call", "call_id": "c1", "name": "f", "arguments": '{"a":1}'},
                {"type": "function_call", "call_id": "c2", "name": "g", "arguments": "{}"},
                {"type": "function_call_output", "call_id": "c1", "output": "done"},
            ],
            "tools": [{"type": "function", "name": "f", "parameters": {"type": "object"}},
                      {"type": "web_search"}],
            "tool_choice": {"type": "function", "name": "f"},
            "max_output_tokens": 50,
            "text": {"format": {"type": "json_object"}},
        })
        roles = [m["role"] for m in chat["messages"]]
        self.assertEqual(roles, ["system", "system", "user", "assistant", "tool"])
        self.assertEqual(chat["messages"][2]["content"], "hi")
        self.assertEqual([c["id"] for c in chat["messages"][3]["tool_calls"]], ["c1", "c2"])
        self.assertEqual(chat["messages"][4], {"role": "tool", "tool_call_id": "c1", "content": "done"})
        self.assertEqual([t["function"]["name"] for t in chat["tools"]], ["f"])
        self.assertEqual(chat["tool_choice"], {"type": "function", "function": {"name": "f"}})
        self.assertEqual(chat["max_tokens"], 50)
        self.assertEqual(chat["response_format"], {"type": "json_object"})
        self.assertEqual(server.responses_to_chat_request({"input": "x"})["messages"],
                         [{"role": "user", "content": "x"}])
        with self.assertRaises(server.ProxyError):
            server.responses_to_chat_request({"previous_response_id": "resp_1"})

    def test_responses_builder_text_reasoning_and_tools(self):
        events = [
            {"choices": [{"index": 0, "delta": {"reasoning_content": "hmm"}}]},
            chunk("Hel"), chunk("lo"),
            chunk(tool={"index": 0, "id": "call_a", "function": {"name": "f", "arguments": '{"x"'}}),
            chunk(tool={"index": 0, "function": {"arguments": ":1}"}}),
            chunk(tool={"index": 1, "id": "call_b", "function": {"name": "g"}}),
            chunk(finish="tool_calls", usage={"prompt_tokens": 3, "completion_tokens": 4}),
        ]
        b = server.ResponsesBuilder("m", {})
        stream = b.start()
        for e in events:
            stream += b.feed(e)
        stream += b.finish()
        types = [e["type"] for e in stream]
        self.assertEqual(types[:2], ["response.created", "response.in_progress"])
        self.assertEqual(types[-1], "response.completed")
        self.assertEqual([e["sequence_number"] for e in stream], list(range(len(stream))))
        self.assertEqual(types.count("response.output_item.added"), 4)
        self.assertEqual(types.count("response.output_item.done"), 4)
        out = stream[-1]["response"]["output"]
        self.assertEqual([i["type"] for i in out], ["reasoning", "message", "function_call", "function_call"])
        self.assertEqual(out[0]["summary"][0]["text"], "hmm")
        self.assertEqual(out[1]["content"][0]["text"], "Hello")
        self.assertEqual((out[2]["call_id"], out[2]["name"], out[2]["arguments"]), ("call_a", "f", '{"x":1}'))
        self.assertEqual(out[3]["arguments"], "{}")
        self.assertEqual(stream[-1]["response"]["output_text"], "Hello")
        self.assertEqual(stream[-1]["response"]["usage"]["total_tokens"], 7)
        self.assertTrue(all(i["status"] == "completed" for i in out if "status" in i))

    def test_metrics_save_and_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "stats.json"
            m = server.Metrics(path)
            m.record({"status": 200, "ms": 100, "model": "a", "tokens_in": 1000,
                      "tokens_out": 250, "tokens_cached": 400}, inference=True)
            m.record({"status": 429, "ms": 5, "model": "a",
                      "error": 'upstream rejected request (HTTP 429): {"type":"FreeUsageLimitError"}'},
                     inference=True)
            self.assertTrue(m.dirty)
            self.assertIsNotNone(m.save())
            self.assertFalse(m.dirty)
            again = server.Metrics(path)
            self.assertTrue(again.load())
            snap = again.snapshot()
            self.assertEqual((snap["requests"], snap["errors"]), (2, 1))
            self.assertEqual((snap["tokens_in"], snap["tokens_out"], snap["tokens_cached"]), (1000, 250, 400))
            self.assertEqual(snap["models"]["a"]["tokens_in"], 1000)
            self.assertEqual(len(snap["log"]), 2)
            self.assertEqual(snap["limits"]["requests_24h"], 2)
            self.assertIsNotNone(snap["limits"]["last_limit_hit"])
            self.assertIsNotNone(snap["saved_at"])
            again.reset()
            self.assertEqual(again.snapshot()["requests"], 0)

    def test_metrics_load_old_bucket_format(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "stats.json"
            hour = int(time.time() // 3600)
            minute = int(time.time() // 60)
            path.write_text(json.dumps({"version": 1, "requests": 3, "minutes": {str(minute): [3, 1]},
                                        "hours": {str(hour): 3}}), encoding="utf-8")
            m = server.Metrics(path)
            self.assertTrue(m.load())
            snap = m.snapshot()
            self.assertEqual(snap["timeline"][-1], {"requests": 3, "errors": 1, "tokens_in": 0, "tokens_out": 0})
            self.assertEqual(snap["limits"]["requests_24h"], 3)

    def test_metrics_load_ignores_broken_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "stats.json"
            path.write_text("{kaputt", encoding="utf-8")
            self.assertFalse(server.Metrics(path).load())
            self.assertFalse(server.Metrics(Path(tmp) / "missing.json").load())

    def test_limit_error_detection(self):
        hit = server.Metrics.is_limit_error
        self.assertTrue(hit({"status": 429, "error": "upstream rejected request (HTTP 429): FreeUsageLimitError"}))
        self.assertTrue(hit({"status": 200, "error": "Free usage exceeded, subscribe to Go"}))
        self.assertFalse(hit({"status": 500, "error": "internal error"}))
        self.assertFalse(hit({"status": 429, "error": 'upstream rejected request (HTTP 429): {"error":{"type":"server_error","message":"Upstream request failed: Endpoint is unavailable."}}'}))
        self.assertFalse(hit({"status": 200}))

    def test_pricing_match_uses_exact_paid_model(self):
        p = fake_pricing()
        self.assertEqual(p.match("muse-spark-1.3-contributor-free"), "meta/muse-spark-1.3-contributor")
        self.assertEqual(p.match("glm-5.3-flash-free"), "z-ai/glm-5.3-flash")
        self.assertEqual(p.match("longcat-2.5-preview-free"), "inclusionai/longcat-2.5")
        self.assertIsNone(p.match("big-pickle"))
        self.assertIsNone(p.match("muse-spark-free"))

    def test_usage_cost(self):
        price = {"prompt": 1e-06, "completion": 2e-06, "cache_read": 1e-07}
        cost = server.usage_cost({"tokens_in": 1000, "tokens_out": 500, "tokens_cached": 200}, price)
        self.assertAlmostEqual(cost, 800 * 1e-06 + 200 * 1e-07 + 500 * 2e-06)
        self.assertIsNone(server.usage_cost({"tokens_in": 1}, None))
        no_cache_price = {"prompt": 1e-06, "completion": 2e-06}
        self.assertAlmostEqual(server.usage_cost({"tokens_in": 10, "tokens_cached": 5}, no_cache_price), 1e-05)

    def test_pricing_refresh_parses_openrouter(self):
        payload = {"data": [
            {"id": "z-ai/glm-5.3-flash", "pricing": {"prompt": "0.00000002", "completion": "0.0000007",
                                                     "input_cache_read": "0.00000001"}},
            {"id": "broken/model", "pricing": {"prompt": "x"}},
        ]}
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(server, "_json_get", lambda url, timeout: payload):
            p = server.Pricing(Path(tmp) / "pricing.json")
            self.assertTrue(p.refresh())
            self.assertEqual(set(p.prices), {"z-ai/glm-5.3-flash"})
            self.assertEqual(p.prices["z-ai/glm-5.3-flash"]["cache_read"], 1e-08)
            cached = server.Pricing(Path(tmp) / "pricing.json")
            self.assertEqual(cached.prices, p.prices)
        with mock.patch.object(server, "_json_get", side_effect=OSError("offline")):
            p = server.Pricing()
            self.assertFalse(p.refresh())
            self.assertIn("offline", p.info()["error"])

    def test_load_config_defaults_and_env(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(server, "CONFIG_FILE", Path(tmp) / "c.json"):
            self.assertEqual(server.load_config()["access"], "network")
            server.CONFIG_FILE.write_text('{"access": "bogus", "port": "x", "extra": 1}', encoding="utf-8")
            config = server.load_config()
            self.assertEqual((config["access"], config["port"]), ("network", 8787))
            self.assertNotIn("extra", config)
            with mock.patch.dict(server.os.environ, {"OCFREE_ACCESS": "local", "OCFREE_PORT": "9001"}):
                config = server.load_config()
                overrides = server.env_overrides()
            self.assertEqual((config["access"], config["port"]), ("network", 8787))
            self.assertEqual(overrides, {"access": "local", "port": 9001})

    def test_metrics_timeline_and_models(self):
        m = server.Metrics()
        m.record({"status": 200, "ms": 100, "model": "a", "tokens_in": 1, "tokens_out": 2}, inference=True)
        m.record({"status": 502, "ms": 300, "model": "a"}, inference=True)
        m.record({"status": 404, "ms": 1}, inference=False)
        snap = m.snapshot()
        self.assertEqual((snap["requests"], snap["errors"]), (2, 1))
        self.assertEqual(snap["models"]["a"]["avg_ms"], 200)
        self.assertEqual(len(snap["timeline"]), m.TIMELINE_MINUTES)
        self.assertEqual(snap["timeline"][-1], {"requests": 2, "errors": 1, "tokens_in": 1, "tokens_out": 2})
        self.assertEqual(len(snap["hourly"]), 24)
        self.assertEqual(snap["hourly"][-1]["requests"], 2)
        self.assertEqual(len(snap["log"]), 3)


class ServerTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.patches = [
            mock.patch.object(server, "KEY_FILE", Path(cls.tmp.name) / "api-key.txt"),
            mock.patch.object(server, "available_models", lambda *a, **k: list(MODELS)),
            mock.patch.object(server, "_lan_ip", lambda: "192.168.1.20"),
            mock.patch.object(server, "_emit", lambda *a, **k: None),
            mock.patch.object(server, "CONFIG_FILE", Path(cls.tmp.name) / "config.json"),
            mock.patch.object(server, "autostart_enabled", lambda: None),
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
        deadline = time.time() + 5
        while threading.active_count() > self.baseline_threads and time.time() < deadline:
            time.sleep(0.01)

    def setUp(self):
        self.settle()
        self.stats_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.stats_dir.cleanup)
        server.METRICS = server.Metrics(Path(self.stats_dir.name) / "stats.json")
        server.PRICING = fake_pricing()
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


    def test_health_and_models_open(self):
        self.assertEqual(self.request("GET", "/health")[0], 200)
        status, body = self.request("GET", "/v1/models")
        self.assertEqual([m["id"] for m in json.loads(body)["data"]], MODELS)

    def test_dashboard_page_served_as_utf8(self):
        status, body = self.request("GET", "/")
        self.assertEqual(status, 200)
        text = body.decode("utf-8")
        self.assertIn("oc-free-api", text)
        self.assertNotRegex(text, "[┬╖ΓÇ]")

    def test_assets_served_without_path_escape(self):
        status, body = self.request("GET", "/assets/fonts/geist.woff2")
        self.assertEqual(status, 200)
        self.assertTrue(body.startswith(b"wOF2"))
        for path in ("/assets/../server.py", "/assets/%2e%2e/api-key.txt", "/assets/fonts/nope.woff2"):
            self.assertEqual(self.request("GET", path)[0], 404, path)

    def test_unknown_path_and_method(self):
        self.assertEqual(self.request("GET", "/nope")[0], 404)
        self.assertEqual(self.request("GET", "/v1/chat/completions")[0], 405)


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


    def test_responses_non_stream(self):
        status, body = self.request("POST", "/v1/responses", {"model": "big-pickle", "input": "hi"})
        self.assertEqual(status, 200)
        resp = json.loads(body)
        self.assertEqual((resp["object"], resp["status"], resp["output_text"]), ("response", "completed", "Hello"))
        self.assertEqual(resp["usage"]["input_tokens"], 5)

    def test_responses_stream(self):
        status, body = self.request("POST", "/v1/responses", {"model": "big-pickle", "input": "hi", "stream": True})
        self.assertEqual(status, 200)
        blocks = [b for b in body.decode().split("\n\n") if b.strip()]
        events = []
        for block in blocks:
            name, data = block.split("\n", 1)
            event = json.loads(data[len("data: "):])
            self.assertEqual(name, "event: " + event["type"])
            events.append(event)
        self.assertEqual(events[-1]["type"], "response.completed")
        text = "".join(e["delta"] for e in events if e["type"] == "response.output_text.delta")
        self.assertEqual(text, "Hello")

    def test_responses_stream_error(self):
        with mock.patch.object(server, "open_upstream_events", fake_upstream(OK_EVENTS, fail_after=1)):
            _, body = self.request("POST", "/v1/responses", {"model": "big-pickle", "input": "x", "stream": True})
        last = json.loads(body.decode().strip().split("\n\n")[-1].split("data: ", 1)[1])
        self.assertEqual(last["type"], "response.failed")
        self.assertEqual(last["response"]["error"]["message"], "upstream exploded")

    def test_responses_needs_key_remotely(self):
        self.as_remote()
        self.assertEqual(self.request("POST", "/v1/responses", {"model": "big-pickle"})[0], 401)


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


    def test_local_requests_need_no_key(self):
        self.assertEqual(self.request("POST", "/api/chat", {"model": "big-pickle", "stream": False})[0], 200)
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


    def test_requests_are_logged_and_counted(self):
        self.request("POST", "/v1/chat/completions", {"model": "big-pickle"})
        self.request("GET", "/api/dashboard/stats")
        self.request("GET", "/api/dashboard/log")
        self.settle()
        stats = json.loads(self.request("GET", "/api/dashboard/stats")[1])
        self.assertEqual(stats["requests"], 1)
        self.assertEqual(stats["tokens_in"] + stats["tokens_out"], 7)
        self.assertEqual(len(stats["log"]), 1)
        entry = stats["log"][0]
        self.assertEqual((entry["path"], entry["status"], entry["model"]),
                         ("/v1/chat/completions", 200, "big-pickle"))

    def test_stats_include_usage_costs_and_limits(self):
        self.request("POST", "/v1/chat/completions", {"model": "muse-spark-1.3-contributor-free"})
        self.settle()
        stats = json.loads(self.request("GET", "/api/dashboard/stats")[1])
        rows = {r["model"]: r for r in stats["usage"]["rows"]}
        muse = rows["muse-spark-1.3-contributor-free"]
        self.assertEqual((muse["tokens_in"], muse["tokens_out"]), (5, 2))
        self.assertAlmostEqual(muse["cost_usd"], 5 * 1e-07 + 2 * 2e-07)
        self.assertEqual(muse["price"]["openrouter_id"], "meta/muse-spark-1.3-contributor")
        self.assertAlmostEqual(muse["price"]["prompt_per_m"], 0.1)
        self.assertIsNone(rows["big-pickle"]["cost_usd"])
        self.assertAlmostEqual(stats["usage"]["total_cost_usd"], muse["cost_usd"])
        self.assertAlmostEqual(stats["log"][-1]["cost_usd"], muse["cost_usd"])
        self.assertEqual(stats["limits"]["requests_24h"], 1)
        self.assertTrue(stats["persistent"])

    def test_save_and_reset_endpoints(self):
        self.request("POST", "/v1/chat/completions", {"model": "big-pickle"})
        self.settle()
        status, body = self.request("POST", "/api/dashboard/save", {})
        self.assertEqual(status, 200)
        saved = json.loads(Path(json.loads(body)["file"]).read_text(encoding="utf-8"))
        self.assertEqual(saved["requests"], 1)
        self.as_remote()
        ok = {"Authorization": f"Bearer {KEY}"}
        self.assertEqual(self.request("POST", "/api/dashboard/save", {}, ok)[0], 200)
        self.assertEqual(self.request("POST", "/api/dashboard/reset", {}, ok)[0], 403)

    def test_reset_endpoint_local(self):
        self.request("POST", "/v1/chat/completions", {"model": "big-pickle"})
        self.settle()
        self.assertEqual(self.request("POST", "/api/dashboard/reset", {})[0], 200)
        self.settle()
        self.assertEqual(server.METRICS.snapshot()["requests"], 0)

    def test_free_limit_hit_is_tracked(self):
        def limited(req, model):
            raise server.ProxyError(
                'upstream rejected request (HTTP 429): {"type":"FreeUsageLimitError"}', 429)
        with mock.patch.object(server, "open_upstream_events", limited):
            status, _ = self.request("POST", "/v1/chat/completions", {"model": "big-pickle"})
        self.assertEqual(status, 429)
        self.settle()
        self.assertIsNotNone(server.METRICS.snapshot()["limits"]["last_limit_hit"])

    def test_models_alias_without_v1(self):
        status, body = self.request("GET", "/models")
        self.assertEqual((status, [m["id"] for m in json.loads(body)["data"]]), (200, MODELS))

    def test_proxied_request_is_not_local(self):
        headers = {"X-Forwarded-For": "203.0.113.9"}
        self.assertEqual(self.request("POST", "/v1/chat/completions", {"model": "big-pickle"}, headers)[0], 401)
        self.assertEqual(self.request("GET", "/api/dashboard/settings", headers=headers)[0], 403)
        state = json.loads(self.request("GET", "/api/dashboard", headers={**headers, "Authorization": f"Bearer {KEY}"})[1])
        self.assertIsNone(state["api_key"])

    def test_settings_roundtrip(self):
        server.CONFIG = dict(server.DEFAULT_CONFIG)
        self.addCleanup(setattr, server, "CONFIG", dict(server.DEFAULT_CONFIG))
        status, body = self.request("GET", "/api/dashboard/settings")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["access"], "network")
        status, body = self.request("POST", "/api/dashboard/settings",
                                    {"access": "local", "port": 9100, "open_browser": False})
        self.assertEqual(status, 200)
        result = json.loads(body)
        self.assertEqual((result["access"], result["port"], result["open_browser"]), ("local", 9100, False))
        self.assertTrue(result["restart_required"])
        saved = json.loads(server.CONFIG_FILE.read_text(encoding="utf-8"))
        self.assertEqual((saved["access"], saved["port"]), ("local", 9100))
        self.assertEqual(self.request("POST", "/api/dashboard/settings", {"port": 70000})[0], 400)
        self.assertEqual(self.request("POST", "/api/dashboard/settings", {"access": "server"})[0], 400)

    def test_settings_admin_only(self):
        self.as_remote()
        ok = {"Authorization": f"Bearer {KEY}"}
        self.assertEqual(self.request("GET", "/api/dashboard/settings", headers=ok)[0], 403)
        self.assertEqual(self.request("POST", "/api/dashboard/settings", {"port": 1}, ok)[0], 403)
        self.assertEqual(self.request("POST", "/api/dashboard/restart", {}, ok)[0], 403)

    def test_restart_request_sets_flag(self):
        self.addCleanup(setattr, server, "RESTART_REQUESTED", False)
        status, body = self.request("POST", "/api/dashboard/restart", {})
        self.assertEqual(status, 200)
        self.assertTrue(json.loads(body)["restarting"])
        self.assertTrue(server.RESTART_REQUESTED)

    def test_dashboard_test_endpoint(self):
        status, body = self.request("POST", "/api/dashboard/test", {"model": "big-pickle"})
        result = json.loads(body)
        self.assertEqual((status, result["text"], result["tokens"]), (200, "Hello", 7))
        self.settle()
        self.assertEqual(server.METRICS.snapshot()["models"]["big-pickle"]["requests"], 1)


if __name__ == "__main__":
    unittest.main()
