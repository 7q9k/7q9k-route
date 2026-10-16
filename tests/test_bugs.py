from __future__ import annotations

import http.client
import io
import json
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import server  # noqa: E402
import test_server  # noqa: E402
from test_server import KEY, OK_EVENTS  # noqa: E402

ServerHarness = test_server.ServerTests.__dict__


class FakeResponse:
    def __init__(self, lines, fail_with=None):
        self.lines = lines
        self.fail_with = fail_with
        self.closed = False

    def __iter__(self):
        yield from self.lines
        if self.fail_with:
            raise self.fail_with

    def close(self):
        self.closed = True


class UpstreamTests(unittest.TestCase):
    def test_connection_reset_while_streaming_is_upstream_error(self):
        response = FakeResponse([b'data: {"choices":[]}\n'], ConnectionResetError("reset"))
        events = server._sse_lines_to_payloads(response)
        next(events)
        with self.assertRaises(server.ProxyError) as ctx:
            next(events)
        self.assertIn("upstream connection lost", str(ctx.exception))
        self.assertTrue(response.closed)

    def test_incomplete_read_is_upstream_error(self):
        response = FakeResponse([], http.client.IncompleteRead(b""))
        with self.assertRaises(server.ProxyError):
            list(server._sse_lines_to_payloads(response))

    def test_redirect_is_not_passed_to_client(self):
        error = urllib.error.HTTPError("https://x", 302, "Found", {}, io.BytesIO(b""))
        with mock.patch.object(server._OPENER, "open", side_effect=error):
            with self.assertRaises(server.ProxyError) as ctx:
                server.upstream_stream("https://x", {})
        self.assertEqual(ctx.exception.status_code, 502)

    def test_rate_limit_status_is_kept(self):
        error = urllib.error.HTTPError("https://x", 429, "Too Many", {}, io.BytesIO(b'{"type":"FreeUsageLimitError"}'))
        with mock.patch.object(server._OPENER, "open", side_effect=error):
            with self.assertRaises(server.ProxyError) as ctx:
                server.upstream_stream("https://x", {})
        self.assertEqual(ctx.exception.status_code, 429)
        self.assertIn("FreeUsageLimitError", str(ctx.exception))

    def test_messages_must_be_a_list_of_objects(self):
        for bad in ("hello", [1, 2], {"role": "user"}):
            with self.assertRaises(server.ProxyError) as ctx:
                server.build_chat_request({"messages": bad})
            self.assertEqual(ctx.exception.status_code, 400)
        self.assertEqual(len(server.build_chat_request({})["messages"]), 1)

    def test_ollama_ignores_broken_tool_calls(self):
        msgs = server.ollama_to_openai_messages(
            [{"role": "assistant", "tool_calls": ["oops", {"function": {"name": "f"}}]}]
        )
        self.assertEqual([c["function"]["name"] for c in msgs[0]["tool_calls"]], ["f"])

    def test_model_preload_needs_no_network(self):
        with mock.patch.object(server, "_MODEL_SNAPSHOT", None), \
                mock.patch.object(server, "_json_get", side_effect=AssertionError("network used")), \
                mock.patch.object(server, "MODEL_CACHE", Path(tempfile.gettempdir()) / "missing-models.json"):
            server.preload_models()
            self.assertEqual(server.available_models(), list(server.FALLBACK_MODELS))


class StateTests(unittest.TestCase):
    def test_snapshot_returns_copies(self):
        m = server.Metrics()
        m.record({"status": 200, "ms": 1, "model": "a"}, inference=True)
        snap = m.snapshot()
        snap["log"][0]["cost_usd"] = 1.0
        self.assertNotIn("cost_usd", m.snapshot()["log"][0])

    def test_parallel_saves_do_not_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            m = server.Metrics(Path(tmp) / "stats.json")
            m.record({"status": 200, "ms": 1, "model": "a"}, inference=True)
            errors = []

            def worker():
                try:
                    for _ in range(10):
                        m.save()
                except Exception as exc:
                    errors.append(exc)

            threads = [threading.Thread(target=worker) for _ in range(8)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            self.assertEqual(errors, [])
            self.assertEqual(json.loads(m.path.read_text(encoding="utf-8"))["requests"], 1)

    def test_corrupt_stats_file_is_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "stats.json"
            path.write_text(json.dumps({"version": 1, "requests": "many", "models": []}), encoding="utf-8")
            m = server.Metrics(path)
            self.assertFalse(m.load())
            self.assertEqual(m.snapshot()["requests"], 0)

    def test_price_refresh_backs_off_after_failure(self):
        p = server.Pricing()
        calls = []
        with mock.patch.object(p, "refresh", side_effect=lambda: calls.append(1) or False):
            p.refresh_async()
            time.sleep(0.2)
            p.refresh_async()
            time.sleep(0.2)
        self.assertEqual(len(calls), 1)

    def test_data_dir_falls_back_when_not_writable(self):
        with tempfile.TemporaryDirectory() as tmp:
            original = server.DATA_DIR
            self.addCleanup(server._use_data_dir, original)
            with mock.patch.object(server, "_writable", return_value=False), \
                    mock.patch.dict(server.os.environ, {"LOCALAPPDATA": tmp}):
                server._ensure_data_dir()
            self.assertEqual(server.DATA_DIR, (Path(tmp) / "oc-free-api").resolve())
            self.assertEqual(server.KEY_FILE.parent, server.DATA_DIR)

    def test_exe_data_lives_in_local_appdata(self):
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(server, "FROZEN", True), \
                mock.patch.object(server.os, "name", "nt"), \
                mock.patch.dict(server.os.environ, {"LOCALAPPDATA": tmp}, clear=False):
            server.os.environ.pop("OCFREE_DATA_DIR", None)
            self.assertEqual(server.default_data_dir(), Path(tmp) / "oc-free-api")
        with mock.patch.dict(server.os.environ, {"OCFREE_DATA_DIR": "X:/custom"}):
            self.assertEqual(server.default_data_dir(), Path("X:/custom"))

    def test_legacy_files_are_moved_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            old, new = Path(tmp) / "Downloads", Path(tmp) / "AppData" / "oc-free-api"
            old.mkdir()
            (old / "api-key.txt").write_text("sk-old", encoding="utf-8")
            (old / "stats.json").write_text("{}", encoding="utf-8")
            (old / "unrelated.txt").write_text("keep", encoding="utf-8")
            new.mkdir(parents=True)
            (new / "stats.json").write_text('{"newer": true}', encoding="utf-8")
            moved = server.migrate_legacy_files(old, new)
            self.assertEqual(moved, ["api-key.txt"])
            self.assertEqual((new / "api-key.txt").read_text(encoding="utf-8"), "sk-old")
            self.assertFalse((old / "api-key.txt").exists())
            self.assertTrue((old / "stats.json").exists())
            self.assertEqual((new / "stats.json").read_text(encoding="utf-8"), '{"newer": true}')
            self.assertTrue((old / "unrelated.txt").exists())
            self.assertEqual(server.migrate_legacy_files(new, new), [])

    def test_port_available(self):
        import socket
        busy = socket.socket()
        busy.bind(("127.0.0.1", 0))
        busy.listen()
        self.addCleanup(busy.close)
        self.assertFalse(server.port_available("127.0.0.1", busy.getsockname()[1]))


class ServerBugTests(unittest.TestCase):
    setUpClass = ServerHarness["setUpClass"]
    tearDownClass = ServerHarness["tearDownClass"]
    setUp = ServerHarness["setUp"]
    settle = ServerHarness["settle"]
    request = ServerHarness["request"]
    as_remote = ServerHarness["as_remote"]

    def test_apps_endpoints(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        home = Path(tmp.name)
        cfg = home / "hermes" / "config.yaml"
        cfg.parent.mkdir(parents=True)
        cfg.write_text("custom_providers: []\n", encoding="utf-8")
        env = mock.patch.dict(server.os.environ, {"USERPROFILE": tmp.name, "HOME": tmp.name,
                                                  "LOCALAPPDATA": tmp.name, "APPDATA": tmp.name})
        env.start()
        self.addCleanup(env.stop)
        server.os.environ.pop("HERMES_HOME", None)
        server.CONFIG = dict(server.DEFAULT_CONFIG, integrations=[])
        self.addCleanup(setattr, server, "CONFIG", dict(server.DEFAULT_CONFIG))
        data_dir = mock.patch.object(server, "DATA_DIR", home / "data")
        data_dir.start()
        self.addCleanup(data_dir.stop)
        apps = {a["id"]: a for a in json.loads(self.request("GET", "/api/dashboard/integrations")[1])["apps"]}
        self.assertEqual(apps["hermes"]["status"], "not_connected")
        status, body = self.request("POST", "/api/dashboard/integrations/connect", {"id": "hermes"})
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["app"]["status"], "connected")
        self.assertEqual(server.CONFIG["integrations"], ["hermes"])
        self.assertIn(KEY, cfg.read_text(encoding="utf-8"))
        self.assertEqual(self.request("POST", "/api/dashboard/integrations/connect", {"id": "nope"})[0], 404)
        self.assertEqual(self.request("POST", "/api/dashboard/integrations/connect", {"id": "opencode"})[0], 409)
        status, body = self.request("POST", "/api/dashboard/integrations/disconnect", {"id": "hermes"})
        self.assertEqual(json.loads(body)["app"]["status"], "not_connected")
        self.assertEqual(server.CONFIG["integrations"], [])
        self.as_remote()
        ok = {"Authorization": f"Bearer {KEY}"}
        self.assertEqual(self.request("GET", "/api/dashboard/integrations", headers=ok)[0], 403)

    def test_own_key(self):
        self.addCleanup(server.KEY_FILE.write_text, KEY, encoding="utf-8")
        self.addCleanup(setattr, server.OpenAICompatibleHandler, "api_key", KEY)
        self.assertEqual(self.request("POST", "/api/dashboard/key", {"key": "short"})[0], 400)
        self.assertEqual(self.request("POST", "/api/dashboard/key", {"key": "has space in it"})[0], 400)
        status, body = self.request("POST", "/api/dashboard/key", {"key": "my-fixed-key-123"})
        self.assertEqual(status, 200)
        self.assertEqual(server.KEY_FILE.read_text(encoding="utf-8").strip(), "my-fixed-key-123")
        self.as_remote()
        ok = {"Authorization": "Bearer my-fixed-key-123"}
        self.assertEqual(self.request("POST", "/v1/chat/completions", {"model": "big-pickle"}, ok)[0], 200)

    def test_host_check_is_exact(self):
        machine = server.socket.gethostname().lower().split(".", 1)[0]
        status, body = self.request("GET", "/api/dashboard", headers={"Host": f"{machine}.evil.example:{self.port}"})
        self.assertEqual(status, 401)
        status, _ = self.request("GET", "/api/dashboard", headers={"Host": f"{machine}:{self.port}"})
        self.assertEqual(status, 200)

    def test_control_characters_in_path_are_cleaned(self):
        import socket
        with socket.create_connection(("127.0.0.1", self.port), timeout=5) as sock:
            sock.sendall(b"GET /x\x1b[2Jy HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n")
            while sock.recv(4096):
                pass
        self.settle()
        self.assertEqual(server.METRICS.snapshot()["log"][-1]["path"], "/x?[2Jy")

    def test_bad_messages_give_400(self):
        status, body = self.request("POST", "/v1/chat/completions", {"model": "big-pickle", "messages": "hi"})
        self.assertEqual(status, 400)

    def test_settings_invalid_request_changes_nothing(self):
        server.CONFIG = dict(server.DEFAULT_CONFIG)
        self.addCleanup(setattr, server, "CONFIG", dict(server.DEFAULT_CONFIG))
        status, _ = self.request("POST", "/api/dashboard/settings", {"access": "local", "port": "abc"})
        self.assertEqual(status, 400)
        self.assertEqual(server.CONFIG["access"], "network")
        status, _ = self.request("POST", "/api/dashboard/settings", {"open_browser": "false"})
        self.assertEqual(status, 400)
        status, _ = self.request("POST", "/api/dashboard/settings", {"port": 80})
        self.assertEqual(status, 400)

    def test_restart_refuses_busy_port(self):
        import socket
        busy = socket.socket()
        busy.bind(("0.0.0.0", 0))
        busy.listen()
        self.addCleanup(busy.close)
        server.CONFIG = dict(server.DEFAULT_CONFIG, port=busy.getsockname()[1])
        self.addCleanup(setattr, server, "CONFIG", dict(server.DEFAULT_CONFIG))
        self.addCleanup(setattr, server, "RESTART_REQUESTED", False)
        status, body = self.request("POST", "/api/dashboard/restart", {})
        self.assertEqual(status, 409)
        self.assertFalse(server.RESTART_REQUESTED)

    def test_stats_do_not_store_costs_in_log(self):
        self.request("POST", "/v1/chat/completions", {"model": "muse-spark-1.3-contributor-free"})
        self.settle()
        self.request("GET", "/api/dashboard/stats")
        self.assertNotIn("cost_usd", server.METRICS.log[-1])

    def test_dashboard_key_comes_from_running_server(self):
        server.KEY_FILE.write_text("sk-edited-by-hand\n", encoding="utf-8")
        self.addCleanup(server.KEY_FILE.write_text, KEY, encoding="utf-8")
        state = json.loads(self.request("GET", "/api/dashboard")[1])
        self.assertEqual(state["api_key"], KEY)

    def test_many_parallel_requests(self):
        results = []

        def worker():
            results.append(self.request("POST", "/v1/chat/completions", {"model": "big-pickle"})[0])

        threads = [threading.Thread(target=worker) for _ in range(30)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.settle()
        self.assertEqual(results.count(200), 30)
        self.assertEqual(server.METRICS.snapshot()["requests"], 30)

    def test_stream_with_lost_upstream_reports_error(self):
        def lost(req, model, timeout=None):
            def gen():
                yield OK_EVENTS[0]
                raise server.ProxyError("upstream connection lost: ConnectionResetError")
            return gen()
        with mock.patch.object(server, "open_upstream_events", lost):
            _, body = self.request("POST", "/v1/chat/completions", {"model": "big-pickle", "stream": True})
        self.assertIn(b"upstream connection lost", body)
        self.assertTrue(body.rstrip().endswith(b"data: [DONE]"))



if __name__ == "__main__":
    unittest.main()
