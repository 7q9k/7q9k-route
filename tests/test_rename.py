"""Tests for the rename to 7q9k route, the 7q9k-Agent app, and
retrying upstream 5xx errors."""

from __future__ import annotations

import io
import sys
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import integrations  # noqa: E402
import server  # noqa: E402
from test_integrations import Base, DSH_PATCH, target  # noqa: E402


def http_error(code: int, body: bytes = b"") -> urllib.error.HTTPError:
    return urllib.error.HTTPError("https://x", code, "err", {}, io.BytesIO(body))


class FakeResponse:
    def __init__(self, lines):
        self._lines = lines

    def __iter__(self):
        return iter(self._lines)

    def close(self):
        pass


class UpstreamRetryTests(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.object(server, "UPSTREAM_RETRY_DELAYS", (0, 0, 0, 0))
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_retries_503_then_streams(self):
        ok = FakeResponse([b'data: {"choices": []}\n', b"data: [DONE]\n"])
        unavailable = http_error(503, b'{"error":{"message":"Endpoint is unavailable"}}')
        with mock.patch.object(server._OPENER, "open", side_effect=[unavailable, unavailable, ok]) as opener:
            events = list(server.upstream_stream("https://x", {"model": "m"}))
        self.assertEqual(events, [{"choices": []}])
        self.assertEqual(opener.call_count, 3)

    def test_gives_up_after_all_retries(self):
        with mock.patch.object(server._OPENER, "open", side_effect=http_error(503)) as opener:
            with self.assertRaises(server.ProxyError) as caught:
                server.upstream_stream("https://x", {"model": "m"})
        self.assertEqual(opener.call_count, len(server.UPSTREAM_RETRY_DELAYS) + 1)
        self.assertEqual(caught.exception.status_code, 502)

    def test_never_retries_rate_limit(self):
        with mock.patch.object(server._OPENER, "open", side_effect=http_error(429)) as opener:
            with self.assertRaises(server.ProxyError) as caught:
                server.upstream_stream("https://x", {"model": "m"})
        self.assertEqual(opener.call_count, 1)
        self.assertEqual(caught.exception.status_code, 429)


class SevenQ9kAgentTests(Base):
    def setUp(self):
        super().setUp()
        self.app = integrations.SevenQ9kAgent()
        self.path = self.home / ".7q9k" / "config.toml"
        self.path.parent.mkdir()
        self.existing = 'model_provider = "deepseek"\n\n[model_providers.deepseek]\nname = "DeepSeek"\n'
        self.path.write_text(self.existing, encoding="utf-8")

    def test_connect_update_disconnect_keeps_other_settings(self):
        self.assertTrue(self.app.installed())
        self.assertEqual(self.app.status(target()), "not_connected")
        self.app.connect(target(), self.backups)
        text = self.path.read_text(encoding="utf-8")
        self.assertTrue(text.startswith(self.existing))
        self.assertIn('[model_providers.7q9k-route]\nname = "7q9k route"\n', text)
        self.assertIn('base_url = "http://127.0.0.1:8787/v1"\nwire_api = "chat"\n', text)
        self.assertEqual(self.app.status(target()), "connected")
        self.assertEqual(self.app.status(target(port=9000)), "outdated")
        self.app.connect(target(port=9000), self.backups)
        text = self.path.read_text(encoding="utf-8")
        self.assertEqual(text.count("[model_providers.7q9k-route]"), 1)
        self.assertIn("127.0.0.1:9000", text)
        self.app.disconnect(self.backups)
        self.assertEqual(self.path.read_text(encoding="utf-8").rstrip("\n"), self.existing.rstrip("\n"))

    def test_respects_codex_home(self):
        other = Path(self.tmp.name) / "custom-home"
        with mock.patch.dict(integrations.os.environ, {"CODEX_HOME": str(other)}):
            self.assertEqual(self.app.config_paths(), [other / "config.toml"])


class LegacyEntryTests(Base):
    def test_hermes_replaces_old_entry(self):
        path = self.local / "hermes" / "config.yaml"
        path.parent.mkdir(parents=True)
        path.write_text(
            'custom_providers:\n  - {"name": "oc-free-api", "base_url": "http://127.0.0.1:8787/v1"}\n',
            encoding="utf-8",
        )
        app = integrations.Hermes()
        self.assertEqual(app.status(target()), "outdated")
        app.connect(target(), self.backups)
        text = path.read_text(encoding="utf-8")
        self.assertNotIn("oc-free-api", text)
        self.assertEqual(text.count('"name": "7q9k route"'), 1)

    def test_deepseek_harness_replaces_old_entry_and_default_provider(self):
        path = self.home / ".dsh" / "profiles" / "desktop" / "cordis.patch.yml"
        path.parent.mkdir(parents=True)
        legacy = DSH_PATCH.replace("    providers: {}", '    providers:\n      oc-free: {"api": "old"}')
        path.write_text(legacy.replace("provider: 7q9k-route", "provider: oc-free"), encoding="utf-8")
        app = integrations.DeepSeekHarness()
        self.assertEqual(app.status(target()), "outdated")
        app.connect(target(), self.backups)
        text = path.read_text(encoding="utf-8")
        self.assertNotIn("oc-free", text)
        self.assertIn("    provider: 7q9k-route\n", text)
        self.assertEqual(app.status(target()), "connected")


class DataMigrationTests(unittest.TestCase):
    def test_moves_old_data_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            old, new = Path(tmp) / "oc-free-api", Path(tmp) / "7q9k-route"
            (old / "backups").mkdir(parents=True)
            (old / "api-key.txt").write_text("sk-oc-free-keep", encoding="utf-8")
            (old / "stats.json").write_text("{}", encoding="utf-8")
            (old / "backups" / "x.bak").write_text("b", encoding="utf-8")
            moved = server.migrate_legacy_data_dir(old, new)
            self.assertIn("api-key.txt", moved)
            self.assertIn("backups", moved)
            self.assertEqual((new / "api-key.txt").read_text(encoding="utf-8"), "sk-oc-free-keep")
            self.assertTrue((new / "backups" / "x.bak").exists())

    def test_keeps_newer_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            old, new = Path(tmp) / "oc-free-api", Path(tmp) / "7q9k-route"
            old.mkdir()
            new.mkdir()
            (old / "api-key.txt").write_text("old", encoding="utf-8")
            (new / "api-key.txt").write_text("new", encoding="utf-8")
            server.migrate_legacy_data_dir(old, new)
            self.assertEqual((new / "api-key.txt").read_text(encoding="utf-8"), "new")


if __name__ == "__main__":
    unittest.main()
