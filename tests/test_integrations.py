from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import integrations  # noqa: E402

DSH_PATCH = """# Your patch layer for this dsh profile
- id: llm-pi-ai
  name: "@deepseek-ai/dsh-llm-pi-ai"
  config:
    providers: {}
- id: ui-chat
  name: "@deepseek-ai/dsh-client-ui-chat"
  config:
    transcriptView: standard
- id: agent-default-model
  name: "@deepseek-ai/dsh-agent-default-model"
  config:
    provider: oc-free
    model: deepseek-v4.1-flash-free
"""

HERMES_CONFIG = """model:
  default: "beam-free"
  provider: "custom"

# =============================================================================
# Custom providers
# =============================================================================
custom_providers:
  - name: 192.168.0.29:8787
    base_url: http://192.168.0.29:8787/v1
    api_key: sk-manual
    models:
      beam-free: {}
# trailing comment
other_section:
  value: 1
"""


def target(key="sk-test-1", port=8787, models=("big-pickle", "glm-5.3-flash-free")):
    return integrations.Target(
        base_url=f"http://127.0.0.1:{port}/v1", api_key=key, models=list(models), default_model=models[0]
    )


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.home = Path(self.tmp.name) / "home"
        self.local = Path(self.tmp.name) / "local"
        self.home.mkdir()
        self.local.mkdir()
        self.backups = Path(self.tmp.name) / "backups"
        patcher = mock.patch.dict(
            integrations.os.environ,
            {"USERPROFILE": str(self.home), "HOME": str(self.home), "LOCALAPPDATA": str(self.local),
             "APPDATA": str(self.local)},
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        for var in ("HERMES_HOME", "XDG_CONFIG_HOME"):
            integrations.os.environ.pop(var, None)


class DeepSeekTests(Base):
    def setUp(self):
        super().setUp()
        self.path = self.home / ".dsh" / "profiles" / "desktop" / "cordis.patch.yml"
        self.path.parent.mkdir(parents=True)
        self.path.write_bytes(DSH_PATCH.replace("\n", "\r\n").encode("utf-8"))
        self.app = integrations.DeepSeekHarness()

    def test_connect_update_disconnect(self):
        self.assertEqual(self.app.status(target()), "not_connected")
        self.app.connect(target(), self.backups)
        raw = self.path.read_bytes()
        self.assertIn(b"\r\n", raw)
        text = raw.decode("utf-8")
        self.assertIn("    providers:\r\n      oc-free: {", text)
        self.assertIn("model: deepseek-v4.1-flash-free", text)
        self.assertEqual(self.app.status(target()), "connected")
        self.assertEqual(self.app.status(target(key="sk-other")), "outdated")
        self.app.connect(target(key="sk-other"), self.backups)
        self.assertEqual(self.path.read_text(encoding="utf-8").count("oc-free: {"), 1)
        self.assertEqual(self.app.status(target(key="sk-other")), "connected")
        self.app.disconnect(self.backups)
        self.assertEqual(self.path.read_bytes().decode("utf-8").replace("\r\n", "\n"), DSH_PATCH)
        self.assertTrue(list(self.backups.glob("cordis.patch.yml.*.bak")))

    def test_keeps_other_providers_and_replaces_block_form(self):
        text = DSH_PATCH.replace("    providers: {}", "    providers:\n      openai:\n        apiKeyEnv: OPENAI_API_KEY\n      oc-free:\n        api: openai-completions\n        baseURL: http://old/v1")
        self.path.write_text(text, encoding="utf-8")
        self.assertEqual(self.app.status(target()), "outdated")
        self.app.connect(target(), self.backups)
        result = self.path.read_text(encoding="utf-8")
        self.assertIn("      openai:\n        apiKeyEnv: OPENAI_API_KEY\n", result)
        self.assertNotIn("http://old/v1", result)
        self.assertEqual(self.app.status(target()), "connected")

    def test_adds_entry_when_missing(self):
        self.path.write_text("- id: ui-chat\n  config:\n    a: 1\n", encoding="utf-8")
        self.app.connect(target(), self.backups)
        result = self.path.read_text(encoding="utf-8")
        self.assertIn("- id: llm-pi-ai\n  name: \"@deepseek-ai/dsh-llm-pi-ai\"\n  config:\n    providers:\n      oc-free: {", result)
        self.assertTrue(result.startswith("- id: ui-chat\n  config:\n    a: 1\n"))

    def test_refuses_unknown_flow_form(self):
        self.path.write_text(DSH_PATCH.replace("providers: {}", "providers: {openai: {}}"), encoding="utf-8")
        with self.assertRaises(integrations.IntegrationError):
            self.app.connect(target(), self.backups)


class HermesTests(Base):
    def setUp(self):
        super().setUp()
        self.path = self.local / "hermes" / "config.yaml"
        self.path.parent.mkdir(parents=True)
        self.path.write_text(HERMES_CONFIG, encoding="utf-8")
        self.app = integrations.Hermes()

    def test_connect_keeps_manual_entry(self):
        self.app.connect(target(), self.backups)
        text = self.path.read_text(encoding="utf-8")
        self.assertIn('custom_providers:\n  - {"name": "oc-free-api"', text)
        self.assertIn("  - name: 192.168.0.29:8787\n    base_url: http://192.168.0.29:8787/v1\n", text)
        self.assertIn("other_section:\n  value: 1\n", text)
        self.assertEqual(self.app.status(target()), "connected")
        self.app.connect(target(key="k2-updated"), self.backups)
        self.assertEqual(self.path.read_text(encoding="utf-8").count('"name": "oc-free-api"'), 1)
        self.app.disconnect(self.backups)
        self.assertEqual(self.path.read_text(encoding="utf-8"), HERMES_CONFIG)

    def test_recognises_entry_rewritten_by_hermes(self):
        t = target()
        block = (
            "custom_providers:\n  - name: oc-free-api\n    base_url: http://127.0.0.1:8787/v1\n"
            f"    api_key: {t.api_key}\n    model: big-pickle\n    models:\n      big-pickle: {{}}\n"
            "      glm-5.3-flash-free: {}\n"
        )
        self.path.write_text(HERMES_CONFIG.replace("custom_providers:\n", block), encoding="utf-8")
        self.assertEqual(self.app.status(t), "connected")
        self.assertEqual(self.app.status(target(port=9000)), "outdated")
        self.app.connect(target(port=9000), self.backups)
        text = self.path.read_text(encoding="utf-8")
        self.assertNotIn("  - name: oc-free-api\n", text)
        self.assertIn("  - name: 192.168.0.29:8787\n", text)

    def test_creates_section_and_empty_list(self):
        self.path.write_text("model:\n  default: x\n", encoding="utf-8")
        self.app.connect(target(), self.backups)
        self.assertIn('\ncustom_providers:\n  - {"name": "oc-free-api"', self.path.read_text(encoding="utf-8"))
        self.path.write_text("custom_providers: []\nlater: 1\n", encoding="utf-8")
        self.app.connect(target(), self.backups)
        self.assertEqual(self.app.status(target()), "connected")
        self.app.disconnect(self.backups)
        self.assertEqual(self.path.read_text(encoding="utf-8"), "custom_providers: []\nlater: 1\n")


class OpenCodeTests(Base):
    def setUp(self):
        super().setUp()
        self.path = self.home / ".config" / "opencode" / "opencode.json"
        self.path.parent.mkdir(parents=True)
        self.path.write_text(json.dumps({"mcp": {"rea": {"type": "local"}}}, indent=2), encoding="utf-8")
        self.app = integrations.OpenCode()

    def test_connect_and_disconnect(self):
        self.app.connect(target(), self.backups)
        data = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertEqual(data["mcp"], {"rea": {"type": "local"}})
        self.assertEqual(data["provider"]["oc-free"]["options"]["baseURL"], "http://127.0.0.1:8787/v1")
        self.assertEqual(self.app.status(target()), "connected")
        self.app.disconnect(self.backups)
        self.assertEqual(json.loads(self.path.read_text(encoding="utf-8")), {"mcp": {"rea": {"type": "local"}}})

    def test_jsonc_is_not_touched(self):
        self.path.write_text('{\n  // comment\n  "mcp": {}\n}\n', encoding="utf-8")
        info = self.app.describe(target())
        self.assertEqual(info["status"], "error")
        with self.assertRaises(integrations.IntegrationError):
            self.app.connect(target(), self.backups)
        self.assertIn("// comment", self.path.read_text(encoding="utf-8"))


class RegistryTests(Base):
    def test_detection_and_sync(self):
        apps = {a["id"]: a for a in integrations.describe_all(target())}
        self.assertEqual(apps["hermes"]["status"], "not_installed")
        cfg = self.local / "hermes" / "config.yaml"
        cfg.parent.mkdir(parents=True)
        cfg.write_text(HERMES_CONFIG, encoding="utf-8")
        self.assertEqual(integrations.sync([], target(), self.backups), [])
        self.assertEqual(integrations.sync(["hermes"], target(), self.backups), ["hermes"])
        self.assertEqual(integrations.sync(["hermes"], target(), self.backups), [])
        self.assertEqual(integrations.sync(["hermes"], target(key="new-key-123"), self.backups), ["hermes"])

    def test_backups_are_limited(self):
        cfg = self.local / "hermes" / "config.yaml"
        cfg.parent.mkdir(parents=True)
        cfg.write_text(HERMES_CONFIG, encoding="utf-8")
        for i in range(14):
            with mock.patch.object(integrations.time, "strftime", return_value=f"2026{i:04d}"):
                integrations.Hermes().connect(target(key=f"key-number-{i}"), self.backups)
        self.assertEqual(len(list(self.backups.glob("config.yaml.*.bak"))), integrations.BACKUPS_KEPT)

    def test_unknown_app(self):
        with self.assertRaises(integrations.IntegrationError):
            integrations.get("nope")


if __name__ == "__main__":
    unittest.main()
