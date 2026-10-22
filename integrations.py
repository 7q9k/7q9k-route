from __future__ import annotations

import json
import os
import re
import shutil
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

PROVIDER_ID = "7q9k-route"
DISPLAY_NAME = "7q9k route"
# Names used before the rename; such entries are replaced on the next connect or sync.
LEGACY_PROVIDER_IDS = ("oc-free",)
LEGACY_DISPLAY_NAMES = ("oc-free-api",)
BACKUPS_KEPT = 10
_LOCK = threading.Lock()


# Claude Code lists gateway models only when the id names claude or anthropic, so /v1/models
# prefixes ids for Anthropic clients and the server strips the prefix again.
ANTHROPIC_MODEL_PREFIX = "anthropic-route-"
MODEL_DESCRIPTION = "Free via 7q9k route"


@dataclass
class Target:
    base_url: str
    api_key: str
    models: list[str]
    default_model: str


class IntegrationError(Exception):
    pass


def _home() -> Path:
    return Path(os.environ.get("USERPROFILE") or os.environ.get("HOME") or Path.home())


def _read_text(path: Path) -> tuple[str, str, bool]:
    raw = path.read_bytes()
    bom = raw.startswith(b"\xef\xbb\xbf")
    text = raw[3:].decode("utf-8") if bom else raw.decode("utf-8")
    newline = "\r\n" if "\r\n" in text else "\n"
    return text.replace("\r\n", "\n"), newline, bom


def _write_text(path: Path, text: str, newline: str, bom: bool, backup_dir: Path) -> None:
    if path.exists():
        backup_dir.mkdir(parents=True, exist_ok=True)
        # Several apps use the same file name in different folders (one per profile),
        # and two writes can land in the same second.
        label = f"{path.parent.name}-{path.name}"
        stamp = time.strftime("%Y%m%d-%H%M%S") + f"-{time.time_ns() % 1_000_000_000:09d}"
        shutil.copy2(path, backup_dir / f"{label}.{stamp}.bak")
        old = sorted(backup_dir.glob(f"{label}.*.bak"))
        for extra in old[:-BACKUPS_KEPT]:
            try:
                extra.unlink()
            except OSError:
                pass
    data = text.replace("\n", newline).encode("utf-8")
    if bom:
        data = b"\xef\xbb\xbf" + data
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".7q9k-route-tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def _flow(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(", ", ": "))


def _block_end(lines: list[str], start: int, indent: int) -> int:
    end = start + 1
    while end < len(lines):
        line = lines[end]
        if line.strip() and not line.lstrip().startswith("#") and len(line) - len(line.lstrip(" ")) <= indent:
            break
        end += 1
    while end > start + 1 and not lines[end - 1].strip():
        end -= 1
    return end


class Integration:
    id = ""
    name = ""
    restart_hint = "Restart the app to load the change."

    def config_paths(self) -> list[Path]:
        return []

    def installed(self) -> bool:
        return any(p.exists() for p in self.config_paths()) or self.app_present()

    def app_present(self) -> bool:
        return False

    def status(self, target: Target) -> str:
        raise NotImplementedError

    def connect(self, target: Target, backup_dir: Path) -> None:
        raise NotImplementedError

    def disconnect(self, backup_dir: Path) -> None:
        raise NotImplementedError

    def describe(self, target: Target) -> dict[str, Any]:
        paths = self.config_paths()
        info: dict[str, Any] = {
            "id": self.id,
            "name": self.name,
            "installed": self.installed(),
            "config": str(paths[0]) if paths else None,
            "restart_hint": self.restart_hint,
        }
        if not info["installed"]:
            info["status"] = "not_installed"
            return info
        try:
            info["status"] = self.status(target)
        except IntegrationError as exc:
            info["status"] = "error"
            info["error"] = str(exc)
        except (OSError, ValueError) as exc:
            info["status"] = "error"
            info["error"] = f"could not read the config: {exc}"
        return info


class DeepSeekHarness(Integration):
    id = "deepseek-harness"
    name = "DeepSeek Harness"
    restart_hint = "DeepSeek Harness picks this up on the next request; restart it if the model list does not update."
    entry = "- id: llm-pi-ai"

    def config_paths(self) -> list[Path]:
        profiles = _home() / ".dsh" / "profiles"
        found = sorted(profiles.glob("*/cordis.patch.yml")) if profiles.exists() else []
        return found or [profiles / "desktop" / "cordis.patch.yml"]

    def app_present(self) -> bool:
        local = Path(os.environ.get("LOCALAPPDATA") or "")
        return (_home() / ".dsh").exists() or (local / "Programs" / "DeepSeek Harness").exists()

    def provider_value(self, target: Target) -> dict[str, Any]:
        return {
            "displayName": DISPLAY_NAME,
            "api": "openai-completions",
            "baseURL": target.base_url,
            "headers": {"Authorization": f"Bearer {target.api_key}"},
            "models": [{"id": m, "name": m} for m in target.models],
        }

    def _find(self, lines: list[str]) -> tuple[int, int, int | None, int | None]:
        start = next((i for i, l in enumerate(lines) if l.rstrip() == self.entry), -1)
        if start < 0:
            return -1, -1, None, None
        end = _block_end(lines, start, 0)
        providers = next((i for i in range(start, end) if re.match(r"^    providers:", lines[i])), None)
        mine = None
        if providers is not None:
            for i in range(providers + 1, _block_end(lines, providers, 4)):
                if any(re.match(rf"^      {re.escape(pid)}:", lines[i])
                       for pid in (PROVIDER_ID, *LEGACY_PROVIDER_IDS)):
                    mine = i
                    break
        return start, end, providers, mine

    def _current(self, path: Path) -> dict[str, Any] | None:
        if not path.exists():
            return None
        lines = _read_text(path)[0].split("\n")
        _, _, _, mine = self._find(lines)
        if mine is None:
            return None
        if not lines[mine].startswith(f"      {PROVIDER_ID}:"):
            return {"_legacy": True}
        value = lines[mine].split(":", 1)[1].strip()
        if not value.startswith("{"):
            return {"_block": True}
        try:
            return json.loads(value)
        except ValueError:
            return {"_block": True}

    def status(self, target: Target) -> str:
        states = []
        for path in self.config_paths():
            current = self._current(path)
            if current is None:
                states.append("not_connected")
            else:
                states.append("connected" if current == self.provider_value(target) else "outdated")
        if "connected" in states and all(s == "connected" for s in states):
            return "connected"
        if any(s in ("connected", "outdated") for s in states):
            return "outdated"
        return "not_connected"

    def connect(self, target: Target, backup_dir: Path) -> None:
        for path in self.config_paths():
            self._apply(path, f"      {PROVIDER_ID}: {_flow(self.provider_value(target))}", backup_dir)

    def disconnect(self, backup_dir: Path) -> None:
        for path in self.config_paths():
            if path.exists():
                self._apply(path, None, backup_dir)

    def _apply(self, path: Path, new_line: str | None, backup_dir: Path) -> None:
        if path.exists():
            text, newline, bom = _read_text(path)
        else:
            text, newline, bom = "", os.linesep if os.name == "nt" else "\n", False
        lines = text.split("\n")
        if lines and lines[-1] == "":
            lines.pop()
        if new_line is not None:
            # Settings that point at the provider by its old id (e.g. the default model).
            legacy = "|".join(re.escape(pid) for pid in LEGACY_PROVIDER_IDS)
            lines = [re.sub(rf"^(\s+provider:\s*)(?:{legacy})(\s*)$", rf"\g<1>{PROVIDER_ID}\2", line)
                     for line in lines]
        start, end, providers, mine = self._find(lines)
        if new_line is None:
            if mine is None:
                return
            stop = _block_end(lines, mine, 6)
            del lines[mine:stop]
            if providers is not None and _block_end(lines, providers, 4) == providers + 1:
                lines[providers] = "    providers: {}"
        elif start < 0:
            lines += [self.entry, '  name: "@deepseek-ai/dsh-llm-pi-ai"', "  config:", "    providers:", new_line]
        elif providers is None:
            config = next((i for i in range(start, end) if lines[i].rstrip() == "  config:"), None)
            if config is None:
                lines[end:end] = ["  config:", "    providers:", new_line]
            else:
                lines.insert(config + 1, "    providers:")
                lines.insert(config + 2, new_line)
        else:
            rest = lines[providers].split(":", 1)[1].strip()
            if rest and rest != "{}" and not rest.startswith("#"):
                raise IntegrationError("providers is written in a form 7q9k route cannot edit safely; add it in the app instead")
            lines[providers] = "    providers:"
            if mine is not None:
                stop = _block_end(lines, mine, 6)
                lines[mine:stop] = [new_line]
            else:
                lines.insert(providers + 1, new_line)
        _write_text(path, "\n".join(lines) + "\n", newline, bom, backup_dir)


class Hermes(Integration):
    id = "hermes"
    name = "Hermes"
    restart_hint = "Restart Hermes, then pick the 7q9k route provider in its model settings."

    def config_paths(self) -> list[Path]:
        if os.environ.get("HERMES_HOME"):
            return [Path(os.environ["HERMES_HOME"]) / "config.yaml"]
        local = os.environ.get("LOCALAPPDATA")
        candidates = []
        if local:
            candidates.append(Path(local) / "hermes" / "config.yaml")
        candidates.append(_home() / ".hermes" / "config.yaml")
        existing = [c for c in candidates if c.exists()]
        return existing[:1] or candidates[:1]

    def provider_value(self, target: Target) -> dict[str, Any]:
        return {
            "name": DISPLAY_NAME,
            "base_url": target.base_url,
            "api_key": target.api_key,
            "model": target.default_model,
            "models": {m: {} for m in target.models},
        }

    @staticmethod
    def _indent(lines: list[str], top: int) -> str:
        # YAML allows the list items at the same indent as the key ("- name: x").
        for line in lines[top + 1:]:
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            match = re.match(r"^( *)- ", line)
            return match.group(1) if match else "  "
        return "  "

    @staticmethod
    def _list_end(lines: list[str], top: int, indent: str) -> int:
        end = top + 1
        while end < len(lines):
            line = lines[end]
            if line.strip() and not line.startswith(" ") and not line.lstrip().startswith("#") \
                    and not (indent == "" and line.startswith("-")):
                break
            end += 1
        while end > top + 1 and not lines[end - 1].strip():
            end -= 1
        return end

    def _find(self, lines: list[str]) -> tuple[int, int | None, str]:
        top = next((i for i, l in enumerate(lines) if re.match(r"^custom_providers:", l)), -1)
        if top < 0:
            return -1, None, "  "
        indent = self._indent(lines, top)
        for i in range(top + 1, self._list_end(lines, top, indent)):
            line = lines[i]
            for name in (DISPLAY_NAME, *LEGACY_DISPLAY_NAMES):
                if re.match(rf"^{indent}- \{{", line) and f'"name": "{name}"' in line:
                    return top, i, indent
                if re.match(rf"^{indent}- name:\s*['\"]?{re.escape(name)}['\"]?\s*$", line):
                    return top, i, indent
        return top, None, indent

    def _item_end(self, lines: list[str], item: int, indent: str) -> int:
        end = item + 1
        while end < len(lines) and (lines[end].startswith(indent + "  ") or not lines[end].strip()):
            end += 1
        while end > item + 1 and not lines[end - 1].strip():
            end -= 1
        return end

    def status(self, target: Target) -> str:
        path = self.config_paths()[0]
        if not path.exists():
            return "not_connected"
        lines = _read_text(path)[0].split("\n")
        _, item, indent = self._find(lines)
        if item is None:
            return "not_connected"
        block = "\n".join(lines[item:self._item_end(lines, item, indent)])
        wanted = self.provider_value(target)
        if lines[item].lstrip().startswith("- {"):
            try:
                current = json.loads(lines[item].lstrip()[2:])
            except ValueError:
                return "outdated"
            return "connected" if current == wanted else "outdated"
        ok = (DISPLAY_NAME in block and target.base_url in block and target.api_key in block
              and all(re.search(rf"^\s+{re.escape(m)}:", block, re.M) for m in target.models))
        return "connected" if ok else "outdated"

    def connect(self, target: Target, backup_dir: Path) -> None:
        self._apply(self.provider_value(target), backup_dir)

    def disconnect(self, backup_dir: Path) -> None:
        if self.config_paths()[0].exists():
            self._apply(None, backup_dir)

    def _apply(self, value: dict[str, Any] | None, backup_dir: Path) -> None:
        path = self.config_paths()[0]
        if path.exists():
            text, newline, bom = _read_text(path)
        else:
            text, newline, bom = "", os.linesep if os.name == "nt" else "\n", False
        lines = text.split("\n")
        if lines and lines[-1] == "":
            lines.pop()
        top, item, indent = self._find(lines)
        if value is None:
            if item is None:
                return
            del lines[item:self._item_end(lines, item, indent)]
            remaining = [l for l in lines[top + 1:self._list_end(lines, top, indent)]
                         if l.strip() and not l.lstrip().startswith("#")]
            if not remaining:
                lines[top] = "custom_providers: []"
            return _write_text(path, "\n".join(lines) + "\n", newline, bom, backup_dir)
        new_line = f"{indent}- {_flow(value)}"
        if top < 0:
            lines += ["custom_providers:", new_line]
        else:
            rest = lines[top].split(":", 1)[1].strip()
            if rest and rest != "[]" and not rest.startswith("#"):
                raise IntegrationError("custom_providers is written in a form 7q9k route cannot edit safely")
            lines[top] = "custom_providers:"
            if item is not None:
                lines[item:self._item_end(lines, item, indent)] = [new_line]
            else:
                lines.insert(top + 1, new_line)
        _write_text(path, "\n".join(lines) + "\n", newline, bom, backup_dir)


class ClaudeCode(Integration):
    id = "claude-code"
    name = "Claude Code"
    restart_hint = "Start a new Claude Code session; /model lists every free model."
    # Marks the env entries as ours, so another local proxy set up by hand is never touched.
    marker = "ROUTE7Q9K_PROVIDER"
    model_keys = ("ANTHROPIC_MODEL", "CLAUDE_CODE_SUBAGENT_MODEL")
    # /model shows one entry per family slot. Each slot gets a different model, under the same
    # id the model list uses, so Claude Code drops it from the list instead of showing it twice.
    families = ("OPUS", "SONNET", "HAIKU", "FABLE")
    family_keys = tuple(
        f"ANTHROPIC_DEFAULT_{family}_MODEL{suffix}" for family in families for suffix in ("", "_NAME", "_DESCRIPTION")
    )
    fixed = {
        # /model lists every model from GET /v1/models.
        "CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY": "1",
        # Otherwise Claude Code adds "[1m]" to the default model.
        "CLAUDE_CODE_DISABLE_1M_CONTEXT": "1",
        # claude.ai connectors need the claude.ai login, which the route replaces; without this
        # Claude Code warns about it at every start.
        "ENABLE_CLAUDEAI_MCP_SERVERS": "0",
    }

    def home(self) -> Path:
        if os.environ.get("CLAUDE_CONFIG_DIR"):
            return Path(os.environ["CLAUDE_CONFIG_DIR"])
        return _home() / ".claude"

    def config_paths(self) -> list[Path]:
        return [self.home() / "settings.json"]

    def app_present(self) -> bool:
        npm = Path(os.environ.get("APPDATA") or _home() / "AppData" / "Roaming") / "npm"
        return (
            self.home().exists()
            or (_home() / ".claude.json").exists()
            or (_home() / ".local" / "bin" / "claude.exe").exists()
            or (_home() / ".local" / "bin" / "claude").exists()
            or (npm / "claude.cmd").exists()
        )

    def env(self, target: Target) -> dict[str, str]:
        # Claude Code appends /v1/messages itself.
        base = target.base_url[:-3] if target.base_url.endswith("/v1") else target.base_url
        values = {"ANTHROPIC_BASE_URL": base, "ANTHROPIC_AUTH_TOKEN": target.api_key}
        values.update({key: target.default_model for key in self.model_keys})
        models = list(dict.fromkeys([target.default_model, *target.models]))
        for i, family in enumerate(self.families):
            model = models[i % len(models)]
            values[f"ANTHROPIC_DEFAULT_{family}_MODEL"] = ANTHROPIC_MODEL_PREFIX + model
            values[f"ANTHROPIC_DEFAULT_{family}_MODEL_NAME"] = model
            values[f"ANTHROPIC_DEFAULT_{family}_MODEL_DESCRIPTION"] = MODEL_DESCRIPTION
        values.update(self.fixed)
        values[self.marker] = PROVIDER_ID
        return values

    def _load(self, path: Path) -> tuple[dict[str, Any], str, bool]:
        if not path.exists():
            return {}, os.linesep if os.name == "nt" else "\n", False
        text, newline, bom = _read_text(path)
        if not text.strip():
            return {}, newline, bom
        try:
            data = json.loads(text)
        except ValueError as exc:
            raise IntegrationError(f"settings.json is not valid JSON ({exc}); fix it first") from exc
        if not isinstance(data, dict):
            raise IntegrationError("settings.json does not hold a JSON object")
        env = data.get("env")
        if env is not None and not isinstance(env, dict):
            raise IntegrationError('"env" in settings.json is not an object')
        return data, newline, bom

    def status(self, target: Target) -> str:
        data, _, _ = self._load(self.config_paths()[0])
        env = data.get("env") or {}
        if env.get(self.marker) != PROVIDER_ID:
            return "not_connected"
        wanted = self.env(target)
        return "connected" if all(env.get(k) == v for k, v in wanted.items()) else "outdated"

    def connect(self, target: Target, backup_dir: Path) -> None:
        path = self.config_paths()[0]
        data, newline, bom = self._load(path)
        env = data.setdefault("env", {})
        env.update(self.env(target))
        _write_text(path, json.dumps(data, indent=2, ensure_ascii=False) + "\n", newline, bom, backup_dir)

    def disconnect(self, backup_dir: Path) -> None:
        path = self.config_paths()[0]
        if not path.exists():
            return
        data, newline, bom = self._load(path)
        env = data.get("env") or {}
        if env.get(self.marker) != PROVIDER_ID:
            return
        for key in ("ANTHROPIC_BASE_URL", "ANTHROPIC_AUTH_TOKEN", *self.model_keys, *self.family_keys, *self.fixed, self.marker):
            env.pop(key, None)
        if not env:
            data.pop("env", None)
        _write_text(path, json.dumps(data, indent=2, ensure_ascii=False) + "\n", newline, bom, backup_dir)


REGISTRY: list[Integration] = [ClaudeCode(), DeepSeekHarness(), Hermes()]


def get(integration_id: str) -> Integration:
    for item in REGISTRY:
        if item.id == integration_id:
            return item
    raise IntegrationError(f"unknown app: {integration_id}")


def describe_all(target: Target) -> list[dict[str, Any]]:
    with _LOCK:
        return [item.describe(target) for item in REGISTRY]


def connect(integration_id: str, target: Target, backup_dir: Path) -> dict[str, Any]:
    item = get(integration_id)
    with _LOCK:
        if not item.installed():
            raise IntegrationError(f"{item.name} is not installed on this computer")
        item.connect(target, backup_dir)
        return item.describe(target)


def disconnect(integration_id: str, target: Target, backup_dir: Path) -> dict[str, Any]:
    item = get(integration_id)
    with _LOCK:
        item.disconnect(backup_dir)
        return item.describe(target)


def sync(enabled: list[str], target: Target, backup_dir: Path, log: Callable[[str], None] | None = None) -> list[str]:
    updated = []
    with _LOCK:
        for item in REGISTRY:
            if item.id not in enabled or not item.installed():
                continue
            try:
                if item.status(target) != "connected":
                    item.connect(target, backup_dir)
                    updated.append(item.id)
            except (IntegrationError, OSError, ValueError) as exc:
                if log:
                    log(f"could not update {item.name}: {exc}")
    return updated
