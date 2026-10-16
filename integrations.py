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

PROVIDER_ID = "oc-free"
DISPLAY_NAME = "oc-free-api"
BACKUPS_KEPT = 10
_LOCK = threading.Lock()


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
        stamp = time.strftime("%Y%m%d-%H%M%S")
        shutil.copy2(path, backup_dir / f"{path.name}.{stamp}.bak")
        old = sorted(backup_dir.glob(f"{path.name}.*.bak"))
        for extra in old[:-BACKUPS_KEPT]:
            try:
                extra.unlink()
            except OSError:
                pass
    data = text.replace("\n", newline).encode("utf-8")
    if bom:
        data = b"\xef\xbb\xbf" + data
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".oc-free-tmp")
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
                if re.match(rf"^      {re.escape(PROVIDER_ID)}:", lines[i]):
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
                raise IntegrationError("providers is written in a form oc-free-api cannot edit safely; add it in the app instead")
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
    restart_hint = "Restart Hermes, then pick the oc-free-api provider in its model settings."

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

    def _find(self, lines: list[str]) -> tuple[int, int | None]:
        top = next((i for i, l in enumerate(lines) if re.match(r"^custom_providers:", l)), -1)
        if top < 0:
            return -1, None
        for i in range(top + 1, _block_end(lines, top, 0)):
            line = lines[i]
            if re.match(r"^  - \{", line) and f'"name": "{DISPLAY_NAME}"' in line:
                return top, i
            if re.match(rf"^  - name:\s*['\"]?{re.escape(DISPLAY_NAME)}['\"]?\s*$", line):
                return top, i
        return top, None

    def _item_end(self, lines: list[str], item: int) -> int:
        end = item + 1
        while end < len(lines) and (lines[end].startswith("    ") or not lines[end].strip()):
            end += 1
        while end > item + 1 and not lines[end - 1].strip():
            end -= 1
        return end

    def status(self, target: Target) -> str:
        path = self.config_paths()[0]
        if not path.exists():
            return "not_connected"
        lines = _read_text(path)[0].split("\n")
        _, item = self._find(lines)
        if item is None:
            return "not_connected"
        block = "\n".join(lines[item:self._item_end(lines, item)])
        wanted = self.provider_value(target)
        if lines[item].lstrip().startswith("- {"):
            try:
                current = json.loads(lines[item].lstrip()[2:])
            except ValueError:
                return "outdated"
            return "connected" if current == wanted else "outdated"
        ok = (target.base_url in block and target.api_key in block
              and all(re.search(rf"^\s+{re.escape(m)}:", block, re.M) for m in target.models))
        return "connected" if ok else "outdated"

    def connect(self, target: Target, backup_dir: Path) -> None:
        self._apply(f"  - {_flow(self.provider_value(target))}", backup_dir)

    def disconnect(self, backup_dir: Path) -> None:
        if self.config_paths()[0].exists():
            self._apply(None, backup_dir)

    def _apply(self, new_line: str | None, backup_dir: Path) -> None:
        path = self.config_paths()[0]
        if path.exists():
            text, newline, bom = _read_text(path)
        else:
            text, newline, bom = "", os.linesep if os.name == "nt" else "\n", False
        lines = text.split("\n")
        if lines and lines[-1] == "":
            lines.pop()
        top, item = self._find(lines)
        if new_line is None:
            if item is None:
                return
            del lines[item:self._item_end(lines, item)]
            remaining = [l for l in lines[top + 1:_block_end(lines, top, 0)] if l.strip() and not l.lstrip().startswith("#")]
            if not remaining:
                lines[top] = "custom_providers: []"
        elif top < 0:
            lines += ["custom_providers:", new_line]
        else:
            rest = lines[top].split(":", 1)[1].strip()
            if rest and rest != "[]" and not rest.startswith("#"):
                raise IntegrationError("custom_providers is written in a form oc-free-api cannot edit safely")
            lines[top] = "custom_providers:"
            if item is not None:
                lines[item:self._item_end(lines, item)] = [new_line]
            else:
                lines.insert(top + 1, new_line)
        _write_text(path, "\n".join(lines) + "\n", newline, bom, backup_dir)


class OpenCode(Integration):
    id = "opencode"
    name = "OpenCode"
    restart_hint = "Restart OpenCode, then choose a model from oc-free-api."

    def config_paths(self) -> list[Path]:
        base = Path(os.environ["XDG_CONFIG_HOME"]) if os.environ.get("XDG_CONFIG_HOME") else _home() / ".config"
        return [base / "opencode" / "opencode.json"]

    def app_present(self) -> bool:
        appdata = Path(os.environ.get("APPDATA") or "")
        return (_home() / ".config" / "opencode").exists() or (appdata / "ai.opencode.desktop").exists()

    def provider_value(self, target: Target) -> dict[str, Any]:
        return {
            "npm": "@ai-sdk/openai-compatible",
            "name": DISPLAY_NAME,
            "options": {"baseURL": target.base_url, "apiKey": target.api_key},
            "models": {m: {"name": m} for m in target.models},
        }

    def _load(self) -> dict[str, Any]:
        path = self.config_paths()[0]
        if not path.exists():
            return {}
        try:
            data = json.loads(_read_text(path)[0] or "{}")
        except ValueError as exc:
            raise IntegrationError("opencode.json contains comments or is not plain JSON; add it in OpenCode instead") from exc
        if not isinstance(data, dict):
            raise IntegrationError("opencode.json has an unexpected format")
        return data

    def status(self, target: Target) -> str:
        current = (self._load().get("provider") or {}).get(PROVIDER_ID)
        if current is None:
            return "not_connected"
        return "connected" if current == self.provider_value(target) else "outdated"

    def connect(self, target: Target, backup_dir: Path) -> None:
        data = self._load()
        providers = data.setdefault("provider", {})
        if not isinstance(providers, dict):
            raise IntegrationError("opencode.json has an unexpected provider section")
        providers[PROVIDER_ID] = self.provider_value(target)
        self._save(data, backup_dir)

    def disconnect(self, backup_dir: Path) -> None:
        data = self._load()
        if PROVIDER_ID in (data.get("provider") or {}):
            del data["provider"][PROVIDER_ID]
            if not data["provider"]:
                del data["provider"]
            self._save(data, backup_dir)

    def _save(self, data: dict[str, Any], backup_dir: Path) -> None:
        path = self.config_paths()[0]
        newline, bom = ("\n", False)
        if path.exists():
            _, newline, bom = _read_text(path)
        _write_text(path, json.dumps(data, indent=2, ensure_ascii=False) + "\n", newline, bom, backup_dir)


REGISTRY: list[Integration] = [DeepSeekHarness(), Hermes(), OpenCode()]


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
