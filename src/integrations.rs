//! Points other apps (Claude Code, DeepSeek Harness, Hermes) at the route by
//! editing their config files, with a backup before every write.

use crate::paths::home;
use crate::util::{pydumps, J};
use regex::Regex;
use serde_json::{json, Map, Value};
use std::env;
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

pub const PROVIDER_ID: &str = "7q9k-route";
pub const DISPLAY_NAME: &str = "7q9k route";
/// Names used before the rename; such entries are replaced on the next connect or sync.
pub const LEGACY_PROVIDER_IDS: [&str; 1] = ["oc-free"];
pub const LEGACY_DISPLAY_NAMES: [&str; 1] = ["oc-free-api"];
pub const BACKUPS_KEPT: usize = 10;
static LOCK: Mutex<()> = Mutex::new(());

/// Claude Code lists gateway models only when the id names claude or anthropic, so /v1/models
/// prefixes ids for Anthropic clients and the server strips the prefix again.
pub const ANTHROPIC_MODEL_PREFIX: &str = "anthropic-route-";
pub const MODEL_DESCRIPTION: &str = "Free via 7q9k route";

#[derive(Debug, Clone)]
pub struct Target {
    pub base_url: String,
    pub api_key: String,
    pub models: Vec<String>,
    pub default_model: String,
}

#[derive(Debug)]
pub enum IntegrationError {
    /// The app or its config is in a state we will not touch (HTTP 409).
    App(String),
    /// The config file is not UTF-8 (HTTP 409).
    Utf8(String),
    /// Reading or writing failed (HTTP 500).
    Io(String),
}

impl fmt::Display for IntegrationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IntegrationError::App(m) | IntegrationError::Utf8(m) | IntegrationError::Io(m) => f.write_str(m),
        }
    }
}

impl From<std::io::Error> for IntegrationError {
    fn from(e: std::io::Error) -> Self {
        IntegrationError::Io(e.to_string())
    }
}

type IResult<T> = Result<T, IntegrationError>;

fn app_err(message: impl Into<String>) -> IntegrationError {
    IntegrationError::App(message.into())
}

fn default_newline() -> &'static str {
    if cfg!(windows) {
        "\r\n"
    } else {
        "\n"
    }
}

fn read_text(path: &Path) -> IResult<(String, &'static str, bool)> {
    let raw = fs::read(path)?;
    let bom = raw.starts_with(b"\xef\xbb\xbf");
    let body = if bom { &raw[3..] } else { &raw[..] };
    let text = String::from_utf8(body.to_vec()).map_err(|e| IntegrationError::Utf8(e.to_string()))?;
    let newline = if text.contains("\r\n") { "\r\n" } else { "\n" };
    Ok((text.replace("\r\n", "\n"), newline, bom))
}

fn write_text(path: &Path, text: &str, newline: &str, bom: bool, backup_dir: &Path) -> IResult<()> {
    if path.exists() {
        fs::create_dir_all(backup_dir)?;
        // Several apps use the same file name in different folders (one per profile),
        // and two writes can land in the same second.
        let parent = path.parent().and_then(|p| p.file_name()).map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        let label = format!("{parent}-{name}");
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.subsec_nanos())
            .unwrap_or(0);
        let stamp = format!("{}-{:09}", chrono::Local::now().format("%Y%m%d-%H%M%S"), nanos);
        fs::copy(path, backup_dir.join(format!("{label}.{stamp}.bak")))?;
        let prefix = format!("{label}.");
        let mut old: Vec<PathBuf> = fs::read_dir(backup_dir)?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| {
                let n = p.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
                n.starts_with(&prefix) && n.ends_with(".bak")
            })
            .collect();
        old.sort();
        if old.len() > BACKUPS_KEPT {
            for extra in &old[..old.len() - BACKUPS_KEPT] {
                let _ = fs::remove_file(extra);
            }
        }
    }
    let mut data = Vec::new();
    if bom {
        data.extend_from_slice(b"\xef\xbb\xbf");
    }
    data.extend_from_slice(text.replace('\n', newline).as_bytes());
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    let tmp = path.with_file_name(format!("{name}.7q9k-route-tmp"));
    fs::write(&tmp, data)?;
    fs::rename(&tmp, path)?;
    Ok(())
}

/// One-line JSON, which is also valid YAML flow style.
fn flow(value: &Value) -> String {
    pydumps(value, false, ", ", ": ")
}

/// `json.dumps(data, indent=2, ensure_ascii=False)`.
fn pretty(value: &Value) -> String {
    serde_json::to_string_pretty(value).unwrap_or_default()
}

fn indent_of(line: &str) -> usize {
    line.len() - line.trim_start_matches(' ').len()
}

fn block_end(lines: &[String], start: usize, indent: usize) -> usize {
    let mut end = start + 1;
    while end < lines.len() {
        let line = &lines[end];
        if !line.trim().is_empty() && !line.trim_start().starts_with('#') && indent_of(line) <= indent {
            break;
        }
        end += 1;
    }
    while end > start + 1 && lines[end - 1].trim().is_empty() {
        end -= 1;
    }
    end
}

fn split_lines(text: &str) -> Vec<String> {
    text.split('\n').map(str::to_string).collect()
}

fn load_lines(path: &Path) -> IResult<(Vec<String>, &'static str, bool)> {
    let (text, newline, bom) = if path.exists() { read_text(path)? } else { (String::new(), default_newline(), false) };
    let mut lines = split_lines(&text);
    if lines.last().map(|l| l.is_empty()).unwrap_or(false) {
        lines.pop();
    }
    Ok((lines, newline, bom))
}

fn local_app_data() -> PathBuf {
    PathBuf::from(env::var_os("LOCALAPPDATA").unwrap_or_default())
}

fn env_path(name: &str) -> Option<PathBuf> {
    env::var_os(name).filter(|v| !v.is_empty()).map(PathBuf::from)
}

fn after_colon(line: &str) -> String {
    line.split_once(':').map(|(_, r)| r.trim().to_string()).unwrap_or_default()
}

pub trait Integration: Sync {
    fn id(&self) -> &'static str;
    fn name(&self) -> &'static str;
    fn restart_hint(&self) -> &'static str {
        "Restart the app to load the change."
    }
    fn config_paths(&self) -> Vec<PathBuf>;
    fn app_present(&self) -> bool {
        false
    }
    fn installed(&self) -> bool {
        self.config_paths().iter().any(|p| p.exists()) || self.app_present()
    }
    fn status(&self, target: &Target) -> IResult<&'static str>;
    fn connect(&self, target: &Target, backup_dir: &Path) -> IResult<()>;
    fn disconnect(&self, backup_dir: &Path) -> IResult<()>;

    fn describe(&self, target: &Target) -> Value {
        let paths = self.config_paths();
        let installed = self.installed();
        let mut info = json!({
            "id": self.id(),
            "name": self.name(),
            "installed": installed,
            "config": paths.first().map(|p| Value::from(p.to_string_lossy().into_owned())).unwrap_or(Value::Null),
            "restart_hint": self.restart_hint(),
        });
        let map = info.as_object_mut().expect("object");
        if !installed {
            map.insert("status".into(), "not_installed".into());
            return info;
        }
        match self.status(target) {
            Ok(status) => {
                map.insert("status".into(), status.into());
            }
            Err(IntegrationError::App(m)) => {
                map.insert("status".into(), "error".into());
                map.insert("error".into(), m.into());
            }
            Err(e) => {
                map.insert("status".into(), "error".into());
                map.insert("error".into(), format!("could not read the config: {e}").into());
            }
        }
        info
    }
}

pub struct DeepSeekHarness;

const DSH_ENTRY: &str = "- id: llm-pi-ai";

impl DeepSeekHarness {
    fn provider_value(&self, target: &Target) -> Value {
        json!({
            "displayName": DISPLAY_NAME,
            "api": "openai-completions",
            "baseURL": target.base_url,
            "headers": {"Authorization": format!("Bearer {}", target.api_key)},
            "models": target.models.iter().map(|m| json!({"id": m, "name": m})).collect::<Vec<_>>(),
        })
    }

    /// (start, end, providers, mine); start is None when the entry is missing.
    fn find(&self, lines: &[String]) -> (Option<usize>, usize, Option<usize>, Option<usize>) {
        let Some(start) = lines.iter().position(|l| l.trim_end() == DSH_ENTRY) else {
            return (None, 0, None, None);
        };
        let end = block_end(lines, start, 0);
        let providers = (start..end).find(|&i| lines[i].starts_with("    providers:"));
        let mut mine = None;
        if let Some(p) = providers {
            let ids: Vec<&str> = std::iter::once(PROVIDER_ID).chain(LEGACY_PROVIDER_IDS).collect();
            for i in p + 1..block_end(lines, p, 4) {
                if ids.iter().any(|pid| lines[i].starts_with(&format!("      {pid}:"))) {
                    mine = Some(i);
                    break;
                }
            }
        }
        (Some(start), end, providers, mine)
    }

    fn current(&self, path: &Path) -> IResult<Option<Value>> {
        if !path.exists() {
            return Ok(None);
        }
        let lines = split_lines(&read_text(path)?.0);
        let (_, _, _, mine) = self.find(&lines);
        let Some(mine) = mine else { return Ok(None) };
        if !lines[mine].starts_with(&format!("      {PROVIDER_ID}:")) {
            return Ok(Some(json!({"_legacy": true})));
        }
        let value = after_colon(&lines[mine]);
        if !value.starts_with('{') {
            return Ok(Some(json!({"_block": true})));
        }
        Ok(Some(serde_json::from_str(&value).unwrap_or_else(|_| json!({"_block": true}))))
    }

    fn apply(&self, path: &Path, new_line: Option<String>, backup_dir: &Path) -> IResult<()> {
        let (mut lines, newline, bom) = load_lines(path)?;
        if new_line.is_some() {
            // Settings that point at the provider by its old id (e.g. the default model).
            let legacy = LEGACY_PROVIDER_IDS.iter().map(|p| regex::escape(p)).collect::<Vec<_>>().join("|");
            let re = Regex::new(&format!(r"^(\s+provider:\s*)(?:{legacy})(\s*)$")).expect("regex");
            lines = lines.into_iter().map(|l| re.replace(&l, format!("${{1}}{PROVIDER_ID}${{2}}")).into_owned()).collect();
        }
        let (start, end, providers, mine) = self.find(&lines);
        match new_line {
            None => {
                let Some(mine) = mine else { return Ok(()) };
                let stop = block_end(&lines, mine, 6);
                lines.drain(mine..stop);
                if let Some(p) = providers {
                    if block_end(&lines, p, 4) == p + 1 {
                        lines[p] = "    providers: {}".into();
                    }
                }
            }
            Some(new_line) => match (start, providers) {
                (None, _) => {
                    lines.extend([
                        DSH_ENTRY.to_string(),
                        "  name: \"@deepseek-ai/dsh-llm-pi-ai\"".into(),
                        "  config:".into(),
                        "    providers:".into(),
                        new_line,
                    ]);
                }
                (Some(start), None) => {
                    match (start..end).find(|&i| lines[i].trim_end() == "  config:") {
                        None => {
                            lines.splice(end..end, ["  config:".to_string(), "    providers:".into(), new_line]);
                        }
                        Some(config) => {
                            lines.insert(config + 1, "    providers:".into());
                            lines.insert(config + 2, new_line);
                        }
                    }
                }
                (Some(_), Some(p)) => {
                    let rest = after_colon(&lines[p]);
                    if !rest.is_empty() && rest != "{}" && !rest.starts_with('#') {
                        return Err(app_err(
                            "providers is written in a form 7q9k route cannot edit safely; add it in the app instead",
                        ));
                    }
                    lines[p] = "    providers:".into();
                    match mine {
                        Some(mine) => {
                            let stop = block_end(&lines, mine, 6);
                            lines.splice(mine..stop, [new_line]);
                        }
                        None => lines.insert(p + 1, new_line),
                    }
                }
            },
        }
        write_text(path, &(lines.join("\n") + "\n"), newline, bom, backup_dir)
    }
}

impl Integration for DeepSeekHarness {
    fn id(&self) -> &'static str {
        "deepseek-harness"
    }
    fn name(&self) -> &'static str {
        "DeepSeek Harness"
    }
    fn restart_hint(&self) -> &'static str {
        "DeepSeek Harness picks this up on the next request; restart it if the model list does not update."
    }

    fn config_paths(&self) -> Vec<PathBuf> {
        let profiles = home().join(".dsh").join("profiles");
        let mut found: Vec<PathBuf> = fs::read_dir(&profiles)
            .map(|rd| {
                rd.filter_map(|e| e.ok())
                    .map(|e| e.path().join("cordis.patch.yml"))
                    .filter(|p| p.is_file())
                    .collect()
            })
            .unwrap_or_default();
        found.sort();
        if found.is_empty() {
            vec![profiles.join("desktop").join("cordis.patch.yml")]
        } else {
            found
        }
    }

    fn app_present(&self) -> bool {
        home().join(".dsh").exists() || local_app_data().join("Programs").join("DeepSeek Harness").exists()
    }

    fn status(&self, target: &Target) -> IResult<&'static str> {
        let wanted = self.provider_value(target);
        let mut states = Vec::new();
        for path in self.config_paths() {
            states.push(match self.current(&path)? {
                None => "not_connected",
                Some(current) if current == wanted => "connected",
                Some(_) => "outdated",
            });
        }
        if states.contains(&"connected") && states.iter().all(|s| *s == "connected") {
            return Ok("connected");
        }
        if states.iter().any(|s| *s == "connected" || *s == "outdated") {
            return Ok("outdated");
        }
        Ok("not_connected")
    }

    fn connect(&self, target: &Target, backup_dir: &Path) -> IResult<()> {
        for path in self.config_paths() {
            let line = format!("      {PROVIDER_ID}: {}", flow(&self.provider_value(target)));
            self.apply(&path, Some(line), backup_dir)?;
        }
        Ok(())
    }

    fn disconnect(&self, backup_dir: &Path) -> IResult<()> {
        for path in self.config_paths() {
            if path.exists() {
                self.apply(&path, None, backup_dir)?;
            }
        }
        Ok(())
    }
}

pub struct Hermes;

impl Hermes {
    fn provider_value(&self, target: &Target) -> Value {
        let models: Map<String, Value> = target.models.iter().map(|m| (m.clone(), json!({}))).collect();
        json!({
            "name": DISPLAY_NAME,
            "base_url": target.base_url,
            "api_key": target.api_key,
            "model": target.default_model,
            "models": models,
        })
    }

    /// YAML allows the list items at the same indent as the key ("- name: x").
    fn indent(lines: &[String], top: usize) -> String {
        for line in &lines[top + 1..] {
            if line.trim().is_empty() || line.trim_start().starts_with('#') {
                continue;
            }
            let spaces = indent_of(line);
            return if line[spaces..].starts_with("- ") { " ".repeat(spaces) } else { "  ".into() };
        }
        "  ".into()
    }

    fn list_end(lines: &[String], top: usize, indent: &str) -> usize {
        let mut end = top + 1;
        while end < lines.len() {
            let line = &lines[end];
            if !line.trim().is_empty()
                && !line.starts_with(' ')
                && !line.trim_start().starts_with('#')
                && !(indent.is_empty() && line.starts_with('-'))
            {
                break;
            }
            end += 1;
        }
        while end > top + 1 && lines[end - 1].trim().is_empty() {
            end -= 1;
        }
        end
    }

    /// (top, item, indent); top is None when there is no custom_providers key.
    fn find(&self, lines: &[String]) -> (Option<usize>, Option<usize>, String) {
        let Some(top) = lines.iter().position(|l| l.starts_with("custom_providers:")) else {
            return (None, None, "  ".into());
        };
        let indent = Self::indent(lines, top);
        let flow_start = format!("{indent}- {{");
        for i in top + 1..Self::list_end(lines, top, &indent) {
            let line = &lines[i];
            for name in std::iter::once(DISPLAY_NAME).chain(LEGACY_DISPLAY_NAMES) {
                if line.starts_with(&flow_start) && line.contains(&format!("\"name\": \"{name}\"")) {
                    return (Some(top), Some(i), indent);
                }
                let re = Regex::new(&format!(
                    r#"^{}- name:\s*['"]?{}['"]?\s*$"#,
                    regex::escape(&indent),
                    regex::escape(name)
                ))
                .expect("regex");
                if re.is_match(line) {
                    return (Some(top), Some(i), indent);
                }
            }
        }
        (Some(top), None, indent)
    }

    fn item_end(lines: &[String], item: usize, indent: &str) -> usize {
        let inner = format!("{indent}  ");
        let mut end = item + 1;
        while end < lines.len() && (lines[end].starts_with(&inner) || lines[end].trim().is_empty()) {
            end += 1;
        }
        while end > item + 1 && lines[end - 1].trim().is_empty() {
            end -= 1;
        }
        end
    }

    fn path(&self) -> PathBuf {
        self.config_paths().remove(0)
    }

    fn apply(&self, value: Option<Value>, backup_dir: &Path) -> IResult<()> {
        let path = self.path();
        let (mut lines, newline, bom) = load_lines(&path)?;
        let (top, item, indent) = self.find(&lines);
        let Some(value) = value else {
            let (Some(top), Some(item)) = (top, item) else { return Ok(()) };
            let stop = Self::item_end(&lines, item, &indent);
            lines.drain(item..stop);
            let end = Self::list_end(&lines, top, &indent);
            let remaining = lines[top + 1..end]
                .iter()
                .any(|l| !l.trim().is_empty() && !l.trim_start().starts_with('#'));
            if !remaining {
                lines[top] = "custom_providers: []".into();
            }
            return write_text(&path, &(lines.join("\n") + "\n"), newline, bom, backup_dir);
        };
        let new_line = format!("{indent}- {}", flow(&value));
        match top {
            None => lines.extend(["custom_providers:".to_string(), new_line]),
            Some(top) => {
                let rest = after_colon(&lines[top]);
                if !rest.is_empty() && rest != "[]" && !rest.starts_with('#') {
                    return Err(app_err("custom_providers is written in a form 7q9k route cannot edit safely"));
                }
                lines[top] = "custom_providers:".into();
                match item {
                    Some(item) => {
                        let stop = Self::item_end(&lines, item, &indent);
                        lines.splice(item..stop, [new_line]);
                    }
                    None => lines.insert(top + 1, new_line),
                }
            }
        }
        write_text(&path, &(lines.join("\n") + "\n"), newline, bom, backup_dir)
    }
}

impl Integration for Hermes {
    fn id(&self) -> &'static str {
        "hermes"
    }
    fn name(&self) -> &'static str {
        "Hermes"
    }
    fn restart_hint(&self) -> &'static str {
        "Restart Hermes, then pick the 7q9k route provider in its model settings."
    }

    fn config_paths(&self) -> Vec<PathBuf> {
        if let Some(dir) = env_path("HERMES_HOME") {
            return vec![dir.join("config.yaml")];
        }
        let mut candidates = Vec::new();
        if let Some(local) = env_path("LOCALAPPDATA") {
            candidates.push(local.join("hermes").join("config.yaml"));
        }
        candidates.push(home().join(".hermes").join("config.yaml"));
        let existing = candidates.iter().find(|c| c.exists()).cloned();
        vec![existing.unwrap_or_else(|| candidates[0].clone())]
    }

    fn status(&self, target: &Target) -> IResult<&'static str> {
        let path = self.path();
        if !path.exists() {
            return Ok("not_connected");
        }
        let lines = split_lines(&read_text(&path)?.0);
        let (_, item, indent) = self.find(&lines);
        let Some(item) = item else { return Ok("not_connected") };
        let block = lines[item..Self::item_end(&lines, item, &indent)].join("\n");
        let wanted = self.provider_value(target);
        let first = lines[item].trim_start();
        if first.starts_with("- {") {
            return Ok(match serde_json::from_str::<Value>(&first[2..]) {
                Ok(current) if current == wanted => "connected",
                _ => "outdated",
            });
        }
        let ok = block.contains(DISPLAY_NAME)
            && block.contains(&target.base_url)
            && block.contains(&target.api_key)
            && target.models.iter().all(|m| {
                Regex::new(&format!(r"(?m)^\s+{}:", regex::escape(m))).map(|re| re.is_match(&block)).unwrap_or(false)
            });
        Ok(if ok { "connected" } else { "outdated" })
    }

    fn connect(&self, target: &Target, backup_dir: &Path) -> IResult<()> {
        self.apply(Some(self.provider_value(target)), backup_dir)
    }

    fn disconnect(&self, backup_dir: &Path) -> IResult<()> {
        if self.path().exists() {
            self.apply(None, backup_dir)?;
        }
        Ok(())
    }
}

pub struct ClaudeCode;

/// Marks the env entries as ours, so another local proxy set up by hand is never touched.
const CC_MARKER: &str = "ROUTE7Q9K_PROVIDER";
const CC_MODEL_KEYS: [&str; 2] = ["ANTHROPIC_MODEL", "CLAUDE_CODE_SUBAGENT_MODEL"];
/// /model shows one entry per family slot. Each slot gets a different model, under the same
/// id the model list uses, so Claude Code drops it from the list instead of showing it twice.
const CC_FAMILIES: [&str; 4] = ["OPUS", "SONNET", "HAIKU", "FABLE"];
const CC_FIXED: [(&str, &str); 3] = [
    // /model lists every model from GET /v1/models.
    ("CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY", "1"),
    // Otherwise Claude Code adds "[1m]" to the default model.
    ("CLAUDE_CODE_DISABLE_1M_CONTEXT", "1"),
    // claude.ai connectors need the claude.ai login, which the route replaces; without this
    // Claude Code warns about it at every start.
    ("ENABLE_CLAUDEAI_MCP_SERVERS", "0"),
];

impl ClaudeCode {
    fn home(&self) -> PathBuf {
        env_path("CLAUDE_CONFIG_DIR").unwrap_or_else(|| home().join(".claude"))
    }

    fn family_keys() -> Vec<String> {
        CC_FAMILIES
            .iter()
            .flat_map(|f| ["", "_NAME", "_DESCRIPTION"].map(|s| format!("ANTHROPIC_DEFAULT_{f}_MODEL{s}")))
            .collect()
    }

    pub fn env(&self, target: &Target) -> Map<String, Value> {
        // Claude Code appends /v1/messages itself.
        let base = target.base_url.strip_suffix("/v1").unwrap_or(&target.base_url);
        let mut values = Map::new();
        values.insert("ANTHROPIC_BASE_URL".into(), base.into());
        values.insert("ANTHROPIC_AUTH_TOKEN".into(), target.api_key.clone().into());
        for key in CC_MODEL_KEYS {
            values.insert(key.into(), target.default_model.clone().into());
        }
        let mut models: Vec<String> = Vec::new();
        for m in std::iter::once(&target.default_model).chain(&target.models) {
            if !models.contains(m) {
                models.push(m.clone());
            }
        }
        for (i, family) in CC_FAMILIES.iter().enumerate() {
            let model = &models[i % models.len()];
            values.insert(format!("ANTHROPIC_DEFAULT_{family}_MODEL"), format!("{ANTHROPIC_MODEL_PREFIX}{model}").into());
            values.insert(format!("ANTHROPIC_DEFAULT_{family}_MODEL_NAME"), model.clone().into());
            values.insert(format!("ANTHROPIC_DEFAULT_{family}_MODEL_DESCRIPTION"), MODEL_DESCRIPTION.into());
        }
        for (k, v) in CC_FIXED {
            values.insert(k.into(), v.into());
        }
        values.insert(CC_MARKER.into(), PROVIDER_ID.into());
        values
    }

    fn load(&self, path: &Path) -> IResult<(Map<String, Value>, &'static str, bool)> {
        if !path.exists() {
            return Ok((Map::new(), default_newline(), false));
        }
        let (text, newline, bom) = read_text(path)?;
        if text.trim().is_empty() {
            return Ok((Map::new(), newline, bom));
        }
        let data: Value = serde_json::from_str(&text)
            .map_err(|e| app_err(format!("settings.json is not valid JSON ({e}); fix it first")))?;
        let Value::Object(data) = data else {
            return Err(app_err("settings.json does not hold a JSON object"));
        };
        if let Some(env) = data.get("env") {
            if !env.is_null() && !env.is_object() {
                return Err(app_err("\"env\" in settings.json is not an object"));
            }
        }
        Ok((data, newline, bom))
    }
}

impl Integration for ClaudeCode {
    fn id(&self) -> &'static str {
        "claude-code"
    }
    fn name(&self) -> &'static str {
        "Claude Code"
    }
    fn restart_hint(&self) -> &'static str {
        "Start a new Claude Code session; /model lists every free model."
    }

    fn config_paths(&self) -> Vec<PathBuf> {
        vec![self.home().join("settings.json")]
    }

    fn app_present(&self) -> bool {
        let npm = env_path("APPDATA").unwrap_or_else(|| home().join("AppData").join("Roaming")).join("npm");
        let local_bin = home().join(".local").join("bin");
        self.home().exists()
            || home().join(".claude.json").exists()
            || local_bin.join("claude.exe").exists()
            || local_bin.join("claude").exists()
            || npm.join("claude.cmd").exists()
    }

    fn status(&self, target: &Target) -> IResult<&'static str> {
        let (data, _, _) = self.load(&self.config_paths()[0])?;
        let env = data.get("env").cloned().unwrap_or(Value::Null);
        if env.g(CC_MARKER) != PROVIDER_ID {
            return Ok("not_connected");
        }
        let wanted = self.env(target);
        Ok(if wanted.iter().all(|(k, v)| env.g(k) == v) { "connected" } else { "outdated" })
    }

    fn connect(&self, target: &Target, backup_dir: &Path) -> IResult<()> {
        let path = self.config_paths().remove(0);
        let (mut data, newline, bom) = self.load(&path)?;
        let env = data.entry("env").or_insert_with(|| json!({}));
        if env.is_null() {
            *env = json!({});
        }
        let map = env.as_object_mut().expect("object");
        for (k, v) in self.env(target) {
            map.insert(k, v);
        }
        write_text(&path, &(pretty(&Value::Object(data)) + "\n"), newline, bom, backup_dir)
    }

    fn disconnect(&self, backup_dir: &Path) -> IResult<()> {
        let path = self.config_paths().remove(0);
        if !path.exists() {
            return Ok(());
        }
        let (mut data, newline, bom) = self.load(&path)?;
        let Some(Value::Object(env)) = data.get_mut("env") else { return Ok(()) };
        if env.get(CC_MARKER).map(|v| v != PROVIDER_ID).unwrap_or(true) {
            return Ok(());
        }
        let mut keys: Vec<String> = vec!["ANTHROPIC_BASE_URL".into(), "ANTHROPIC_AUTH_TOKEN".into()];
        keys.extend(CC_MODEL_KEYS.map(String::from));
        keys.extend(Self::family_keys());
        keys.extend(CC_FIXED.map(|(k, _)| k.to_string()));
        keys.push(CC_MARKER.into());
        for key in keys {
            env.shift_remove(&key);
        }
        if env.is_empty() {
            data.shift_remove("env");
        }
        write_text(&path, &(pretty(&Value::Object(data)) + "\n"), newline, bom, backup_dir)
    }
}

pub static REGISTRY: [&dyn Integration; 3] = [&ClaudeCode, &DeepSeekHarness, &Hermes];

pub fn get(id: &str) -> IResult<&'static dyn Integration> {
    REGISTRY.iter().copied().find(|i| i.id() == id).ok_or_else(|| app_err(format!("unknown app: {id}")))
}

fn lock() -> std::sync::MutexGuard<'static, ()> {
    LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

pub fn describe_all(target: &Target) -> Vec<Value> {
    let _guard = lock();
    REGISTRY.iter().map(|i| i.describe(target)).collect()
}

pub fn connect(id: &str, target: &Target, backup_dir: &Path) -> IResult<Value> {
    let item = get(id)?;
    let _guard = lock();
    if !item.installed() {
        return Err(app_err(format!("{} is not installed on this computer", item.name())));
    }
    item.connect(target, backup_dir)?;
    Ok(item.describe(target))
}

pub fn disconnect(id: &str, target: &Target, backup_dir: &Path) -> IResult<Value> {
    let item = get(id)?;
    let _guard = lock();
    item.disconnect(backup_dir)?;
    Ok(item.describe(target))
}

pub fn sync(enabled: &[String], target: &Target, backup_dir: &Path, log: &dyn Fn(&str)) -> Vec<String> {
    let mut updated = Vec::new();
    let _guard = lock();
    for item in REGISTRY {
        if !enabled.iter().any(|e| e == item.id()) || !item.installed() {
            continue;
        }
        let result = item.status(target).and_then(|status| {
            if status != "connected" {
                item.connect(target, backup_dir)?;
                return Ok(true);
            }
            Ok(false)
        });
        match result {
            Ok(true) => updated.push(item.id().to_string()),
            Ok(false) => {}
            Err(e) => log(&format!("could not update {}: {e}", item.name())),
        }
    }
    updated
}

#[cfg(test)]
mod tests {
    use super::*;

    fn target() -> Target {
        Target {
            base_url: "http://127.0.0.1:8787/v1".into(),
            api_key: "sk-test".into(),
            models: vec!["big-pickle".into(), "other".into()],
            default_model: "big-pickle".into(),
        }
    }

    fn tmp() -> PathBuf {
        let dir = env::temp_dir().join(format!("7q9k-int-{}", crate::util::token_hex(4)));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn deepseek_roundtrip() {
        let dir = tmp();
        let path = dir.join("p").join("cordis.patch.yml");
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(&path, "- id: other\n  x: 1\n- id: llm-pi-ai\n  name: foo\n  config:\n    providers: {}\n").unwrap();
        let dsh = DeepSeekHarness;
        let line = format!("      {PROVIDER_ID}: {}", flow(&dsh.provider_value(&target())));
        dsh.apply(&path, Some(line), &dir.join("b")).unwrap();
        let text = fs::read_to_string(&path).unwrap();
        assert!(text.contains("    providers:\n      7q9k-route: {\"displayName\": \"7q9k route\""), "{text}");
        assert_eq!(dsh.current(&path).unwrap().unwrap(), dsh.provider_value(&target()));
        dsh.apply(&path, None, &dir.join("b")).unwrap();
        let text = fs::read_to_string(&path).unwrap();
        assert!(text.ends_with("  config:\n    providers: {}\n"), "{text}");
        assert_eq!(fs::read_dir(dir.join("b")).unwrap().count(), 2);
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn claude_env_slots() {
        let env = ClaudeCode.env(&target());
        assert_eq!(env["ANTHROPIC_BASE_URL"], "http://127.0.0.1:8787");
        assert_eq!(env["ANTHROPIC_DEFAULT_OPUS_MODEL"], "anthropic-route-big-pickle");
        assert_eq!(env["ANTHROPIC_DEFAULT_SONNET_MODEL_NAME"], "other");
        assert_eq!(env["ANTHROPIC_DEFAULT_HAIKU_MODEL_NAME"], "big-pickle");
        assert_eq!(ClaudeCode::family_keys().len(), 12);
    }
}
