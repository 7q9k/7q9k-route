//! The OpenCode Zen endpoint: model list, SSE streaming and retries.

use crate::convert::chat::{build_chat_request, build_responses_request, merge_deltas, responses_events_to_deltas, usage};
use crate::log::{emit_err, paint};
use crate::paths;
use crate::util::{clip, compact, fmt_g, now, py_round, request_id, session_id, J, PResult, ProxyError};
use serde_json::{json, Value};
use std::env;
use std::fs;
use std::io::{self, BufRead, BufReader, Read};
use std::sync::Mutex;
use std::thread;
use std::time::{Duration, Instant};

pub const DIRECT_URL: &str = "https://opencode.ai/inference/v1/chat/completions";
pub const RESPONSES_URL: &str = "https://opencode.ai/inference/v1/responses";
pub const ZEN_MODELS_URL: &str = "https://opencode.ai/zen/v1/models";
pub const MODELS_DEV_URL: &str = "https://models.dev/api.json";
pub const OPENCODE_VERSION: &str = "2.2.1";
pub const UA_CHAT: &str = "opencode/2.2.1 ai-sdk/provider-utils/4.0.23 runtime/bun/1.3.14";
pub const UA_RESPONSES: &str = "opencode/2.2.1 ai-sdk/provider-utils/4.0.40 runtime/bun/1.3.14";
pub const CLIENT_HEADER: &str = "cli";
pub const PROJECT_HEADER: &str = "global";

pub const DEFAULT_MODEL: &str = "big-pickle";
pub const FALLBACK_MODELS: [&str; 7] = [
    DEFAULT_MODEL,
    "ling-3.0-flash-fin-free",
    "mimo-v2.5-free",
    "muse-spark-1.2-contributor-free",
    "muse-spark-1.3-contributor-free",
    "nemotron-3-ultra-free",
    "nemotron-3.5-lightning-free",
];
pub const RESPONSES_MODELS: [&str; 2] = ["muse-spark-1.2-contributor-free", "muse-spark-1.3-contributor-free"];

pub const UPSTREAM_TIMEOUT: f64 = 900.0;
pub const TEST_TIMEOUT: f64 = 120.0;
pub const TEST_PROMPT: &str = "Reply with exactly the token: OK-DASH";

/// OpenCode answers HTTP 5xx ("Endpoint is unavailable") while a free model's
/// backend briefly drops out; it usually works again within a few seconds.
/// Retried before anything is streamed to the client, so no output repeats.
/// Rate limits (429) are never retried.
pub const UPSTREAM_RETRY_DELAYS: [f64; 4] = [0.5, 1.0, 2.0, 4.0];
const RETRYABLE_UPSTREAM_STATUS: [u16; 4] = [500, 502, 503, 504];

/// A stream of upstream events (chat completion chunks once converted).
pub type Events = Box<dyn Iterator<Item = PResult<Value>> + Send>;

/// The endpoints can be pointed elsewhere for tests.
pub fn url(name: &str, default: &str) -> String {
    env::var(name).ok().filter(|v| !v.is_empty()).unwrap_or_else(|| default.to_string())
}

pub fn chat_url() -> String {
    url("ROUTE7Q9K_CHAT_URL", DIRECT_URL)
}

pub fn responses_url() -> String {
    url("ROUTE7Q9K_RESPONSES_URL", RESPONSES_URL)
}

fn agent(timeout: f64, redirects: u32) -> ureq::Agent {
    let t = Duration::from_secs_f64(timeout);
    ureq::AgentBuilder::new()
        .redirects(redirects)
        .timeout_connect(t)
        .timeout_read(t)
        .timeout_write(t)
        .build()
}

/// GET a JSON document (redirects followed). Any failure is an `Err`.
pub fn json_get(url: &str, timeout: f64) -> Result<Value, String> {
    let response = agent(timeout, 5)
        .get(url)
        .set("Accept", "application/json")
        .set("User-Agent", UA_CHAT)
        .call()
        .map_err(|e| e.to_string())?;
    serde_json::from_reader(response.into_reader()).map_err(|e| e.to_string())
}

// ---------------------------------------------------------------- models

struct ModelState {
    snapshot: Option<Vec<String>>,
    source: &'static str,
    updated: f64,
}

static MODELS: Mutex<ModelState> = Mutex::new(ModelState { snapshot: None, source: "fallback", updated: 0.0 });

fn models_lock() -> std::sync::MutexGuard<'static, ModelState> {
    MODELS.lock().unwrap_or_else(|e| e.into_inner())
}

pub fn free_live_models(zen: &Value, catalog: &Value) -> Vec<String> {
    let zen_models: Vec<String> = zen
        .g("data")
        .items()
        .iter()
        .filter(|item| item.is_object() && item.g("id").truthy())
        .map(|item| item.g("id").pystr())
        .collect();
    let Some(metadata) = catalog.g("opencode").g("models").as_object() else {
        return Vec::new();
    };
    let mut free = Vec::new();
    for (model, details) in metadata {
        if !details.is_object() || details.g("status") == "deprecated" {
            continue;
        }
        let Some(costs) = details.g("cost").as_object() else { continue };
        if zen_models.contains(model)
            && details.g("tool_call") == &Value::Bool(true)
            && !costs.is_empty()
            && costs.values().all(|c| c.as_f64() == Some(0.0))
        {
            free.push(model.clone());
        }
    }
    free.sort();
    free
}

fn sorted_unique(mut models: Vec<String>) -> Vec<String> {
    models.sort();
    models.dedup();
    models
}

pub fn load_model_cache() -> Vec<String> {
    let Ok(text) = fs::read_to_string(paths::model_cache()) else { return Vec::new() };
    let Ok(Value::Array(items)) = serde_json::from_str::<Value>(&text) else { return Vec::new() };
    let mut models = Vec::new();
    for item in items {
        match item {
            Value::String(s) if !s.is_empty() => models.push(s),
            _ => return Vec::new(),
        }
    }
    sorted_unique(models)
}

fn save_model_cache(models: &[String]) {
    let _ = fs::write(paths::model_cache(), format!("{}\n", crate::util::dumps(&json!(models))));
}

fn preload_locked(state: &mut ModelState) {
    let cached = load_model_cache();
    state.source = if cached.is_empty() { "fallback" } else { "cache" };
    state.snapshot = Some(if cached.is_empty() { FALLBACK_MODELS.iter().map(|s| s.to_string()).collect() } else { cached });
    state.updated = now();
}

pub fn preload_models() {
    let mut state = models_lock();
    if state.snapshot.is_none() {
        preload_locked(&mut state);
    }
}

pub fn available_models(timeout: f64, refresh: bool) -> Vec<String> {
    {
        let state = models_lock();
        if let (Some(snapshot), false) = (&state.snapshot, refresh) {
            return snapshot.clone();
        }
    }
    let fetched = json_get(&url("ROUTE7Q9K_ZEN_MODELS_URL", ZEN_MODELS_URL), timeout)
        .and_then(|zen| Ok((zen, json_get(&url("ROUTE7Q9K_MODELS_DEV_URL", MODELS_DEV_URL), timeout)?)));
    let models = fetched.map(|(zen, catalog)| free_live_models(&zen, &catalog)).unwrap_or_default();
    let mut state = models_lock();
    if !models.is_empty() {
        save_model_cache(&models);
        state.snapshot = Some(models);
        state.source = "live";
        state.updated = now();
    } else if state.snapshot.is_none() {
        preload_locked(&mut state);
    }
    state.snapshot.clone().unwrap_or_default()
}

pub fn model_source() -> Value {
    let state = models_lock();
    json!({"source": state.source, "updated": state.updated})
}

pub fn uses_responses_api(model: &str) -> bool {
    RESPONSES_MODELS.contains(&model)
}

// ---------------------------------------------------------------- streaming

fn io_error_name(err: &io::Error) -> &'static str {
    match err.kind() {
        io::ErrorKind::ConnectionReset => "ConnectionResetError",
        io::ErrorKind::ConnectionAborted => "ConnectionAbortedError",
        io::ErrorKind::BrokenPipe => "BrokenPipeError",
        io::ErrorKind::UnexpectedEof => "IncompleteRead",
        io::ErrorKind::InvalidData => "IncompleteRead",
        _ => "OSError",
    }
}

fn is_timeout(err: &io::Error) -> bool {
    matches!(err.kind(), io::ErrorKind::TimedOut | io::ErrorKind::WouldBlock)
}

/// Reads `data:` lines from an SSE body until `[DONE]` or the end.
pub struct SseStream {
    reader: Option<Box<dyn BufRead + Send>>,
}

impl SseStream {
    pub fn new(reader: Box<dyn Read + Send>) -> Self {
        SseStream { reader: Some(Box::new(BufReader::with_capacity(64 * 1024, reader))) }
    }
}

impl Iterator for SseStream {
    type Item = PResult<Value>;

    fn next(&mut self) -> Option<PResult<Value>> {
        loop {
            let reader = self.reader.as_mut()?;
            let mut raw = Vec::new();
            match reader.read_until(b'\n', &mut raw) {
                Ok(0) => {
                    self.reader = None;
                    return None;
                }
                Ok(_) => {}
                Err(err) => {
                    self.reader = None;
                    return Some(Err(if is_timeout(&err) {
                        ProxyError::new("upstream timed out while streaming.", 504)
                    } else {
                        ProxyError::upstream(format!("upstream connection lost: {}", io_error_name(&err)))
                    }));
                }
            }
            let text = String::from_utf8_lossy(&raw);
            let line = text.trim();
            let Some(payload) = line.strip_prefix("data:") else { continue };
            let payload = payload.trim();
            if payload == "[DONE]" {
                self.reader = None;
                return None;
            }
            if payload.is_empty() {
                continue;
            }
            return Some(match serde_json::from_str(payload) {
                Ok(value) => Ok(value),
                Err(_) => {
                    self.reader = None;
                    Err(ProxyError::upstream(format!("upstream sent invalid JSON: {}", clip(payload, 200))))
                }
            });
        }
    }
}

fn transport_error(err: &ureq::Transport) -> ProxyError {
    let timed_out = std::error::Error::source(err)
        .and_then(|s| s.downcast_ref::<io::Error>())
        .map(is_timeout)
        .unwrap_or(false);
    if timed_out {
        return ProxyError::new("upstream timed out.", 504);
    }
    let reason = err.message().map(str::to_string).unwrap_or_else(|| err.kind().to_string());
    let mut error = ProxyError::upstream(format!("upstream connection failed: {reason}"));
    error.retryable = true;
    error
}

fn http_error(code: u16, response: ureq::Response) -> ProxyError {
    let mut body = Vec::new();
    let _ = response.into_reader().take(64 * 1024).read_to_end(&mut body);
    let detail = clip(&String::from_utf8_lossy(&body), 500);
    let status = if (400..500).contains(&code) { code } else { 502 };
    let mut error = ProxyError::new(format!("upstream rejected request (HTTP {code}): {detail}"), status);
    error.retryable = RETRYABLE_UPSTREAM_STATUS.contains(&code);
    error
}

fn upstream_stream_once(url: &str, body: &Value, timeout: f64) -> PResult<Events> {
    let session = session_id();
    let ua = if url.contains("responses") { UA_RESPONSES } else { UA_CHAT };
    let result = agent(timeout, 0)
        .post(url)
        .set("Authorization", "Bearer public")
        .set("Content-Type", "application/json")
        .set("Accept", "*/*")
        .set("User-Agent", ua)
        .set("x-opencode-session", &session)
        .set("x-opencode-session-id", &session)
        .set("x-opencode-client", CLIENT_HEADER)
        .set("x-opencode-project", PROJECT_HEADER)
        .set("x-opencode-request", &request_id())
        .send_string(&compact(body));
    match result {
        Ok(response) if (200..300).contains(&response.status()) => Ok(Box::new(SseStream::new(response.into_reader()))),
        Ok(response) => Err(http_error(response.status(), response)),
        Err(ureq::Error::Status(code, response)) => Err(http_error(code, response)),
        Err(ureq::Error::Transport(err)) => Err(transport_error(&err)),
    }
}

pub fn upstream_stream(url: &str, body: &Value, timeout: f64) -> PResult<Events> {
    let mut delays = UPSTREAM_RETRY_DELAYS.iter();
    loop {
        match upstream_stream_once(url, body, timeout) {
            Ok(events) => return Ok(events),
            Err(err) => {
                let delay = if err.retryable { delays.next() } else { None };
                let Some(&delay) = delay else { return Err(err) };
                let model = match body.g("model") {
                    Value::Null if !body.has("model") => "?".to_string(),
                    v => v.pystr(),
                };
                emit_err(&format!(
                    "{}{model}: {} - retrying in {}s",
                    paint("[7q9k route] ", "dim"),
                    clip(&err.message, 160),
                    fmt_g(delay)
                ));
                thread::sleep(Duration::from_secs_f64(delay));
            }
        }
    }
}

/// Opens the upstream stream for `req`, as chat completion chunks.
pub fn open_upstream_events(req: &Value, model: &str, timeout: f64) -> PResult<Events> {
    let mut req = req.clone();
    crate::util::set(&mut req, "model", model);
    if uses_responses_api(model) {
        let body = build_responses_request(&req)?;
        let events = upstream_stream(&responses_url(), &body, timeout)?;
        return Ok(Box::new(responses_events_to_deltas(events)));
    }
    let body = build_chat_request(&req)?;
    upstream_stream(&chat_url(), &body, timeout)
}

pub fn run_model_test(model: &str, prompt: Option<&str>) -> PResult<Value> {
    let prompt = prompt.filter(|p| !p.is_empty()).unwrap_or(TEST_PROMPT);
    let req = json!({"model": model, "messages": [{"role": "user", "content": prompt}], "stream": false});
    let started = Instant::now();
    let completion = merge_deltas(open_upstream_events(&req, model, TEST_TIMEOUT)?, model)?;
    let latency_ms = py_round(started.elapsed().as_secs_f64() * 1000.0);
    let text = completion.g("choices").items().first().map(|c| c.g("message").g("content").or_str()).unwrap_or_default();
    let usage = usage(completion.g("usage"));
    Ok(json!({
        "ok": true,
        "text": text,
        "latency_ms": latency_ms,
        "tokens": usage.g("total_tokens").clone(),
        "usage": usage,
    }))
}
