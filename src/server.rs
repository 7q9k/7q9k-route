//! The HTTP server: routing, access checks, the dashboard API and the
//! OpenAI, Responses, Anthropic and Ollama endpoints.

use crate::assets;
use crate::config::{self, access_host, lan_ip, port_available, KEY_PATTERN};
use crate::convert::anthropic::{self, AnthropicStream, CLAUDE_MODEL_ALIASES};
use crate::convert::chat::{merge_deltas, usage as chat_usage, validate_messages};
use crate::convert::ollama::{
    json_obj, normalize_model_name, ollama_chat_response, ollama_details, ollama_generate_response,
    ollama_model_entries, ollama_now, ollama_options, ollama_to_openai_messages, ollama_usage,
};
use crate::convert::responses::{build_responses_object, responses_to_chat_request, ResponsesBuilder};
use crate::integrations::{self, IntegrationError, Target, ANTHROPIC_MODEL_PREFIX, MODEL_DESCRIPTION};
use crate::log::{emit, emit_err, logging_to_file, paint, status_color, status_word};
use crate::metrics::Metrics;
use crate::paths;
use crate::pricing::{usage_cost, usage_report, Pricing};
use crate::upstream::{
    available_models, model_source, open_upstream_events, run_model_test, uses_responses_api, Events, DEFAULT_MODEL,
    UPSTREAM_TIMEOUT,
};
use crate::util::{compare_digest, dumps, now, now_secs, py_round, pydumps, set, token_hex, PResult, ProxyError, J};
use regex::Regex;
use serde_json::{json, Map, Value};
use std::collections::BTreeMap;
use std::io::{self, BufRead, BufReader, Read, Write};
use std::net::{IpAddr, SocketAddr, TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, LazyLock, Mutex, MutexGuard, OnceLock, RwLock};
use std::thread;
use std::time::{Duration, Instant};

pub const APP_VERSION: &str = env!("CARGO_PKG_VERSION");
pub const MAX_BODY_BYTES: i64 = 32 * 1024 * 1024;
const MODELS_TIMEOUT: f64 = 15.0;
const SOCKET_TIMEOUT: Duration = Duration::from_secs(300);
const MAX_LINE: usize = 65536;
const MAX_HEADERS: usize = 100;

// ---- Process-wide state -----------------------------------------------------

#[derive(Default)]
pub struct Bound {
    pub host: String,
    pub port: u16,
    pub open: bool,
    /// --host / --port or the environment; they win over the dashboard settings.
    pub cli_host: Option<String>,
    pub cli_port: Option<u16>,
}

static BOUND: LazyLock<Mutex<Bound>> = LazyLock::new(|| Mutex::new(Bound::default()));
static API_KEY: RwLock<String> = RwLock::new(String::new());
static METRICS: OnceLock<Metrics> = OnceLock::new();
static PRICING: OnceLock<Arc<Pricing>> = OnceLock::new();
pub static RESTART_REQUESTED: AtomicBool = AtomicBool::new(false);
static STOP: AtomicBool = AtomicBool::new(false);

pub fn bound() -> MutexGuard<'static, Bound> {
    BOUND.lock().unwrap_or_else(|e| e.into_inner())
}

pub fn api_key() -> String {
    API_KEY.read().unwrap_or_else(|e| e.into_inner()).clone()
}

pub fn set_api_key(key: &str) {
    *API_KEY.write().unwrap_or_else(|e| e.into_inner()) = key.to_string();
}

/// Sets the metrics store; the first call wins.
pub fn init_metrics(metrics: Metrics) -> &'static Metrics {
    let _ = METRICS.set(metrics);
    self::metrics()
}

pub fn metrics() -> &'static Metrics {
    METRICS.get_or_init(|| Metrics::new(None))
}

pub fn init_pricing(pricing: Pricing) -> &'static Arc<Pricing> {
    let _ = PRICING.set(Arc::new(pricing));
    self::pricing()
}

pub fn pricing() -> &'static Arc<Pricing> {
    PRICING.get_or_init(|| Arc::new(Pricing::new(None)))
}

pub fn route_default_model(models: &[String]) -> String {
    if models.iter().any(|m| m == DEFAULT_MODEL) {
        DEFAULT_MODEL.to_string()
    } else {
        models.first().cloned().unwrap_or_else(|| DEFAULT_MODEL.to_string())
    }
}

pub fn public_urls(port: u16) -> Value {
    let lan = if bound().open { Some(lan_ip()) } else { None };
    let network = lan.filter(|ip| ip != "127.0.0.1").map(|ip| format!("http://{ip}:{port}"));
    json!({"local": format!("http://127.0.0.1:{port}"), "network": network})
}

pub fn settings_payload() -> Value {
    let config = config::config().clone();
    let b = bound();
    let wanted_host = b.cli_host.clone().unwrap_or_else(|| access_host(&config.access).unwrap_or("0.0.0.0").into());
    let wanted_port = b.cli_port.unwrap_or(config.port);
    json!({
        "ok": true,
        "access": config.access,
        "port": config.port,
        "open_browser": config.open_browser,
        "autostart": config::autostart_enabled(),
        "data_dir": paths::data_dir().to_string_lossy(),
        "can_open_folder": cfg!(windows),
        "running": {"host": b.host, "port": b.port},
        "restart_required": wanted_host != b.host || wanted_port != b.port,
        "overridden": b.cli_host.is_some() || b.cli_port.is_some(),
    })
}

pub fn dashboard_state(local: bool) -> Value {
    let models = available_models(MODELS_TIMEOUT, false);
    let (port, open) = {
        let b = bound();
        (b.port, b.open)
    };
    let key = if local {
        let key = api_key();
        if key.is_empty() {
            config::api_key_cache().map(Value::from).unwrap_or(Value::Null)
        } else {
            Value::from(key)
        }
    } else {
        Value::Null
    };
    json!({
        "ok": true,
        "version": APP_VERSION,
        "local": local,
        "urls": public_urls(port),
        "api_key": key,
        "models": models
            .iter()
            .map(|m| json!({"id": m, "api": if uses_responses_api(m) { "responses" } else { "chat" }}))
            .collect::<Vec<_>>(),
        "model_list": model_source(),
        "lan_ip": if open { Some(lan_ip()) } else { None },
        "port": port,
        "default_model": route_default_model(&models),
    })
}

pub fn integration_target() -> Target {
    let models = available_models(MODELS_TIMEOUT, false);
    Target {
        base_url: format!("http://127.0.0.1:{}/v1", bound().port),
        api_key: api_key(),
        default_model: route_default_model(&models),
        models,
    }
}

pub fn sync_integrations() -> Vec<String> {
    let enabled = config::config().integrations.clone();
    if enabled.is_empty() {
        return Vec::new();
    }
    let log = |m: &str| emit(&format!("[7q9k route] {m}"));
    let updated = integrations::sync(&enabled, &integration_target(), &paths::backups_dir(), &log);
    for item in &updated {
        if let Ok(app) = integrations::get(item) {
            emit(&format!("[7q9k route] updated {}", app.name()));
        }
    }
    updated
}

pub fn sync_integrations_async() {
    thread::spawn(sync_integrations);
}

// ---- Routing ------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq)]
enum Access {
    Open,
    Client,
    Dashboard,
    Admin,
}

type Handler = fn(&mut Conn) -> HResult;

const ROUTES: &[(&str, &str, Access, Handler)] = &[
    ("GET", "/", Access::Open, Conn::get_dashboard_page),
    ("GET", "/dashboard", Access::Open, Conn::get_dashboard_page),
    ("GET", "/health", Access::Open, Conn::get_health),
    ("GET", "/favicon.ico", Access::Open, Conn::get_favicon),
    ("GET", "/v1/models", Access::Open, Conn::get_models),
    ("GET", "/models", Access::Open, Conn::get_models),
    ("GET", "/api/v1/models", Access::Open, Conn::get_models),
    ("GET", "/api/tags", Access::Open, Conn::get_ollama_tags),
    ("GET", "/api/ps", Access::Open, Conn::get_ollama_ps),
    ("GET", "/api/version", Access::Open, Conn::get_version),
    ("GET", "/version", Access::Open, Conn::get_version),
    ("GET", "/v1/props", Access::Open, Conn::get_props),
    ("GET", "/props", Access::Open, Conn::get_props),
    ("POST", "/api/show", Access::Open, Conn::post_ollama_show),
    ("POST", "/v1/chat/completions", Access::Client, Conn::post_chat_completions),
    ("POST", "/chat/completions", Access::Client, Conn::post_chat_completions),
    ("POST", "/v1/responses", Access::Client, Conn::post_responses),
    ("POST", "/responses", Access::Client, Conn::post_responses),
    ("POST", "/api/chat", Access::Client, Conn::post_ollama_chat),
    ("POST", "/api/generate", Access::Client, Conn::post_ollama_generate),
    ("POST", "/v1/messages", Access::Client, Conn::post_anthropic_messages),
    ("POST", "/v1/messages/count_tokens", Access::Client, Conn::post_anthropic_count_tokens),
    ("GET", "/api/dashboard", Access::Dashboard, Conn::get_dashboard_state),
    ("GET", "/api/dashboard/stats", Access::Dashboard, Conn::get_dashboard_stats),
    ("GET", "/api/dashboard/log", Access::Dashboard, Conn::get_dashboard_log),
    ("POST", "/api/dashboard/test", Access::Dashboard, Conn::post_dashboard_test),
    ("POST", "/api/dashboard/refresh", Access::Dashboard, Conn::post_refresh_models),
    ("POST", "/api/dashboard/save", Access::Dashboard, Conn::post_save_stats),
    ("POST", "/api/dashboard/prices", Access::Dashboard, Conn::post_refresh_prices),
    ("GET", "/api/dashboard/settings", Access::Admin, Conn::get_settings),
    ("POST", "/api/dashboard/settings", Access::Admin, Conn::post_settings),
    ("POST", "/api/dashboard/restart", Access::Admin, Conn::post_restart),
    ("POST", "/api/dashboard/reset", Access::Admin, Conn::post_reset_stats),
    ("POST", "/api/dashboard/key", Access::Admin, Conn::post_new_key),
    ("POST", "/api/dashboard/stop", Access::Admin, Conn::post_stop_server),
    ("POST", "/api/dashboard/open-folder", Access::Admin, Conn::post_open_folder),
    ("GET", "/api/dashboard/integrations", Access::Admin, Conn::get_integrations),
    ("POST", "/api/dashboard/integrations/connect", Access::Admin, Conn::post_integration_connect),
    ("POST", "/api/dashboard/integrations/disconnect", Access::Admin, Conn::post_integration_disconnect),
];

const INFERENCE_PATHS: [&str; 8] = [
    "/v1/chat/completions",
    "/chat/completions",
    "/v1/responses",
    "/responses",
    "/api/chat",
    "/api/generate",
    "/api/dashboard/test",
    "/v1/messages",
];
const OLLAMA_PATHS: [&str; 3] = ["/api/chat", "/api/generate", "/api/show"];
const ANTHROPIC_PATHS: [&str; 2] = ["/v1/messages", "/v1/messages/count_tokens"];
const QUIET_PATHS: [&str; 11] = [
    "/",
    "/dashboard",
    "/favicon.ico",
    "/health",
    "/api/dashboard",
    "/api/dashboard/stats",
    "/api/dashboard/log",
    "/api/dashboard/settings",
    "/api/dashboard/integrations",
    "/assets/fonts/geist.woff2",
    "/assets/fonts/geist-mono.woff2",
];

// ---- Errors -----------------------------------------------------------------

enum HErr {
    Proxy(ProxyError),
    /// The client went away (BrokenPipe, ConnectionReset, ConnectionAborted).
    Gone,
    Internal(String),
}

type HResult = Result<(), HErr>;

fn client_gone(e: &io::Error) -> bool {
    matches!(e.kind(), io::ErrorKind::BrokenPipe | io::ErrorKind::ConnectionReset | io::ErrorKind::ConnectionAborted)
}

impl From<ProxyError> for HErr {
    fn from(e: ProxyError) -> Self {
        HErr::Proxy(e)
    }
}

impl From<io::Error> for HErr {
    fn from(e: io::Error) -> Self {
        if client_gone(&e) {
            HErr::Gone
        } else {
            HErr::Internal(format!("OSError: {e}"))
        }
    }
}

/// An error inside a stream: from upstream, or from writing to the client.
enum StreamErr {
    Proxy(ProxyError),
    Io(io::Error),
}

impl From<ProxyError> for StreamErr {
    fn from(e: ProxyError) -> Self {
        StreamErr::Proxy(e)
    }
}

impl From<io::Error> for StreamErr {
    fn from(e: io::Error) -> Self {
        StreamErr::Io(e)
    }
}

impl StreamErr {
    fn message(&self) -> String {
        match self {
            StreamErr::Proxy(e) => e.message.clone(),
            StreamErr::Io(e) => e.to_string(),
        }
    }

    fn status(&self) -> u16 {
        match self {
            StreamErr::Proxy(e) => e.status,
            StreamErr::Io(_) => 502,
        }
    }

    fn gone(&self) -> bool {
        matches!(self, StreamErr::Io(e) if client_gone(e))
    }
}

fn no_events() -> ProxyError {
    ProxyError::upstream("upstream returned no events.")
}

// ---- Usage tap ------------------------------------------------------------------

/// Passes upstream events through and keeps the latest usage for the log.
struct Tap {
    inner: Events,
    usage: Arc<Mutex<Option<Value>>>,
}

impl Iterator for Tap {
    type Item = PResult<Value>;

    fn next(&mut self) -> Option<Self::Item> {
        let item = self.inner.next()?;
        if let Ok(event) = &item {
            if event.g("usage").is_object() {
                *self.usage.lock().unwrap_or_else(|e| e.into_inner()) = Some(chat_usage(event.g("usage")));
            }
        }
        Some(item)
    }
}

// ---- Requests -----------------------------------------------------------------

struct Request {
    method: String,
    target: String,
    version: String,
    headers: Vec<(String, String)>,
}

impl Request {
    fn header(&self, name: &str) -> Option<&str> {
        self.headers.iter().find(|(k, _)| k.eq_ignore_ascii_case(name)).map(|(_, v)| v.as_str())
    }

    fn header_or(&self, name: &str) -> &str {
        self.header(name).unwrap_or("")
    }
}

enum ReadErr {
    /// Connection closed or nothing usable; close without a response.
    Closed,
    /// Bad request: send this status and message, then close.
    Bad(u16, String),
}

fn read_line(reader: &mut BufReader<TcpStream>, limit: usize) -> Result<(Vec<u8>, bool), io::Error> {
    let mut line = Vec::new();
    let n = reader.by_ref().take(limit as u64 + 1).read_until(b'\n', &mut line)?;
    Ok((line, n > limit))
}

fn read_request(reader: &mut BufReader<TcpStream>) -> Result<Request, ReadErr> {
    let (raw, too_long) = read_line(reader, MAX_LINE).map_err(|_| ReadErr::Closed)?;
    if too_long {
        return Err(ReadErr::Bad(414, "Request-URI Too Long".into()));
    }
    if raw.is_empty() {
        return Err(ReadErr::Closed);
    }
    let line = String::from_utf8_lossy(&raw).trim_end_matches(['\r', '\n']).to_string();
    let words: Vec<&str> = line.split_whitespace().collect();
    if words.is_empty() {
        return Err(ReadErr::Closed);
    }
    if words.len() != 3 || !words[2].starts_with("HTTP/") {
        return Err(ReadErr::Bad(400, format!("Bad request syntax ({line:?})")));
    }
    let version = words[2].to_string();
    let numbers = &version[5..];
    let valid = numbers.split_once('.').is_some_and(|(a, b)| {
        !a.is_empty() && !b.is_empty() && a.bytes().all(|c| c.is_ascii_digit()) && b.bytes().all(|c| c.is_ascii_digit())
    });
    if !valid {
        return Err(ReadErr::Bad(400, format!("Bad request version ({version:?})")));
    }
    if version.as_str() >= "HTTP/2.0" {
        return Err(ReadErr::Bad(505, format!("Invalid HTTP version ({numbers})")));
    }
    let mut headers = Vec::new();
    loop {
        let (raw, too_long) = read_line(reader, MAX_LINE).map_err(|_| ReadErr::Closed)?;
        if too_long {
            return Err(ReadErr::Bad(431, "Line too long".into()));
        }
        if raw.is_empty() || raw == b"\r\n" || raw == b"\n" {
            break;
        }
        let text = String::from_utf8_lossy(&raw).trim_end_matches(['\r', '\n']).to_string();
        if text.starts_with([' ', '\t']) {
            // Folded continuation line.
            if let Some((_, value)) = headers.last_mut() {
                let value: &mut String = value;
                value.push(' ');
                value.push_str(text.trim());
            }
            continue;
        }
        if let Some((name, value)) = text.split_once(':') {
            headers.push((name.trim().to_string(), value.trim().to_string()));
        }
        if headers.len() > MAX_HEADERS {
            return Err(ReadErr::Bad(431, "Too many headers".into()));
        }
    }
    Ok(Request { method: words[0].to_string(), target: words[1].to_string(), version, headers })
}

/// `urllib.parse.urlsplit(target).path`.
fn url_path(target: &str) -> String {
    let mut rest = target;
    if let Some(i) = rest.find("://") {
        let scheme = &rest[..i];
        if !scheme.is_empty() && scheme.chars().all(|c| c.is_ascii_alphanumeric() || "+-.".contains(c)) {
            rest = &rest[i + 1..];
        }
    }
    if let Some(after) = rest.strip_prefix("//") {
        rest = after.find(['/', '?', '#']).map(|i| &after[i..]).unwrap_or("");
    }
    let end = rest.find(['?', '#']).unwrap_or(rest.len());
    rest[..end].to_string()
}

fn percent_decode(text: &str) -> String {
    let bytes = text.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() + 0 && i + 2 <= bytes.len() - 1 {
            if let Ok(b) = u8::from_str_radix(&text[i + 1..i + 3], 16) {
                out.push(b);
                i += 3;
                continue;
            }
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn reason(code: u16) -> &'static str {
    match code {
        100 => "Continue",
        200 => "OK",
        204 => "No Content",
        400 => "Bad Request",
        401 => "Unauthorized",
        402 => "Payment Required",
        403 => "Forbidden",
        404 => "Not Found",
        405 => "Method Not Allowed",
        408 => "Request Timeout",
        409 => "Conflict",
        411 => "Length Required",
        413 => "Request Entity Too Large",
        414 => "Request-URI Too Long",
        415 => "Unsupported Media Type",
        422 => "Unprocessable Entity",
        429 => "Too Many Requests",
        431 => "Request Header Fields Too Large",
        451 => "Unavailable For Legal Reasons",
        500 => "Internal Server Error",
        501 => "Not Implemented",
        502 => "Bad Gateway",
        503 => "Service Unavailable",
        504 => "Gateway Timeout",
        505 => "HTTP Version Not Supported",
        _ => "",
    }
}

fn http_date() -> String {
    chrono::Utc::now().format("%a, %d %b %Y %H:%M:%S GMT").to_string()
}

fn machine_name() -> String {
    #[cfg(windows)]
    {
        use windows_sys::Win32::System::SystemInformation::{ComputerNamePhysicalDnsHostname, GetComputerNameExW};
        let mut buf = [0u16; 256];
        let mut len = buf.len() as u32;
        if unsafe { GetComputerNameExW(ComputerNamePhysicalDnsHostname, buf.as_mut_ptr(), &mut len) } != 0 {
            return String::from_utf16_lossy(&buf[..len as usize]);
        }
        std::env::var("COMPUTERNAME").unwrap_or_default()
    }
    #[cfg(not(windows))]
    {
        std::fs::read_to_string("/proc/sys/kernel/hostname")
            .or_else(|_| std::fs::read_to_string("/etc/hostname"))
            .map(|s| s.trim().to_string())
            .or_else(|_| std::env::var("HOSTNAME"))
            .unwrap_or_default()
    }
}

static MACHINE: LazyLock<String> =
    LazyLock::new(|| machine_name().to_lowercase().split('.').next().unwrap_or("").to_string());

/// `urlsplit("//" + host).hostname`.
fn host_name(host: &str) -> String {
    let host = host.rsplit_once('@').map(|(_, h)| h).unwrap_or(host);
    if let Some(rest) = host.strip_prefix('[') {
        return rest.split(']').next().unwrap_or("").to_lowercase();
    }
    host.split(':').next().unwrap_or("").to_lowercase()
}

// ---- Connections --------------------------------------------------------------

struct Conn {
    reader: BufReader<TcpStream>,
    out: TcpStream,
    peer: SocketAddr,
    req: Request,
    path: String,
    t0: Instant,
    /// The status line already sent, 0 before.
    status: u16,
    model: Option<String>,
    usage: Arc<Mutex<Option<Value>>>,
    error: Option<String>,
    body_read: bool,
    close: bool,
}

fn handle_connection(stream: TcpStream) {
    let _ = stream.set_read_timeout(Some(SOCKET_TIMEOUT));
    let _ = stream.set_write_timeout(Some(SOCKET_TIMEOUT));
    let _ = stream.set_nodelay(true);
    let Ok(peer) = stream.peer_addr() else { return };
    let Ok(out) = stream.try_clone() else { return };
    let mut reader = BufReader::new(stream);
    loop {
        let req = match read_request(&mut reader) {
            Ok(req) => req,
            Err(ReadErr::Closed) => return,
            Err(ReadErr::Bad(code, message)) => {
                let mut out = &out;
                let _ = out.write_all(&error_page(code, &message, true));
                return;
            }
        };
        let close = {
            let connection = req.header_or("Connection").to_lowercase();
            if connection == "close" {
                true
            } else {
                !(connection == "keep-alive" || req.version.as_str() >= "HTTP/1.1")
            }
        };
        let mut conn = Conn {
            reader,
            out: out.try_clone().unwrap_or_else(|_| out.try_clone().expect("socket")),
            peer,
            path: url_path(&req.target),
            req,
            t0: Instant::now(),
            status: 0,
            model: None,
            usage: Arc::new(Mutex::new(None)),
            error: None,
            body_read: false,
            close,
        };
        if conn.req.version.as_str() >= "HTTP/1.1" && conn.req.header_or("Expect").eq_ignore_ascii_case("100-continue") {
            if conn.out.write_all(b"HTTP/1.1 100 Continue\r\n\r\n").is_err() {
                return;
            }
        }
        match conn.req.method.as_str() {
            "GET" | "POST" | "OPTIONS" => conn.dispatch(),
            method => {
                let body = method != "HEAD";
                let _ = conn.out.write_all(&error_page(501, &format!("Unsupported method ('{method}')"), body));
                return;
            }
        }
        let _ = conn.out.flush();
        if conn.close || STOP.load(Ordering::SeqCst) {
            return;
        }
        reader = conn.reader;
    }
}

/// The plain error page of Python's http.server, for requests that never reach a handler.
fn error_page(code: u16, message: &str, body: bool) -> Vec<u8> {
    let explain = reason(code);
    let html = format!(
        "<!DOCTYPE HTML>\n<html lang=\"en\">\n    <head>\n        <meta charset=\"utf-8\">\n        <title>Error response</title>\n    </head>\n    <body>\n        <h1>Error response</h1>\n        <p>Error code: {code}</p>\n        <p>Message: {}.</p>\n        <p>Error code explanation: {code} - {explain}.</p>\n    </body>\n</html>\n",
        html_escape(message)
    );
    let mut out = format!(
        "HTTP/1.1 {code} {explain}\r\nServer: 7q9k-route/{APP_VERSION}\r\nDate: {}\r\nConnection: close\r\nContent-Type: text/html;charset=utf-8\r\nContent-Length: {}\r\n\r\n",
        http_date(),
        html.len()
    )
    .into_bytes();
    if body {
        out.extend_from_slice(html.as_bytes());
    }
    out
}

fn html_escape(text: &str) -> String {
    text.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;").replace('"', "&quot;").replace('\'', "&#x27;")
}

impl Conn {
    fn dispatch(&mut self) {
        let method = self.req.method.clone();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| self.route(&method)));
        match result {
            Ok(Ok(())) => {}
            Ok(Err(HErr::Proxy(e))) => self.fail(e.status, &e.message),
            Ok(Err(HErr::Gone)) => self.close = true,
            Ok(Err(HErr::Internal(message))) => self.fail(500, &format!("internal error: {message}")),
            Err(panic) => {
                let text = panic
                    .downcast_ref::<String>()
                    .cloned()
                    .or_else(|| panic.downcast_ref::<&str>().map(|s| s.to_string()))
                    .unwrap_or_default();
                self.fail(500, &format!("internal error: panic: {text}"));
            }
        }
        self.finish();
    }

    fn route(&mut self, method: &str) -> HResult {
        if method == "OPTIONS" {
            return self.send_head(204, &[("Content-Length", "0".into())]).map_err(HErr::from);
        }
        let mut route = ROUTES.iter().find(|(m, p, _, _)| *m == method && *p == self.path).map(|(_, _, a, h)| (*a, *h));
        if route.is_none() && method == "GET" && self.path.starts_with("/assets/") {
            route = Some((Access::Open, Conn::get_asset as Handler));
        }
        let Some((access, handler)) = route else {
            if ROUTES.iter().any(|(_, p, _, _)| *p == self.path) {
                return Err(ProxyError::new(format!("method {method} not allowed for {}", self.path), 405).into());
            }
            return Err(ProxyError::new(format!("unknown path: {}", self.path), 404).into());
        };
        self.check_access(access)?;
        handler(self)
    }

    fn fail(&mut self, code: u16, message: &str) {
        if self.error.is_none() {
            self.error = Some(message.to_string());
        }
        if self.status != 0 {
            self.close = true;
            return;
        }
        self.drain_body();
        let payload = if OLLAMA_PATHS.contains(&self.path.as_str()) {
            json!({"error": message})
        } else if ANTHROPIC_PATHS.contains(&self.path.as_str()) {
            AnthropicStream::fail(message, code).swap_remove(0).1
        } else {
            json!({"error": {"message": message, "type": "route_error", "code": code}})
        };
        if self.send_json(code, &payload).is_err() {
            self.close = true;
        }
    }

    fn drain_body(&mut self) {
        if self.body_read {
            return;
        }
        self.body_read = true;
        let Ok(length) = self.req.header_or("Content-Length").trim().parse::<i64>() else { return };
        if 0 < length && length <= 1024 * 1024 {
            let _ = io::copy(&mut self.reader.by_ref().take(length as u64), &mut io::sink());
        }
    }

    fn finish(&mut self) {
        let method = self.req.method.as_str();
        let status = self.status;
        if method == "OPTIONS" || (method == "GET" && status < 400 && QUIET_PATHS.contains(&self.path.as_str())) {
            return;
        }
        let path: String = self.path.chars().map(|c| if c.is_control() { '?' } else { c }).take(200).collect();
        let mut entry = json!({
            "t": chrono::Local::now().format("%H:%M:%S").to_string(),
            "ts": now(),
            "method": method,
            "path": path,
            "status": status,
            "ms": py_round(self.t0.elapsed().as_secs_f64() * 1000.0),
            "client": if self.is_loopback() { "local".to_string() } else { self.client_ip() },
        });
        let inference = self.model.is_some() && INFERENCE_PATHS.contains(&self.path.as_str());
        if let Some(model) = &self.model {
            let usage = self.usage.lock().unwrap_or_else(|e| e.into_inner()).clone().unwrap_or(Value::Null);
            set(&mut entry, "model", model.as_str());
            set(&mut entry, "tokens_in", usage.g("prompt_tokens").int());
            set(&mut entry, "tokens_out", usage.g("completion_tokens").int());
            set(&mut entry, "tokens_cached", usage.g("prompt_tokens_details").g("cached_tokens").int());
        }
        if let Some(error) = &self.error {
            if !error.is_empty() {
                set(&mut entry, "error", error.chars().take(300).collect::<String>());
            }
        }
        console_line(&entry);
        metrics().record(entry, inference);
    }

    // ---- Access -----------------------------------------------------------------

    fn via_proxy(&self) -> bool {
        ["X-Forwarded-For", "Forwarded", "X-Real-IP"].iter().any(|h| !self.req.header_or(h).is_empty())
    }

    fn is_loopback(&self) -> bool {
        if self.via_proxy() {
            return false;
        }
        match self.peer.ip() {
            IpAddr::V4(ip) => ip.is_loopback(),
            IpAddr::V6(ip) => ip.to_ipv4_mapped().map(|v4| v4.is_loopback()).unwrap_or(ip.is_loopback()),
        }
    }

    fn host_ok(&self) -> bool {
        let host = self.req.header_or("Host").trim().to_lowercase();
        if host.is_empty() {
            return true;
        }
        let hostname = host_name(&host);
        if hostname == "localhost" || hostname.ends_with(".localhost") {
            return true;
        }
        if hostname.split('%').next().unwrap_or("").parse::<IpAddr>().is_ok() {
            return true;
        }
        let machine = MACHINE.as_str();
        [machine.to_string(), format!("{machine}.local"), format!("{machine}.lan"), format!("{machine}.home")]
            .contains(&hostname)
    }

    fn origin_ok(&self) -> bool {
        if self.req.header_or("Sec-Fetch-Site").to_lowercase() == "cross-site" {
            return false;
        }
        let Some(origin) = self.req.header("Origin") else { return true };
        let netloc = origin
            .split_once(':')
            .and_then(|(_, rest)| rest.strip_prefix("//"))
            .map(|rest| rest[..rest.find(['/', '?', '#']).unwrap_or(rest.len())].to_lowercase())
            .unwrap_or_default();
        !netloc.is_empty() && netloc == self.req.header_or("Host").trim().to_lowercase()
    }

    fn trusted(&self) -> bool {
        self.is_loopback() && self.host_ok() && self.origin_ok()
    }

    fn presented_key(&self) -> Option<String> {
        let auth = self.req.header_or("Authorization");
        if auth.len() >= 7 && auth.is_char_boundary(7) && auth[..7].eq_ignore_ascii_case("bearer ") {
            return Some(auth[7..].trim().to_string());
        }
        let key = self.req.header_or("x-api-key").trim();
        (!key.is_empty()).then(|| key.to_string())
    }

    fn authorized(&self) -> bool {
        self.presented_key().is_some_and(|key| compare_digest(key.as_bytes(), api_key().as_bytes()))
    }

    fn client_ip(&self) -> String {
        let forwarded = self.req.header_or("X-Forwarded-For").split(',').next().unwrap_or("").trim().to_string();
        if forwarded.is_empty() {
            self.peer.ip().to_string()
        } else {
            forwarded
        }
    }

    fn check_access(&self, access: Access) -> PResult<()> {
        match access {
            Access::Open => Ok(()),
            Access::Admin if self.trusted() => Ok(()),
            Access::Admin => Err(ProxyError::new("only allowed from the server machine itself.", 403)),
            _ if self.trusted() || self.authorized() => Ok(()),
            _ => Err(ProxyError::new(
                "Missing or invalid API key. Pass it as 'Authorization: Bearer <key>' (key file: api-key.txt). \
                 Requests from other machines always need it.",
                401,
            )),
        }
    }

    // ---- Sending ----------------------------------------------------------------

    fn head_bytes(&mut self, code: u16, headers: &[(&str, String)]) -> Vec<u8> {
        self.status = code;
        let mut head = format!(
            "HTTP/1.1 {code} {}\r\nServer: 7q9k-route/{APP_VERSION}\r\nDate: {}\r\n",
            reason(code),
            http_date()
        );
        for (name, value) in headers {
            head.push_str(&format!("{name}: {value}\r\n"));
            if name.eq_ignore_ascii_case("Connection") && value.eq_ignore_ascii_case("close") {
                self.close = true;
            }
        }
        head.push_str("\r\n");
        head.into_bytes()
    }

    fn send_head(&mut self, code: u16, headers: &[(&str, String)]) -> io::Result<()> {
        let head = self.head_bytes(code, headers);
        self.out.write_all(&head)
    }

    fn send_with_body(&mut self, code: u16, headers: &[(&str, String)], body: &[u8]) -> io::Result<()> {
        let mut data = self.head_bytes(code, headers);
        data.extend_from_slice(body);
        self.out.write_all(&data)
    }

    fn send_json(&mut self, code: u16, payload: &Value) -> io::Result<()> {
        let data = dumps(payload);
        self.send_bytes(code, data.as_bytes(), "application/json")
    }

    fn send_bytes(&mut self, code: u16, data: &[u8], content_type: &str) -> io::Result<()> {
        self.send_with_body(
            code,
            &[
                ("Content-Type", content_type.into()),
                ("Content-Length", data.len().to_string()),
                ("Cache-Control", "no-store".into()),
                ("X-Content-Type-Options", "nosniff".into()),
                ("Connection", "close".into()),
            ],
            data,
        )
    }

    fn ok(&mut self, payload: Value) -> HResult {
        self.send_json(200, &payload).map_err(HErr::from)
    }

    fn start_chunked(&mut self, content_type: &str) -> io::Result<()> {
        self.send_head(
            200,
            &[
                ("Content-Type", content_type.into()),
                ("Cache-Control", "no-cache".into()),
                ("X-Accel-Buffering", "no".into()),
                ("Transfer-Encoding", "chunked".into()),
            ],
        )
    }

    fn chunk(&mut self, data: &[u8]) -> io::Result<()> {
        let mut out = format!("{:x}\r\n", data.len()).into_bytes();
        out.extend_from_slice(data);
        out.extend_from_slice(b"\r\n");
        self.out.write_all(&out)?;
        self.out.flush()
    }

    fn end_chunked(&mut self) {
        let _ = self.out.write_all(b"0\r\n\r\n");
        let _ = self.out.flush();
        self.close = true;
    }

    // ---- Request helpers ----------------------------------------------------------

    fn read_json(&mut self) -> PResult<Value> {
        if self.req.header_or("Transfer-Encoding").to_lowercase().contains("chunked") {
            return Err(ProxyError::new("chunked request bodies are not supported; send Content-Length.", 411));
        }
        let raw_length = self.req.header_or("Content-Length").trim().to_string();
        let length: i64 = if raw_length.is_empty() {
            0
        } else {
            raw_length
                .trim_start_matches('+')
                .replace('_', "")
                .parse()
                .map_err(|_| ProxyError::new("invalid Content-Length header.", 400))?
        };
        if length < 0 {
            return Err(ProxyError::new("invalid Content-Length header.", 400));
        }
        if length > MAX_BODY_BYTES {
            return Err(ProxyError::new(
                format!("request body too large ({length} bytes, max {MAX_BODY_BYTES})."),
                413,
            ));
        }
        let mut raw = Vec::with_capacity(length as usize);
        if length > 0 {
            if let Err(e) = self.reader.by_ref().take(length as u64).read_to_end(&mut raw) {
                self.body_read = true;
                return Err(ProxyError::new(format!("invalid JSON body: {e}"), 400));
            }
        }
        self.body_read = true;
        if raw.is_empty() {
            return Ok(json!({}));
        }
        let text = std::str::from_utf8(&raw).map_err(|e| ProxyError::new(format!("invalid JSON body: {e}"), 400))?;
        let req: Value =
            serde_json::from_str(text).map_err(|e| ProxyError::new(format!("invalid JSON body: {e}"), 400))?;
        if !req.is_object() {
            return Err(ProxyError::new("request body must be a JSON object", 400));
        }
        Ok(req)
    }

    fn resolve_model(&mut self, name: &Value) -> PResult<String> {
        let model = normalize_model_name(name)?;
        let models = available_models(MODELS_TIMEOUT, false);
        if !models.is_empty() && !models.contains(&model) {
            return Err(ProxyError::new(
                format!(
                    "model '{model}' is not an available free model. Call GET /v1/models or /api/tags for the current list ({})",
                    models.join(", ")
                ),
                404,
            ));
        }
        self.model = Some(model.clone());
        Ok(model)
    }

    fn open_upstream(&mut self, req: &Value, model: &str) -> PResult<Events> {
        let inner = open_upstream_events(req, model, UPSTREAM_TIMEOUT)?;
        Ok(Box::new(Tap { inner, usage: Arc::clone(&self.usage) }))
    }

    fn set_usage(&mut self, usage: Value) {
        *self.usage.lock().unwrap_or_else(|e| e.into_inner()) = Some(usage);
    }

    // ---- Pages and model lists ------------------------------------------------------

    fn get_dashboard_page(&mut self) -> HResult {
        let page = assets::DASHBOARD_HTML;
        self.send_with_body(
            200,
            &[
                ("Content-Type", "text/html; charset=utf-8".into()),
                ("Content-Length", page.len().to_string()),
                ("Cache-Control", "no-store".into()),
                ("X-Frame-Options", "DENY".into()),
                ("Referrer-Policy", "no-referrer".into()),
                ("Connection", "close".into()),
            ],
            page,
        )?;
        Ok(())
    }

    fn get_asset(&mut self) -> HResult {
        let name = percent_decode(&self.path["/assets/".len()..]);
        let Some((content_type, data)) = assets::find(&name) else {
            return Err(ProxyError::new(format!("unknown path: {}", self.path), 404).into());
        };
        self.send_with_body(
            200,
            &[
                ("Content-Type", content_type.into()),
                ("Content-Length", data.len().to_string()),
                ("Cache-Control", "public, max-age=86400".into()),
            ],
            data,
        )?;
        Ok(())
    }

    fn get_favicon(&mut self) -> HResult {
        let (content_type, data) = assets::find("icon.ico").expect("icon");
        self.send_with_body(
            200,
            &[
                ("Content-Type", content_type.into()),
                ("Content-Length", data.len().to_string()),
                ("Cache-Control", "public, max-age=86400".into()),
            ],
            data,
        )?;
        Ok(())
    }

    fn get_health(&mut self) -> HResult {
        self.ok(json!({"ok": true, "version": APP_VERSION}))
    }

    fn get_models(&mut self) -> HResult {
        let models = available_models(MODELS_TIMEOUT, false);
        let prefix = if self.req.header_or("anthropic-version").is_empty() { "" } else { ANTHROPIC_MODEL_PREFIX };
        let data: Vec<Value> = models
            .iter()
            .map(|m| {
                json!({
                    "id": format!("{prefix}{m}"), "object": "model", "created": 0, "owned_by": "opencode",
                    "type": "model", "display_name": m, "created_at": "1970-01-01T00:00:00Z",
                    "description": MODEL_DESCRIPTION,
                })
            })
            .collect();
        self.ok(json!({
            "object": "list",
            "data": data,
            "has_more": false,
            "first_id": models.first().map(|m| format!("{prefix}{m}")),
            "last_id": models.last().map(|m| format!("{prefix}{m}")),
        }))
    }

    fn get_ollama_tags(&mut self) -> HResult {
        self.ok(json!({"models": ollama_model_entries()}))
    }

    fn get_ollama_ps(&mut self) -> HResult {
        let models: Vec<Value> = ollama_model_entries()
            .iter()
            .map(|e| {
                json!({
                    "name": e.g("name"), "model": e.g("model"), "size": 0, "size_vram": 0,
                    "digest": e.g("digest"), "details": e.g("details"), "expires_at": "0001-01-01T00:00:00Z",
                })
            })
            .collect();
        self.ok(json!({"models": models}))
    }

    fn get_version(&mut self) -> HResult {
        self.ok(json!({"version": format!("7q9k-route/{APP_VERSION}")}))
    }

    fn get_props(&mut self) -> HResult {
        let models = available_models(MODELS_TIMEOUT, false);
        let model = models.first().cloned().unwrap_or_else(|| DEFAULT_MODEL.to_string());
        self.ok(json!({
            "model": model,
            "family": "opencode-free",
            "prompt": "",
            "n_keep": 0,
            "n_discard": 0,
            "n_predict": -1,
            "n_ctx": 131072,
            "temperature": 1.0,
            "top_k": 40,
            "top_p": 0.95,
            "total_slots": 1,
            "modalities": {"vision": false, "audio_in": false, "audio_out": false},
            "default_generation_settings": {
                "id": "slot-0", "port": 0, "model": model, "n_predict": -1, "n_keep": 0, "truncation": 50, "n_ctx": 131072,
            },
        }))
    }

    // ---- Dashboard ------------------------------------------------------------------

    fn get_dashboard_state(&mut self) -> HResult {
        let local = self.trusted();
        self.ok(dashboard_state(local))
    }

    fn get_settings(&mut self) -> HResult {
        self.ok(settings_payload())
    }

    fn post_settings(&mut self) -> HResult {
        let req = self.read_json()?;
        let mut access = None;
        let mut port = None;
        if req.has("access") {
            match req.g("access").as_str().filter(|a| access_host(a).is_some()) {
                Some(a) => access = Some(a.to_string()),
                None => return Err(ProxyError::new("access must be local or network.", 400).into()),
            }
        }
        if req.has("port") {
            let value = match req.g("port") {
                Value::Number(n) => n.as_i64(),
                Value::String(s) => s.trim().trim_start_matches('+').parse::<i64>().ok(),
                _ => None,
            };
            let Some(value) = value else { return Err(ProxyError::new("port must be a number.", 400).into()) };
            if !(1024..=65535).contains(&value) {
                return Err(ProxyError::new("port must be between 1024 and 65535.", 400).into());
            }
            port = Some(value as u16);
        }
        for flag in ["open_browser", "autostart"] {
            if req.has(flag) && !matches!(req.g(flag), Value::Null | Value::Bool(_)) {
                return Err(ProxyError::new(format!("{flag} must be true or false."), 400).into());
            }
        }
        if let Value::Bool(on) = req.g("autostart") {
            config::set_autostart(*on)?;
        }
        {
            let mut config = config::config();
            if let Some(access) = access {
                config.access = access;
            }
            if let Some(port) = port {
                config.port = port;
            }
            if req.has("open_browser") {
                config.open_browser = req.g("open_browser").truthy();
            }
        }
        config::save_config().map_err(|e| ProxyError::new(format!("could not write config.json: {e}"), 500))?;
        self.ok(settings_payload())
    }

    fn post_restart(&mut self) -> HResult {
        let (host, port, bound_port) = {
            let config = config::config().clone();
            let b = bound();
            (
                b.cli_host.clone().unwrap_or_else(|| access_host(&config.access).unwrap_or("0.0.0.0").into()),
                b.cli_port.unwrap_or(config.port),
                b.port,
            )
        };
        // Only a new port can be taken by someone else; the current one is ours
        // and is released before the new process binds it.
        if port != bound_port && !port_available(&host, port) {
            return Err(ProxyError::new(
                format!("port {port} is already used by another program. Pick another port."),
                409,
            )
            .into());
        }
        RESTART_REQUESTED.store(true, Ordering::SeqCst);
        self.ok(json!({"ok": true, "restarting": true, "port": port}))?;
        thread::spawn(shutdown);
        Ok(())
    }

    fn get_dashboard_stats(&mut self) -> HResult {
        let pricing = pricing();
        pricing.refresh_async(false);
        let mut snapshot = metrics().snapshot();
        if let Some(Value::Array(log)) = snapshot.as_object_mut().and_then(|s| s.get_mut("log")) {
            for entry in log.iter_mut() {
                let model = entry.g("model").or_str();
                if !model.is_empty() {
                    let found = pricing.for_model(&model);
                    let cost = usage_cost(entry, found.as_ref().map(|(_, _, p)| p));
                    set(entry, "cost_usd", cost);
                }
            }
        }
        let usage = usage_report(&snapshot, pricing);
        let mut payload = Map::new();
        payload.insert("ok".into(), Value::Bool(true));
        if let Value::Object(map) = snapshot {
            payload.extend(map);
        }
        payload.insert("usage".into(), usage);
        self.ok(Value::Object(payload))
    }

    fn post_save_stats(&mut self) -> HResult {
        let metrics = metrics();
        let Some(path) = metrics.path.clone() else {
            return Err(ProxyError::new("saving is disabled for this server instance.", 409).into());
        };
        let saved_at = metrics.save().map_err(|e| ProxyError::new(format!("could not write {}: {e}", file_name(&path)), 500))?;
        self.ok(json!({"ok": true, "saved_at": saved_at, "file": path.to_string_lossy()}))
    }

    fn post_reset_stats(&mut self) -> HResult {
        let metrics = metrics();
        metrics.reset();
        if let Some(path) = &metrics.path {
            metrics.save().map_err(|e| ProxyError::new(format!("could not write {}: {e}", file_name(path)), 500))?;
        }
        self.ok(json!({"ok": true}))
    }

    fn post_refresh_prices(&mut self) -> HResult {
        let pricing = pricing();
        if !pricing.refresh(20.0) {
            return Err(ProxyError::new(
                format!("OpenRouter prices could not be loaded: {}", pricing.info().g("error").pystr()),
                502,
            )
            .into());
        }
        let mut payload = Map::new();
        payload.insert("ok".into(), Value::Bool(true));
        if let Value::Object(map) = pricing.info() {
            payload.extend(map);
        }
        self.ok(Value::Object(payload))
    }

    fn get_dashboard_log(&mut self) -> HResult {
        let log = metrics().snapshot().g("log").clone();
        self.ok(json!({"ok": true, "log": log}))
    }

    fn post_refresh_models(&mut self) -> HResult {
        let models = available_models(MODELS_TIMEOUT, true);
        sync_integrations_async();
        let mut payload = Map::new();
        payload.insert("ok".into(), Value::Bool(true));
        payload.insert("models".into(), json!(models));
        if let Value::Object(map) = model_source() {
            payload.extend(map);
        }
        self.ok(Value::Object(payload))
    }

    fn post_dashboard_test(&mut self) -> HResult {
        let req = self.read_json()?;
        let model = self.resolve_model(req.g("model"))?;
        let result = run_model_test(&model, req.g("prompt").as_str())?;
        self.set_usage(result.g("usage").clone());
        self.ok(result)
    }

    fn post_new_key(&mut self) -> HResult {
        let req = self.read_json()?;
        let custom = req.g("key").or_str().trim().to_string();
        let new_key = if custom.is_empty() {
            config::regenerate_api_key()?
        } else {
            if !KEY_PATTERN.is_match(&custom) {
                return Err(ProxyError::new(
                    "the key needs 8 to 200 characters: letters, digits and . _ ~ + / = -",
                    400,
                )
                .into());
            }
            config::write_key(&custom)?
        };
        set_api_key(&new_key);
        sync_integrations_async();
        self.ok(json!({"ok": true, "api_key": new_key}))
    }

    fn get_integrations(&mut self) -> HResult {
        let enabled = config::config().integrations.clone();
        let mut apps = integrations::describe_all(&integration_target());
        for app in apps.iter_mut() {
            let on = app.g("id").as_str().is_some_and(|id| enabled.iter().any(|e| e == id));
            set(app, "auto_update", on);
        }
        self.ok(json!({"ok": true, "apps": apps, "backups": paths::backups_dir().to_string_lossy()}))
    }

    fn integration_id(&mut self) -> PResult<String> {
        let req = self.read_json()?;
        let id = req.g("id").or_str();
        integrations::get(&id).map_err(|e| ProxyError::new(e.to_string(), 404))?;
        Ok(id)
    }

    fn set_enabled(id: &str, enabled: bool) {
        {
            let mut config = config::config();
            config.integrations.retain(|i| i != id);
            if enabled {
                config.integrations.push(id.to_string());
            }
        }
        let _ = config::save_config();
    }

    fn post_integration_change(&mut self, connect: bool) -> HResult {
        let id = self.integration_id()?;
        let target = integration_target();
        let backups = paths::backups_dir();
        let result = if connect {
            integrations::connect(&id, &target, &backups)
        } else {
            integrations::disconnect(&id, &target, &backups)
        };
        let mut app = result.map_err(|e| match e {
            IntegrationError::App(m) => ProxyError::new(m, 409),
            IntegrationError::Utf8(m) => ProxyError::new(format!("the app config is not valid UTF-8: {m}"), 409),
            IntegrationError::Io(m) => ProxyError::new(format!("could not write the app config: {m}"), 500),
        })?;
        Self::set_enabled(&id, connect);
        set(&mut app, "auto_update", connect);
        self.ok(json!({"ok": true, "app": app}))
    }

    fn post_integration_connect(&mut self) -> HResult {
        self.post_integration_change(true)
    }

    fn post_integration_disconnect(&mut self) -> HResult {
        self.post_integration_change(false)
    }

    fn post_open_folder(&mut self) -> HResult {
        if !cfg!(windows) {
            return Err(ProxyError::new("opening folders is only supported on Windows.", 400).into());
        }
        let dir = paths::data_dir();
        std::fs::create_dir_all(&dir)?;
        std::process::Command::new("explorer").arg(&dir).spawn()?;
        self.ok(json!({"ok": true, "path": dir.to_string_lossy()}))
    }

    fn post_stop_server(&mut self) -> HResult {
        self.ok(json!({"ok": true, "stopping": true}))?;
        thread::spawn(shutdown);
        Ok(())
    }

    // ---- OpenAI chat completions -------------------------------------------------------

    fn post_chat_completions(&mut self) -> HResult {
        let req = self.read_json()?;
        validate_messages(req.g("messages"))?;
        let model = self.resolve_model(req.g("model"))?;
        let deltas = self.open_upstream(&req, &model)?;
        if req.g("stream").truthy() {
            self.openai_stream(deltas, &model);
            Ok(())
        } else {
            let completion = merge_deltas(deltas, &model)?;
            self.ok(completion)
        }
    }

    fn openai_stream(&mut self, deltas: Events, model: &str) {
        let completion_id = format!("chatcmpl-{}", token_hex(12));
        let created = now_secs();
        let sse = |payload: &Value| format!("data: {}\n\n", pydumps(payload, true, ",", ":")).into_bytes();
        let result = (|| -> Result<(), StreamErr> {
            self.start_chunked("text/event-stream; charset=utf-8")?;
            let mut seen = false;
            for event in deltas {
                let event = event?;
                seen = true;
                let choices: Vec<Value> = event
                    .g("choices")
                    .items()
                    .iter()
                    .map(|choice| {
                        let mut choice = if choice.is_object() { choice.clone() } else { json!({}) };
                        let index = choice.g("index").int();
                        set(&mut choice, "index", index);
                        choice
                    })
                    .collect();
                let event_model = event.g("model").or_str();
                let mut chunk = json!({
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": if event_model.is_empty() { model.to_string() } else { event_model },
                    "choices": choices,
                });
                if event.g("usage").truthy() {
                    set(&mut chunk, "usage", event.g("usage").clone());
                }
                self.chunk(&sse(&chunk))?;
            }
            if !seen {
                return Err(no_events().into());
            }
            Ok(())
        })();
        if let Err(err) = result {
            if err.gone() {
                self.error = Some("client disconnected".into());
                self.close = true;
                return;
            }
            let message = err.message();
            self.error = Some(message.clone());
            if self.status == 0 {
                // The stream never started; the caller-facing error is a JSON response.
                self.fail(err.status(), &message);
                return;
            }
            let _ = self.chunk(&sse(&json!({"error": {"message": message, "type": "route_error", "code": err.status()}})));
        }
        let _ = self.chunk(b"data: [DONE]\n\n");
        self.end_chunked();
    }

    // ---- OpenAI Responses API -------------------------------------------------------

    fn post_responses(&mut self) -> HResult {
        let req = self.read_json()?;
        let model = self.resolve_model(req.g("model"))?;
        let deltas = self.open_upstream(&responses_to_chat_request(&req)?, &model)?;
        if req.g("stream").truthy() {
            self.responses_stream(deltas, &model, &req);
            Ok(())
        } else {
            let object = build_responses_object(deltas, &model, &req)?;
            self.ok(object)
        }
    }

    fn responses_stream(&mut self, deltas: Events, model: &str, req: &Value) {
        let mut builder = ResponsesBuilder::new(model, req);
        fn send(conn: &mut Conn, events: Vec<Value>) -> io::Result<()> {
            for event in events {
                let data = format!("event: {}\ndata: {}\n\n", event.g("type").pystr(), pydumps(&event, true, ",", ":"));
                conn.chunk(data.as_bytes())?;
            }
            Ok(())
        }
        let result = (|| -> Result<(), StreamErr> {
            self.start_chunked("text/event-stream; charset=utf-8")?;
            send(self, builder.start())?;
            let mut seen = false;
            for event in deltas {
                let event = event?;
                seen = true;
                let out = builder.feed(&event)?;
                send(self, out)?;
            }
            if !seen {
                return Err(no_events().into());
            }
            send(self, builder.finish())?;
            Ok(())
        })();
        if let Err(err) = result {
            if err.gone() {
                self.error = Some("client disconnected".into());
                self.close = true;
                return;
            }
            let message = err.message();
            self.error = Some(message.clone());
            if self.status == 0 {
                self.fail(err.status(), &message);
                return;
            }
            let _ = send(self, builder.fail(&message, "upstream_error"));
        }
        self.end_chunked();
    }

    // ---- Anthropic Messages API -------------------------------------------------------

    fn resolve_anthropic_model(&mut self, name: &Value) -> PResult<String> {
        static SUFFIX: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\[[^\]]*\]$").expect("regex"));
        let raw = name.or_str();
        let mut text = SUFFIX.replace(raw.trim(), "").into_owned(); // "[1m]" context suffix
        if let Some(rest) = text.strip_prefix(ANTHROPIC_MODEL_PREFIX) {
            text = rest.to_string();
        }
        let bare = match text.strip_prefix("opencode/") {
            Some(rest) => rest.to_string(),
            None => text.clone(),
        };
        let models = available_models(MODELS_TIMEOUT, false);
        let lowered = bare.to_lowercase();
        if !models.is_empty()
            && !models.contains(&bare)
            && (bare.is_empty() || lowered.starts_with("claude") || CLAUDE_MODEL_ALIASES.contains(&lowered.as_str()))
        {
            // Claude Code asks for Claude models unless every model setting points here.
            text = route_default_model(&models);
        }
        self.resolve_model(&Value::String(text))
    }

    fn post_anthropic_messages(&mut self) -> HResult {
        let req = self.read_json()?;
        validate_messages(req.g("messages"))?;
        let model = self.resolve_anthropic_model(req.g("model"))?;
        let thinking = anthropic::wants_thinking(&req);
        let estimate = anthropic::estimate_tokens(&req);
        let deltas = self.open_upstream(&anthropic::to_chat_request(&req)?, &model)?;
        if req.g("stream").truthy() {
            self.anthropic_stream(deltas, &model, thinking, estimate);
            Ok(())
        } else {
            let completion = merge_deltas(deltas, &model)?;
            self.ok(anthropic::message(&completion, thinking))
        }
    }

    fn post_anthropic_count_tokens(&mut self) -> HResult {
        let req = self.read_json()?;
        validate_messages(req.g("messages"))?;
        self.ok(json!({"input_tokens": anthropic::estimate_tokens(&req)}))
    }

    fn anthropic_stream(&mut self, deltas: Events, model: &str, thinking: bool, estimate: i64) {
        let mut builder = AnthropicStream::new(model, thinking, estimate);
        fn send(conn: &mut Conn, events: Vec<(String, Value)>) -> io::Result<()> {
            for (name, payload) in events {
                let data = format!("event: {name}\ndata: {}\n\n", pydumps(&payload, false, ",", ":"));
                conn.chunk(data.as_bytes())?;
            }
            Ok(())
        }
        let result = (|| -> Result<(), StreamErr> {
            self.start_chunked("text/event-stream; charset=utf-8")?;
            send(self, builder.start())?;
            let mut seen = false;
            for event in deltas {
                let event = event?;
                seen = true;
                let out = builder.feed(&event)?;
                send(self, out)?;
            }
            if !seen {
                return Err(no_events().into());
            }
            let out = builder.finish()?;
            send(self, out)?;
            Ok(())
        })();
        if let Err(err) = result {
            if err.gone() {
                self.error = Some("client disconnected".into());
                self.close = true;
                return;
            }
            let message = err.message();
            self.error = Some(message.clone());
            if self.status == 0 {
                self.fail(err.status(), &message);
                return;
            }
            let _ = send(self, AnthropicStream::fail(&message, err.status()));
        }
        self.end_chunked();
    }

    // ---- Ollama API ---------------------------------------------------------------

    fn post_ollama_chat(&mut self) -> HResult {
        let req = self.read_json()?;
        let model = self.resolve_model(req.g("model"))?;
        let mut openai_req = ollama_options(&req);
        set(&mut openai_req, "messages", ollama_to_openai_messages(req.g("messages")));
        let deltas = self.open_upstream(&openai_req, &model)?;
        let include_thinking = req.g("think").truthy();
        if req.g("stream") != &Value::Bool(false) {
            self.ollama_stream(deltas, &model, include_thinking, true);
            Ok(())
        } else {
            let completion = merge_deltas(deltas, &model)?;
            self.ok(ollama_chat_response(&completion, include_thinking))
        }
    }

    fn post_ollama_generate(&mut self) -> HResult {
        let req = self.read_json()?;
        let model = self.resolve_model(req.g("model"))?;
        let mut messages = Vec::new();
        if req.g("system").truthy() && !req.g("raw").truthy() {
            messages.push(json!({"role": "system", "content": req.g("system").pystr()}));
        }
        messages.push(json!({"role": "user", "content": req.g("prompt").or_str()}));
        let mut openai_req = ollama_options(&req);
        set(&mut openai_req, "messages", messages);
        let deltas = self.open_upstream(&openai_req, &model)?;
        if req.g("stream") != &Value::Bool(false) {
            self.ollama_stream(deltas, &model, req.g("think").truthy(), false);
            Ok(())
        } else {
            let completion = merge_deltas(deltas, &model)?;
            self.ok(ollama_generate_response(&completion))
        }
    }

    fn post_ollama_show(&mut self) -> HResult {
        let req = self.read_json()?;
        let raw = if req.g("name").truthy() { req.g("name") } else { req.g("model") };
        let name = normalize_model_name(raw)?;
        self.ok(json!({
            "name": name,
            "model": name,
            "modified_at": ollama_now(),
            "license": "personal-use (see repository)",
            "modelfile": format!("FROM {name}"),
            "parameters": "",
            "template": "{{ .Prompt }}",
            "details": ollama_details(),
            "capabilities": ["completion", "streaming", "tools"],
            "model_info": {
                "general.architecture": "opencode-free",
                "general.name": name,
                "general.parameter_count": 0,
                "general.quantization_level": "",
            },
        }))
    }

    fn ollama_stream(&mut self, deltas: Events, model: &str, include_thinking: bool, chat: bool) {
        let created = ollama_now();
        let mut calls: BTreeMap<i64, (String, String)> = BTreeMap::new();
        let mut finish = "stop".to_string();
        let mut usage = Value::Null;
        let line = |conn: &mut Conn, payload: &Value| -> io::Result<()> {
            conn.chunk(format!("{}\n", pydumps(payload, true, ",", ":")).as_bytes())
        };
        let piece = |field: &str, text: String| -> Value {
            let mut base = json!({"model": model, "created_at": created, "done": false});
            if chat {
                let mut message = json!({"role": "assistant"});
                set(&mut message, field, text);
                set(&mut base, "message", message);
            } else {
                set(&mut base, if field == "content" { "response" } else { field }, text);
            }
            base
        };
        let result = (|| -> Result<(), StreamErr> {
            self.start_chunked("application/x-ndjson")?;
            let mut seen = false;
            for event in deltas {
                let event = event?;
                seen = true;
                if event.g("usage").is_object() {
                    usage = event.g("usage").clone();
                }
                for choice in event.g("choices").items() {
                    if choice.g("finish_reason").truthy() {
                        finish = choice.g("finish_reason").pystr();
                    }
                    let delta = choice.g("delta");
                    if delta.g("content").truthy() {
                        line(self, &piece("content", delta.g("content").pystr()))?;
                    }
                    let thought =
                        if delta.g("reasoning_content").truthy() { delta.g("reasoning_content") } else { delta.g("reasoning") };
                    if include_thinking && thought.truthy() {
                        line(self, &piece("thinking", thought.pystr()))?;
                    }
                    for raw_call in delta.g("tool_calls").items() {
                        let item = calls.entry(raw_call.g("index").int()).or_default();
                        let function = raw_call.g("function");
                        if function.g("name").truthy() {
                            item.0 = function.g("name").pystr();
                        }
                        item.1.push_str(&function.g("arguments").or_str());
                    }
                }
            }
            if !seen {
                return Err(no_events().into());
            }
            if finish == "content_filter" {
                return Err(ProxyError::new("upstream blocked the response by content filter.", 400).into());
            }
            let tool_calls: Vec<Value> = calls
                .values()
                .filter(|(name, _)| !name.is_empty())
                .map(|(name, arguments)| {
                    json!({"function": {"name": name, "arguments": json_obj(&Value::String(arguments.clone()))}})
                })
                .collect();
            let mut final_line = json!({"model": model, "created_at": created, "done": true});
            if chat {
                let mut message = json!({"role": "assistant", "content": ""});
                if !tool_calls.is_empty() {
                    set(&mut message, "tool_calls", tool_calls.clone());
                }
                set(&mut final_line, "message", message);
            } else {
                set(&mut final_line, "response", "");
            }
            set(&mut final_line, "done_reason", if tool_calls.is_empty() { finish.clone() } else { "tool_calls".into() });
            let (prompt_eval, eval) = ollama_usage(&usage);
            set(&mut final_line, "prompt_eval_count", prompt_eval);
            set(&mut final_line, "eval_count", eval);
            line(self, &final_line)?;
            Ok(())
        })();
        if let Err(err) = result {
            if err.gone() {
                self.error = Some("client disconnected".into());
                self.close = true;
                return;
            }
            let message = err.message();
            self.error = Some(message.clone());
            if self.status == 0 {
                self.fail(err.status(), &message);
                return;
            }
            let _ = line(self, &json!({"error": message}));
        }
        self.end_chunked();
    }
}

fn file_name(path: &std::path::Path) -> String {
    path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default()
}

fn console_line(entry: &Value) {
    let code = entry.g("status").int() as u16;
    let word = status_word(code);
    let code_text = if code == 0 { "---".to_string() } else { code.to_string() };
    let mut line = format!(
        "{}  {} {:<26} {}",
        paint(entry.g("t").as_str().unwrap_or(""), "dim"),
        paint(&format!("{:<6}", entry.g("method").pystr()), "bright"),
        entry.g("path").pystr(),
        paint(&code_text, if code != 0 { status_color(code) } else { "red" })
    );
    if !word.is_empty() {
        line += &format!(" {}", paint(&format!("{word:<14}"), "dim"));
    }
    line += &format!(" {}", paint(&format!("{:>6}ms", entry.g("ms").int()), "dim"));
    if entry.g("model").truthy() {
        line += &format!("  {}", paint(&entry.g("model").pystr(), "cyan"));
        let tokens = entry.g("tokens_in").int() + entry.g("tokens_out").int();
        if tokens != 0 {
            line += &paint(&format!(" {tokens} tok"), "dim");
        }
    }
    if entry.g("client").as_str() != Some("local") {
        line += &paint(&format!("  [{}]", entry.g("client").pystr()), "magenta");
    }
    if entry.g("error").truthy() {
        let error: String = entry.g("error").pystr().chars().take(120).collect();
        line += &format!("  {}", paint(&error, "red"));
    }
    emit_err(&format!("{}{line}", paint("[7q9k route] ", "dim")));
}

pub fn print_banner(port: u16, host: &str, key: &str, models: &[String]) {
    let mut shown = models.iter().take(5).cloned().collect::<Vec<_>>().join(", ");
    if models.len() > 5 {
        shown.push_str(" …");
    }
    emit("");
    emit(&format!(
        "{}   {}",
        paint("7q9k route", "bright"),
        paint(&format!("v{APP_VERSION} · OpenCode free models · OpenAI-compatible"), "dim")
    ));
    emit(&format!("  {}{}", paint("Dashboard ", "cyan"), paint(&format!("http://localhost:{port}/"), "green")));
    if host == "0.0.0.0" || host == "::" {
        emit(&format!(
            "  {}{}{}",
            paint("Network   ", "cyan"),
            paint(&format!("http://{}:{port}/", lan_ip()), "green"),
            paint("  (LAN access requires the API key)", "dim")
        ));
    }
    emit(&format!("  {}{}", paint("Base URL  ", "cyan"), paint(&format!("http://127.0.0.1:{port}/v1"), "bright")));
    emit(&format!("  {}{}", paint("API-Key   ", "cyan"), paint(key, "yellow")));
    emit(&format!("  {}{}", paint("Models    ", "cyan"), paint(&format!("{} available: {shown}", models.len()), "bright")));
    let stop_hint = if logging_to_file() { "dashboard > Settings > Stop server" } else { "Ctrl+C or close this window" };
    emit(&format!("  {}{}", paint("Stop      ", "cyan"), paint(stop_hint, "dim")));
    emit("");
}

pub fn print_window_banner(port: u16) {
    let urls = public_urls(port);
    emit("");
    emit("  7q9k route is running");
    emit("");
    emit(&format!("  Dashboard        http://localhost:{port}"));
    emit(&format!("  Base URL         {}/v1", urls.g("local").pystr()));
    if let Some(network) = urls.g("network").as_str() {
        emit(&format!("  On your network  {network}"));
    }
    emit("");
    emit("  This window closes in a few seconds. The server keeps running");
    emit("  in the background. Stop it in the dashboard under Settings.");
    emit("");
}

// ---- Serving ------------------------------------------------------------------

/// Binds the listening socket, retrying until `wait` seconds have passed.
pub fn bind(host: &str, port: u16, wait: f64) -> io::Result<TcpListener> {
    let deadline = Instant::now() + Duration::from_secs_f64(wait);
    loop {
        match config::bind_listener(host, port) {
            Ok(listener) => return Ok(listener),
            Err(e) if Instant::now() >= deadline => return Err(e),
            Err(_) => thread::sleep(Duration::from_millis(300)),
        }
    }
}

/// Stops `serve`: sets the flag and wakes the blocking accept.
pub fn shutdown() {
    STOP.store(true, Ordering::SeqCst);
    let (host, port) = {
        let b = bound();
        (b.host.clone(), b.port)
    };
    let host = match host.as_str() {
        "0.0.0.0" | "" => "127.0.0.1".to_string(),
        "::" => "::1".to_string(),
        other => other.to_string(),
    };
    if let Ok(addr) = config::bind_addr(&host, port) {
        let _ = TcpStream::connect_timeout(&addr, Duration::from_secs(2));
    }
}

pub fn stopping() -> bool {
    STOP.load(Ordering::SeqCst)
}

/// Serves until `shutdown` is called; one thread per connection.
pub fn serve(listener: TcpListener) {
    for stream in listener.incoming() {
        if STOP.load(Ordering::SeqCst) {
            break;
        }
        match stream {
            Ok(stream) => {
                thread::spawn(move || handle_connection(stream));
            }
            Err(_) => thread::sleep(Duration::from_millis(10)),
        }
    }
}

/// Whether a 7q9k route already answers on `port`.
pub fn probe_running(port: u16) -> bool {
    let agent = ureq::AgentBuilder::new().timeout(Duration::from_secs_f64(1.5)).build();
    let Ok(response) = agent.get(&format!("http://127.0.0.1:{port}/health")).call() else { return false };
    let Ok(data) = serde_json::from_reader::<_, Value>(response.into_reader()) else { return false };
    data.g("ok") == &Value::Bool(true) && data.has("version")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paths_and_hosts() {
        assert_eq!(url_path("/v1/models?x=1"), "/v1/models");
        assert_eq!(url_path("http://localhost:8787/health"), "/health");
        assert_eq!(url_path("//evil/x"), "/x");
        assert_eq!(host_name("localhost:8787"), "localhost");
        assert_eq!(host_name("[::1]:8787"), "::1");
        assert_eq!(percent_decode("fonts%2Fgeist.woff2"), "fonts/geist.woff2");
        assert_eq!(percent_decode("a%2"), "a%2");
    }
}
