//! Starts the real binary against a fake upstream and talks to it over HTTP.

use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::process::{Child, Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

const KEY: &str = "sk-test-e2e-key";

fn free_port() -> u16 {
    TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port()
}

fn sse(events: &[Value]) -> String {
    let mut out: String = events.iter().map(|e| format!("data: {e}\n\n")).collect();
    out.push_str("data: [DONE]\n\n");
    out
}

/// A tiny upstream: the model lists and a streaming chat endpoint.
fn fake_upstream() -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    thread::spawn(move || {
        for stream in listener.incoming().flatten() {
            thread::spawn(move || serve_fake(stream));
        }
    });
    port
}

fn serve_fake(stream: TcpStream) {
    let mut reader = BufReader::new(stream.try_clone().unwrap());
    let mut out = stream;
    let mut line = String::new();
    if reader.read_line(&mut line).unwrap_or(0) == 0 {
        return;
    }
    let path = line.split_whitespace().nth(1).unwrap_or("").to_string();
    let mut length = 0;
    loop {
        let mut header = String::new();
        reader.read_line(&mut header).unwrap();
        if header.trim().is_empty() {
            break;
        }
        if let Some((k, v)) = header.split_once(':') {
            if k.eq_ignore_ascii_case("content-length") {
                length = v.trim().parse().unwrap();
            }
        }
    }
    let mut body = vec![0; length];
    reader.read_exact(&mut body).unwrap();
    let (content_type, data) = match path.as_str() {
        "/zen/v1/models" => ("application/json", json!({"data": [{"id": "big-pickle"}, {"id": "other-free"}]}).to_string()),
        "/api.json" => {
            let free = json!({"tool_call": true, "cost": {"input": 0, "output": 0}});
            ("application/json", json!({"opencode": {"models": {"big-pickle": free, "other-free": free}}}).to_string())
        }
        "/chat" => {
            let req: Value = serde_json::from_slice(&body).unwrap();
            let model = req["model"].as_str().unwrap_or("").to_string();
            let tools = req.get("tools").is_some_and(|t| t.as_array().is_some_and(|a| !a.is_empty()));
            let mut events = vec![
                json!({"model": model, "choices": [{"index": 0, "delta": {"role": "assistant", "content": "Hello"}}]}),
                json!({"model": model, "choices": [{"index": 0, "delta": {"content": " ünïcode"}}]}),
            ];
            if tools {
                events.push(json!({"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\":"}}]}}]}));
                events.push(json!({"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {"arguments": "\"Berlin\"}"}}]}}]}));
                events.push(json!({"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}));
            } else {
                events.push(json!({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}));
            }
            events.push(json!({"choices": [], "usage": {"prompt_tokens": 11, "completion_tokens": 3, "total_tokens": 14}}));
            ("text/event-stream", sse(&events))
        }
        _ => {
            let _ = out.write_all(b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n");
            return;
        }
    };
    let head = format!("HTTP/1.1 200 OK\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", data.len());
    let _ = out.write_all(head.as_bytes());
    let _ = out.write_all(data.as_bytes());
}

struct Route {
    child: Child,
    port: u16,
    data_dir: std::path::PathBuf,
}

impl Drop for Route {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = std::fs::remove_dir_all(&self.data_dir);
    }
}

fn start_route() -> Route {
    let upstream = fake_upstream();
    let port = free_port();
    let data_dir = std::env::temp_dir().join(format!("7q9k-e2e-{port}"));
    let _ = std::fs::remove_dir_all(&data_dir);
    let base = format!("http://127.0.0.1:{upstream}");
    let child = Command::new(env!("CARGO_BIN_EXE_7q9k-route"))
        .args(["--port", &port.to_string(), "--host", "127.0.0.1", "--no-browser", "--key", KEY])
        .env("ROUTE7Q9K_DATA_DIR", &data_dir)
        .env("ROUTE7Q9K_ZEN_MODELS_URL", format!("{base}/zen/v1/models"))
        .env("ROUTE7Q9K_MODELS_DEV_URL", format!("{base}/api.json"))
        .env("ROUTE7Q9K_CHAT_URL", format!("{base}/chat"))
        .env("ROUTE7Q9K_RESPONSES_URL", format!("{base}/responses"))
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    let route = Route { child, port, data_dir };
    let deadline = Instant::now() + Duration::from_secs(20);
    while TcpStream::connect(("127.0.0.1", port)).is_err() {
        assert!(Instant::now() < deadline, "route did not start");
        thread::sleep(Duration::from_millis(100));
    }
    route
}

/// Sends a raw request and returns (status, headers, body).
fn raw(port: u16, method: &str, path: &str, headers: &[(&str, &str)], body: Option<&Value>) -> (u16, String, String) {
    let mut stream = TcpStream::connect(("127.0.0.1", port)).unwrap();
    stream.set_read_timeout(Some(Duration::from_secs(30))).unwrap();
    let data = body.map(|b| b.to_string()).unwrap_or_default();
    let mut req = format!("{method} {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nConnection: close\r\n");
    for (k, v) in headers {
        req.push_str(&format!("{k}: {v}\r\n"));
    }
    if body.is_some() {
        req.push_str(&format!("Content-Type: application/json\r\nContent-Length: {}\r\n", data.len()));
    }
    req.push_str("\r\n");
    req.push_str(&data);
    stream.write_all(req.as_bytes()).unwrap();
    let mut response = Vec::new();
    let _ = stream.read_to_end(&mut response);
    let text = String::from_utf8_lossy(&response).into_owned();
    let (head, body) = text.split_once("\r\n\r\n").unwrap_or((&text, ""));
    let status = head.split_whitespace().nth(1).unwrap_or("0").parse().unwrap();
    let body = if head.to_lowercase().contains("transfer-encoding: chunked") { dechunk(body) } else { body.to_string() };
    (status, head.to_string(), body)
}

fn dechunk(mut body: &str) -> String {
    let mut out = String::new();
    while let Some((size, rest)) = body.split_once("\r\n") {
        let n = usize::from_str_radix(size.trim(), 16).unwrap_or(0);
        if n == 0 {
            break;
        }
        out.push_str(&rest[..n]);
        body = &rest[n + 2..];
    }
    out
}

fn get(port: u16, path: &str) -> (u16, Value) {
    let (status, _, body) = raw(port, "GET", path, &[], None);
    (status, serde_json::from_str(&body).unwrap_or(Value::Null))
}

fn post(port: u16, path: &str, body: Value) -> (u16, String) {
    let (status, _, body) = raw(port, "POST", path, &[], Some(&body));
    (status, body)
}

fn chat(content: &str) -> Value {
    json!([{"role": "user", "content": content}])
}

#[test]
fn end_to_end() {
    let route = start_route();
    let port = route.port;

    // Open endpoints.
    let (status, health) = get(port, "/health");
    assert_eq!(status, 200);
    assert_eq!(health["ok"], true);
    let (_, models) = get(port, "/v1/models");
    let ids: Vec<&str> = models["data"].as_array().unwrap().iter().map(|m| m["id"].as_str().unwrap()).collect();
    assert_eq!(ids, ["big-pickle", "other-free"]);
    let (_, tags) = get(port, "/api/tags");
    assert_eq!(tags["models"].as_array().unwrap().len(), 2);
    let (status, _, page) = raw(port, "GET", "/", &[], None);
    assert_eq!(status, 200);
    assert!(page.contains("<html"), "dashboard page");
    let (status, head, _) = raw(port, "GET", "/favicon.ico", &[], None);
    assert_eq!(status, 200);
    assert!(head.contains("image/"), "{head}");

    // Errors.
    assert_eq!(get(port, "/nope").0, 404);
    assert_eq!(raw(port, "GET", "/v1/chat/completions", &[], None).0, 405);
    assert_eq!(raw(port, "OPTIONS", "/v1/models", &[], None).0, 204);
    let (status, _, body) = raw(port, "POST", "/v1/chat/completions", &[], Some(&json!([1])));
    assert_eq!(status, 400, "{body}");
    let (status, body) = post(port, "/v1/chat/completions", json!({"model": "nope", "messages": chat("x")}));
    assert_eq!(status, 404, "{body}");

    // Access rules: a forwarded request needs the key, admin stays local.
    let proxied = [("X-Forwarded-For", "10.0.0.5")];
    let body = json!({"model": "big-pickle", "messages": chat("hi")});
    assert_eq!(raw(port, "POST", "/v1/chat/completions", &proxied, Some(&body)).0, 401);
    let auth = format!("Bearer {KEY}");
    let with_key = [("X-Forwarded-For", "10.0.0.5"), ("Authorization", auth.as_str())];
    assert_eq!(raw(port, "POST", "/v1/chat/completions", &with_key, Some(&body)).0, 200);
    assert_eq!(raw(port, "GET", "/api/dashboard/settings", &with_key, None).0, 403);
    assert_eq!(raw(port, "GET", "/api/dashboard/settings", &[("Origin", "http://evil.example")], None).0, 403);
    assert_eq!(get(port, "/api/dashboard/settings").0, 200);

    // OpenAI chat completions.
    let (status, text) = post(port, "/v1/chat/completions", body.clone());
    assert_eq!(status, 200);
    let completion: Value = serde_json::from_str(&text).unwrap();
    assert_eq!(completion["choices"][0]["message"]["content"], "Hello ünïcode");
    assert_eq!(completion["usage"]["prompt_tokens"], 11);
    let (status, text) = post(port, "/v1/chat/completions", json!({"model": "big-pickle", "messages": chat("hi"), "stream": true}));
    assert_eq!(status, 200);
    assert!(text.contains("chat.completion.chunk") && text.ends_with("data: [DONE]\n\n"), "{text}");
    assert!(text.contains("\\u00fc"), "stream should be ASCII-escaped: {text}");
    let tools = json!([{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}}]);
    let (_, text) = post(port, "/v1/chat/completions", json!({"model": "big-pickle", "messages": chat("weather"), "tools": tools}));
    let completion: Value = serde_json::from_str(&text).unwrap();
    assert_eq!(completion["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"], "{\"city\":\"Berlin\"}");

    // Responses API.
    let (status, text) = post(port, "/v1/responses", json!({"model": "big-pickle", "input": "hi"}));
    assert_eq!(status, 200, "{text}");
    assert!(text.contains("Hello \\u00fcn\\u00efcode") || text.contains("Hello ünïcode"), "{text}");
    let (_, text) = post(port, "/v1/responses", json!({"model": "big-pickle", "input": "hi", "stream": true}));
    assert!(text.contains("event: response.completed"), "{text}");

    // Anthropic Messages API, with a Claude model name mapped to the default.
    let claude = json!({"model": "claude-sonnet-4-5", "max_tokens": 100, "messages": chat("hi")});
    let (status, text) = post(port, "/v1/messages", claude.clone());
    assert_eq!(status, 200, "{text}");
    let message: Value = serde_json::from_str(&text).unwrap();
    assert_eq!(message["type"], "message");
    assert_eq!(message["content"][0]["text"], "Hello ünïcode");
    let mut streaming = claude.clone();
    streaming["stream"] = json!(true);
    let (_, text) = post(port, "/v1/messages", streaming);
    assert!(text.contains("event: message_stop"), "{text}");
    let (_, text) = post(port, "/v1/messages/count_tokens", claude);
    assert!(serde_json::from_str::<Value>(&text).unwrap()["input_tokens"].as_i64().unwrap() > 0);
    let (status, text) = post(port, "/v1/messages", json!({"model": "big-pickle", "messages": "x"}));
    assert_eq!(status, 400);
    assert_eq!(serde_json::from_str::<Value>(&text).unwrap()["type"], "error");

    // Ollama API.
    let (status, text) = post(port, "/api/chat", json!({"model": "big-pickle", "messages": chat("hi")}));
    assert_eq!(status, 200);
    let lines: Vec<Value> = text.lines().map(|l| serde_json::from_str(l).unwrap()).collect();
    assert_eq!(lines.last().unwrap()["done"], true);
    assert_eq!(lines.last().unwrap()["eval_count"], 3);
    let (_, text) = post(port, "/api/chat", json!({"model": "big-pickle", "messages": chat("hi"), "stream": false}));
    assert_eq!(serde_json::from_str::<Value>(&text).unwrap()["message"]["content"], "Hello ünïcode");
    let (_, text) = post(port, "/api/generate", json!({"model": "big-pickle", "prompt": "hi", "stream": false}));
    assert_eq!(serde_json::from_str::<Value>(&text).unwrap()["response"], "Hello ünïcode");
    let (status, text) = post(port, "/api/show", json!({"name": "big-pickle"}));
    assert_eq!(status, 200, "{text}");

    // Dashboard.
    let (_, state) = get(port, "/api/dashboard");
    assert_eq!(state["api_key"], KEY);
    assert_eq!(state["default_model"], "big-pickle");
    let (status, text) = post(port, "/api/dashboard/test", json!({"model": "big-pickle"}));
    assert_eq!(status, 200, "{text}");
    let (_, stats) = get(port, "/api/dashboard/stats");
    assert!(stats["requests"].as_i64().unwrap() >= 10, "{stats}");
    assert!(stats["tokens_in"].as_i64().unwrap() >= 11);
    assert!(stats["log"].as_array().unwrap().iter().any(|e| e["status"] == 401));
    let (status, _) = post(port, "/api/dashboard/save", json!({}));
    assert_eq!(status, 200);
    assert!(route.data_dir.join("stats.json").is_file());
    let (status, text) = post(port, "/api/dashboard/clear-errors", json!({}));
    assert_eq!(status, 200, "{text}");
    let (_, stats) = get(port, "/api/dashboard/stats");
    assert_eq!(stats["errors"], 0, "{stats}");
    assert!(stats["requests"].as_i64().unwrap() >= 10, "{stats}");
    assert!(stats["log"].as_array().unwrap().iter().all(|e| e["status"].as_i64().unwrap_or(0) < 400), "{stats}");
    let (status, text) = post(port, "/api/dashboard/key", json!({"key": "bad key"}));
    assert_eq!(status, 400, "{text}");
    let (status, text) = post(port, "/api/dashboard/settings", json!({"port": 80}));
    assert_eq!(status, 400, "{text}");

    // Stop from the dashboard.
    let (status, text) = post(port, "/api/dashboard/stop", json!({}));
    assert_eq!(status, 200, "{text}");
    let mut route = route;
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        if let Some(code) = route.child.try_wait().unwrap() {
            assert!(code.success());
            break;
        }
        assert!(Instant::now() < deadline, "route did not stop");
        thread::sleep(Duration::from_millis(100));
    }
}
