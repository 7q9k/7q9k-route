//! Anthropic Messages API, used by Claude Code. Requests are turned into Chat
//! Completions for OpenCode and the answer is turned back into Anthropic blocks.

use super::chat::{usage, validate_messages};
use super::ollama::json_obj;
use crate::util::{is_number, pydumps, set, token_hex, J, PResult, ProxyError};
use regex::Regex;
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::sync::OnceLock;

/// Thinking blocks need a signature; Claude Code sends it back but nothing checks it here.
pub const THINKING_SIGNATURE: &str = "7q9k-route";
/// Model names Claude Code uses on its own (aliases, background tasks).
pub const CLAUDE_MODEL_ALIASES: [&str; 6] = ["opus", "sonnet", "haiku", "default", "best", "opusplan"];
pub const ANTHROPIC_MODEL_PREFIX: &str = "anthropic-route-";

pub fn stop_reason(reason: &str) -> &'static str {
    match reason {
        "stop" => "end_turn",
        "length" => "max_tokens",
        "tool_calls" | "function_call" => "tool_use",
        "content_filter" => "refusal",
        _ => "end_turn",
    }
}

pub fn error_type(status: u16) -> &'static str {
    match status {
        400 => "invalid_request_error",
        401 => "authentication_error",
        403 => "permission_error",
        404 => "not_found_error",
        413 => "request_too_large",
        429 => "rate_limit_error",
        _ => "api_error",
    }
}

fn py_json(v: &Value) -> String {
    pydumps(v, false, ", ", ": ")
}

fn blocks(content: &Value) -> Vec<Value> {
    match content {
        Value::String(s) => vec![json!({"type": "text", "text": s})],
        Value::Array(items) => items
            .iter()
            .filter_map(|b| match b {
                Value::String(s) => Some(json!({"type": "text", "text": s})),
                Value::Object(_) => Some(b.clone()),
                _ => None,
            })
            .collect(),
        _ => Vec::new(),
    }
}

fn text_of(content: &Value) -> String {
    blocks(content)
        .iter()
        .filter(|b| b.g("type") == "text" && b.g("text").truthy())
        .map(|b| b.g("text").pystr())
        .collect::<Vec<_>>()
        .join("\n\n")
}

fn image(block: &Value) -> Option<Value> {
    let source = block.g("source");
    let url = if source.g("type") == "base64" && source.g("data").truthy() {
        let media = if source.g("media_type").truthy() { source.g("media_type").pystr() } else { "image/png".into() };
        format!("data:{media};base64,{}", source.g("data").pystr())
    } else if source.g("type") == "url" && source.g("url").truthy() {
        source.g("url").pystr()
    } else {
        return None;
    };
    Some(json!({"type": "image_url", "image_url": {"url": url}}))
}

/// Text for the tool message and images, which tool messages cannot carry.
fn tool_result(block: &Value) -> (String, Vec<Value>) {
    let content = block.g("content");
    let (mut text, images) = if content.is_array() {
        let (mut texts, mut images) = (Vec::new(), Vec::new());
        for part in blocks(content) {
            if part.g("type") == "text" {
                texts.push(part.g("text").or_str());
            } else if part.g("type") == "image" {
                if let Some(img) = image(&part) {
                    images.push(img);
                    texts.push("[image attached below]".to_string());
                }
            }
        }
        (texts.join("\n"), images)
    } else if content.is_null() {
        (String::new(), Vec::new())
    } else {
        (content.pystr(), Vec::new())
    };
    if block.g("is_error").truthy() {
        text = format!("Error: {text}");
    }
    if text.is_empty() {
        text = "(no output)".into();
    }
    (text, images)
}

pub fn to_chat_messages(req: &Value) -> PResult<Vec<Value>> {
    let mut messages = Vec::new();
    let system = text_of(req.g("system"));
    if !system.trim().is_empty() {
        messages.push(json!({"role": "system", "content": system}));
    }
    for message in validate_messages(req.g("messages"))? {
        let blocks = blocks(message.g("content"));
        if message.g("role") == "assistant" {
            let mut texts = String::new();
            let mut calls: Vec<Value> = Vec::new();
            for block in &blocks {
                match block.g("type").as_str() {
                    Some("text") => texts.push_str(&block.g("text").or_str()),
                    Some("tool_use" | "server_tool_use") => {
                        let arguments = if block.g("input").is_null() { json!({}) } else { block.g("input").clone() };
                        let id = if block.g("id").truthy() { block.g("id").pystr() } else { format!("toolu_{}", calls.len()) };
                        calls.push(json!({
                            "id": id,
                            "type": "function",
                            "function": {"name": block.g("name").or_str(), "arguments": py_json(&arguments)},
                        }));
                    }
                    _ => {}
                }
            }
            let content = if texts.is_empty() { Value::Null } else { Value::from(texts) };
            if content.is_null() && calls.is_empty() {
                continue;
            }
            let mut entry = json!({"role": "assistant", "content": content});
            if !calls.is_empty() {
                set(&mut entry, "tool_calls", calls);
            }
            messages.push(entry);
            continue;
        }
        let mut parts: Vec<Value> = Vec::new();
        let mut images: Vec<Value> = Vec::new();
        for block in &blocks {
            match block.g("type").as_str() {
                Some("tool_result") => {
                    let (text, found) = tool_result(block);
                    messages.push(json!({"role": "tool", "tool_call_id": block.g("tool_use_id").or_str(), "content": text}));
                    images.extend(found);
                }
                Some("text") if block.g("text").truthy() => parts.push(json!({"type": "text", "text": block.g("text").pystr()})),
                Some("image") => parts.extend(image(block)),
                Some("document") => {
                    let source = block.g("source");
                    if source.g("type") == "text" && source.g("data").truthy() {
                        parts.push(json!({"type": "text", "text": source.g("data").pystr()}));
                    }
                }
                _ => {}
            }
        }
        images.extend(parts);
        let parts = images;
        if !parts.is_empty() {
            let content = if parts.iter().all(|p| p["type"] == "text") {
                Value::from(parts.iter().map(|p| p["text"].pystr()).collect::<Vec<_>>().join("\n\n"))
            } else {
                Value::Array(parts)
            };
            messages.push(json!({"role": "user", "content": content}));
        }
    }
    Ok(messages)
}

pub fn tools(tools: &Value) -> Vec<Value> {
    let mut result = Vec::new();
    for tool in tools.items() {
        // Typed tools (web_search_..., text_editor_...) run on Anthropic's side.
        let kind = tool.g("type");
        if !tool.is_object() || !tool.g("name").truthy() || !(kind.is_null() || kind == "custom") {
            continue;
        }
        let schema = tool.g("input_schema");
        result.push(json!({
            "type": "function",
            "function": {
                "name": tool.g("name").pystr(),
                "description": tool.g("description").or_str(),
                "parameters": if schema.is_object() { schema.clone() } else { json!({"type": "object", "properties": {}}) },
            },
        }));
    }
    result
}

fn tool_choice(choice: &Value) -> Option<Value> {
    if !choice.is_object() {
        return None;
    }
    let kind = choice.g("type");
    if kind == "tool" && choice.g("name").truthy() {
        return Some(json!({"type": "function", "function": {"name": choice.g("name").pystr()}}));
    }
    match kind.as_str() {
        Some("auto") => Some(json!("auto")),
        Some("any") => Some(json!("required")),
        Some("none") => Some(json!("none")),
        _ => None,
    }
}

pub fn to_chat_request(req: &Value) -> PResult<Value> {
    // max_tokens is not passed on: Claude Code asks for more than most free models allow.
    let mut chat = json!({"model": req.g("model").clone(), "messages": to_chat_messages(req)?});
    let tools = tools(req.g("tools"));
    if !tools.is_empty() {
        set(&mut chat, "tools", tools);
        if let Some(choice) = tool_choice(req.g("tool_choice")) {
            set(&mut chat, "tool_choice", choice);
        }
        if req.g("tool_choice").g("disable_parallel_tool_use").truthy() {
            set(&mut chat, "parallel_tool_calls", false);
        }
    }
    for key in ["temperature", "top_p"] {
        if is_number(req.g(key)) {
            set(&mut chat, key, req.g(key).clone());
        }
    }
    let stops: Vec<Value> = req.g("stop_sequences").items().iter().take(4).map(|s| Value::from(s.pystr())).collect();
    if !stops.is_empty() {
        set(&mut chat, "stop", stops);
    }
    Ok(chat)
}

pub fn wants_thinking(req: &Value) -> bool {
    matches!(req.g("thinking").g("type").as_str(), Some("enabled" | "adaptive"))
}

pub fn estimate_tokens(req: &Value) -> i64 {
    // Rough count (about 4 characters per token); OpenCode has no counting endpoint.
    let text = pydumps(
        &json!([req.g("system").clone(), req.g("messages").clone(), req.g("tools").clone()]),
        false,
        ",",
        ":",
    );
    (text.chars().count() as i64 / 4).max(1)
}

fn anthropic_usage(raw: &Value) -> Value {
    let u = usage(raw);
    let cached = u["prompt_tokens_details"]["cached_tokens"].int();
    json!({
        "input_tokens": (u["prompt_tokens"].int() - cached).max(0),
        "output_tokens": u["completion_tokens"].int(),
        "cache_read_input_tokens": cached,
        "cache_creation_input_tokens": 0,
    })
}

fn tool_id(value: &Value) -> String {
    static BAD: OnceLock<Regex> = OnceLock::new();
    let bad = BAD.get_or_init(|| Regex::new(r"[^A-Za-z0-9_-]").unwrap());
    let raw = if value.truthy() { value.pystr() } else { String::new() };
    let text = bad.replace_all(&raw, "_").into_owned();
    if text.is_empty() { format!("toolu_{}", token_hex(12)) } else { text }
}

pub fn message(completion: &Value, thinking: bool) -> Value {
    let choice = completion.g("choices").items().first().cloned().unwrap_or(Value::Null);
    let message = choice.g("message");
    let mut content = Vec::new();
    let reasoning = if message.g("reasoning_content").truthy() { message.g("reasoning_content") } else { message.g("reasoning") };
    if thinking && reasoning.truthy() {
        content.push(json!({"type": "thinking", "thinking": reasoning.pystr(), "signature": THINKING_SIGNATURE}));
    }
    if message.g("content").truthy() {
        content.push(json!({"type": "text", "text": message.g("content").pystr()}));
    }
    for call in message.g("tool_calls").items() {
        let function = call.g("function");
        content.push(json!({
            "type": "tool_use",
            "id": tool_id(call.g("id")),
            "name": function.g("name").or_str(),
            "input": json_obj(function.g("arguments")),
        }));
    }
    let reason = if message.g("tool_calls").truthy() {
        "tool_use"
    } else {
        let finish = if choice.g("finish_reason").truthy() { choice.g("finish_reason").pystr() } else { "stop".into() };
        stop_reason(&finish)
    };
    json!({
        "id": format!("msg_{}", token_hex(12)),
        "type": "message",
        "role": "assistant",
        "model": completion.g("model").clone(),
        "content": content,
        "stop_reason": reason,
        "stop_sequence": null,
        "usage": anthropic_usage(completion.g("usage")),
    })
}

pub type SseEvent = (String, Value);

#[derive(Default)]
struct Call {
    id: String,
    name: String,
    arguments: String,
}

/// Turns Chat Completions deltas into Anthropic stream events.
///
/// Text and thinking stream as they arrive; tool calls are sent whole at the
/// end, because OpenAI-style deltas may interleave several calls.
pub struct AnthropicStream {
    pub model: String,
    thinking: bool,
    input_tokens: i64,
    pub id: String,
    index: i64,
    open: Option<&'static str>,
    calls: BTreeMap<i64, Call>,
    finish_reason: String,
    usage: Option<Value>,
}

fn event(kind: &str, fields: Value) -> SseEvent {
    let mut data = json!({"type": kind});
    if let (Value::Object(target), Value::Object(extra)) = (&mut data, fields) {
        target.extend(extra);
    }
    (kind.to_string(), data)
}

impl AnthropicStream {
    pub fn new(model: &str, thinking: bool, input_tokens: i64) -> Self {
        AnthropicStream {
            model: model.to_string(),
            thinking,
            input_tokens,
            id: format!("msg_{}", token_hex(12)),
            index: -1,
            open: None,
            calls: BTreeMap::new(),
            finish_reason: "stop".into(),
            usage: None,
        }
    }

    pub fn start(&self) -> Vec<SseEvent> {
        let message = json!({
            "id": self.id,
            "type": "message",
            "role": "assistant",
            "model": self.model,
            "content": [],
            "stop_reason": null,
            "stop_sequence": null,
            "usage": {"input_tokens": self.input_tokens, "output_tokens": 0},
        });
        vec![event("message_start", json!({"message": message})), event("ping", json!({}))]
    }

    fn close(&mut self) -> Vec<SseEvent> {
        let Some(open) = self.open.take() else { return Vec::new() };
        let mut events = Vec::new();
        if open == "thinking" {
            events.push(event(
                "content_block_delta",
                json!({"index": self.index, "delta": {"type": "signature_delta", "signature": THINKING_SIGNATURE}}),
            ));
        }
        events.push(event("content_block_stop", json!({"index": self.index})));
        events
    }

    fn open_block(&mut self, kind: &'static str, block: Value) -> Vec<SseEvent> {
        let mut events = self.close();
        self.index += 1;
        self.open = Some(kind);
        events.push(event("content_block_start", json!({"index": self.index, "content_block": block})));
        events
    }

    fn delta(&mut self, kind: &'static str, delta: Value) -> Vec<SseEvent> {
        let mut events = Vec::new();
        if self.open != Some(kind) {
            let block = if kind == "thinking" {
                json!({"type": "thinking", "thinking": "", "signature": ""})
            } else {
                json!({"type": "text", "text": ""})
            };
            events = self.open_block(kind, block);
        }
        events.push(event("content_block_delta", json!({"index": self.index, "delta": delta})));
        events
    }

    pub fn feed(&mut self, ev: &Value) -> PResult<Vec<SseEvent>> {
        if ev.g("error").truthy() {
            return Err(ProxyError::upstream(format!("upstream inference error: {}", ev.g("error").pystr())));
        }
        if ev.g("model").truthy() {
            self.model = ev.g("model").pystr();
        }
        if ev.g("usage").is_object() {
            self.usage = Some(ev.g("usage").clone());
        }
        let mut events = Vec::new();
        for choice in ev.g("choices").items() {
            if choice.g("finish_reason").truthy() {
                self.finish_reason = choice.g("finish_reason").pystr();
            }
            let delta = choice.g("delta");
            let thought = if delta.g("reasoning_content").truthy() { delta.g("reasoning_content") } else { delta.g("reasoning") };
            if self.thinking && thought.truthy() {
                events.extend(self.delta("thinking", json!({"type": "thinking_delta", "thinking": thought.pystr()})));
            }
            if delta.g("content").truthy() {
                events.extend(self.delta("text", json!({"type": "text_delta", "text": delta.g("content").pystr()})));
            }
            for raw in delta.g("tool_calls").items() {
                let item = self.calls.entry(raw.g("index").int()).or_default();
                if raw.g("id").truthy() {
                    item.id = raw.g("id").pystr();
                }
                let function = raw.g("function");
                if function.g("name").truthy() {
                    item.name = function.g("name").pystr();
                }
                item.arguments.push_str(&function.g("arguments").or_str());
            }
        }
        Ok(events)
    }

    pub fn finish(&mut self) -> PResult<Vec<SseEvent>> {
        if self.finish_reason == "content_filter" && self.calls.is_empty() && self.open.is_none() {
            return Err(ProxyError::new("upstream blocked the response by content filter.", 400));
        }
        let mut events = self.close();
        let calls: Vec<Call> = std::mem::take(&mut self.calls).into_values().filter(|c| !c.name.is_empty()).collect();
        for call in &calls {
            events.extend(self.open_block(
                "tool_use",
                json!({"type": "tool_use", "id": tool_id(&Value::from(call.id.clone())), "name": call.name, "input": {}}),
            ));
            let partial = py_json(&json_obj(&Value::from(call.arguments.clone())));
            events.push(event(
                "content_block_delta",
                json!({"index": self.index, "delta": {"type": "input_json_delta", "partial_json": partial}}),
            ));
            events.extend(self.close());
        }
        let reason = if calls.is_empty() { stop_reason(&self.finish_reason) } else { "tool_use" };
        let usage = match &self.usage {
            Some(u) => anthropic_usage(u),
            None => json!({"input_tokens": self.input_tokens, "output_tokens": 0}),
        };
        events.push(event("message_delta", json!({"delta": {"stop_reason": reason, "stop_sequence": null}, "usage": usage})));
        events.push(event("message_stop", json!({})));
        Ok(events)
    }

    pub fn fail(message: &str, status: u16) -> Vec<SseEvent> {
        vec![(
            "error".to_string(),
            json!({"type": "error", "error": {"type": error_type(status), "message": message}}),
        )]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_conversion() {
        let req = json!({
            "model": "claude-x",
            "system": [{"type": "text", "text": "sys"}],
            "max_tokens": 32000,
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": [{"type": "text", "text": "ok"},
                    {"type": "tool_use", "id": "t1", "name": "Read", "input": {"path": "ä"}}]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "is_error": true, "content": [{"type": "text", "text": "boom"}]},
                    {"type": "image", "source": {"type": "base64", "data": "AAA"}},
                    {"type": "text", "text": "next"}]},
            ],
            "tools": [{"name": "Read", "input_schema": {"type": "object"}}, {"type": "web_search_20250305", "name": "web"}],
            "tool_choice": {"type": "any", "disable_parallel_tool_use": true},
            "temperature": 0.5,
            "stop_sequences": ["a", "b", "c", "d", "e"],
        });
        let chat = to_chat_request(&req).unwrap();
        let m = chat["messages"].items();
        assert_eq!(m[0]["content"], "sys");
        assert_eq!(m[2]["tool_calls"][0]["function"]["arguments"], "{\"path\": \"ä\"}");
        assert_eq!(m[3]["content"], "Error: boom");
        assert_eq!(m[4]["content"][0]["image_url"]["url"], "data:image/png;base64,AAA");
        assert_eq!(chat["tools"].items().len(), 1);
        assert_eq!(chat["tool_choice"], "required");
        assert_eq!(chat["parallel_tool_calls"], false);
        assert_eq!(chat["stop"].items().len(), 4);
        assert!(chat.get("max_tokens").is_none());
    }

    #[test]
    fn stream_events() {
        let mut s = AnthropicStream::new("m", true, 7);
        let mut events = s.start();
        events.extend(s.feed(&json!({"choices": [{"delta": {"reasoning": "hm"}}]})).unwrap());
        events.extend(s.feed(&json!({"choices": [{"delta": {"content": "Hi"}}]})).unwrap());
        events.extend(s.feed(&json!({"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c.1", "function": {"name": "f", "arguments": "{\"a\":"}}]}}]})).unwrap());
        events.extend(s.feed(&json!({"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "1}"}}]}, "finish_reason": "tool_calls"}]})).unwrap());
        events.extend(s.finish().unwrap());
        let kinds: Vec<&str> = events.iter().map(|(k, _)| k.as_str()).collect();
        assert_eq!(kinds[..2], ["message_start", "ping"]);
        assert_eq!(kinds.last(), Some(&"message_stop"));
        let sig = events.iter().find(|(_, d)| d["delta"]["type"] == "signature_delta");
        assert!(sig.is_some());
        let tool = events.iter().find(|(_, d)| d["content_block"]["type"] == "tool_use").unwrap();
        assert_eq!(tool.1["content_block"]["id"], "c_1");
        assert_eq!(tool.1["index"], 2);
        let partial = events.iter().find(|(_, d)| d["delta"]["type"] == "input_json_delta").unwrap();
        assert_eq!(partial.1["delta"]["partial_json"], "{\"a\": 1}");
        let delta = events.iter().find(|(k, _)| k == "message_delta").unwrap();
        assert_eq!(delta.1["delta"]["stop_reason"], "tool_use");
        assert_eq!(delta.1["usage"]["input_tokens"], 7);
    }

    #[test]
    fn message_from_completion() {
        let completion = json!({"model": "m", "choices": [{"message": {"content": "x"}, "finish_reason": "length"}],
                                "usage": {"prompt_tokens": 10, "completion_tokens": 2, "prompt_tokens_details": {"cached_tokens": 4}}});
        let msg = message(&completion, false);
        assert_eq!(msg["stop_reason"], "max_tokens");
        assert_eq!(msg["usage"]["input_tokens"], 6);
        assert_eq!(msg["usage"]["cache_read_input_tokens"], 4);
    }

    #[test]
    fn content_filter_without_output_fails() {
        let mut s = AnthropicStream::new("m", false, 1);
        s.feed(&json!({"choices": [{"delta": {}, "finish_reason": "content_filter"}]})).unwrap();
        assert_eq!(s.finish().unwrap_err().status, 400);
        assert!(estimate_tokens(&json!({"messages": []})) >= 1);
    }
}
