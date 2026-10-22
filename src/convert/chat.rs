//! Upstream request bodies (chat completions and responses) and the merge of
//! streamed chunks into one chat completion.

use crate::upstream::{Events, DEFAULT_MODEL};
use crate::util::{clip, pydumps, dumps, now_secs, session_id, set, token_hex, J, PResult, ProxyError};
use serde_json::{json, Map, Value};
use std::collections::{BTreeMap, HashMap};

pub const COMPAT_DESCRIPTION: &str = "Compatibility marker; unavailable. Never call.";

/// The marker tools OpenCode's free tier expects in every request.
pub fn compat_tool_parameters(name: &str) -> Value {
    match name {
        "bash" => json!({
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "timeout": {"type": "integer", "minimum": 1},
                "workdir": {"type": "string"},
            },
            "required": ["command"],
        }),
        _ => json!({
            "type": "object",
            "properties": {
                "filePath": {"type": "string"},
                "offset": {"type": "integer", "minimum": 1},
                "limit": {"type": "integer", "minimum": 1},
            },
            "required": ["filePath"],
        }),
    }
}

/// `dict.get(key, default)`: the default only when the key is missing.
pub fn get_or(v: &Value, key: &str, default: Value) -> Value {
    match v.as_object().and_then(|o| o.get(key)) {
        Some(found) => found.clone(),
        None => default,
    }
}

fn mesh_tool_names(tools: &[Value]) -> Vec<String> {
    let mut names = Vec::new();
    for tool in tools {
        if !tool.is_object() {
            continue;
        }
        if tool.g("type") == "function" && tool.g("function").is_object() {
            names.push(tool.g("function").g("name").or_str());
        } else if tool.g("name").truthy() {
            names.push(tool.g("name").pystr());
        }
    }
    names
}

pub fn ensure_required_tools(tools: &[Value]) -> Vec<Value> {
    let mut result: Vec<Value> = tools.iter().filter(|t| t.is_object()).cloned().collect();
    let names = mesh_tool_names(&result);
    for required in ["bash", "read"] {
        if !names.iter().any(|n| n == required) {
            result.insert(
                0,
                json!({
                    "type": "function",
                    "function": {
                        "name": required,
                        "description": COMPAT_DESCRIPTION,
                        "parameters": compat_tool_parameters(required),
                    },
                }),
            );
        }
    }
    result
}

pub fn ensure_responses_required_tools(tools: Vec<Value>) -> Vec<Value> {
    let mut tools: Vec<Value> = tools.into_iter().filter(|t| t.is_object()).collect();
    let names: Vec<String> = tools.iter().map(|t| t.g("name").or_str()).collect();
    for required in ["bash", "read"] {
        if !names.iter().any(|n| n == required) {
            tools.insert(
                0,
                json!({
                    "type": "function",
                    "name": required,
                    "description": COMPAT_DESCRIPTION,
                    "parameters": compat_tool_parameters(required),
                    "strict": false,
                }),
            );
        }
    }
    tools
}

const PASSTHROUGH_KEYS: [&str; 11] = [
    "temperature",
    "top_p",
    "max_tokens",
    "stop",
    "presence_penalty",
    "frequency_penalty",
    "seed",
    "response_format",
    "parallel_tool_calls",
    "logit_bias",
    "user",
];

pub fn validate_messages(messages: &Value) -> PResult<Vec<Value>> {
    match messages {
        Value::Null => Ok(Vec::new()),
        Value::Array(items) if items.iter().all(Value::is_object) => Ok(items.clone()),
        _ => Err(ProxyError::new("'messages' must be a list of message objects.", 400)),
    }
}

fn model_or_default(req: &Value) -> Value {
    if req.g("model").truthy() {
        req.g("model").clone()
    } else {
        Value::from(DEFAULT_MODEL)
    }
}

pub fn build_chat_request(req: &Value) -> PResult<Value> {
    let mut messages = validate_messages(req.g("messages"))?;
    if !messages.iter().any(|m| m.g("role") == "system") {
        messages.insert(0, json!({"role": "system", "content": "You are opencode"}));
    }
    let tools = req.g("tools").as_array();
    let mut body = json!({
        "model": model_or_default(req),
        "messages": messages,
        "tools": ensure_required_tools(tools.map(Vec::as_slice).unwrap_or(&[])),
        "stream": true,
        "stream_options": {"include_usage": true},
    });
    for key in PASSTHROUGH_KEYS {
        if !req.g(key).is_null() {
            set(&mut body, key, req.g(key).clone());
        }
    }
    if tools.map(|t| !t.is_empty()).unwrap_or(false) && !req.g("tool_choice").is_null() {
        set(&mut body, "tool_choice", req.g("tool_choice").clone());
    }
    Ok(body)
}

/// The text of a message's content (string, or the text of its parts).
pub fn message_text(content: &Value) -> String {
    match content {
        Value::String(s) => s.clone(),
        Value::Array(parts) => parts
            .iter()
            .filter(|p| p.is_object() && !p.g("text").is_null())
            .map(|p| p.g("text").or_str())
            .collect::<Vec<_>>()
            .join("\n"),
        Value::Null => String::new(),
        other => other.pystr(),
    }
}

pub fn responses_content(content: &Value, output: bool) -> Vec<Value> {
    let text_type = if output { "output_text" } else { "input_text" };
    let Value::Array(parts) = content else {
        let text = message_text(content);
        return if text.is_empty() { Vec::new() } else { vec![json!({"type": text_type, "text": text})] };
    };
    let mut result = Vec::new();
    for part in parts {
        if !part.is_object() {
            result.push(json!({"type": text_type, "text": part.pystr()}));
            continue;
        }
        let kind = part.g("type").or_str();
        if matches!(kind.as_str(), "text" | "input_text" | "output_text") {
            result.push(json!({"type": text_type, "text": part.g("text").or_str()}));
        } else if matches!(kind.as_str(), "image_url" | "input_image") && !output {
            let mut url = part.g("image_url").clone();
            if url.is_object() {
                url = url.g("url").clone();
            }
            if url.truthy() {
                result.push(json!({"type": "input_image", "image_url": url.pystr()}));
            }
        }
    }
    result
}

fn responses_tools(tools: &[Value]) -> Vec<Value> {
    tools
        .iter()
        .map(|tool| {
            let function = tool.g("function");
            let parameters = if function.g("parameters").truthy() { function.g("parameters").clone() } else { json!({"type": "object"}) };
            json!({
                "type": "function",
                "name": function.g("name").or_str(),
                "description": function.g("description").or_str(),
                "parameters": parameters,
                "strict": get_or(function, "strict", Value::Bool(false)).truthy(),
            })
        })
        .collect()
}

fn responses_input(messages: &[Value]) -> (Vec<Value>, String) {
    let mut instructions = vec!["You are opencode".to_string()];
    let mut items = Vec::new();
    for message in messages {
        let role = if message.g("role").truthy() { message.g("role").pystr() } else { "user".to_string() };
        if role == "system" {
            let text = message_text(message.g("content"));
            if !text.is_empty() {
                instructions.push(text);
            }
            continue;
        }
        if role == "tool" {
            let call_id = message.g("tool_call_id").or_str();
            if !call_id.is_empty() {
                items.push(json!({
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": message_text(message.g("content")),
                }));
            }
            continue;
        }
        if role == "assistant" {
            for detail in message.g("reasoning_details").items() {
                if detail.g("type") == "reasoning" {
                    items.push(detail.clone());
                }
            }
            let content = responses_content(message.g("content"), true);
            if !content.is_empty() {
                items.push(json!({"role": "assistant", "content": content}));
            }
            for call in message.g("tool_calls").items() {
                if !call.is_object() {
                    continue;
                }
                let function = call.g("function");
                let arguments = match function.g("arguments") {
                    v if !v.truthy() => "{}".to_string(),
                    Value::String(s) => s.clone(),
                    v => pydumps(v, true, ",", ":"),
                };
                let call_id = if call.g("id").truthy() { call.g("id").pystr() } else { call.g("call_id").or_str() };
                if !call_id.is_empty() {
                    items.push(json!({
                        "type": "function_call",
                        "call_id": call_id,
                        "name": function.g("name").or_str(),
                        "arguments": arguments,
                    }));
                }
            }
            continue;
        }
        let content = responses_content(message.g("content"), false);
        if !content.is_empty() {
            items.push(json!({"role": role, "content": content}));
        }
    }
    (items, instructions.join("\n\n"))
}

pub fn build_responses_request(req: &Value) -> PResult<Value> {
    let tools: Vec<Value> = req.g("tools").items().iter().filter(|t| t.is_object()).cloned().collect();
    let (input, instructions) = responses_input(&validate_messages(req.g("messages"))?);
    let converted = if tools.is_empty() { Vec::new() } else { responses_tools(&tools) };
    let mut body = json!({
        "model": model_or_default(req),
        "input": input,
        "instructions": instructions,
        "tools": ensure_responses_required_tools(converted),
        "stream": true,
        "store": false,
        "include": ["reasoning.encrypted_content"],
        "prompt_cache_key": session_id(),
    });
    for (key, target) in [("temperature", "temperature"), ("top_p", "top_p"), ("max_tokens", "max_output_tokens")] {
        if !req.g(key).is_null() {
            set(&mut body, target, req.g(key).clone());
        }
    }
    if !tools.is_empty() && !req.g("tool_choice").is_null() {
        let mut choice = req.g("tool_choice").clone();
        if choice.g("function").is_object() {
            choice = json!({"type": "function", "name": choice.g("function").g("name").or_str()});
        }
        set(&mut body, "tool_choice", choice);
    }
    Ok(body)
}

fn delta_chunk(model: &str, delta: Value, finish_reason: Value) -> Value {
    json!({"model": model, "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}]})
}

/// Turns Responses API events into chat completion chunks.
pub struct ResponsesDeltas {
    events: Events,
    seen: bool,
    saw_tool_call: bool,
    call_indexes: HashMap<i64, usize>,
    model: String,
    done: bool,
}

pub fn responses_events_to_deltas(events: Events) -> ResponsesDeltas {
    ResponsesDeltas { events, seen: false, saw_tool_call: false, call_indexes: HashMap::new(), model: String::new(), done: false }
}

impl ResponsesDeltas {
    fn convert(&mut self, event: Value) -> PResult<Option<Value>> {
        self.seen = true;
        let kind = event.g("type").or_str();
        let response = if event.g("response").truthy() { event.g("response").clone() } else { json!({}) };
        if response.g("model").truthy() {
            self.model = response.g("model").pystr();
        }
        if kind == "error" || event.g("error").truthy() {
            return Err(ProxyError::upstream(format!("upstream inference error: {}", clip(&dumps(&event), 500))));
        }
        let model = self.model.clone();
        match kind.as_str() {
            "response.output_text.delta" => {
                Ok(Some(delta_chunk(&model, json!({"content": event.g("delta").or_str()}), Value::Null)))
            }
            "response.reasoning_summary_text.delta" => {
                Ok(Some(delta_chunk(&model, json!({"reasoning_content": event.g("delta").or_str()}), Value::Null)))
            }
            "response.output_item.done" => {
                let item = if event.g("item").truthy() { event.g("item").clone() } else { json!({}) };
                if item.g("type") == "reasoning" {
                    return Ok(Some(delta_chunk(&model, json!({"reasoning_details": [item]}), Value::Null)));
                }
                if item.g("type") == "function_call" {
                    self.saw_tool_call = true;
                    let output_index = get_or(&event, "output_index", json!(0)).int();
                    let next = self.call_indexes.len();
                    let index = *self.call_indexes.entry(output_index).or_insert(next);
                    let arguments = if item.g("arguments").truthy() { item.g("arguments").pystr() } else { "{}".to_string() };
                    let call = json!({
                        "index": index,
                        "id": item.g("call_id").or_str(),
                        "type": "function",
                        "function": {"name": item.g("name").or_str(), "arguments": arguments},
                    });
                    return Ok(Some(delta_chunk(&model, json!({"tool_calls": [call]}), Value::Null)));
                }
                Ok(None)
            }
            "response.failed" | "response.incomplete" => Err(ProxyError::upstream(format!(
                "upstream inference did not complete: {}",
                clip(&dumps(&response), 500)
            ))),
            "response.completed" => {
                let status = response.g("status");
                if !status.is_null() && status != "completed" {
                    return Err(ProxyError::upstream(format!(
                        "upstream inference did not complete: {}",
                        clip(&dumps(&response), 500)
                    )));
                }
                let usage = if response.g("usage").truthy() { response.g("usage").clone() } else { json!({}) };
                let finish = if self.saw_tool_call { "tool_calls" } else { "stop" };
                let mut chunk = delta_chunk(&model, json!({}), Value::from(finish));
                set(
                    &mut chunk,
                    "usage",
                    json!({
                        "prompt_tokens": get_or(&usage, "input_tokens", json!(0)),
                        "completion_tokens": get_or(&usage, "output_tokens", json!(0)),
                        "total_tokens": get_or(&usage, "total_tokens", json!(0)),
                        "prompt_tokens_details": get_or(&usage, "input_tokens_details", json!({})),
                    }),
                );
                Ok(Some(chunk))
            }
            _ => Ok(None),
        }
    }
}

impl Iterator for ResponsesDeltas {
    type Item = PResult<Value>;

    fn next(&mut self) -> Option<PResult<Value>> {
        while !self.done {
            match self.events.next() {
                None => {
                    self.done = true;
                    if !self.seen {
                        return Some(Err(ProxyError::upstream("upstream returned no events.")));
                    }
                }
                Some(Err(err)) => {
                    self.done = true;
                    return Some(Err(err));
                }
                Some(Ok(event)) => match self.convert(event) {
                    Ok(Some(chunk)) => return Some(Ok(chunk)),
                    Ok(None) => {}
                    Err(err) => {
                        self.done = true;
                        return Some(Err(err));
                    }
                },
            }
        }
        None
    }
}

/// `int(x or 0)` of `raw.get(key, raw.get(alt, 0))`.
fn count(raw: &Value, key: &str, alt: &str) -> i64 {
    get_or(raw, key, get_or(raw, alt, json!(0))).int()
}

/// Normalised OpenAI usage.
pub fn usage(value: &Value) -> Value {
    let empty = json!({});
    let raw = if value.is_object() { value } else { &empty };
    let prompt = count(raw, "prompt_tokens", "input");
    let completion = count(raw, "completion_tokens", "output");
    let total_raw = get_or(raw, "total_tokens", json!(prompt + completion));
    let total = if total_raw.truthy() { total_raw.int() } else { prompt + completion };
    let cached = get_or(raw.g("prompt_tokens_details"), "cached_tokens", json!(0)).int();
    json!({
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
        "prompt_tokens_details": {"cached_tokens": cached},
    })
}

struct CallBuf {
    id: String,
    kind: String,
    name: String,
    arguments: String,
}

/// Collects a streamed answer into one `chat.completion`.
pub fn merge_deltas(events: Events, model: &str) -> PResult<Value> {
    let mut content = String::new();
    let mut reasoning = String::new();
    let mut reasoning_details: Vec<Value> = Vec::new();
    let mut calls: BTreeMap<i64, CallBuf> = BTreeMap::new();
    let mut finish_reason = Value::from("stop");
    let mut usage_value: Option<Value> = None;
    let mut selected = model.to_string();
    let mut seen = false;

    for event in events {
        let event = event?;
        seen = true;
        if event.g("error").truthy() {
            return Err(ProxyError::upstream(format!("upstream inference error: {}", event.g("error").pystr())));
        }
        if event.g("model").truthy() {
            selected = event.g("model").pystr();
        }
        if event.g("usage").is_object() {
            usage_value = Some(event.g("usage").clone());
        }
        for choice in event.g("choices").items() {
            if choice.g("finish_reason").truthy() {
                finish_reason = choice.g("finish_reason").clone();
            }
            let delta = choice.g("delta");
            if !delta.g("content").is_null() {
                content.push_str(&delta.g("content").pystr());
            }
            for key in ["reasoning_content", "reasoning"] {
                if !delta.g(key).is_null() {
                    reasoning.push_str(&delta.g(key).pystr());
                    break;
                }
            }
            match delta.g("reasoning_details") {
                Value::Null => {}
                Value::Array(items) => reasoning_details.extend(items.iter().cloned()),
                other => reasoning_details.push(other.clone()),
            }
            for raw_call in delta.g("tool_calls").items() {
                let index = get_or(raw_call, "index", json!(0)).int();
                if raw_call.g("type") == "custom" {
                    return Err(ProxyError::upstream(format!(
                        "custom tool calls are not supported; the client requested a custom tool: {}",
                        raw_call.g("name").pystr()
                    )));
                }
                let item = calls.entry(index).or_insert_with(|| CallBuf {
                    id: String::new(),
                    kind: "function".into(),
                    name: String::new(),
                    arguments: String::new(),
                });
                if raw_call.g("id").truthy() {
                    item.id = raw_call.g("id").pystr();
                }
                if raw_call.g("type").truthy() {
                    item.kind = raw_call.g("type").pystr();
                }
                let function = raw_call.g("function");
                if function.g("name").truthy() {
                    item.name = function.g("name").pystr();
                }
                item.arguments.push_str(&function.g("arguments").or_str());
            }
        }
    }
    if !seen {
        return Err(ProxyError::upstream("upstream returned no events."));
    }
    if finish_reason == "content_filter" {
        return Err(ProxyError::new("upstream blocked the response by content filter.", 400));
    }

    let tool_calls: Vec<Value> = calls
        .iter()
        .map(|(index, call)| {
            let id = if call.id.is_empty() { format!("call_{index}") } else { call.id.clone() };
            json!({"id": id, "type": "function", "function": {"name": call.name, "arguments": call.arguments}})
        })
        .collect();

    let mut message = Map::new();
    message.insert("role".into(), "assistant".into());
    message.insert("content".into(), if content.is_empty() { Value::Null } else { content.into() });
    if !reasoning.is_empty() {
        message.insert("reasoning".into(), reasoning.clone().into());
        message.insert("reasoning_content".into(), reasoning.into());
    }
    if !reasoning_details.is_empty() {
        message.insert("reasoning_details".into(), reasoning_details.into());
    }
    let has_calls = !tool_calls.is_empty();
    if has_calls {
        message.insert("tool_calls".into(), tool_calls.into());
    }
    Ok(json!({
        "id": format!("chatcmpl-{}", token_hex(12)),
        "object": "chat.completion",
        "created": now_secs(),
        "model": selected,
        "choices": [{
            "index": 0,
            "finish_reason": if has_calls { Value::from("tool_calls") } else { finish_reason },
            "message": Value::Object(message),
        }],
        "usage": usage_value.unwrap_or_else(|| usage(&Value::Null)),
    }))
}

/// Iterates over plain values, for tests and canned streams.
pub fn events_from(values: Vec<Value>) -> Events {
    Box::new(values.into_iter().map(Ok))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chat_request_adds_system_prompt_and_marker_tools() {
        let body = build_chat_request(&json!({"messages": [{"role": "user", "content": "hi"}], "temperature": 0.2})).unwrap();
        assert_eq!(body["model"], "big-pickle");
        assert_eq!(body["messages"][0], json!({"role": "system", "content": "You are opencode"}));
        let names: Vec<_> = body["tools"].items().iter().map(|t| t["function"]["name"].clone()).collect();
        assert_eq!(names, vec![json!("read"), json!("bash")]);
        assert_eq!(body["temperature"], 0.2);
        assert!(body.get("tool_choice").is_none());
    }

    #[test]
    fn bad_messages_are_rejected() {
        let err = build_chat_request(&json!({"messages": "hi"})).unwrap_err();
        assert_eq!(err.status, 400);
    }

    #[test]
    fn merge_collects_text_and_tool_calls() {
        let events = events_from(vec![
            json!({"model": "m1", "choices": [{"delta": {"content": "Hel"}}]}),
            json!({"choices": [{"delta": {"content": "lo", "reasoning_content": "think"}}]}),
            json!({"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c1", "function": {"name": "f", "arguments": "{\"a\""}}]}}]}),
            json!({"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": ":1}"}}]}, "finish_reason": "stop"}],
                   "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7}}),
        ]);
        let done = merge_deltas(events, "big-pickle").unwrap();
        assert_eq!(done["model"], "m1");
        let message = &done["choices"][0]["message"];
        assert_eq!(message["content"], "Hello");
        assert_eq!(message["reasoning"], "think");
        assert_eq!(message["tool_calls"][0]["function"]["arguments"], "{\"a\":1}");
        assert_eq!(done["choices"][0]["finish_reason"], "tool_calls");
        assert_eq!(done["usage"]["total_tokens"], 7);
    }

    #[test]
    fn merge_rejects_empty_and_filtered_streams() {
        assert_eq!(merge_deltas(events_from(vec![]), "m").unwrap_err().message, "upstream returned no events.");
        let filtered = events_from(vec![json!({"choices": [{"delta": {}, "finish_reason": "content_filter"}]})]);
        assert_eq!(merge_deltas(filtered, "m").unwrap_err().status, 400);
    }

    #[test]
    fn responses_events_become_chunks() {
        let events = events_from(vec![
            json!({"type": "response.created", "response": {"model": "muse"}}),
            json!({"type": "response.output_text.delta", "delta": "Hi"}),
            json!({"type": "response.output_item.done", "output_index": 2, "item": {"type": "function_call", "call_id": "c", "name": "f", "arguments": ""}}),
            json!({"type": "response.completed", "response": {"status": "completed", "usage": {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3}}}),
        ]);
        let chunks: Vec<Value> = responses_events_to_deltas(events).map(Result::unwrap).collect();
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0]["choices"][0]["delta"]["content"], "Hi");
        assert_eq!(chunks[1]["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"], "{}");
        assert_eq!(chunks[2]["choices"][0]["finish_reason"], "tool_calls");
        assert_eq!(chunks[2]["usage"]["total_tokens"], 3);
    }

    #[test]
    fn responses_request_shape() {
        let req = json!({
            "model": "muse-spark-1.2-contributor-free",
            "messages": [
                {"role": "system", "content": "Be short."},
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "function": {"name": "f", "arguments": {"x": 1}}}]},
                {"role": "tool", "tool_call_id": "c1", "content": "out"},
            ],
            "tools": [{"type": "function", "function": {"name": "f"}}],
            "tool_choice": {"type": "function", "function": {"name": "f"}},
            "max_tokens": 50,
        });
        let body = build_responses_request(&req).unwrap();
        assert_eq!(body["instructions"], "You are opencode\n\nBe short.");
        assert_eq!(body["input"][0], json!({"role": "user", "content": [{"type": "input_text", "text": "hi"}]}));
        assert_eq!(body["input"][1]["arguments"], "{\"x\":1}");
        assert_eq!(body["input"][2]["type"], "function_call_output");
        assert_eq!(body["max_output_tokens"], 50);
        assert_eq!(body["tool_choice"], json!({"type": "function", "name": "f"}));
        let names: Vec<_> = body["tools"].items().iter().map(|t| t["name"].clone()).collect();
        assert_eq!(names, vec![json!("read"), json!("bash"), json!("f")]);
    }

    #[test]
    fn usage_defaults() {
        assert_eq!(usage(&json!({"input": 2, "output": 3}))["total_tokens"], 5);
        assert_eq!(usage(&Value::Null)["prompt_tokens_details"]["cached_tokens"], 0);
    }
}
