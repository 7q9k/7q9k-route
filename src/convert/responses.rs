//! OpenAI Responses API: requests become chat completions, chunks become
//! Responses events.

use super::chat::{get_or, message_text, usage};
use crate::upstream::Events;
use crate::util::{dumps, now_secs, set, token_hex, J, PResult, ProxyError};
use serde_json::{json, Value};
use std::collections::BTreeMap;

fn parts_to_chat(content: &Value) -> Value {
    let Value::Array(items) = content else {
        return Value::from(message_text(content));
    };
    let mut parts: Vec<Value> = Vec::new();
    for part in items {
        if let Value::String(s) = part {
            parts.push(json!({"type": "text", "text": s}));
            continue;
        }
        if !part.is_object() {
            continue;
        }
        let kind = part.g("type").or_str();
        match kind.as_str() {
            "input_text" | "output_text" | "text" | "summary_text" => {
                parts.push(json!({"type": "text", "text": part.g("text").or_str()}))
            }
            "refusal" => parts.push(json!({"type": "text", "text": part.g("refusal").or_str()})),
            "input_image" => {
                let mut url = part.g("image_url").clone();
                if url.is_object() {
                    url = url.g("url").clone();
                }
                if url.truthy() {
                    parts.push(json!({"type": "image_url", "image_url": {"url": url.pystr()}}));
                }
            }
            _ => {}
        }
    }
    if parts.iter().all(|p| p["type"] == "text") {
        return Value::from(parts.iter().map(|p| p["text"].pystr()).collect::<String>());
    }
    Value::Array(parts)
}

fn tool_output_text(output: &Value) -> String {
    match output {
        Value::String(s) => s.clone(),
        Value::Array(_) => message_text(output),
        Value::Null => String::new(),
        other => dumps(other),
    }
}

pub fn responses_to_chat_request(req: &Value) -> PResult<Value> {
    if req.g("previous_response_id").truthy() {
        return Err(ProxyError::new(
            "previous_response_id is not supported (stateless proxy); send the full conversation in 'input' and set store=false.",
            400,
        ));
    }
    let mut messages: Vec<Value> = Vec::new();
    if req.g("instructions").truthy() {
        messages.push(json!({"role": "system", "content": req.g("instructions").pystr()}));
    }
    let items: Vec<Value> = match req.g("input") {
        Value::String(s) => vec![json!({"role": "user", "content": s})],
        Value::Array(a) => a.clone(),
        _ => Vec::new(),
    };
    for item in &items {
        if !item.is_object() {
            continue;
        }
        let kind = if item.g("type").truthy() { item.g("type").pystr() } else { "message".to_string() };
        match kind.as_str() {
            "message" => {
                let mut role = if item.g("role").truthy() { item.g("role").pystr() } else { "user".to_string() };
                if role == "developer" {
                    role = "system".into();
                }
                messages.push(json!({"role": role, "content": parts_to_chat(item.g("content"))}));
            }
            "function_call" => {
                let id = if item.g("call_id").truthy() { item.g("call_id").pystr() } else { item.g("id").or_str() };
                let arguments = if item.g("arguments").truthy() { item.g("arguments").pystr() } else { "{}".into() };
                let call = json!({
                    "id": id,
                    "type": "function",
                    "function": {"name": item.g("name").or_str(), "arguments": arguments},
                });
                let last = messages.last_mut().filter(|m| m.g("role") == "assistant");
                match last {
                    Some(Value::Object(m)) if m.contains_key("tool_calls") && m["tool_calls"].is_array() => {
                        m.get_mut("tool_calls").and_then(Value::as_array_mut).map(|calls| calls.push(call));
                    }
                    Some(Value::Object(m)) if !m.get("tool_calls").map(J::truthy).unwrap_or(false) => {
                        m.insert("tool_calls".into(), json!([call]));
                    }
                    _ => messages.push(json!({"role": "assistant", "content": null, "tool_calls": [call]})),
                }
            }
            "function_call_output" => messages.push(json!({
                "role": "tool",
                "tool_call_id": item.g("call_id").or_str(),
                "content": tool_output_text(item.g("output")),
            })),
            _ => {}
        }
    }

    let mut chat = json!({"model": req.g("model").clone(), "messages": messages});
    let mut tools = Vec::new();
    for tool in req.g("tools").items() {
        if tool.g("type") == "function" && tool.g("name").truthy() {
            let parameters = if tool.g("parameters").truthy() {
                tool.g("parameters").clone()
            } else {
                json!({"type": "object", "properties": {}})
            };
            tools.push(json!({
                "type": "function",
                "function": {
                    "name": tool.g("name").pystr(),
                    "description": tool.g("description").or_str(),
                    "parameters": parameters,
                },
            }));
        }
    }
    if !tools.is_empty() {
        set(&mut chat, "tools", tools);
        let choice = req.g("tool_choice");
        if choice.is_object() && choice.g("type") == "function" {
            set(&mut chat, "tool_choice", json!({"type": "function", "function": {"name": choice.g("name").or_str()}}));
        } else if matches!(choice.as_str(), Some("auto" | "none" | "required")) {
            set(&mut chat, "tool_choice", choice.clone());
        }
        if !req.g("parallel_tool_calls").is_null() {
            set(&mut chat, "parallel_tool_calls", req.g("parallel_tool_calls").truthy());
        }
    }
    for key in ["temperature", "top_p", "user"] {
        if !req.g(key).is_null() {
            set(&mut chat, key, req.g(key).clone());
        }
    }
    if !req.g("max_output_tokens").is_null() {
        set(&mut chat, "max_tokens", req.g("max_output_tokens").clone());
    }
    let fmt = req.g("text").g("format");
    if fmt.g("type") == "json_object" {
        set(&mut chat, "response_format", json!({"type": "json_object"}));
    } else if fmt.g("type") == "json_schema" {
        let name = if fmt.g("name").truthy() { fmt.g("name").pystr() } else { "output".into() };
        let schema = if fmt.g("schema").truthy() { fmt.g("schema").clone() } else { json!({}) };
        set(
            &mut chat,
            "response_format",
            json!({
                "type": "json_schema",
                "json_schema": {"name": name, "schema": schema, "strict": get_or(fmt, "strict", json!(false)).truthy()},
            }),
        );
    }
    Ok(chat)
}

pub fn responses_usage(raw: &Value) -> Value {
    let u = usage(raw);
    json!({
        "input_tokens": u["prompt_tokens"],
        "input_tokens_details": {"cached_tokens": u["prompt_tokens_details"]["cached_tokens"]},
        "output_tokens": u["completion_tokens"],
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": u["total_tokens"],
    })
}

/// Builds Responses API events (and the final object) from chat chunks.
pub struct ResponsesBuilder {
    pub id: String,
    created: i64,
    model: String,
    req: Value,
    output: Vec<Value>,
    usage: Option<Value>,
    status: &'static str,
    error: Value,
    seq: i64,
    open: Option<usize>,
    tool_items: BTreeMap<i64, Value>,
}

impl ResponsesBuilder {
    pub fn new(model: &str, req: &Value) -> Self {
        ResponsesBuilder {
            id: format!("resp_{}", token_hex(12)),
            created: now_secs(),
            model: model.to_string(),
            req: req.clone(),
            output: Vec::new(),
            usage: None,
            status: "in_progress",
            error: Value::Null,
            seq: 0,
            open: None,
            tool_items: BTreeMap::new(),
        }
    }

    fn event(&mut self, kind: &str, fields: Value) -> Value {
        let mut event = json!({"type": kind, "sequence_number": self.seq});
        if let (Value::Object(target), Value::Object(extra)) = (&mut event, fields) {
            target.extend(extra);
        }
        self.seq += 1;
        event
    }

    pub fn response(&self) -> Value {
        let req = &self.req;
        let mut body = json!({
            "id": self.id,
            "object": "response",
            "created_at": self.created,
            "status": self.status,
            "error": self.error,
            "incomplete_details": null,
            "instructions": req.g("instructions").clone(),
            "model": self.model,
            "output": self.output,
            "parallel_tool_calls": get_or(req, "parallel_tool_calls", json!(true)).truthy(),
            "previous_response_id": null,
            "store": false,
            "temperature": req.g("temperature").clone(),
            "top_p": req.g("top_p").clone(),
            "max_output_tokens": req.g("max_output_tokens").clone(),
            "tool_choice": get_or(req, "tool_choice", json!("auto")),
            "tools": if req.g("tools").truthy() { req.g("tools").clone() } else { json!([]) },
            "text": if req.g("text").truthy() { req.g("text").clone() } else { json!({"format": {"type": "text"}}) },
            "usage": self.usage.as_ref().map(responses_usage).unwrap_or(Value::Null),
            "metadata": if req.g("metadata").truthy() { req.g("metadata").clone() } else { json!({}) },
        });
        if self.status == "completed" {
            let text: String = self
                .output
                .iter()
                .filter(|item| item.g("type") == "message")
                .flat_map(|item| item.g("content").items().iter())
                .map(|part| get_or(part, "text", json!("")).pystr())
                .collect();
            set(&mut body, "output_text", text);
        }
        body
    }

    pub fn start(&mut self) -> Vec<Value> {
        let created = self.response();
        let first = self.event("response.created", json!({"response": created}));
        let progress = self.response();
        let second = self.event("response.in_progress", json!({"response": progress}));
        vec![first, second]
    }

    fn close_open(&mut self) -> Vec<Value> {
        let Some(index) = self.open.take() else { return Vec::new() };
        let mut events = Vec::new();
        let item = self.output[index].clone();
        let id = item.g("id").clone();
        match item.g("type").as_str() {
            Some("message") => {
                let part = item.g("content").items()[0].clone();
                let text = part.g("text").clone();
                events.push(self.event(
                    "response.output_text.done",
                    json!({"item_id": id, "output_index": index, "content_index": 0, "text": text}),
                ));
                events.push(self.event(
                    "response.content_part.done",
                    json!({"item_id": id, "output_index": index, "content_index": 0, "part": part}),
                ));
            }
            Some("reasoning") => {
                let summary = item.g("summary").items()[0].clone();
                let text = summary.g("text").clone();
                events.push(self.event(
                    "response.reasoning_summary_text.done",
                    json!({"item_id": id, "output_index": index, "summary_index": 0, "text": text}),
                ));
                events.push(self.event(
                    "response.reasoning_summary_part.done",
                    json!({"item_id": id, "output_index": index, "summary_index": 0, "part": summary}),
                ));
            }
            Some("function_call") => {
                let arguments = item.g("arguments").clone();
                events.push(self.event(
                    "response.function_call_arguments.done",
                    json!({"item_id": id, "output_index": index, "arguments": arguments}),
                ));
            }
            _ => {}
        }
        set(&mut self.output[index], "status", "completed");
        let done = self.output[index].clone();
        events.push(self.event("response.output_item.done", json!({"output_index": index, "item": done})));
        events
    }

    fn open_item(&mut self, item: Value) -> Vec<Value> {
        let mut events = self.close_open();
        self.output.push(item.clone());
        let index = self.output.len() - 1;
        self.open = Some(index);
        let kind = item.g("type").or_str();
        let mut shown = item.clone();
        match kind.as_str() {
            "message" => set(&mut shown, "content", json!([])),
            "reasoning" => set(&mut shown, "summary", json!([])),
            _ => {}
        }
        events.push(self.event("response.output_item.added", json!({"output_index": index, "item": shown})));
        let id = item.g("id").clone();
        if kind == "message" {
            events.push(self.event(
                "response.content_part.added",
                json!({"item_id": id, "output_index": index, "content_index": 0,
                       "part": {"type": "output_text", "text": "", "annotations": []}}),
            ));
        } else if kind == "reasoning" {
            events.push(self.event(
                "response.reasoning_summary_part.added",
                json!({"item_id": id, "output_index": index, "summary_index": 0,
                       "part": {"type": "summary_text", "text": ""}}),
            ));
        }
        events
    }

    fn open_kind(&self) -> Option<String> {
        self.open.map(|i| self.output[i].g("type").or_str())
    }

    fn append_text(&mut self, index: usize, list: &str, text: &str) {
        if let Some(Value::String(current)) = self.output[index].get_mut(list).and_then(|l| l.get_mut(0)).and_then(|p| p.get_mut("text")) {
            current.push_str(text);
        }
    }

    pub fn feed(&mut self, event: &Value) -> PResult<Vec<Value>> {
        let mut events = Vec::new();
        if event.g("error").truthy() {
            return Err(ProxyError::upstream(format!("upstream inference error: {}", event.g("error").pystr())));
        }
        if event.g("usage").is_object() {
            self.usage = Some(event.g("usage").clone());
        }
        for choice in event.g("choices").items() {
            let delta = choice.g("delta");
            let thought = if delta.g("reasoning_content").truthy() { delta.g("reasoning_content") } else { delta.g("reasoning") };
            if thought.truthy() {
                let thought = thought.pystr();
                if self.open_kind().as_deref() != Some("reasoning") {
                    let item = json!({"id": format!("rs_{}", token_hex(8)), "type": "reasoning",
                                      "summary": [{"type": "summary_text", "text": ""}]});
                    events.extend(self.open_item(item));
                }
                let index = self.open.unwrap_or(0);
                self.append_text(index, "summary", &thought);
                let id = self.output[index].g("id").clone();
                events.push(self.event(
                    "response.reasoning_summary_text.delta",
                    json!({"item_id": id, "output_index": index, "summary_index": 0, "delta": thought}),
                ));
            }
            if delta.g("content").truthy() {
                let text = delta.g("content").pystr();
                if self.open_kind().as_deref() != Some("message") {
                    let item = json!({"id": format!("msg_{}", token_hex(8)), "type": "message",
                                      "status": "in_progress", "role": "assistant",
                                      "content": [{"type": "output_text", "text": "", "annotations": []}]});
                    events.extend(self.open_item(item));
                }
                let index = self.open.unwrap_or(0);
                self.append_text(index, "content", &text);
                let id = self.output[index].g("id").clone();
                events.push(self.event(
                    "response.output_text.delta",
                    json!({"item_id": id, "output_index": index, "content_index": 0, "delta": text}),
                ));
            }
            // Tool calls are buffered and sent in finish(): upstream may interleave
            // argument chunks with text, reasoning or other calls.
            for raw in delta.g("tool_calls").items() {
                let index = get_or(raw, "index", json!(0)).int();
                let function = raw.g("function");
                let item = self.tool_items.entry(index).or_insert_with(|| {
                    json!({"id": format!("fc_{}", token_hex(8)), "type": "function_call",
                           "status": "in_progress", "call_id": "", "name": "", "arguments": ""})
                });
                if raw.g("id").truthy() && !item.g("call_id").truthy() {
                    set(item, "call_id", raw.g("id").pystr());
                }
                if function.g("name").truthy() && !item.g("name").truthy() {
                    set(item, "name", function.g("name").pystr());
                }
                let arguments = format!("{}{}", item.g("arguments").or_str(), function.g("arguments").or_str());
                set(item, "arguments", arguments);
            }
            if choice.g("finish_reason") == "content_filter" {
                return Err(ProxyError::new("upstream blocked the response by content filter.", 400));
            }
        }
        Ok(events)
    }

    pub fn finish(&mut self) -> Vec<Value> {
        let mut events = self.close_open();
        let items = std::mem::take(&mut self.tool_items);
        for (_, mut item) in items {
            if !item.g("call_id").truthy() {
                set(&mut item, "call_id", format!("call_{}", token_hex(6)));
            }
            let arguments = if item.g("arguments").truthy() { item.g("arguments").or_str() } else { "{}".to_string() };
            set(&mut item, "arguments", "");
            events.extend(self.open_item(item));
            let index = self.open.unwrap_or(0);
            set(&mut self.output[index], "arguments", arguments.clone());
            let id = self.output[index].g("id").clone();
            events.push(self.event(
                "response.function_call_arguments.delta",
                json!({"item_id": id, "output_index": index, "delta": arguments}),
            ));
            events.extend(self.close_open());
        }
        self.status = "completed";
        let response = self.response();
        events.push(self.event("response.completed", json!({"response": response})));
        events
    }

    pub fn fail(&mut self, message: &str, code: &str) -> Vec<Value> {
        self.status = "failed";
        self.error = json!({"code": code, "message": message});
        let response = self.response();
        vec![self.event("response.failed", json!({"response": response}))]
    }
}

pub fn build_responses_object(deltas: Events, model: &str, req: &Value) -> PResult<Value> {
    let mut builder = ResponsesBuilder::new(model, req);
    let mut seen = false;
    for event in deltas {
        seen = true;
        builder.feed(&event?)?;
    }
    if !seen {
        return Err(ProxyError::upstream("upstream returned no events."));
    }
    builder.finish();
    Ok(builder.response())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::convert::chat::events_from;

    #[test]
    fn input_items_become_messages() {
        let req = json!({
            "model": "big-pickle",
            "instructions": "Be brief.",
            "input": [
                {"role": "developer", "content": [{"type": "input_text", "text": "dev"}]},
                {"role": "user", "content": "hi"},
                {"type": "function_call", "call_id": "c1", "name": "f", "arguments": "{}"},
                {"type": "function_call", "call_id": "c2", "name": "g"},
                {"type": "function_call_output", "call_id": "c1", "output": {"ok": true}},
            ],
            "tools": [{"type": "function", "name": "f"}],
            "tool_choice": {"type": "function", "name": "f"},
            "max_output_tokens": 9,
            "text": {"format": {"type": "json_schema", "schema": {"type": "object"}}},
        });
        let chat = responses_to_chat_request(&req).unwrap();
        let messages = chat["messages"].items();
        assert_eq!(messages[0], json!({"role": "system", "content": "Be brief."}));
        assert_eq!(messages[1], json!({"role": "system", "content": "dev"}));
        assert_eq!(messages[3]["tool_calls"].items().len(), 2);
        assert_eq!(messages[4]["content"], "{\"ok\": true}");
        assert_eq!(chat["tool_choice"]["function"]["name"], "f");
        assert_eq!(chat["max_tokens"], 9);
        assert_eq!(chat["response_format"]["json_schema"]["name"], "output");
    }

    #[test]
    fn previous_response_id_is_refused() {
        assert_eq!(responses_to_chat_request(&json!({"previous_response_id": "r"})).unwrap_err().status, 400);
    }

    #[test]
    fn builder_emits_ordered_events() {
        let mut builder = ResponsesBuilder::new("big-pickle", &json!({}));
        let mut events = builder.start();
        events.extend(builder.feed(&json!({"choices": [{"delta": {"reasoning_content": "hm"}}]})).unwrap());
        events.extend(builder.feed(&json!({"choices": [{"delta": {"content": "Hi"}}]})).unwrap());
        events.extend(builder.feed(&json!({"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c", "function": {"name": "f", "arguments": "{}"}}]}}]})).unwrap());
        events.extend(builder.finish());
        let kinds: Vec<String> = events.iter().map(|e| e["type"].pystr()).collect();
        assert_eq!(kinds.first().unwrap(), "response.created");
        assert_eq!(kinds.last().unwrap(), "response.completed");
        for (i, e) in events.iter().enumerate() {
            assert_eq!(e["sequence_number"], i as i64);
        }
        let done = events.last().unwrap();
        assert_eq!(done["response"]["output_text"], "Hi");
        let output = done["response"]["output"].items();
        assert_eq!(output.len(), 3);
        assert_eq!(output[0]["status"], "completed");
        assert_eq!(output[2]["call_id"], "c");
        assert_eq!(output[2]["arguments"], "{}");
        assert!(kinds.contains(&"response.function_call_arguments.delta".to_string()));
    }

    #[test]
    fn object_from_deltas() {
        let deltas = events_from(vec![json!({"choices": [{"delta": {"content": "ok"}}], "usage": {"prompt_tokens": 1, "completion_tokens": 1}})]);
        let object = build_responses_object(deltas, "m", &json!({"parallel_tool_calls": false})).unwrap();
        assert_eq!(object["status"], "completed");
        assert_eq!(object["parallel_tool_calls"], false);
        assert_eq!(object["usage"]["total_tokens"], 2);
    }
}
