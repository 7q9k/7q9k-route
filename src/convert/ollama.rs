//! Ollama API (/api/chat, /api/generate, /api/tags) on top of chat completions.

use crate::upstream::{available_models, DEFAULT_MODEL};
use crate::util::{compact, set, J, PResult, ProxyError};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

static CALL_ID_SEQ: AtomicU64 = AtomicU64::new(1);

fn next_call_id() -> String {
    format!("call_{}", CALL_ID_SEQ.fetch_add(1, Ordering::Relaxed))
}

pub fn json_obj(text: &Value) -> Value {
    let raw = match text {
        Value::String(s) if !s.trim().is_empty() => s.as_str(),
        _ => "{}",
    };
    match serde_json::from_str::<Value>(raw) {
        Ok(v @ Value::Object(_)) => v,
        _ => json!({}),
    }
}

pub fn ollama_now() -> String {
    chrono::Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string()
}

pub fn ollama_usage(usage: &Value) -> (i64, i64) {
    (usage.g("prompt_tokens").int(), usage.g("completion_tokens").int())
}

pub fn ollama_details() -> Value {
    json!({
        "format": "api",
        "family": "opencode-free",
        "families": ["opencode-free"],
        "parameter_size": "",
        "parent_model": "",
        "quantization_level": "",
    })
}

pub fn ollama_model_entries() -> Vec<Value> {
    available_models(15.0, false)
        .into_iter()
        .map(|model| {
            json!({
                "name": model,
                "model": model,
                "modified_at": ollama_now(),
                "size": 0,
                "digest": format!("7q9k-route-{model}"),
                "details": ollama_details(),
                "capabilities": ["completion", "streaming", "tools"],
            })
        })
        .collect()
}

pub fn normalize_model_name(name: &Value) -> PResult<String> {
    let full = if name.truthy() { name.pystr() } else { String::new() };
    let mut text = full.trim();
    if let Some((provider, rest)) = text.split_once('/') {
        if provider != "opencode" {
            return Err(ProxyError::new(
                "only models from the 'opencode' provider are supported (e.g. 'opencode/big-pickle').",
                400,
            ));
        }
        text = rest;
    }
    let text = text.strip_suffix(":latest").unwrap_or(text);
    Ok(if text.is_empty() { DEFAULT_MODEL.to_string() } else { text.to_string() })
}

pub fn ollama_to_openai_messages(messages: &Value) -> Vec<Value> {
    let mut result = Vec::new();
    let mut ids_by_name: HashMap<String, Vec<String>> = HashMap::new();
    let mut fifo_ids: Vec<String> = Vec::new();
    for message in messages.items() {
        if !message.is_object() {
            continue;
        }
        let role = message.g("role").or_str();
        if role == "assistant" && message.g("tool_calls").is_array() {
            let mut calls = Vec::new();
            for call in message.g("tool_calls").items() {
                if !call.is_object() {
                    continue;
                }
                let function = call.g("function");
                let name = function.g("name").or_str();
                let raw = function.g("arguments");
                let arguments = match raw {
                    Value::String(s) if !s.is_empty() => s.clone(),
                    v if !v.truthy() => "{}".to_string(),
                    v => compact(v),
                };
                let call_id = next_call_id();
                calls.push(json!({"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}));
                ids_by_name.entry(name).or_default().push(call_id.clone());
                fifo_ids.push(call_id);
            }
            let mut item = json!({"role": "assistant", "tool_calls": calls});
            if message.g("content").truthy() {
                set(&mut item, "content", message.g("content").clone());
            }
            result.push(item);
            continue;
        }
        if role == "tool" {
            let call_id = if message.g("tool_call_id").truthy() {
                message.g("tool_call_id").clone()
            } else {
                let name = if message.g("tool_name").truthy() { message.g("tool_name").pystr() } else { message.g("name").or_str() };
                let by_name = ids_by_name.get_mut(&name).filter(|ids| !name.is_empty() && !ids.is_empty());
                let id = if let Some(ids) = by_name {
                    ids.remove(0)
                } else if !fifo_ids.is_empty() {
                    fifo_ids.remove(0)
                } else {
                    next_call_id()
                };
                Value::from(id)
            };
            let content = if message.g("content").truthy() { message.g("content").clone() } else { json!("") };
            result.push(json!({"role": "tool", "tool_call_id": call_id, "content": content}));
            continue;
        }
        let content = if message.g("content").truthy() { message.g("content").clone() } else { json!("") };
        result.push(json!({"role": if role.is_empty() { "user".to_string() } else { role }, "content": content}));
    }
    result
}

pub fn ollama_options(req: &Value) -> Value {
    let options = req.g("options");
    let mut out = json!({"model": req.g("model").clone(), "stream": req.g("stream").truthy()});
    for key in ["temperature", "top_p", "seed"] {
        if !options.g(key).is_null() {
            set(&mut out, key, options.g(key).clone());
        }
    }
    if !options.g("num_predict").is_null() {
        set(&mut out, "max_tokens", options.g("num_predict").clone());
    }
    let stop = options.g("stop");
    if !stop.is_null() {
        set(&mut out, "stop", if stop.is_array() { stop.clone() } else { json!([stop]) });
    }
    if req.g("format") == "json" {
        set(&mut out, "response_format", json!({"type": "json_object"}));
    }
    let mut tools = Vec::new();
    for tool in req.g("tools").items() {
        if !tool.is_object() {
            continue;
        }
        let mut function = if tool.g("function").is_object() { tool.g("function").clone() } else { json!({}) };
        if !function.has("parameters") && function.has("arguments") {
            let arguments = function["arguments"].clone();
            set(&mut function, "parameters", arguments);
        }
        tools.push(json!({"type": "function", "function": function}));
    }
    if !tools.is_empty() {
        set(&mut out, "tools", tools);
        if !req.g("tool_choice").is_null() {
            set(&mut out, "tool_choice", req.g("tool_choice").clone());
        }
    }
    out
}

fn first_choice(completion: &Value) -> &Value {
    completion.g("choices").items().first().unwrap_or(&Value::Null)
}

fn done_reason(completion: &Value) -> Value {
    let reason = first_choice(completion).g("finish_reason");
    if reason.truthy() { reason.clone() } else { json!("stop") }
}

pub fn ollama_chat_response(completion: &Value, include_thinking: bool) -> Value {
    let message = first_choice(completion).g("message");
    let content = if message.g("content").truthy() { message.g("content").clone() } else { json!("") };
    let mut out = json!({"role": "assistant", "content": content});
    if include_thinking && message.g("reasoning").truthy() {
        set(&mut out, "thinking", message.g("reasoning").pystr());
    }
    let calls: Vec<Value> = message
        .g("tool_calls")
        .items()
        .iter()
        .map(|call| json!({"function": {"name": call.g("function").g("name").clone(), "arguments": json_obj(call.g("function").g("arguments"))}}))
        .collect();
    if !calls.is_empty() {
        set(&mut out, "tool_calls", calls);
    }
    let (prompt, eval) = ollama_usage(completion.g("usage"));
    json!({
        "model": completion.g("model").clone(),
        "created_at": ollama_now(),
        "message": out,
        "done_reason": done_reason(completion),
        "done": true,
        "total_duration": 0,
        "load_duration": 0,
        "prompt_eval_count": prompt,
        "eval_count": eval,
    })
}

pub fn ollama_generate_response(completion: &Value) -> Value {
    let message = first_choice(completion).g("message");
    let (prompt, eval) = ollama_usage(completion.g("usage"));
    json!({
        "model": completion.g("model").clone(),
        "created_at": ollama_now(),
        "response": if message.g("content").truthy() { message.g("content").clone() } else { json!("") },
        "done_reason": done_reason(completion),
        "done": true,
        "total_duration": 0,
        "load_duration": 0,
        "prompt_eval_count": prompt,
        "eval_count": eval,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_names() {
        assert_eq!(normalize_model_name(&json!("opencode/big-pickle:latest")).unwrap(), "big-pickle");
        assert_eq!(normalize_model_name(&json!("")).unwrap(), DEFAULT_MODEL);
        assert_eq!(normalize_model_name(&json!("openai/gpt")).unwrap_err().status, 400);
    }

    #[test]
    fn tool_results_get_call_ids() {
        let messages = json!([
            {"role": "assistant", "tool_calls": [{"function": {"name": "a", "arguments": {"x": 1}}}, {"function": {"name": "b"}}]},
            {"role": "tool", "tool_name": "b", "content": "B"},
            {"role": "tool", "content": "A"},
            {"role": "system"},
        ]);
        let out = ollama_to_openai_messages(&messages);
        let calls = out[0]["tool_calls"].items();
        assert_eq!(calls[0]["function"]["arguments"], "{\"x\":1}");
        assert_eq!(calls[1]["function"]["arguments"], "{}");
        assert_eq!(out[1]["tool_call_id"], calls[1]["id"]);
        assert_eq!(out[2]["tool_call_id"], calls[0]["id"]);
        assert_eq!(out[3], json!({"role": "system", "content": ""}));
    }

    #[test]
    fn options_map_to_openai() {
        let req = json!({"model": "m", "options": {"num_predict": 5, "stop": "x", "seed": 1}, "format": "json",
                         "tools": [{"function": {"name": "f", "arguments": {"type": "object"}}}]});
        let out = ollama_options(&req);
        assert_eq!(out["max_tokens"], 5);
        assert_eq!(out["stop"], json!(["x"]));
        assert_eq!(out["stream"], false);
        assert_eq!(out["tools"][0]["function"]["parameters"]["type"], "object");
        assert_eq!(out["response_format"]["type"], "json_object");
    }

    #[test]
    fn chat_response_shape() {
        let completion = json!({"model": "m", "choices": [{"message": {"content": "hi", "reasoning": "r",
            "tool_calls": [{"function": {"name": "f", "arguments": "{\"a\":1}"}}]}, "finish_reason": null}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2}});
        let out = ollama_chat_response(&completion, true);
        assert_eq!(out["message"]["thinking"], "r");
        assert_eq!(out["message"]["tool_calls"][0]["function"]["arguments"]["a"], 1);
        assert_eq!(out["done_reason"], "stop");
        assert_eq!(out["eval_count"], 2);
        assert_eq!(ollama_generate_response(&completion)["response"], "hi");
    }
}
