//! Request counters, timelines and the request log shown on the dashboard.

use crate::util::{now, py_round, J};
use serde_json::{json, Map, Value};
use std::collections::{BTreeMap, VecDeque};
use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;
use std::time::{Duration, UNIX_EPOCH};

pub const TIMELINE_MINUTES: i64 = 30;
pub const SAVE_VERSION: i64 = 1;
const LOG_SIZE: usize = 200;

#[derive(Clone, Default)]
struct ModelStats {
    requests: f64,
    errors: f64,
    tokens_in: f64,
    tokens_out: f64,
    tokens_cached: f64,
    latency_total: f64,
    last_ms: f64,
}

const STAT_KEYS: [&str; 7] = ["requests", "errors", "tokens_in", "tokens_out", "tokens_cached", "latency_total", "last_ms"];

impl ModelStats {
    fn get_mut(&mut self, key: &str) -> &mut f64 {
        match key {
            "requests" => &mut self.requests,
            "errors" => &mut self.errors,
            "tokens_in" => &mut self.tokens_in,
            "tokens_out" => &mut self.tokens_out,
            "tokens_cached" => &mut self.tokens_cached,
            "latency_total" => &mut self.latency_total,
            _ => &mut self.last_ms,
        }
    }

    fn to_json(&self) -> Value {
        let n = |x: f64| if x.fract() == 0.0 { json!(x as i64) } else { json!(x) };
        json!({
            "requests": n(self.requests), "errors": n(self.errors), "tokens_in": n(self.tokens_in),
            "tokens_out": n(self.tokens_out), "tokens_cached": n(self.tokens_cached),
            "latency_total": self.latency_total, "last_ms": n(self.last_ms),
        })
    }
}

struct Counters {
    since: f64,
    requests: i64,
    errors: i64,
    tokens_in: i64,
    tokens_out: i64,
    tokens_cached: i64,
    latency_total: f64,
    models: Vec<(String, ModelStats)>,
    minutes: BTreeMap<i64, [i64; 4]>,
    hours: BTreeMap<i64, [i64; 4]>,
    limit_hits: Vec<f64>,
    log: VecDeque<Value>,
    saved_at: Option<f64>,
}

impl Counters {
    fn new() -> Self {
        Counters {
            since: now(),
            requests: 0,
            errors: 0,
            tokens_in: 0,
            tokens_out: 0,
            tokens_cached: 0,
            latency_total: 0.0,
            models: Vec::new(),
            minutes: BTreeMap::new(),
            hours: BTreeMap::new(),
            limit_hits: Vec::new(),
            log: VecDeque::new(),
            saved_at: None,
        }
    }

    fn model(&mut self, name: &str) -> &mut ModelStats {
        let index = match self.models.iter().position(|(n, _)| n == name) {
            Some(i) => i,
            None => {
                self.models.push((name.to_string(), ModelStats::default()));
                self.models.len() - 1
            }
        };
        &mut self.models[index].1
    }

    fn push_log(&mut self, entry: Value) {
        self.log.push_back(entry);
        while self.log.len() > LOG_SIZE {
            self.log.pop_front();
        }
    }
}

pub struct Metrics {
    inner: Mutex<Counters>,
    save_lock: Mutex<()>,
    pub path: Option<PathBuf>,
    started: f64,
    dirty: AtomicBool,
}

pub fn is_limit_error(entry: &Value) -> bool {
    let error = entry.g("error").or_str();
    error.contains("FreeUsageLimit") || error.contains("Free usage exceeded")
}

fn point(bucket: &[i64; 4]) -> Value {
    json!({"requests": bucket[0], "errors": bucket[1], "tokens_in": bucket[2], "tokens_out": bucket[3]})
}

fn num(v: &Value) -> Option<f64> {
    if v.is_boolean() {
        return Some(if v.as_bool() == Some(true) { 1.0 } else { 0.0 });
    }
    v.as_f64()
}

fn bucket_from(v: &Value) -> Option<[i64; 4]> {
    let items = v.as_array()?;
    let mut out = [0i64; 4];
    for (i, x) in items.iter().take(4).enumerate() {
        out[i] = num(x)? as i64;
    }
    Some(out)
}

impl Metrics {
    pub fn new(path: Option<PathBuf>) -> Self {
        Metrics {
            inner: Mutex::new(Counters::new()),
            save_lock: Mutex::new(()),
            path,
            started: now(),
            dirty: AtomicBool::new(false),
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Counters> {
        self.inner.lock().unwrap_or_else(|e| e.into_inner())
    }

    pub fn is_dirty(&self) -> bool {
        self.dirty.load(Ordering::Relaxed)
    }

    pub fn record(&self, entry: Value, inference: bool) {
        let failed = (entry.g("status").int() >= 400 || entry.g("error").truthy()) as i64;
        let tokens_in = entry.g("tokens_in").int();
        let tokens_out = entry.g("tokens_out").int();
        let tokens_cached = entry.g("tokens_cached").int();
        let ms = entry.g("ms").as_f64().unwrap_or(0.0);
        let model = entry.g("model").or_str();
        let limit = is_limit_error(&entry);
        let mut c = self.lock();
        c.push_log(entry);
        self.dirty.store(true, Ordering::Relaxed);
        if !inference {
            return;
        }
        c.requests += 1;
        c.errors += failed;
        c.tokens_in += tokens_in;
        c.tokens_out += tokens_out;
        c.tokens_cached += tokens_cached;
        c.latency_total += ms;
        let stats = c.model(&model);
        stats.requests += 1.0;
        stats.errors += failed as f64;
        stats.tokens_in += tokens_in as f64;
        stats.tokens_out += tokens_out as f64;
        stats.tokens_cached += tokens_cached as f64;
        stats.latency_total += ms;
        stats.last_ms = ms;
        let t = now();
        let minute = (t / 60.0).floor() as i64;
        let hour = (t / 3600.0).floor() as i64;
        let counters = &mut *c;
        for (buckets, key) in [(&mut counters.minutes, minute), (&mut counters.hours, hour)] {
            let bucket = buckets.entry(key).or_insert([0; 4]);
            bucket[0] += 1;
            bucket[1] += failed;
            bucket[2] += tokens_in;
            bucket[3] += tokens_out;
        }
        c.minutes.retain(|m, _| *m > minute - TIMELINE_MINUTES);
        c.hours.retain(|h, _| *h > hour - 48);
        if limit {
            c.limit_hits.push(now());
            let excess = c.limit_hits.len().saturating_sub(50);
            c.limit_hits.drain(..excess);
        }
    }

    pub fn snapshot(&self) -> Value {
        let empty = [0i64; 4];
        let c = self.lock();
        let t = now();
        let now_minute = (t / 60.0).floor() as i64;
        let hour = (t / 3600.0).floor() as i64;
        let timeline: Vec<Value> = (now_minute - TIMELINE_MINUTES + 1..=now_minute)
            .map(|m| point(c.minutes.get(&m).unwrap_or(&empty)))
            .collect();
        let hourly: Vec<Value> = (hour - 23..=hour).map(|h| point(c.hours.get(&h).unwrap_or(&empty))).collect();
        let mut models = Map::new();
        for (name, s) in &c.models {
            let avg = if s.requests != 0.0 { json!(py_round(s.latency_total / s.requests)) } else { Value::Null };
            models.insert(
                name.clone(),
                json!({
                    "requests": s.requests as i64,
                    "errors": s.errors as i64,
                    "tokens_in": s.tokens_in as i64,
                    "tokens_out": s.tokens_out as i64,
                    "tokens_cached": s.tokens_cached as i64,
                    "tokens": (s.tokens_in + s.tokens_out) as i64,
                    "avg_ms": avg,
                    "last_ms": py_round(s.last_ms),
                }),
            );
        }
        let window = |n: i64| c.hours.iter().filter(|(h, _)| **h > hour - n).map(|(_, b)| b[0]).sum::<i64>();
        json!({
            "limits": {
                "requests_5h": window(5),
                "requests_24h": window(24),
                "last_limit_hit": c.limit_hits.last(),
                "limit_hits_24h": c.limit_hits.iter().filter(|x| **x > t - 86400.0).count(),
            },
            "uptime_s": py_round(t - self.started),
            "since": c.since,
            "saved_at": c.saved_at,
            "persistent": self.path.is_some(),
            "requests": c.requests,
            "errors": c.errors,
            "tokens_in": c.tokens_in,
            "tokens_out": c.tokens_out,
            "tokens_cached": c.tokens_cached,
            "avg_ms": if c.requests != 0 { json!(py_round(c.latency_total / c.requests as f64)) } else { Value::Null },
            "models": models,
            "timeline": timeline,
            "hourly": hourly,
            "log": c.log.iter().cloned().collect::<Vec<_>>(),
        })
    }

    fn state(c: &Counters) -> Value {
        let models: Map<String, Value> = c.models.iter().map(|(n, s)| (n.clone(), s.to_json())).collect();
        let minutes: Map<String, Value> = c.minutes.iter().map(|(k, v)| (k.to_string(), json!(v))).collect();
        let hours: Map<String, Value> = c.hours.iter().map(|(k, v)| (k.to_string(), json!(v))).collect();
        json!({
            "version": SAVE_VERSION,
            "since": c.since,
            "requests": c.requests,
            "errors": c.errors,
            "tokens_in": c.tokens_in,
            "tokens_out": c.tokens_out,
            "tokens_cached": c.tokens_cached,
            "latency_total": c.latency_total,
            "models": models,
            "minutes": minutes,
            "hours": hours,
            "limit_hits": c.limit_hits,
            "log": c.log.iter().cloned().collect::<Vec<_>>(),
        })
    }

    pub fn save(&self) -> std::io::Result<Option<f64>> {
        let Some(path) = &self.path else { return Ok(None) };
        let _guard = self.save_lock.lock().unwrap_or_else(|e| e.into_inner());
        let data = {
            let c = self.lock();
            self.dirty.store(false, Ordering::Relaxed);
            serde_json::to_string(&Self::state(&c)).unwrap_or_default()
        };
        let tmp = path.with_file_name(format!("{}.tmp", path.file_name().unwrap_or_default().to_string_lossy()));
        if let Err(e) = fs::write(&tmp, data).and_then(|_| fs::rename(&tmp, path)) {
            self.dirty.store(true, Ordering::Relaxed);
            return Err(e);
        }
        let t = now();
        self.lock().saved_at = Some(t);
        Ok(Some(t))
    }

    pub fn load(&self) -> bool {
        let Some(path) = &self.path else { return false };
        let Ok(text) = fs::read_to_string(path) else { return false };
        let Ok(state) = serde_json::from_str::<Value>(&text) else { return false };
        if !state.is_object() || state.g("version").as_i64() != Some(SAVE_VERSION) {
            return false;
        }
        match Self::parse_state(&state) {
            Some(mut counters) => {
                counters.saved_at = fs::metadata(path)
                    .and_then(|m| m.modified())
                    .ok()
                    .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
                    .map(|d| d.as_secs_f64());
                *self.lock() = counters;
                self.dirty.store(false, Ordering::Relaxed);
                true
            }
            None => {
                *self.lock() = Counters::new();
                false
            }
        }
    }

    fn parse_state(state: &Value) -> Option<Counters> {
        let int = |key: &str| -> Option<i64> {
            match state.g(key) {
                Value::Null => Some(0),
                v => num(v).map(|x| x as i64),
            }
        };
        let mut c = Counters::new();
        if let Some(since) = num(state.g("since")).filter(|x| *x != 0.0) {
            c.since = since;
        }
        c.requests = int("requests")?;
        c.errors = int("errors")?;
        c.tokens_in = int("tokens_in")?;
        c.tokens_out = int("tokens_out")?;
        c.tokens_cached = int("tokens_cached")?;
        c.latency_total = match state.g("latency_total") {
            Value::Null => 0.0,
            v => num(v)?,
        };
        if let Some(models) = state.g("models").as_object() {
            for (name, raw) in models {
                let mut stats = ModelStats::default();
                for key in STAT_KEYS {
                    if let Some(x) = raw.get(key).filter(|v| v.is_number()).and_then(Value::as_f64) {
                        *stats.get_mut(key) = x;
                    }
                }
                c.models.push((name.clone(), stats));
            }
        }
        if let Some(minutes) = state.g("minutes").as_object() {
            for (k, v) in minutes {
                if matches!(v.as_array().map(Vec::len), Some(2 | 4)) {
                    c.minutes.insert(k.parse().ok()?, bucket_from(v)?);
                }
            }
        }
        if let Some(hours) = state.g("hours").as_object() {
            for (k, v) in hours {
                if let Some(x) = v.as_f64().filter(|_| v.is_number()) {
                    c.hours.insert(k.parse().ok()?, [x as i64, 0, 0, 0]);
                } else if v.as_array().map(Vec::len) == Some(4) {
                    c.hours.insert(k.parse().ok()?, bucket_from(v)?);
                }
            }
        }
        let hits: Vec<f64> = state.g("limit_hits").items().iter().filter(|v| v.is_number()).filter_map(Value::as_f64).collect();
        c.limit_hits = hits[hits.len().saturating_sub(50)..].to_vec();
        for entry in state.g("log").items().iter().filter(|e| e.is_object()) {
            c.push_log(entry.clone());
        }
        Some(c)
    }

    pub fn reset(&self) {
        *self.lock() = Counters::new();
        self.dirty.store(true, Ordering::Relaxed);
    }

    pub fn autosave_loop(&self, interval: f64) {
        loop {
            std::thread::sleep(Duration::from_secs_f64(interval));
            if self.is_dirty() {
                let _ = self.save();
            }
        }
    }
}

#[allow(dead_code)]

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn records_and_reloads() {
        let path = std::env::temp_dir().join(format!("7q9k-stats-{}.json", crate::util::token_hex(4)));
        let metrics = Metrics::new(Some(path.clone()));
        metrics.record(json!({"status": 200, "model": "m", "ms": 120.0, "tokens_in": 5, "tokens_out": 3}), true);
        metrics.record(json!({"status": 429, "model": "m", "ms": 10.0, "error": "FreeUsageLimit"}), true);
        metrics.record(json!({"status": 200, "model": "-", "ms": 1.0}), false);
        let snap = metrics.snapshot();
        assert_eq!(snap["requests"], 2);
        assert_eq!(snap["errors"], 1);
        assert_eq!(snap["avg_ms"], 65);
        assert_eq!(snap["models"]["m"]["tokens"], 8);
        assert_eq!(snap["limits"]["limit_hits_24h"], 1);
        assert_eq!(snap["timeline"].items().len(), 30);
        assert_eq!(snap["hourly"].items().len(), 24);
        assert_eq!(snap["log"].items().len(), 3);
        metrics.save().unwrap();
        let again = Metrics::new(Some(path.clone()));
        assert!(again.load());
        let snap = again.snapshot();
        assert_eq!(snap["requests"], 2);
        assert_eq!(snap["models"]["m"]["avg_ms"], 65);
        assert_eq!(snap["limits"]["requests_24h"], 2);
        let _ = fs::remove_file(path);
    }
}
