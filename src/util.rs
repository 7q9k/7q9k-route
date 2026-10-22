//! Small helpers shared by every module: loose JSON access (the request
//! bodies come from many clients and are read the forgiving way), ids and time.

use rand::Rng;
use serde_json::{Map, Value};
use std::fmt;
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

pub type Obj = Map<String, Value>;

static NULL: Value = Value::Null;

/// An error that ends a request with an HTTP status.
#[derive(Debug, Clone)]
pub struct ProxyError {
    pub message: String,
    pub status: u16,
    /// Set for upstream failures worth retrying (5xx, connection errors).
    pub retryable: bool,
}

impl ProxyError {
    pub fn new(message: impl Into<String>, status: u16) -> Self {
        ProxyError { message: message.into(), status, retryable: false }
    }

    pub fn upstream(message: impl Into<String>) -> Self {
        Self::new(message, 502)
    }
}

impl fmt::Display for ProxyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for ProxyError {}

pub type PResult<T> = Result<T, ProxyError>;

/// Forgiving access to JSON values, close to `dict.get` with falsy defaults.
pub trait J {
    /// The member `key`, or null when missing or when this is not an object.
    fn g(&self, key: &str) -> &Value;
    /// Python truthiness: null, false, 0, "", [] and {} are false.
    fn truthy(&self) -> bool;
    /// `str(value)`; strings are taken as they are.
    fn pystr(&self) -> String;
    /// `str(value or "")`.
    fn or_str(&self) -> String;
    /// `int(value or 0)` without raising.
    fn int(&self) -> i64;
    /// The array items, or nothing when this is not an array.
    fn items(&self) -> &[Value];
    fn obj(&self) -> Option<&Obj>;
    fn has(&self, key: &str) -> bool;
}

impl J for Value {
    fn g(&self, key: &str) -> &Value {
        match self {
            Value::Object(map) => map.get(key).unwrap_or(&NULL),
            _ => &NULL,
        }
    }

    fn truthy(&self) -> bool {
        match self {
            Value::Null => false,
            Value::Bool(b) => *b,
            Value::Number(n) => n.as_f64().map(|f| f != 0.0).unwrap_or(true),
            Value::String(s) => !s.is_empty(),
            Value::Array(a) => !a.is_empty(),
            Value::Object(o) => !o.is_empty(),
        }
    }

    fn pystr(&self) -> String {
        match self {
            Value::String(s) => s.clone(),
            Value::Null => "None".into(),
            Value::Bool(true) => "True".into(),
            Value::Bool(false) => "False".into(),
            Value::Number(n) => n.to_string(),
            other => other.to_string(),
        }
    }

    fn or_str(&self) -> String {
        if self.truthy() {
            self.pystr()
        } else {
            String::new()
        }
    }

    fn int(&self) -> i64 {
        match self {
            Value::Number(n) => n.as_i64().unwrap_or_else(|| n.as_f64().unwrap_or(0.0) as i64),
            Value::Bool(b) => *b as i64,
            Value::String(s) => s.trim().parse().unwrap_or(0),
            _ => 0,
        }
    }

    fn items(&self) -> &[Value] {
        match self {
            Value::Array(a) => a.as_slice(),
            _ => &[],
        }
    }

    fn obj(&self) -> Option<&Obj> {
        self.as_object()
    }

    fn has(&self, key: &str) -> bool {
        self.as_object().map(|o| o.contains_key(key)).unwrap_or(false)
    }
}

/// Whether a value is a JSON number (Python `int`/`float` that is not a bool).
pub fn is_number(v: &Value) -> bool {
    v.is_number()
}

/// Compact JSON (`separators=(",", ":")`).
pub fn compact(v: &Value) -> String {
    serde_json::to_string(v).unwrap_or_default()
}

/// The first `n` characters of `text`.
pub fn clip(text: &str, n: usize) -> String {
    text.chars().take(n).collect()
}

pub fn now() -> f64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs_f64()).unwrap_or(0.0)
}

pub fn now_secs() -> i64 {
    now() as i64
}

pub fn token_hex(bytes: usize) -> String {
    let mut rng = rand::thread_rng();
    (0..bytes).map(|_| format!("{:02x}", rng.gen::<u8>())).collect()
}

const BASE62: &[u8] = b"0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";

static ID_STATE: Mutex<(u64, u64)> = Mutex::new((0, 0));

/// OpenCode style ids: 6 bytes of time and counter, then 14 random base62 characters.
pub fn oc_id(prefix: &str, descending: bool) -> String {
    let value = {
        let mut state = ID_STATE.lock().unwrap_or_else(|e| e.into_inner());
        let now_ms = (now() * 1000.0) as u64;
        if now_ms != state.0 {
            *state = (now_ms, 0);
        }
        state.1 += 1;
        let mut value = (now_ms << 12) | state.1;
        if descending {
            value = !value & ((1u64 << 56) - 1);
        }
        value
    };
    let bytes = value.to_be_bytes();
    let hex: String = bytes[2..8].iter().map(|b| format!("{b:02x}")).collect();
    let mut rng = rand::thread_rng();
    let tail: String = (0..14).map(|_| BASE62[rng.gen_range(0..BASE62.len())] as char).collect();
    format!("{prefix}_{hex}{tail}")
}

pub fn session_id() -> String {
    oc_id("ses", true)
}

pub fn request_id() -> String {
    oc_id("msg", false)
}

/// `time.strftime("%Y-%m-%dT%H:%M:%SZ", gmtime())`.
pub fn utc_stamp() -> String {
    chrono::Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string()
}

/// Constant time comparison of two byte strings.
pub fn compare_digest(a: &[u8], b: &[u8]) -> bool {
    let mut diff = (a.len() ^ b.len()) as u8 | ((a.len() ^ b.len()) >> 8) as u8;
    let n = a.len().max(b.len());
    for i in 0..n {
        let x = a.get(i).copied().unwrap_or(0);
        let y = b.get(i).copied().unwrap_or(0);
        diff |= x ^ y;
    }
    diff == 0 && a.len() == b.len()
}

/// Builds a JSON object from key/value pairs.
#[macro_export]
macro_rules! obj {
    ($($k:expr => $v:expr),* $(,)?) => {{
        #[allow(unused_mut)]
        let mut m = serde_json::Map::new();
        $( m.insert(String::from($k), serde_json::Value::from($v)); )*
        serde_json::Value::Object(m)
    }};
}

/// Inserts into a JSON value that is known to be an object.
pub fn set(target: &mut Value, key: &str, value: impl Into<Value>) {
    if let Value::Object(map) = target {
        map.insert(key.to_string(), value.into());
    }
}

/// Python round(): halves go to the even neighbour.
pub fn py_round(x: f64) -> i64 {
    let r = x.round();
    if (x - x.trunc()).abs() == 0.5 {
        let t = x.trunc();
        if (t as i64) % 2 == 0 {
            t as i64
        } else {
            r as i64
        }
    } else {
        r as i64
    }
}

/// `json.dumps` with Python's spacing and escaping rules.
pub fn pydumps(v: &Value, ensure_ascii: bool, item_sep: &str, key_sep: &str) -> String {
    let mut out = String::new();
    dump_into(&mut out, v, ensure_ascii, item_sep, key_sep);
    out
}

/// `json.dumps(v)` with the defaults.
pub fn dumps(v: &Value) -> String {
    pydumps(v, true, ", ", ": ")
}

fn dump_into(out: &mut String, v: &Value, ascii: bool, item_sep: &str, key_sep: &str) {
    match v {
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Number(n) => out.push_str(&n.to_string()),
        Value::String(s) => dump_str(out, s, ascii),
        Value::Array(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push_str(item_sep);
                }
                dump_into(out, item, ascii, item_sep, key_sep);
            }
            out.push(']');
        }
        Value::Object(map) => {
            out.push('{');
            for (i, (k, item)) in map.iter().enumerate() {
                if i > 0 {
                    out.push_str(item_sep);
                }
                dump_str(out, k, ascii);
                out.push_str(key_sep);
                dump_into(out, item, ascii, item_sep, key_sep);
            }
            out.push('}');
        }
    }
}

fn dump_str(out: &mut String, s: &str, ascii: bool) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c if ascii && (c as u32) > 0x7f => {
                let mut buf = [0u16; 2];
                for unit in c.encode_utf16(&mut buf) {
                    out.push_str(&format!("\\u{:04x}", unit));
                }
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

/// `f"{x:g}"` for the short delays printed in log lines.
pub fn fmt_g(x: f64) -> String {
    if x.fract() == 0.0 && x.abs() < 1e16 {
        format!("{}", x as i64)
    } else {
        format!("{x}")
    }
}

/// `f"{n:,}"`.
pub fn thousands(n: i64) -> String {
    let digits = n.unsigned_abs().to_string();
    let mut out = String::new();
    for (i, c) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(c);
    }
    if n < 0 {
        format!("-{out}")
    } else {
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn dumps_like_python() {
        let v = json!({"a": [1, "ä\n"], "b": null});
        let u = String::from("\\") + "u";
        assert_eq!(dumps(&v), format!("{{\"a\": [1, \"{u}00e4\\n\"], \"b\": null}}"));
        assert_eq!(pydumps(&v, false, ",", ":"), "{\"a\":[1,\"ä\\n\"],\"b\":null}");
        let (emoji, del) = (char::from_u32(0x1f600).unwrap(), char::from_u32(0x7f).unwrap());
        assert_eq!(dumps(&json!(format!("{emoji}{del}"))), format!("\"{u}d83d{u}de00{del}\""));
        assert_eq!(fmt_g(0.5), "0.5");
        assert_eq!(fmt_g(2.0), "2");
        assert_eq!(thousands(1234567), "1,234,567");
        assert_eq!(thousands(12), "12");
    }

    #[test]
    fn ids_have_the_opencode_shape() {
        let id = oc_id("ses", true);
        assert!(id.starts_with("ses_"));
        assert_eq!(id.len(), 4 + 12 + 14);
        let a = request_id();
        let b = request_id();
        assert!(a[4..16] < b[4..16], "ascending ids sort by time");
        let c = session_id();
        let d = session_id();
        assert!(c[4..16] > d[4..16], "descending ids sort newest first");
    }

    #[test]
    fn loose_access() {
        let v = json!({"a": {"b": 3}, "s": "", "n": null, "f": 2.7});
        assert_eq!(v.g("a").g("b").int(), 3);
        assert_eq!(v.g("missing").g("x"), &Value::Null);
        assert!(!v.g("s").truthy());
        assert_eq!(v.g("n").or_str(), "");
        assert_eq!(v.g("f").int(), 2);
        assert_eq!(json!(5).pystr(), "5");
    }

    #[test]
    fn rounding_matches_python() {
        assert_eq!(py_round(2.5), 2);
        assert_eq!(py_round(3.5), 4);
        assert_eq!(py_round(2.4), 2);
        assert_eq!(py_round(-1.5), -2);
    }

    #[test]
    fn digest() {
        assert!(compare_digest(b"abc", b"abc"));
        assert!(!compare_digest(b"abc", b"abd"));
        assert!(!compare_digest(b"abc", b"abcd"));
    }
}
