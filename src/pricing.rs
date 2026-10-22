//! What the used tokens would cost on OpenRouter, for the dashboard.

use crate::upstream::{available_models, json_get};
use crate::util::{dumps, now, J};
use serde_json::{json, Map, Value};
use std::collections::{BTreeSet, HashMap};
use std::fs;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

pub const OPENROUTER_MODELS_URL: &str = "https://openrouter.ai/api/v1/models";
/// Daily ECB reference rate, used to show the OpenRouter value in euros.
pub const FX_URL: &str = "https://api.frankfurter.dev/v1/latest?base=USD&symbols=EUR";
pub const PRICING_TTL: f64 = 6.0 * 3600.0;

#[derive(Clone, Debug, PartialEq)]
pub struct Price {
    pub prompt: f64,
    pub completion: f64,
    pub cache_read: Option<f64>,
}

impl Price {
    fn to_json(&self) -> Value {
        let mut v = json!({"prompt": self.prompt, "completion": self.completion});
        if let Some(c) = self.cache_read {
            v["cache_read"] = json!(c);
        }
        v
    }
}

#[derive(Default)]
struct State {
    fetched: f64,
    prices: HashMap<String, Price>,
    error: Option<String>,
    usd_eur: Option<f64>,
    fx_date: Option<String>,
    refreshing: bool,
    last_attempt: f64,
    /// model -> (OpenRouter id, exact match?)
    matches: HashMap<String, (Option<String>, bool)>,
}

pub struct Pricing {
    state: Mutex<State>,
    pub path: Option<PathBuf>,
}

fn price(value: &Value) -> Option<f64> {
    let x = match value {
        Value::Number(n) => n.as_f64()?,
        Value::String(s) => s.trim().parse::<f64>().ok()?,
        Value::Bool(b) => *b as i64 as f64,
        _ => return None,
    };
    (x >= 0.0).then_some(x)
}

/// `os.path.commonprefix` of two strings, in characters.
fn common_prefix(a: &[char], b: &[char]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

/// `difflib.SequenceMatcher(None, a, b).ratio()` (no junk, short strings).
pub fn sequence_ratio(a: &str, b: &str) -> f64 {
    let (a, b): (Vec<char>, Vec<char>) = (a.chars().collect(), b.chars().collect());
    let total = a.len() + b.len();
    if total == 0 {
        return 1.0;
    }
    let mut b2j: HashMap<char, Vec<usize>> = HashMap::new();
    for (j, c) in b.iter().enumerate() {
        b2j.entry(*c).or_default().push(j);
    }
    let longest = |alo: usize, ahi: usize, blo: usize, bhi: usize| -> (usize, usize, usize) {
        let (mut besti, mut bestj, mut bestsize) = (alo, blo, 0usize);
        let mut j2len: HashMap<usize, usize> = HashMap::new();
        for i in alo..ahi {
            let mut next: HashMap<usize, usize> = HashMap::new();
            if let Some(js) = b2j.get(&a[i]) {
                for &j in js {
                    if j < blo {
                        continue;
                    }
                    if j >= bhi {
                        break;
                    }
                    let k = j.checked_sub(1).and_then(|p| j2len.get(&p)).copied().unwrap_or(0) + 1;
                    next.insert(j, k);
                    if k > bestsize {
                        besti = i + 1 - k;
                        bestj = j + 1 - k;
                        bestsize = k;
                    }
                }
            }
            j2len = next;
        }
        (besti, bestj, bestsize)
    };
    let mut matched = 0;
    let mut queue = vec![(0, a.len(), 0, b.len())];
    while let Some((alo, ahi, blo, bhi)) = queue.pop() {
        let (i, j, k) = longest(alo, ahi, blo, bhi);
        if k > 0 {
            matched += k;
            if alo < i && blo < j {
                queue.push((alo, i, blo, j));
            }
            if i + k < ahi && j + k < bhi {
                queue.push((i + k, ahi, j + k, bhi));
            }
        }
    }
    2.0 * matched as f64 / total as f64
}

/// The paid OpenRouter model for an OpenCode model id.
///
/// Exact names win. Otherwise the closest paid model of the same family is
/// used (e.g. minimax-m3.1-flash -> minimax/minimax-m3), so new free models
/// get a value without adding them by hand. Free (`:free`), batch and
/// `~latest` aliases are skipped because they do not reflect the real price.
pub fn find_match<'a>(model: &str, ids: impl IntoIterator<Item = &'a String>) -> (Option<String>, bool) {
    let base = model.strip_suffix("-free").unwrap_or(model);
    let mut candidates = vec![base];
    if let Some(stripped) = base.strip_suffix("-preview") {
        candidates.push(stripped);
    }
    let ids: BTreeSet<&String> = ids.into_iter().filter(|i| !i.contains(':') && !i.starts_with('~') && i.contains('/')).collect();
    let name = |id: &str| id.split_once('/').map(|(_, n)| n.to_string()).unwrap_or_default();
    for candidate in &candidates {
        for id in &ids {
            if name(id) == *candidate {
                return (Some((*id).clone()), true);
            }
        }
    }
    let family = base.split('-').next().unwrap_or("");
    let base_chars: Vec<char> = base.chars().collect();
    let mut best: Option<(&String, (usize, f64, i64))> = None;
    for id in &ids {
        let n = name(id);
        if n.split('-').next().unwrap_or("") != family {
            continue;
        }
        let n_chars: Vec<char> = n.chars().collect();
        let key = (common_prefix(&base_chars, &n_chars), sequence_ratio(base, &n), -(n_chars.len() as i64));
        // max() keeps the first of equal keys.
        let better = match &best {
            None => true,
            Some((_, k)) => key.0 > k.0 || (key.0 == k.0 && (key.1 > k.1 || (key.1 == k.1 && key.2 > k.2))),
        };
        if better {
            best = Some((id, key));
        }
    }
    match best {
        Some((id, _)) => (Some(id.clone()), false),
        None => (None, false),
    }
}

impl Pricing {
    pub fn new(path: Option<PathBuf>) -> Self {
        let pricing = Pricing { state: Mutex::new(State::default()), path };
        pricing.load();
        pricing
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn load(&self) {
        let Some(path) = &self.path else { return };
        let Ok(text) = fs::read_to_string(path) else { return };
        let Ok(data) = serde_json::from_str::<Value>(&text) else { return };
        let (Some(fetched), Some(prices)) = (data.g("fetched").as_f64(), data.g("prices").as_object()) else { return };
        let mut parsed = HashMap::new();
        for (k, v) in prices {
            let (Some(prompt), Some(completion)) = (v.g("prompt").as_f64(), v.g("completion").as_f64()) else { return };
            parsed.insert(k.clone(), Price { prompt, completion, cache_read: v.g("cache_read").as_f64() });
        }
        let mut s = self.lock();
        s.fetched = fetched;
        s.prices = parsed;
        if data.g("usd_eur").truthy() {
            s.usd_eur = data.g("usd_eur").as_f64();
            s.fx_date = data.g("fx_date").as_str().map(String::from);
        }
    }

    /// Fetch USD -> EUR. A failure keeps the last known rate.
    fn refresh_rate(&self, timeout: f64) {
        let Ok(data) = json_get(FX_URL, timeout) else { return };
        let Some(rate) = price(data.g("rates").g("EUR")).filter(|r| *r > 0.0) else { return };
        let mut s = self.lock();
        s.usd_eur = Some(rate);
        let date = data.g("date").or_str();
        s.fx_date = if date.is_empty() { None } else { Some(date) };
    }

    pub fn refresh(&self, timeout: f64) -> bool {
        self.refresh_rate(timeout);
        let result = json_get(OPENROUTER_MODELS_URL, timeout).and_then(|data| {
            let mut prices = HashMap::new();
            for item in data.g("data").items() {
                let id = item.g("id").or_str();
                let pricing = item.g("pricing");
                let (Some(prompt), Some(completion)) = (price(pricing.g("prompt")), price(pricing.g("completion"))) else { continue };
                if id.is_empty() {
                    continue;
                }
                prices.insert(id, Price { prompt, completion, cache_read: price(pricing.g("input_cache_read")) });
            }
            if prices.is_empty() {
                return Err("no prices in response".to_string());
            }
            Ok(prices)
        });
        let prices = match result {
            Ok(p) => p,
            Err(e) => {
                self.lock().error = Some(e);
                return false;
            }
        };
        let data = {
            let mut s = self.lock();
            s.prices = prices;
            s.fetched = now();
            s.error = None;
            s.matches.clear();
            let prices: Map<String, Value> = s.prices.iter().map(|(k, v)| (k.clone(), v.to_json())).collect();
            json!({"fetched": s.fetched, "prices": prices, "usd_eur": s.usd_eur, "fx_date": s.fx_date})
        };
        if let Some(path) = &self.path {
            let _ = fs::write(path, dumps(&data));
        }
        true
    }

    pub fn refresh_async(self: &Arc<Self>, force: bool) {
        let t = now();
        {
            let mut s = self.lock();
            if s.refreshing {
                return;
            }
            // A missing euro rate (older pricing.json) is fetched without waiting for the TTL.
            let fresh = t - s.fetched < PRICING_TTL && s.usd_eur.is_some();
            if !force && (fresh || t - s.last_attempt < 300.0) {
                return;
            }
            s.refreshing = true;
            s.last_attempt = t;
        }
        let me = Arc::clone(self);
        std::thread::spawn(move || {
            me.refresh(20.0);
            me.lock().refreshing = false;
        });
    }

    fn lookup(&self, model: &str) -> (Option<String>, bool) {
        if let Some(found) = self.lock().matches.get(model) {
            return found.clone();
        }
        let ids: Vec<String> = self.lock().prices.keys().cloned().collect();
        let found = find_match(model, &ids);
        self.lock().matches.insert(model.to_string(), found.clone());
        found
    }

    pub fn matched(&self, model: &str) -> Option<String> {
        self.lookup(model).0
    }

    /// (OpenRouter id, similar?, price)
    pub fn for_model(&self, model: &str) -> Option<(String, bool, Price)> {
        let (id, exact) = self.lookup(model);
        let id = id?;
        let price = self.lock().prices.get(&id)?.clone();
        Some((id, !exact, price))
    }

    pub fn info(&self) -> Value {
        let s = self.lock();
        json!({
            "source": "openrouter",
            "fetched": if s.fetched != 0.0 { json!(s.fetched) } else { Value::Null },
            "error": s.error,
            "models": s.prices.len(),
            "usd_eur": s.usd_eur,
            "fx_date": s.fx_date,
        })
    }
}

pub fn usage_cost(stats: &Value, price: Option<&Price>) -> Option<f64> {
    let price = price?;
    let tokens_in = stats.g("tokens_in").int();
    let cached = stats.g("tokens_cached").int().min(tokens_in);
    let uncached = tokens_in - cached;
    let cache_rate = price.cache_read.unwrap_or(price.prompt);
    Some(uncached as f64 * price.prompt + cached as f64 * cache_rate + stats.g("tokens_out").int() as f64 * price.completion)
}

pub fn usage_report(snapshot: &Value, pricing: &Pricing) -> Value {
    let mut names: BTreeSet<String> = snapshot.g("models").obj().map(|m| m.keys().cloned().collect()).unwrap_or_default();
    names.extend(available_models(15.0, false));
    let mut rows = Vec::new();
    let mut total = 0.0;
    let mut unpriced = 0i64;
    for name in names {
        let stats = match snapshot.g("models").g(&name) {
            Value::Null => json!({"requests": 0, "tokens_in": 0, "tokens_out": 0, "tokens_cached": 0, "tokens": 0}),
            s => s.clone(),
        };
        let found = pricing.for_model(&name);
        let cost = usage_cost(&stats, found.as_ref().map(|(_, _, p)| p));
        match cost {
            Some(c) => total += c,
            None => unpriced += stats.g("tokens").int(),
        }
        let price = found.map(|(id, similar, p)| {
            json!({
                "openrouter_id": id,
                "similar": similar,
                "prompt_per_m": p.prompt * 1_000_000.0,
                "completion_per_m": p.completion * 1_000_000.0,
                "cache_read_per_m": p.cache_read.map(|c| c * 1_000_000.0),
            })
        });
        rows.push(json!({
            "model": name,
            "requests": stats.g("requests").int(),
            "tokens_in": stats.g("tokens_in").int(),
            "tokens_out": stats.g("tokens_out").int(),
            "tokens_cached": stats.g("tokens_cached").int(),
            "cost_usd": cost,
            "price": price,
        }));
    }
    json!({"rows": rows, "total_cost_usd": total, "unpriced_tokens": unpriced, "pricing": pricing.info()})
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ratio_matches_difflib() {
        // Values from Python's difflib.
        assert!((sequence_ratio("abcd", "bcde") - 0.75).abs() < 1e-9);
        assert!((sequence_ratio("minimax-m3.1-flash", "minimax-m3") - 0.7142857142857143).abs() < 1e-9);
        assert_eq!(sequence_ratio("", ""), 1.0);
    }

    #[test]
    fn finds_exact_and_similar() {
        let ids: Vec<String> = ["minimax/minimax-m3", "minimax/minimax-m2.1", "minimax/minimax-m3:free", "x/glm-5", "~x/glm-latest", "z/glm-5.1"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert_eq!(find_match("glm-5-free", &ids), (Some("x/glm-5".into()), true));
        assert_eq!(find_match("minimax-m3.1-flash-free", &ids), (Some("minimax/minimax-m3".into()), false));
        assert_eq!(find_match("big-pickle", &ids), (None, false));
    }

    #[test]
    fn cost_uses_cache_rate() {
        let p = Price { prompt: 1.0, completion: 2.0, cache_read: Some(0.5) };
        let cost = usage_cost(&json!({"tokens_in": 10, "tokens_cached": 4, "tokens_out": 3}), Some(&p)).unwrap();
        assert_eq!(cost, 6.0 + 2.0 + 6.0);
    }
}
