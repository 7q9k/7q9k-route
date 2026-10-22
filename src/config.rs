//! Settings from the dashboard (config.json), the API key file and "start with Windows".

use crate::paths;
use crate::util::{token_hex, PResult, ProxyError};
use regex::Regex;
use serde_json::{json, Value};
use std::env;
use std::fs;
use std::io;
use std::net::{SocketAddr, ToSocketAddrs, UdpSocket};
use std::sync::{LazyLock, Mutex, MutexGuard};

pub const DEFAULT_PORT: u16 = 8787;

#[derive(Debug, Clone, PartialEq)]
pub struct Config {
    pub access: String,
    pub port: u16,
    pub open_browser: bool,
    pub integrations: Vec<String>,
}

impl Default for Config {
    fn default() -> Self {
        Config { access: "network".into(), port: DEFAULT_PORT, open_browser: true, integrations: Vec::new() }
    }
}

impl Config {
    pub fn to_json(&self) -> Value {
        json!({
            "access": self.access,
            "port": self.port,
            "open_browser": self.open_browser,
            "integrations": self.integrations,
        })
    }
}

pub fn access_host(access: &str) -> Option<&'static str> {
    match access {
        "local" => Some("127.0.0.1"),
        "network" => Some("0.0.0.0"),
        _ => None,
    }
}

static CONFIG: LazyLock<Mutex<Config>> = LazyLock::new(|| Mutex::new(Config::default()));

pub fn config() -> MutexGuard<'static, Config> {
    CONFIG.lock().unwrap_or_else(|e| e.into_inner())
}

/// Python `int(value)` for the port field.
fn port_value(v: &Value) -> Option<i64> {
    match v {
        Value::Number(n) => n.as_i64().or_else(|| n.as_f64().map(|f| f.trunc() as i64)),
        Value::Bool(b) => Some(*b as i64),
        Value::String(s) => s.trim().parse().ok(),
        _ => None,
    }
}

pub fn parse_config(stored: &Value) -> Config {
    let mut config = Config::default();
    let Some(map) = stored.as_object() else { return config };
    if let Some(access) = map.get("access").and_then(Value::as_str) {
        if access_host(access).is_some() {
            config.access = access.to_string();
        }
    }
    if let Some(port) = map.get("port").and_then(port_value) {
        if (1..=65535).contains(&port) {
            config.port = port as u16;
        }
    }
    if let Some(open) = map.get("open_browser") {
        // Python keeps whatever was stored and reads it with bool().
        config.open_browser = crate::util::J::truthy(open);
    }
    if let Some(Value::Array(items)) = map.get("integrations") {
        config.integrations = items.iter().filter_map(|i| i.as_str().map(String::from)).collect();
    }
    config
}

pub fn load_config() -> Config {
    let stored = fs::read_to_string(paths::config_file())
        .ok()
        .and_then(|t| serde_json::from_str::<Value>(&t).ok())
        .unwrap_or(Value::Null);
    parse_config(&stored)
}

pub struct EnvOverrides {
    pub access: Option<String>,
    pub port: Option<u16>,
}

pub fn env_overrides() -> EnvOverrides {
    let access = env::var("ROUTE7Q9K_ACCESS").ok().filter(|a| access_host(a).is_some());
    let port = env::var("ROUTE7Q9K_PORT")
        .ok()
        .filter(|p| !p.is_empty() && p.bytes().all(|b| b.is_ascii_digit()))
        .and_then(|p| p.parse::<u32>().ok())
        .filter(|p| (1..=65535).contains(p))
        .map(|p| p as u16);
    EnvOverrides { access, port }
}

/// CONFIG holds what the user set; environment overrides are applied at
/// start only and never written to config.json.
pub fn save_config() -> io::Result<()> {
    let data = serde_json::to_string_pretty(&config().to_json()).unwrap_or_default();
    fs::create_dir_all(paths::data_dir())?;
    let file = paths::config_file();
    let tmp = file.with_file_name("config.json.tmp");
    fs::write(&tmp, data)?;
    fs::rename(&tmp, &file)
}

// ---- API key ----------------------------------------------------------------

pub static KEY_PATTERN: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^[A-Za-z0-9._~+/=-]{8,200}$").expect("regex"));

pub fn api_key_cache() -> Option<String> {
    let text = fs::read_to_string(paths::key_file()).ok()?;
    let key = text.trim().to_string();
    if key.is_empty() {
        None
    } else {
        Some(key)
    }
}

pub fn write_key(key: &str) -> io::Result<String> {
    fs::create_dir_all(paths::data_dir())?;
    fs::write(paths::key_file(), format!("{key}\n"))?;
    Ok(key.to_string())
}

pub fn regenerate_api_key() -> io::Result<String> {
    write_key(&format!("sk-7q9k-{}", token_hex(20)))
}

pub fn ensure_api_key() -> io::Result<String> {
    match api_key_cache() {
        Some(key) => Ok(key),
        None => regenerate_api_key(),
    }
}

// ---- Network ----------------------------------------------------------------

pub fn bind_addr(host: &str, port: u16) -> io::Result<SocketAddr> {
    let host = host.trim_start_matches('[').trim_end_matches(']');
    (host, port)
        .to_socket_addrs()?
        .next()
        .ok_or_else(|| io::Error::new(io::ErrorKind::AddrNotAvailable, format!("cannot resolve {host}")))
}

/// Binds a listening socket the way the server does (exclusive on Windows,
/// dual stack for "::").
pub fn bind_listener(host: &str, port: u16) -> io::Result<std::net::TcpListener> {
    use socket2::{Domain, Protocol, Socket, Type};
    let addr = bind_addr(host, port)?;
    let socket = Socket::new(Domain::for_address(addr), Type::STREAM, Some(Protocol::TCP))?;
    #[cfg(windows)]
    exclusive(&socket);
    #[cfg(not(windows))]
    socket.set_reuse_address(true)?;
    if addr.is_ipv6() && addr.ip().is_unspecified() {
        let _ = socket.set_only_v6(false);
    }
    socket.bind(&addr.into())?;
    socket.listen(128)?;
    Ok(socket.into())
}

/// On Windows SO_REUSEADDR lets a second socket bind a port that is in use,
/// so a busy port would not be detected. Use exclusive binding there instead.
#[cfg(windows)]
fn exclusive(socket: &socket2::Socket) {
    use std::os::windows::io::AsRawSocket;
    use windows_sys::Win32::Networking::WinSock::{setsockopt, SOL_SOCKET, SO_EXCLUSIVEADDRUSE};
    let on: i32 = 1;
    unsafe {
        setsockopt(
            socket.as_raw_socket() as usize,
            SOL_SOCKET,
            SO_EXCLUSIVEADDRUSE,
            &on as *const i32 as *const u8,
            std::mem::size_of::<i32>() as i32,
        );
    }
}

pub fn port_available(host: &str, port: u16) -> bool {
    bind_listener(host, port).is_ok()
}

pub fn lan_ip() -> String {
    let probe = || -> io::Result<String> {
        let socket = UdpSocket::bind("0.0.0.0:0")?;
        socket.connect("9.9.9.9:53")?;
        Ok(socket.local_addr()?.ip().to_string())
    };
    probe().unwrap_or_else(|_| "127.0.0.1".into())
}

// ---- Start with Windows -----------------------------------------------------

pub const AUTOSTART_KEY: &str = r"Software\Microsoft\Windows\CurrentVersion\Run";
pub const AUTOSTART_NAME: &str = "7q9k-route";
/// Name used before the rename to 7q9k route; replaced on start.
pub const LEGACY_AUTOSTART_NAME: &str = "oc-free-api";

pub fn autostart_command() -> String {
    let exe = env::current_exe().map(|p| p.to_string_lossy().into_owned()).unwrap_or_default();
    format!("\"{exe}\" --background")
}

#[cfg(windows)]
mod registry {
    use super::*;
    use winreg::enums::{HKEY_CURRENT_USER, KEY_QUERY_VALUE, KEY_SET_VALUE};
    use winreg::RegKey;

    fn run_key(flags: u32) -> io::Result<RegKey> {
        RegKey::predef(HKEY_CURRENT_USER).open_subkey_with_flags(AUTOSTART_KEY, flags)
    }

    pub fn value() -> Option<String> {
        run_key(KEY_QUERY_VALUE).ok()?.get_value::<String, _>(AUTOSTART_NAME).ok()
    }

    /// Removes the pre-rename `oc-free-api` entry; returns whether it existed.
    pub fn take_legacy() -> bool {
        let Ok(key) = run_key(KEY_QUERY_VALUE | KEY_SET_VALUE) else { return false };
        key.get_raw_value(LEGACY_AUTOSTART_NAME).is_ok() && key.delete_value(LEGACY_AUTOSTART_NAME).is_ok()
    }

    pub fn set(enabled: bool) -> io::Result<()> {
        let key = run_key(KEY_SET_VALUE)?;
        if enabled {
            key.set_value(AUTOSTART_NAME, &autostart_command())
        } else {
            match key.delete_value(AUTOSTART_NAME) {
                Err(e) if e.kind() != io::ErrorKind::NotFound => Err(e),
                _ => Ok(()),
            }
        }
    }

    pub fn enabled() -> bool {
        run_key(KEY_QUERY_VALUE).map(|k| k.get_raw_value(AUTOSTART_NAME).is_ok()).unwrap_or(false)
    }
}

pub fn refresh_autostart() {
    #[cfg(windows)]
    {
        // Keep "start with Windows" on across the rename to 7q9k route.
        if registry::take_legacy() {
            let _ = registry::set(true);
            return;
        }
        if let Some(value) = registry::value() {
            if value != autostart_command() {
                let _ = registry::set(true);
            }
        }
    }
}

pub fn autostart_enabled() -> Option<bool> {
    #[cfg(windows)]
    {
        Some(registry::enabled())
    }
    #[cfg(not(windows))]
    {
        None
    }
}

pub fn set_autostart(enabled: bool) -> PResult<()> {
    #[cfg(windows)]
    {
        registry::set(enabled).map_err(|e| ProxyError::new(format!("could not change autostart: {e}"), 500))
    }
    #[cfg(not(windows))]
    {
        let _ = enabled;
        Err(ProxyError::new("start with Windows is only available on Windows.", 400))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_falls_back_on_bad_values() {
        let c = parse_config(&json!({"access": "x", "port": 70000, "integrations": ["a", 1], "other": 1}));
        assert_eq!(c, Config { integrations: vec!["a".into()], ..Config::default() });
        let c = parse_config(&json!({"access": "local", "port": "9000", "open_browser": false}));
        assert_eq!((c.access.as_str(), c.port, c.open_browser), ("local", 9000, false));
        assert_eq!(
            serde_json::to_string_pretty(&Config::default().to_json()).unwrap(),
            "{\n  \"access\": \"network\",\n  \"port\": 8787,\n  \"open_browser\": true,\n  \"integrations\": []\n}"
        );
    }

    #[test]
    fn key_pattern() {
        assert!(KEY_PATTERN.is_match("sk-7q9k-abc"));
        assert!(!KEY_PATTERN.is_match("short"));
        assert!(!KEY_PATTERN.is_match("has space here"));
    }

    #[test]
    fn busy_port_is_detected() {
        let listener = bind_listener("127.0.0.1", 0).unwrap();
        let port = listener.local_addr().unwrap().port();
        assert!(!port_available("127.0.0.1", port));
        drop(listener);
        assert!(port_available("127.0.0.1", port));
    }
}
