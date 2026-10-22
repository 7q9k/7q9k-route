#![cfg_attr(all(windows, not(debug_assertions)), windows_subsystem = "windows")]

use route7q9k::config::{self, access_host};
use route7q9k::log::{self, emit, paint};
use route7q9k::metrics::Metrics;
use route7q9k::paths;
use route7q9k::pricing::Pricing;
use route7q9k::server;
use route7q9k::upstream::{available_models, preload_models};
use route7q9k::util::thousands;
use std::env;
use std::io::Write;
use std::process::{Command, ExitCode};
use std::thread;
use std::time::Duration;

/// The release build on Windows is a window-less app that opens its own console.
const WINDOWS_APP: bool = cfg!(all(windows, not(debug_assertions)));
const MODELS_TIMEOUT: f64 = 15.0;
const AUTOSAVE_INTERVAL: f64 = 2.0;

const HELP: &str = "usage: 7q9k-route [-h] [--host HOST] [--port PORT] [--key KEY] [--no-save]
                  [--refresh-models] [--no-browser] [--background]

OpenAI-compatible local API for OpenCode Zen free models.

options:
  -h, --help        show this help message and exit
  --host HOST       bind address; overrides the dashboard setting
  --port PORT       port; overrides the dashboard setting
  --key KEY         use a specific API key instead of the generated one
  --no-save         do not load or save usage statistics
  --refresh-models  refresh the model list on start
  --no-browser      do not open the dashboard on start
  --background      start without a window (Windows)";

#[derive(Default)]
struct Args {
    host: Option<String>,
    port: Option<u16>,
    key: Option<String>,
    no_save: bool,
    refresh_models: bool,
    no_browser: bool,
    background: bool,
}

fn parse_args() -> Result<Args, String> {
    let mut args = Args::default();
    let mut items = env::args().skip(1);
    while let Some(item) = items.next() {
        let (name, inline) = match item.split_once('=') {
            Some((n, v)) if n.starts_with("--") => (n.to_string(), Some(v.to_string())),
            _ => (item.clone(), None),
        };
        let mut value = |flag: &str| {
            inline.clone().or_else(|| items.next()).ok_or_else(|| format!("argument {flag}: expected one argument"))
        };
        match name.as_str() {
            "-h" | "--help" => {
                println!("{HELP}");
                std::process::exit(0);
            }
            "--host" => args.host = Some(value("--host")?),
            "--port" => {
                let raw = value("--port")?;
                let port: i64 =
                    raw.trim().parse().map_err(|_| format!("argument --port: invalid int value: '{raw}'"))?;
                if !(1..=65535).contains(&port) {
                    return Err("--port must be between 1 and 65535".into());
                }
                args.port = Some(port as u16);
            }
            "--key" => args.key = Some(value("--key")?),
            "--no-save" => args.no_save = true,
            "--refresh-models" => args.refresh_models = true,
            "--no-browser" => args.no_browser = true,
            "--background" => args.background = true,
            other => return Err(format!("unrecognized arguments: {other}")),
        }
    }
    Ok(args)
}

fn open_browser(url: &str) {
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        let _ = Command::new("rundll32").args(["url.dll,FileProtocolHandler", url]).creation_flags(0x0800_0000).spawn();
    }
    #[cfg(target_os = "macos")]
    let _ = Command::new("open").arg(url).spawn();
    #[cfg(all(unix, not(target_os = "macos")))]
    let _ = Command::new("xdg-open").arg(url).spawn();
}

fn restart_process() {
    let Ok(exe) = env::current_exe() else { return };
    let mut command = Command::new(exe);
    command.args(env::args_os().skip(1)).env("ROUTE7Q9K_RESTARTED", "1");
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        // DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW
        let _ = command.creation_flags(0x0000_0008 | 0x0000_0200 | 0x0800_0000).spawn();
    }
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        let _ = command.exec();
    }
}

fn main() -> ExitCode {
    let args = match parse_args() {
        Ok(args) => args,
        Err(message) => {
            eprintln!("usage: 7q9k-route [-h] [--host HOST] [--port PORT] [--key KEY] [--no-save] ...");
            eprintln!("7q9k-route: error: {message}");
            return ExitCode::from(2);
        }
    };
    match run(args) {
        Ok(code) => code,
        Err(message) => {
            emit("");
            emit("  7q9k route could not start:");
            emit(&format!("  {message}"));
            if let Ok(mut file) = std::fs::OpenOptions::new().create(true).append(true).open(paths::log_file()) {
                let _ = writeln!(file, "{message}");
            }
            if !log::logging_to_file() && WINDOWS_APP {
                emit(&format!("  Details are in {}. This window closes in 30 seconds.", paths::log_file().display()));
                thread::sleep(Duration::from_secs(30));
                log::windows_console_close();
            }
            ExitCode::from(1)
        }
    }
}

fn run(args: Args) -> Result<ExitCode, String> {
    let restarted = env::var("ROUTE7Q9K_RESTARTED").as_deref() == Ok("1");
    env::remove_var("ROUTE7Q9K_RESTARTED");
    let mut show_window = WINDOWS_APP && !args.background && !restarted;
    if show_window && !log::windows_console_open() {
        show_window = false;
    }
    paths::ensure_data_dir();
    if WINDOWS_APP && env::var_os("ROUTE7Q9K_DATA_DIR").is_none() {
        paths::migrate_legacy_files(&paths::base_dir(), &paths::data_dir());
        paths::migrate_legacy_data_dir(&paths::legacy_data_dir(), &paths::data_dir());
    }
    if !show_window && (WINDOWS_APP || (cfg!(windows) && (restarted || args.background))) {
        log::open_log_file();
    }

    let stored = config::load_config();
    *config::config() = stored.clone();
    let overrides = config::env_overrides();
    let cli_host = args.host.clone().or_else(|| overrides.access.as_deref().and_then(access_host).map(String::from));
    let cli_port = args.port.or(overrides.port);
    let host = cli_host.clone().unwrap_or_else(|| access_host(&stored.access).unwrap_or("0.0.0.0").to_string());
    let port = cli_port.unwrap_or(stored.port);
    if let Some(key) = args.key.clone().or_else(|| env::var("ROUTE7Q9K_API_KEY").ok()).filter(|k| !k.is_empty()) {
        config::write_key(key.trim()).map_err(|e| format!("could not write the API key: {e}"))?;
    }
    let browser = WINDOWS_APP && stored.open_browser && !(args.no_browser || args.background || restarted);
    let dashboard_url = format!("http://localhost:{port}/");

    if !restarted && server::probe_running(port) {
        emit("");
        emit("  7q9k route is already running.");
        emit(&format!("  Dashboard  {dashboard_url}"));
        emit("");
        if browser {
            open_browser(&dashboard_url);
        }
        if show_window {
            thread::sleep(Duration::from_secs(4));
            log::windows_console_close();
        }
        return Ok(ExitCode::SUCCESS);
    }

    let api_key = config::ensure_api_key().map_err(|e| format!("could not write the API key: {e}"))?;
    let models = if args.refresh_models {
        available_models(MODELS_TIMEOUT, true)
    } else {
        preload_models();
        available_models(MODELS_TIMEOUT, false)
    };
    config::refresh_autostart();
    let metrics = server::init_metrics(Metrics::new(if args.no_save { None } else { Some(paths::stats_file()) }));
    let restored = metrics.load();
    server::init_pricing(Pricing::new(Some(paths::pricing_file()))).refresh_async(false);
    server::set_api_key(&api_key);

    let listener = match server::bind(&host, port, if restarted { 15.0 } else { 0.0 }) {
        Ok(listener) => listener,
        Err(e) => {
            emit(&format!("  Cannot listen on {host}:{port}: {e}"));
            if show_window {
                emit("  Another program may use this port. This window closes in 15 seconds.");
                thread::sleep(Duration::from_secs(15));
                log::windows_console_close();
            }
            return Ok(ExitCode::from(1));
        }
    };
    {
        let mut bound = server::bound();
        bound.host = host.clone();
        bound.port = port;
        bound.open = host == "0.0.0.0" || host == "::";
        bound.cli_host = cli_host;
        bound.cli_port = cli_port;
    }

    if !log::logging_to_file() {
        log::console_init(port);
    }
    if show_window {
        server::print_window_banner(port);
    } else {
        server::print_banner(port, &host, &api_key, &models);
        if restored {
            let snap = metrics.snapshot();
            let tokens = snap["tokens_in"].as_i64().unwrap_or(0) + snap["tokens_out"].as_i64().unwrap_or(0);
            let name = paths::stats_file().file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
            emit(&format!(
                "  {}{}",
                paint("Stats     ", "cyan"),
                paint(
                    &format!(
                        "loaded from {name}: {} requests, {} tokens",
                        snap["requests"].as_i64().unwrap_or(0),
                        thousands(tokens)
                    ),
                    "dim"
                )
            ));
            emit("");
        }
    }
    if metrics.path.is_some() {
        thread::spawn(move || metrics.autosave_loop(AUTOSAVE_INTERVAL));
    }
    let refresh_models = args.refresh_models;
    thread::spawn(move || {
        if !refresh_models {
            available_models(MODELS_TIMEOUT, true);
        }
        server::sync_integrations();
    });
    if browser {
        let url = dashboard_url.clone();
        thread::spawn(move || {
            thread::sleep(Duration::from_millis(300));
            open_browser(&url);
        });
    }
    if show_window {
        thread::spawn(|| {
            thread::sleep(Duration::from_secs(6));
            log::open_log_file();
            log::windows_console_close();
        });
    }
    let _ = ctrlc::set_handler(|| {
        if !server::stopping() {
            emit("\nShutting down.");
            server::shutdown();
        }
    });

    server::serve(listener);
    if metrics.path.is_some() {
        let _ = metrics.save();
    }
    if server::RESTART_REQUESTED.load(std::sync::atomic::Ordering::SeqCst) {
        restart_process();
    }
    Ok(ExitCode::SUCCESS)
}
