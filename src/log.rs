//! Console and log file output.

use crate::paths;
use std::fs::{self, File, OpenOptions};
use std::io::{self, IsTerminal, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

pub const LOG_MAX_BYTES: u64 = 5_000_000;

static VT: AtomicBool = AtomicBool::new(false);
static LOG: Mutex<Option<(File, u64)>> = Mutex::new(None);
/// Set once the console is gone (the window closed, or never had one).
static NO_CONSOLE: AtomicBool = AtomicBool::new(false);

pub fn vt() -> bool {
    VT.load(Ordering::Relaxed)
}

pub fn set_vt(on: bool) {
    VT.store(on, Ordering::Relaxed);
}

pub fn set_no_console(on: bool) {
    NO_CONSOLE.store(on, Ordering::Relaxed);
}

pub fn paint(text: &str, color: &str) -> String {
    if !vt() {
        return text.to_string();
    }
    let code = match color {
        "green" => "\x1b[92m",
        "bright" => "\x1b[1m",
        "dim" => "\x1b[90m",
        "cyan" => "\x1b[96m",
        "yellow" => "\x1b[93m",
        "red" => "\x1b[91m",
        "magenta" => "\x1b[95m",
        _ => "",
    };
    format!("{code}{text}\x1b[0m")
}

pub fn logging_to_file() -> bool {
    LOG.lock().unwrap_or_else(|e| e.into_inner()).is_some()
}

fn rotated_name() -> std::path::PathBuf {
    paths::file("server.log.1")
}

fn open_append() -> Option<(File, u64)> {
    let file = OpenOptions::new().create(true).append(true).open(paths::log_file()).ok()?;
    let size = file.metadata().map(|m| m.len()).unwrap_or(0);
    Some((file, size))
}

/// Writes a line to the log file when one is open, else to stdout or stderr.
pub fn emit(line: &str) {
    emit_to(line, false)
}

pub fn emit_err(line: &str) {
    emit_to(line, true)
}

fn emit_to(line: &str, stderr: bool) {
    let mut log = LOG.lock().unwrap_or_else(|e| e.into_inner());
    if let Some((_, size)) = log.as_ref() {
        if *size >= LOG_MAX_BYTES {
            *log = None;
            let _ = fs::rename(paths::log_file(), rotated_name());
            *log = open_append();
        }
    }
    if let Some((file, size)) = log.as_mut() {
        let data = format!("{line}\n");
        if file.write_all(data.as_bytes()).is_ok() {
            *size += data.len() as u64;
        }
        let _ = file.flush();
        return;
    }
    drop(log);
    if NO_CONSOLE.load(Ordering::Relaxed) {
        return;
    }
    if stderr {
        let mut out = io::stderr().lock();
        let _ = writeln!(out, "{line}");
        let _ = out.flush();
    } else {
        let mut out = io::stdout().lock();
        let _ = writeln!(out, "{line}");
        let _ = out.flush();
    }
}

pub fn open_log_file() {
    let _ = fs::create_dir_all(paths::data_dir());
    let file = paths::log_file();
    if fs::metadata(&file).map(|m| m.len() > 1_000_000).unwrap_or(false) {
        let _ = fs::rename(&file, rotated_name());
    }
    *LOG.lock().unwrap_or_else(|e| e.into_inner()) = open_append();
    set_vt(false);
}

pub fn console_init(port: u16) {
    set_vt(enable_vt(port));
}

#[cfg(windows)]
fn enable_vt(port: u16) -> bool {
    use windows_sys::Win32::System::Console::{
        GetConsoleMode, GetStdHandle, SetConsoleMode, SetConsoleOutputCP, SetConsoleTitleW,
        ENABLE_VIRTUAL_TERMINAL_PROCESSING, STD_ERROR_HANDLE, STD_OUTPUT_HANDLE,
    };
    let title = wide(&format!("7q9k route - Port {port}"));
    unsafe {
        SetConsoleTitleW(title.as_ptr());
        SetConsoleOutputCP(65001);
        for id in [STD_OUTPUT_HANDLE, STD_ERROR_HANDLE] {
            let handle = GetStdHandle(id);
            let mut mode = 0u32;
            if GetConsoleMode(handle, &mut mode) == 0 {
                return false;
            }
            if SetConsoleMode(handle, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING) == 0 {
                return false;
            }
        }
    }
    true
}

#[cfg(not(windows))]
fn enable_vt(_port: u16) -> bool {
    io::stderr().is_terminal()
}

/// Whether stdout goes somewhere a person reads.
pub fn stdout_is_terminal() -> bool {
    io::stdout().is_terminal()
}

#[cfg(windows)]
pub fn wide(text: &str) -> Vec<u16> {
    text.encode_utf16().chain(std::iter::once(0)).collect()
}

#[cfg(windows)]
fn set_console_icon() {
    use windows_sys::Win32::System::Console::GetConsoleWindow;
    use windows_sys::Win32::UI::WindowsAndMessaging::{LoadImageW, SendMessageW, IMAGE_ICON, LR_LOADFROMFILE, WM_SETICON};
    let path = std::env::temp_dir().join("7q9k-route-icon.ico");
    if !path.exists() && fs::write(&path, crate::assets::ICON_ICO).is_err() {
        return;
    }
    let name = wide(&path.to_string_lossy());
    unsafe {
        let hwnd = GetConsoleWindow();
        if hwnd.is_null() {
            return;
        }
        for (which, size) in [(0usize, 16i32), (1, 32)] {
            let handle = LoadImageW(std::ptr::null_mut(), name.as_ptr(), IMAGE_ICON, size, size, LR_LOADFROMFILE);
            if !handle.is_null() {
                SendMessageW(hwnd, WM_SETICON, which, handle as isize);
            }
        }
    }
}

/// Opens a console window for a build that has none (the window app).
#[cfg(windows)]
pub fn windows_console_open() -> bool {
    use windows_sys::Win32::System::Console::{AllocConsole, SetConsoleOutputCP, SetConsoleTitleW};
    unsafe {
        if AllocConsole() == 0 {
            return false;
        }
        let title = wide("7q9k route");
        SetConsoleTitleW(title.as_ptr());
        SetConsoleOutputCP(65001);
    }
    set_console_icon();
    set_no_console(false);
    true
}

#[cfg(not(windows))]
pub fn windows_console_open() -> bool {
    false
}

#[cfg(windows)]
pub fn windows_console_close() {
    set_no_console(true);
    unsafe {
        windows_sys::Win32::System::Console::FreeConsole();
    }
}

#[cfg(not(windows))]
pub fn windows_console_close() {
    set_no_console(true);
}

pub fn status_word(code: u16) -> &'static str {
    match code {
        200 | 201 | 204 => "OK",
        400 => "BAD REQUEST",
        401 => "UNAUTHORIZED",
        403 => "FORBIDDEN",
        404 => "NOT FOUND",
        411 => "LENGTH REQUIRED",
        413 => "TOO LARGE",
        429 => "RATE LIMITED",
        451 => "BLOCKED",
        500 => "SERVER ERROR",
        502 => "UPSTREAM ERROR",
        503 => "SERVICE UNAVAILABLE",
        504 => "UPSTREAM TIMEOUT",
        _ => "",
    }
}

pub fn status_color(code: u16) -> &'static str {
    if (200..300).contains(&code) {
        "green"
    } else if code == 429 {
        "magenta"
    } else if (400..500).contains(&code) {
        "yellow"
    } else {
        "red"
    }
}

/// Last lines of the log file, for the dashboard.
pub fn tail(max_lines: usize) -> Vec<String> {
    let text = fs::read(paths::log_file()).map(|b| String::from_utf8_lossy(&b).into_owned()).unwrap_or_default();
    let lines: Vec<&str> = text.lines().collect();
    let start = lines.len().saturating_sub(max_lines);
    lines[start..].iter().map(|s| s.to_string()).collect()
}
