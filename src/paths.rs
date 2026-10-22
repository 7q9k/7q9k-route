//! Where the key, stats, settings and caches live.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::RwLock;

pub const DATA_FILES: [&str; 7] = [
    "api-key.txt",
    "stats.json",
    "config.json",
    "pricing.json",
    "models.json",
    "server.log",
    "server.log.1",
];

static DATA_DIR: RwLock<Option<PathBuf>> = RwLock::new(None);

fn local_app_data() -> PathBuf {
    env::var_os("LOCALAPPDATA")
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| home().join("AppData").join("Local"))
}

pub fn home() -> PathBuf {
    env::var_os("USERPROFILE")
        .filter(|v| !v.is_empty())
        .or_else(|| env::var_os("HOME").filter(|v| !v.is_empty()))
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."))
}

/// The folder that holds the executable.
pub fn base_dir() -> PathBuf {
    env::current_exe()
        .ok()
        .and_then(|p| fs::canonicalize(p).ok())
        .and_then(|p| p.parent().map(Path::to_path_buf))
        .unwrap_or_else(|| PathBuf::from("."))
}

pub fn default_data_dir() -> PathBuf {
    if let Some(dir) = env::var_os("ROUTE7Q9K_DATA_DIR").filter(|v| !v.is_empty()) {
        return PathBuf::from(dir);
    }
    if cfg!(windows) {
        return local_app_data().join("7q9k-route");
    }
    base_dir()
}

fn absolute(path: &Path) -> PathBuf {
    if let Ok(real) = fs::canonicalize(path) {
        return strip_verbatim(real);
    }
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        env::current_dir().map(|d| d.join(path)).unwrap_or_else(|_| path.to_path_buf())
    }
}

/// `canonicalize` on Windows returns `\\?\C:\...`; the plain form reads better.
fn strip_verbatim(path: PathBuf) -> PathBuf {
    let text = path.to_string_lossy();
    if let Some(rest) = text.strip_prefix(r"\\?\") {
        if !rest.starts_with("UNC") {
            return PathBuf::from(rest);
        }
    }
    path
}

pub fn data_dir() -> PathBuf {
    if let Some(dir) = DATA_DIR.read().unwrap_or_else(|e| e.into_inner()).as_ref() {
        return dir.clone();
    }
    let dir = absolute(&default_data_dir());
    *DATA_DIR.write().unwrap_or_else(|e| e.into_inner()) = Some(dir.clone());
    dir
}

pub fn use_data_dir(path: &Path) {
    *DATA_DIR.write().unwrap_or_else(|e| e.into_inner()) = Some(absolute(path));
}

pub fn file(name: &str) -> PathBuf {
    data_dir().join(name)
}

pub fn stats_file() -> PathBuf {
    file("stats.json")
}
pub fn pricing_file() -> PathBuf {
    file("pricing.json")
}
pub fn config_file() -> PathBuf {
    file("config.json")
}
pub fn log_file() -> PathBuf {
    file("server.log")
}
pub fn model_cache() -> PathBuf {
    file("models.json")
}
pub fn key_file() -> PathBuf {
    file("api-key.txt")
}
pub fn backups_dir() -> PathBuf {
    file("backups")
}

pub fn writable(path: &Path) -> bool {
    if fs::create_dir_all(path).is_err() {
        return false;
    }
    let probe = path.join(".write-test");
    fs::write(&probe, "ok").is_ok() && fs::remove_file(&probe).is_ok()
}

pub fn ensure_data_dir() {
    if writable(&data_dir()) {
        return;
    }
    let base = env::var_os("LOCALAPPDATA").filter(|v| !v.is_empty()).map(PathBuf::from).unwrap_or_else(home);
    use_data_dir(&base.join("7q9k-route"));
    let _ = fs::create_dir_all(data_dir());
}

fn same_dir(a: &Path, b: &Path) -> Option<bool> {
    Some(fs::canonicalize(a).ok()? == fs::canonicalize(b).ok()?)
}

fn move_path(old: &Path, new: &Path) -> bool {
    if fs::rename(old, new).is_ok() {
        return true;
    }
    // Different drives: copy, then remove.
    if old.is_file() && fs::copy(old, new).is_ok() {
        let _ = fs::remove_file(old);
        return true;
    }
    false
}

pub fn migrate_legacy_files(source: &Path, target: &Path) -> Vec<String> {
    let mut moved = Vec::new();
    match same_dir(source, target) {
        Some(true) => return moved,
        Some(false) => {}
        // A missing source holds nothing to move; a missing target is created below.
        None => {
            if !source.exists() {
                return moved;
            }
        }
    }
    for name in DATA_FILES {
        let (old, new) = (source.join(name), target.join(name));
        if old.is_file() && !new.exists() && fs::create_dir_all(target).is_ok() && move_path(&old, &new) {
            moved.push(name.to_string());
        }
    }
    moved
}

/// Data folder used before the rename from oc-free-api.
pub fn legacy_data_dir() -> PathBuf {
    local_app_data().join("oc-free-api")
}

/// Moves key, stats, settings and app backups from the old data folder.
pub fn migrate_legacy_data_dir(source: &Path, target: &Path) -> Vec<String> {
    let mut moved = migrate_legacy_files(source, target);
    let (old_backups, new_backups) = (source.join("backups"), target.join("backups"));
    if old_backups.is_dir() && !new_backups.exists() && fs::rename(&old_backups, &new_backups).is_ok() {
        moved.push("backups".to_string());
    }
    moved
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn migrates_missing_files_only() {
        let root = env::temp_dir().join(format!("7q9k-migrate-{}", crate::util::token_hex(4)));
        let (old, new) = (root.join("old"), root.join("new"));
        fs::create_dir_all(old.join("backups")).unwrap();
        fs::create_dir_all(&new).unwrap();
        fs::write(old.join("api-key.txt"), "sk-old\n").unwrap();
        fs::write(old.join("stats.json"), "{}").unwrap();
        fs::write(new.join("stats.json"), "{\"keep\":1}").unwrap();
        let moved = migrate_legacy_data_dir(&old, &new);
        assert_eq!(moved, vec!["api-key.txt".to_string(), "backups".to_string()]);
        assert_eq!(fs::read_to_string(new.join("stats.json")).unwrap(), "{\"keep\":1}");
        assert!(old.join("stats.json").exists());
        assert!(new.join("backups").is_dir());
        let _ = fs::remove_dir_all(root);
    }
}
