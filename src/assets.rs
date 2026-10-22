//! Dashboard page, fonts and icons, built into the binary.

pub const DASHBOARD_HTML: &[u8] = include_bytes!("../dashboard.html");
pub const ICON_ICO: &[u8] = include_bytes!("../assets/icon.ico");
pub const ICON_PNG: &[u8] = include_bytes!("../assets/icon.png");

/// Files served under /assets/: (path, content type, bytes).
pub const FILES: [(&str, &str, &[u8]); 4] = [
    ("fonts/geist.woff2", "font/woff2", include_bytes!("../assets/fonts/geist.woff2")),
    ("fonts/geist-mono.woff2", "font/woff2", include_bytes!("../assets/fonts/geist-mono.woff2")),
    ("icon.ico", "image/x-icon", ICON_ICO),
    ("icon.png", "image/png", ICON_PNG),
];

pub fn find(path: &str) -> Option<(&'static str, &'static [u8])> {
    FILES.iter().find(|(name, _, _)| *name == path).map(|(_, kind, data)| (*kind, *data))
}
