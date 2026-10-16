import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "build"


def version() -> str:
    match = re.search(r'^APP_VERSION = "([\d.]+)"', (ROOT / "server.py").read_text(encoding="utf-8"), re.M)
    if not match:
        raise SystemExit("APP_VERSION not found in server.py")
    return match.group(1)


def version_file(ver: str) -> Path:
    parts = (ver.split(".") + ["0", "0", "0"])[:4]
    nums = ", ".join(str(int(p)) for p in parts)
    text = f"""VSVersionInfo(
  ffi=FixedFileInfo(filevers=({nums}), prodvers=({nums}), mask=0x3f, flags=0x0, OS=0x40004,
                    fileType=0x1, subtype=0x0, date=(0, 0)),
  kids=[
    StringFileInfo([StringTable('040904B0', [
      StringStruct('CompanyName', '7q9k'),
      StringStruct('FileDescription', 'oc-free-api'),
      StringStruct('FileVersion', '{ver}'),
      StringStruct('InternalName', 'oc-free-api'),
      StringStruct('OriginalFilename', 'oc-free-api.exe'),
      StringStruct('ProductName', 'oc-free-api'),
      StringStruct('ProductVersion', '{ver}')])]),
    VarFileInfo([VarStruct('Translation', [1033, 1200])])
  ]
)
"""
    BUILD.mkdir(exist_ok=True)
    path = BUILD / "version_info.txt"
    path.write_text(text, encoding="utf-8")
    return path


def main() -> None:
    ver = version()
    cmd = [
        sys.executable, "-m", "PyInstaller", "--onefile", "--noconsole", "--noconfirm",
        "--name", "oc-free-api",
        "--icon", str(ROOT / "assets" / "icon.ico"),
        "--version-file", str(version_file(ver)),
        "--add-data", f"{ROOT / 'dashboard.html'};.",
        "--add-data", f"{ROOT / 'assets'};assets",
        "--distpath", str(ROOT / "dist"),
        "--workpath", str(BUILD),
        "--specpath", str(BUILD),
        str(ROOT / "server.py"),
    ]
    subprocess.run(cmd, check=True)
    print(f"built dist/oc-free-api.exe ({ver})")


if __name__ == "__main__":
    main()
