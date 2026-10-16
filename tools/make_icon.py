import struct
import sys
import zlib
from pathlib import Path

BG = (17, 17, 19)
EDGE = (46, 46, 52)
RING = (237, 237, 238)
DOT = (74, 222, 128)
SIZES = (16, 24, 32, 48, 64, 128, 256)
SS = 4


def coverage(size: int):
    def shape(x: float, y: float):
        s = size
        r = s * 0.22
        inside_x = min(max(x, r), s - r)
        inside_y = min(max(y, r), s - r)
        if (x - inside_x) ** 2 + (y - inside_y) ** 2 > r * r:
            return None
        edge = s * 0.035
        if (x - inside_x) ** 2 + (y - inside_y) ** 2 > (r - edge) ** 2 and (
            x < r or x > s - r or y < r or y > s - r
        ) or x < edge or y < edge or x > s - edge or y > s - edge:
            color = EDGE
        else:
            color = BG
        cx, cy = s * 0.44, s * 0.5
        d2 = (x - cx) ** 2 + (y - cy) ** 2
        outer, inner = s * 0.25, s * 0.25 - s * (0.105 if s >= 32 else 0.13)
        if inner * inner <= d2 <= outer * outer:
            color = RING
        dx, dy = s * 0.77, s * 0.71
        if (x - dx) ** 2 + (y - dy) ** 2 <= (s * 0.085) ** 2:
            color = DOT
        return color

    rows = []
    for py in range(size):
        row = bytearray()
        for px in range(size):
            acc = [0, 0, 0, 0]
            for sy in range(SS):
                for sx in range(SS):
                    c = shape(px + (sx + 0.5) / SS, py + (sy + 0.5) / SS)
                    if c is not None:
                        acc[0] += c[0]
                        acc[1] += c[1]
                        acc[2] += c[2]
                        acc[3] += 1
            n = SS * SS
            a = acc[3]
            if a:
                row += bytes((acc[0] // a, acc[1] // a, acc[2] // a, round(255 * a / n)))
            else:
                row += bytes((0, 0, 0, 0))
        rows.append(bytes(row))
    return rows


def png(size: int) -> bytes:
    raw = b"".join(b"\x00" + r for r in coverage(size))

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    header = struct.pack(">IIBBBBB", size, size, 8, 6, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"")


def main(out: Path) -> None:
    images = [png(s) for s in SIZES]
    header = struct.pack("<HHH", 0, 1, len(images))
    offset = 6 + 16 * len(images)
    entries = b""
    for size, data in zip(SIZES, images):
        dim = 0 if size >= 256 else size
        entries += struct.pack("<BBBBHHII", dim, dim, 0, 0, 1, 32, len(data), offset)
        offset += len(data)
    out.write_bytes(header + entries + b"".join(images))
    (out.with_suffix(".png")).write_bytes(images[-1])


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("assets/icon.ico"))
