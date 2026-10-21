"""Draws the 7q9k route icon: the 7q9k hexagon (same purple gradient as
7q9k-Agent) with a white winding route: a start dot, an S-shaped path and a
destination ring. Writes a multi-size .ico and a 256px .png.

Geometry uses a 112x112 viewBox like the 7q9k-Agent logo. Small sizes use a
simpler, bolder route so it stays readable in the taskbar and as a favicon.
"""

import math
import struct
import sys
import zlib
from pathlib import Path

SIZES = (16, 24, 32, 48, 64, 128, 256)
SS = 5  # supersampling per axis

HEXAGON = [(56, 10), (95.8, 33), (95.8, 79), (56, 102), (16.2, 79), (16.2, 33)]
ROUNDING = 8.0  # half of the design's round-joined outline stroke
GRADIENT = ((0.0, (0xD8, 0xC2, 0xFF)), (0.45, (0x9B, 0x6B, 0xFF)), (1.0, (0x6A, 0x38, 0xE6)))
WHITE = (255, 255, 255)


def arc(cx, cy, r, a0, a1, steps=24):
    """Points on a circle from angle a0 to a1 (degrees, y pointing down)."""
    return [(cx + r * math.cos(math.radians(a0 + (a1 - a0) * i / steps)),
             cy + r * math.sin(math.radians(a0 + (a1 - a0) * i / steps))) for i in range(steps + 1)]


# Route per variant: path, start dot (center, radius) and destination ring
# (center, radius, thickness).
DETAILED = {
    # Right along the bottom, U-turn up, left, U-turn up, right to the goal.
    "path": [(38, 74)] + arc(66, 65, 9, 90, -90) + arc(46, 47, 9, 90, 270) + [(70, 38)],
    "stroke": 7.0,
    "dot": ((38, 74), 9.0),
    "ring": ((75, 38), 8.5, 5.5),
}
BOLD = {
    # One step: right, up, right.
    "path": [(34, 74), (56, 74), (56, 40), (74, 40)],
    "stroke": 12.0,
    "dot": ((34, 74), 11.0),
    "ring": ((76, 40), 12.0, 8.0),
}


def seg_dist(px, py, a, b):
    (ax, ay), (bx, by) = a, b
    dx, dy = bx - ax, by - ay
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def line_dist(px, py, pts):
    return min(seg_dist(px, py, pts[i], pts[i + 1]) for i in range(len(pts) - 1))


def inside_hexagon(px, py):
    inside, j = False, len(HEXAGON) - 1
    for i, (xi, yi) in enumerate(HEXAGON):
        xj, yj = HEXAGON[j]
        if (yi > py) != (yj > py) and px < (xj - xi) * (py - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


def mix(a, b, t):
    return tuple(a[i] + (b[i] - a[i]) * t for i in range(3))


def gradient(t):
    t = max(0.0, min(1.0, t))
    for (t0, c0), (t1, c1) in zip(GRADIENT, GRADIENT[1:]):
        if t <= t1:
            return mix(c0, c1, (t - t0) / (t1 - t0))
    return GRADIENT[-1][1]


def color_at(x, y, mark):
    """Color of a point in the 112 viewBox, or None outside the icon."""
    closed = HEXAGON + [HEXAGON[0]]
    if not inside_hexagon(x, y) and line_dist(x, y, closed) > ROUNDING:
        return None
    (dx, dy), dot_r = mark["dot"]
    (rx, ry), ring_r, ring_w = mark["ring"]
    ring = math.hypot(x - rx, y - ry)
    on_mark = (
        (line_dist(x, y, mark["path"]) <= mark["stroke"] / 2 and ring > ring_r - ring_w / 2)
        or math.hypot(x - dx, y - dy) <= dot_r
        or abs(ring - ring_r) <= ring_w / 2
    )
    if on_mark:
        return WHITE
    base = gradient((y - 2) / 108)
    sheen = 0.3 * max(0.0, 1 - math.hypot((x - 56) / 112, (y - 2) / 112) / 0.8)
    return mix(base, WHITE, sheen)


def pixels(size):
    mark = BOLD if size <= 32 else DETAILED
    # Map the icon square onto the viewBox, keeping the hexagon centered.
    lo, hi = 2.0, 110.0
    rows = []
    for py in range(size):
        row = bytearray()
        for px in range(size):
            acc, hits = [0.0, 0.0, 0.0], 0
            for sy in range(SS):
                for sx in range(SS):
                    x = lo + (px + (sx + 0.5) / SS) / size * (hi - lo)
                    y = lo + (py + (sy + 0.5) / SS) / size * (hi - lo)
                    c = color_at(x, y, mark)
                    if c is not None:
                        acc = [acc[i] + c[i] for i in range(3)]
                        hits += 1
            if hits:
                row += bytes(round(v / hits) for v in acc) + bytes((round(255 * hits / (SS * SS)),))
            else:
                row += bytes(4)
        rows.append(bytes(row))
    return rows


def png(size: int) -> bytes:
    raw = b"".join(b"\x00" + r for r in pixels(size))

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
