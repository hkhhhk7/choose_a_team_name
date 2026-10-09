"""Make a printable A4 sheet of AprilTags (tag36h11).

    python tools/generate_tags.py                       # every tag in configs/bottles.yaml
    python tools/generate_tags.py --ids 1-6 --size-mm 30  # specific IDs, one size

Print the PNG at 100% scale ("actual size", not "fit to page"), then measure
one black square with a ruler: its edge must match tag_size_m (or
cap_tag_size_m) in configs/bottles.yaml.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_lab  # noqa: E402

DPI = 300
A4_MM = (210, 297)


def mm_to_px(mm: float) -> int:
    return int(round(mm / 25.4 * DPI))


def parse_ids(specs: list[str]) -> list[int]:
    ids: list[int] = []
    for spec in specs:
        for part in spec.split(","):
            if "-" in part:
                a, b = part.split("-")
                ids += list(range(int(a), int(b) + 1))
            elif part:
                ids.append(int(part))
    return ids


def tags_from_lab(path: str) -> list[tuple[int, float, str]]:
    lab = load_lab(path)
    out = []
    for b in lab["bottles"]:
        out.append((b["body_tag"], lab["tag_sizes"][b["body_tag"]], f"{b['name']} BODY"))
        if b.get("cap_tag") is not None:
            out.append((b["cap_tag"], lab["tag_sizes"][b["cap_tag"]], f"{b['name']} CAP"))
    for c in lab["containers"]:
        out.append((c["tag"], lab["tag_sizes"][c["tag"]], c["name"]))
    if lab.get("table_tag") is not None:
        out.append((lab["table_tag"], lab["tag_sizes"][lab["table_tag"]], "TABLE (tape down)"))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lab", default="configs/bottles.yaml")
    ap.add_argument("--ids", nargs="+", help="print these IDs instead of the lab config")
    ap.add_argument("--size-mm", type=float, default=30.0, help="black-square edge for --ids")
    ap.add_argument("--out", default="tags.pdf")
    args = ap.parse_args()

    if args.ids:
        tags = [(i, args.size_mm / 1000, f"id {i}") for i in parse_ids(args.ids)]
    else:
        tags = tags_from_lab(args.lab)

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    page_w, page_h = mm_to_px(A4_MM[0]), mm_to_px(A4_MM[1])
    page = np.full((page_h, page_w), 255, np.uint8)

    margin = mm_to_px(10)  # page margin
    pad = mm_to_px(6)  # white quiet zone around each tag (also cutting room)
    label_h = mm_to_px(7)
    x, y, row_h = margin, margin, 0
    for tag_id, size_m, label in tags:
        tag_px = mm_to_px(size_m * 1000)
        cell_w = tag_px + 2 * pad
        cell_h = tag_px + 2 * pad + label_h
        if x + cell_w > page_w - margin:  # next row
            x, y, row_h = margin, y + row_h, 0
        if y + cell_h > page_h - margin:
            raise SystemExit("tags do not fit on one A4 page; use smaller sizes or fewer IDs")
        tx, ty = x + pad, y + pad
        page[ty:ty + tag_px, tx:tx + tag_px] = cv2.aruco.generateImageMarker(dictionary, tag_id, tag_px)
        cv2.putText(page, label, (tx, ty + tag_px + mm_to_px(4)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, 0, 2, cv2.LINE_AA)
        cv2.putText(page, f"id {tag_id}, {size_m * 1000:g} mm", (tx, ty + tag_px + mm_to_px(6.5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, 90, 1, cv2.LINE_AA)
        cv2.rectangle(page, (x, y), (x + cell_w, y + cell_h), 190, 1)  # cut line
        x += cell_w + mm_to_px(4)
        row_h = max(row_h, cell_h + mm_to_px(4))

    # 50 mm ruler to check the print scale
    rx, ry = margin, page_h - margin - mm_to_px(8)
    cv2.rectangle(page, (rx, ry), (rx + mm_to_px(50), ry + mm_to_px(3)), 0, -1)
    cv2.putText(page, "this bar must measure exactly 50 mm (print at 100% / actual size)",
                (rx, ry - mm_to_px(2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 0, 2, cv2.LINE_AA)

    if args.out.lower().endswith(".pdf"):
        Path(args.out).write_bytes(_a4_pdf(page))
    else:
        ok, buf = cv2.imencode(".png", page)
        Path(args.out).write_bytes(_set_png_dpi(buf.tobytes(), DPI))
    print(f"wrote {args.out}: {len(tags)} tags on A4. Print at 100% / actual size.")


def _a4_pdf(gray: np.ndarray) -> bytes:
    """One A4 page holding the grayscale image edge to edge (exact physical size)."""
    import zlib

    h, w = gray.shape
    img = zlib.compress(gray.tobytes(), 9)
    pw, ph = A4_MM[0] / 25.4 * 72, A4_MM[1] / 25.4 * 72
    content = f"q {pw:.3f} 0 0 {ph:.3f} 0 0 cm /Im0 Do Q".encode()
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {pw:.3f} {ph:.3f}] "
        f"/Resources << /XObject << /Im0 4 0 R >> >> /Contents 5 0 R >>".encode(),
        f"<< /Type /XObject /Subtype /Image /Width {w} /Height {h} /ColorSpace /DeviceGray "
        f"/BitsPerComponent 8 /Filter /FlateDecode /Length {len(img)} >>\nstream\n".encode()
        + img + b"\nendstream",
        f"<< /Length {len(content)} >>\nstream\n".encode() + content + b"\nendstream",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, body in enumerate(objs, 1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{o:010d} 00000 n \n".encode() for o in offsets)
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return bytes(out)


def _set_png_dpi(png: bytes, dpi: int) -> bytes:
    """Insert a pHYs chunk so printers know the physical size."""
    import struct
    import zlib

    ppm = int(round(dpi / 0.0254))
    data = b"pHYs" + struct.pack(">IIB", ppm, ppm, 1)
    chunk = struct.pack(">I", 9) + data + struct.pack(">I", zlib.crc32(data) & 0xFFFFFFFF)
    ihdr_end = 8 + 8 + 13 + 4  # signature + IHDR length/type + data + crc
    return png[:ihdr_end] + chunk + png[ihdr_end:]


if __name__ == "__main__":
    main()
