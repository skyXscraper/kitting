#!/usr/bin/env python3
"""
Writing-station OCR — EasyOCR only.

Handwriting pattern:
  Line 1 (top):    <ply_no>
  Line 2 (bottom): <start>-<end>

After OCR, optional confusion check (only if master_list validates):
  OCR digit 4 → also try as 9
  OCR digit 9 → also try as 7
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import cv2
import easyocr
import numpy as np


DIGIT_ALLOWLIST = "0123456789.- "
CONFUSION_PAIRS = (("4", "9"), ("9", "4"), ("7", "9"), ("9", "7"))


# ---------------------------------------------------------------------------
# Roll geometry
# ---------------------------------------------------------------------------


def detect_sheet_roll_mask(bgr: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int] | None]:
    h, w = bgr.shape[:2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    bright = cv2.inRange(hsv, np.array([0, 0, 135]), np.array([180, 95, 255]))
    yellow = cv2.inRange(hsv, np.array([12, 50, 70]), np.array([40, 255, 255]))
    bright[yellow > 0] = 0
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    bright[gray < 90] = 0

    search = bright.copy()
    search[: int(h * 0.30), :] = 0
    search[int(h * 0.80) :, :] = 0
    search[:, : int(w * 0.18)] = 0
    search[:, int(w * 0.90) :] = 0

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    search = cv2.morphologyEx(search, cv2.MORPH_CLOSE, kernel, iterations=3)
    search = cv2.morphologyEx(search, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(search, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros((h, w), dtype=np.uint8), None

    cx, cy = w * 0.52, h * 0.55
    best, best_score = None, -1.0
    for c in contours:
        area = cv2.contourArea(c)
        if area < (h * w) * 0.01:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        if bw / max(bh, 1) < 1.2:
            continue
        bx, by = x + bw / 2, y + bh / 2
        score = area / (1.0 + np.hypot(bx - cx, by - cy) * 0.02)
        if score > best_score:
            best_score, best = score, c
    if best is None:
        best = max(contours, key=cv2.contourArea)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(mask, [best], -1, 255, thickness=-1)
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)), iterations=1)
    x, y, bw, bh = cv2.boundingRect(mask)
    pad = 8
    box = (max(0, x - pad), max(0, y - pad), min(w, x + bw + pad), min(h, y + bh + pad))
    return mask, box


def detect_red_mask(bgr: np.ndarray, sat_min: int = 55, val_min: int = 45) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0, sat_min, val_min]), np.array([12, 255, 255])) | cv2.inRange(
        hsv, np.array([165, sat_min, val_min]), np.array([180, 255, 255])
    )
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    mask = cv2.bitwise_or(mask, cv2.inRange(lab[:, :, 1], 138, 255))
    return mask


def handwriting_roi_inside_roll(
    bgr: np.ndarray, roll_mask: np.ndarray, roll_box: tuple[int, int, int, int]
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    x1, y1, x2, y2 = roll_box
    red = cv2.bitwise_and(detect_red_mask(bgr), roll_mask)
    contours, _ = cv2.findContours(red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    usable = [c for c in contours if cv2.contourArea(c) >= 30]
    if not usable:
        return bgr[y1:y2, x1:x2].copy(), roll_box
    xs, ys, x2s, y2s = [], [], [], []
    for c in usable:
        x, y, bw, bh = cv2.boundingRect(c)
        xs.append(x)
        ys.append(y)
        x2s.append(x + bw)
        y2s.append(y + bh)
    hx1, hy1, hx2, hy2 = min(xs), min(ys), max(x2s), max(y2s)
    pad_x = int((hx2 - hx1) * 0.45) + 20
    pad_y = int((hy2 - hy1) * 0.55) + 20
    hx1, hy1 = max(x1, hx1 - pad_x), max(y1, hy1 - pad_y)
    hx2, hy2 = min(x2, hx2 + pad_x), min(y2, hy2 + pad_y)
    return bgr[hy1:hy2, hx1:hx2].copy(), (hx1, hy1, hx2, hy2)


def mask_outside_roll(bgr: np.ndarray, roll_mask: np.ndarray) -> np.ndarray:
    out = bgr.copy()
    out[roll_mask == 0] = 0
    return out


def bbox_center_inside_mask(bbox, mask: np.ndarray) -> bool:
    pts = np.array(bbox, dtype=np.float32)
    cx, cy = float(pts[:, 0].mean()), float(pts[:, 1].mean())
    h, w = mask.shape[:2]
    x, y = int(np.clip(cx, 0, w - 1)), int(np.clip(cy, 0, h - 1))
    return bool(mask[y, x] > 0)


def quad_to_xyxy(bbox, pad: int = 8) -> tuple[int, int, int, int]:
    pts = np.array(bbox, dtype=np.float32)
    return (
        int(pts[:, 0].min()) - pad,
        int(pts[:, 1].min()) - pad,
        int(pts[:, 0].max()) + pad,
        int(pts[:, 1].max()) + pad,
    )


def clamp_xyxy(box, w, h):
    x1, y1, x2, y2 = box
    return max(0, x1), max(0, y1), min(w, x2), min(h, y2)


def box_iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    aa = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    ba = max(0, bx2 - bx1) * max(0, by2 - by1)
    return inter / max(aa + ba - inter, 1)


def nms_boxes(boxes, scores, iou_thresh=0.4):
    order = sorted(range(len(boxes)), key=lambda i: scores[i], reverse=True)
    keep = []
    while order:
        i = order.pop(0)
        keep.append(i)
        order = [j for j in order if box_iou(boxes[i], boxes[j]) < iou_thresh]
    return keep


# ---------------------------------------------------------------------------
# Preprocess (mild / medium — not over-aggressive)
# ---------------------------------------------------------------------------


def preprocess_variant(bgr: np.ndarray, mode: str = "mild") -> np.ndarray:
    """
    Red ink → dark strokes on white.
    mild: soft isolation + upscale (preserves 9 loops)
    medium: slightly stronger close + upscale
    """
    if mode == "raw_upscale":
        h, w = bgr.shape[:2]
        up = cv2.resize(bgr, (w * 3, h * 3), interpolation=cv2.INTER_CUBIC)
        return up

    sat = 50 if mode == "mild" else 65
    mask = detect_red_mask(bgr, sat_min=sat, val_min=40)
    k = 2 if mode == "mild" else 3
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    # reconnect mesh-broken strokes without swallowing holes in 9
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1 if mode == "mild" else 2)
    if mode == "medium":
        mask = cv2.dilate(mask, kernel, iterations=1)

    canvas = np.full(bgr.shape[:2], 255, dtype=np.uint8)
    canvas[mask > 0] = 0
    # light blur only — no hard Otsu (Otsu often opens 9 → 4)
    canvas = cv2.GaussianBlur(canvas, (3, 3), 0)
    h, w = canvas.shape[:2]
    scale = 3 if mode == "mild" else 4
    canvas = cv2.resize(canvas, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
    return cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)


# ---------------------------------------------------------------------------
# Text helpers / master list / confusion
# ---------------------------------------------------------------------------


def sanitize(text: str) -> str:
    t = (text or "").strip()
    t = t.replace("O", "0").replace("o", "0").replace("Q", "9").replace("q", "9")
    t = t.replace("—", "-").replace("–", "-").replace("_", "-")
    t = re.sub(r"[^\d.\- ]", "", t)
    return re.sub(r"\s+", " ", t).strip()


def load_master_list(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def parse_fields(texts: list[str]) -> dict:
    cleaned = [sanitize(t) for t in texts if sanitize(t)]
    joined = " ".join(cleaned)
    ranges = re.findall(r"\d+(?:\.\d+)?\s*-\s*\d+(?:\.\d+)?", joined)
    # also allow "4.9" "19.5" merged later
    serial = None
    for t in cleaned:
        if re.fullmatch(r"\d{1,4}", t):
            serial = t
            break
    if serial is None:
        m = re.search(r"(?<![\d.])(\d{1,4})(?![\d.])", joined)
        if m:
            serial = m.group(1)
    range_c = ranges[0].replace(" ", "") if ranges else None
    if range_c is None:
        # try join two decimal-like fragments
        decimals = [t for t in cleaned if re.fullmatch(r"\d+\.\d+", t)]
        if len(decimals) >= 2:
            range_c = f"{decimals[0]}-{decimals[1]}"
    return {"serial_candidate": serial, "range_candidate": range_c, "texts": cleaned}


def grammar_ok_serial(s: str | None) -> bool:
    return bool(s and re.fullmatch(r"\d{1,4}", s))


def grammar_ok_range(r: str | None) -> bool:
    return bool(r and re.fullmatch(r"\d+(?:\.\d+)?-\d+(?:\.\d+)?", r))


def confusion_variants(text: str) -> list[str]:
    """Generate alternate readings by swapping known confusion pairs."""
    text = sanitize(text)
    if not text:
        return []
    out = {text}
    chars = list(text)
    for i, ch in enumerate(chars):
        for a, b in CONFUSION_PAIRS:
            if ch == a:
                alt = chars.copy()
                alt[i] = b
                out.add("".join(alt))
    # Full-string transliterations (44→99, 19.5→17.5, 4.9→4.7)
    out.add(text.translate(str.maketrans("49", "94")))
    out.add(text.translate(str.maketrans("79", "97")))
    out.add(text.translate(str.maketrans("47", "74")))
    # normalize dash forms like 19-5 → 19.5
    if re.fullmatch(r"\d+-\d+", text):
        out.add(text.replace("-", ".", 1))
    return sorted(out)


def hamming(a: str, b: str) -> int:
    if len(a) != len(b):
        return max(len(a), len(b))
    return sum(x != y for x, y in zip(a, b))


def master_lookup(serial: str | None, master: list[dict], max_dist: int = 2) -> dict:
    """
    Resolve OCR ply number against master list (ply_no column; legacy: serial).
    Priority: exact → exact via confusion variant → nearest hamming.
    """
    if not serial or not master:
        return {"status": "no_master_or_serial", "match": None, "distance": None}

    def row_key(r: dict) -> str | None:
        for k in ("ply_no", "serial"):
            if r.get(k) not in (None, ""):
                return str(r[k]).strip()
        return None

    by_serial = {}
    for r in master:
        key = row_key(r)
        if key:
            by_serial[key] = r

    if serial in by_serial:
        return {"status": "exact", "match": by_serial[serial], "distance": 0, "resolved_serial": serial}

    variants = confusion_variants(serial)
    # Prefer confusion variants that are exact master hits; break ties by
    # MORE digits changed (44→99 beats 44→94).
    exact_hits = []
    for v in variants:
        if v in by_serial:
            exact_hits.append((hamming(serial, v) if len(serial) == len(v) else 0, v))
    if exact_hits:
        exact_hits.sort(key=lambda x: -x[0])  # most swaps first
        _, v = exact_hits[0]
        return {
            "status": "exact",
            "match": by_serial[v],
            "distance": 0,
            "resolved_serial": v,
            "ocr_serial": serial,
            "via_variant": v,
            "note": f"OCR '{serial}' → confusion exact '{v}'",
        }

    best_row, best_dist, best_from = None, 999, None
    for key, row in by_serial.items():
        for cand in set(variants) | {serial}:
            if len(cand) != len(key):
                continue
            d = hamming(cand, key)
            if d < best_dist:
                best_dist, best_row, best_from = d, row, cand

    if best_row is not None and best_dist <= max_dist:
        return {
            "status": "nearest",
            "match": best_row,
            "distance": best_dist,
            "resolved_serial": str(best_row.get("ply_no") or best_row.get("serial")).strip(),
            "ocr_serial": serial,
            "via_variant": best_from,
            "note": (
                f"OCR '{serial}' → master ply_no "
                f"'{best_row.get('ply_no') or best_row.get('serial')}' (dist={best_dist})"
            ),
        }

    return {"status": "invalid", "match": None, "distance": best_dist if best_row else None, "ocr_serial": serial}


def confidence_action(conf: float, master_status: str) -> str:
    if master_status == "exact" and conf >= 0.55:
        return "auto"
    if master_status == "nearest" and conf >= 0.35:
        return "confirm"  # show OCR vs master pick
    if conf >= 0.7 and master_status == "exact":
        return "auto"
    if conf < 0.35 or master_status == "invalid":
        return "manual"
    return "confirm"


# ---------------------------------------------------------------------------
# EasyOCR
# ---------------------------------------------------------------------------


def read_best(reader: easyocr.Reader, img_bgr: np.ndarray) -> tuple[str, float]:
    results = reader.readtext(img_bgr, detail=1, paragraph=False, allowlist=DIGIT_ALLOWLIST)
    if not results:
        return "", 0.0
    # pick highest conf, prefer longer digit strings on ties
    best = max(results, key=lambda r: (float(r[2]), len(sanitize(r[1]))))
    return sanitize(best[1]), float(best[2])


def read_all_variants(reader: easyocr.Reader, crop_bgr: np.ndarray) -> list[dict]:
    """Run EasyOCR on raw + mild + medium preprocess; collect candidates."""
    variants = [
        ("raw", crop_bgr),
        ("raw_upscale", preprocess_variant(crop_bgr, "raw_upscale")),
        ("mild", preprocess_variant(crop_bgr, "mild")),
        ("medium", preprocess_variant(crop_bgr, "medium")),
    ]
    out = []
    for name, img in variants:
        text, conf = read_best(reader, img)
        out.append({"variant": name, "text": text, "confidence": conf})
    return out


def digit_confusion_candidates(text: str) -> list[str]:
    """
    EasyOCR often reads:
      true 9 → as 4
      true 7 → as 9

    So when we see OCR digits 4 or 9, also try:
      4 → 9
      9 → 7

    Returns original + all substitution combinations (only those two rules).
    """
    text = sanitize(text)
    if not text:
        return []

    # Positions that can be remapped
    idxs = [i for i, ch in enumerate(text) if ch in ("4", "9")]
    out = {text}
    if not idxs:
        return [text]

    # All subsets of remaps (2^n, n is small for ply/lengths)
    n = len(idxs)
    for mask in range(1, 1 << n):
        chars = list(text)
        for bit in range(n):
            if mask & (1 << bit):
                i = idxs[bit]
                if chars[i] == "4":
                    chars[i] = "9"
                elif chars[i] == "9":
                    chars[i] = "7"
        out.add("".join(chars))
    return sorted(out)


def master_index(master: list[dict]) -> dict[str, dict]:
    by_ply = {}
    for r in master:
        key = str(r.get("ply_no") or r.get("serial") or "").strip()
        if key:
            by_ply[key] = r
    return by_ply


def normalize_range_str(s: str | None) -> str:
    s = sanitize(s or "")
    return s.replace(" ", "")


def ranges_equal(a: str | None, b: str | None) -> bool:
    return normalize_range_str(a) == normalize_range_str(b)


def parse_range_floats(s: str | None) -> tuple[float, float] | None:
    s = normalize_range_str(s)
    m = re.fullmatch(r"(\d+\.?\d*)-(\d+\.?\d*)", s)
    if not m:
        return None
    try:
        return float(m.group(1)), float(m.group(2))
    except ValueError:
        return None


def range_distance(ocr_range: str | None, expected: str | None) -> float:
    """Lower is better; large number if unparseable."""
    a = parse_range_floats(ocr_range)
    b = parse_range_floats(expected)
    if not a or not b:
        return 1e9
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def apply_confusion_master_check(
    ply_ocr: str | None,
    range_ocr: str | None,
    length_frags: list[dict],
    master: list[dict],
    soup_ranges: list[str] | None = None,
) -> dict:
    """
    Keep raw OCR, but also try 4→9 / 9→7 remaps and accept only if master_list
    has that ply_no (and start-end matches when possible).
    """
    by_ply = master_index(master)
    result = {
        "applied": False,
        "ply_corrected": None,
        "range_corrected": None,
        "rule": "OCR 4→try 9; OCR 9→try 7; accept only if in master_list",
        "candidates_tried": [],
    }
    if not ply_ocr or not by_ply:
        return result

    ply_options = digit_confusion_candidates(ply_ocr)

    # Build range options from full range string and/or fragment confusion
    range_options = set()
    if range_ocr:
        range_options.add(normalize_range_str(range_ocr))
        for c in digit_confusion_candidates(range_ocr):
            range_options.add(normalize_range_str(c))

    frag_texts = []
    for f in length_frags:
        t = sanitize(f.get("text", ""))
        if re.fullmatch(r"\d+\.\d+", t):
            frag_texts.append(t)
    # unique preserve
    seen = set()
    frag_texts = [t for t in frag_texts if not (t in seen or seen.add(t))]
    if len(frag_texts) >= 2:
        left_opts = digit_confusion_candidates(frag_texts[0])
        right_opts = digit_confusion_candidates(frag_texts[1])
        for a in left_opts:
            for b in right_opts:
                try:
                    fa, fb = float(a), float(b)
                    lo, hi = (a, b) if fa <= fb else (b, a)
                    range_options.add(f"{lo}-{hi}")
                except ValueError:
                    range_options.add(f"{a}-{b}")

    for s in soup_ranges or []:
        range_options.add(normalize_range_str(s))
        for c in digit_confusion_candidates(s):
            range_options.add(normalize_range_str(c))

    # Prefer ply candidates that exist in master; among those, prefer range match
    hits = []
    for ply in ply_options:
        result["candidates_tried"].append(ply)
        row = by_ply.get(ply)
        if not row:
            continue
        expected = normalize_range_str(f"{row.get('start')}-{row.get('end')}")
        range_hit = None
        range_ok = False
        for ro in range_options:
            if ranges_equal(ro, expected):
                range_hit = ro
                range_ok = True
                break
        hits.append(
            {
                "ply": ply,
                "row": row,
                "expected_range": expected,
                "range_corrected": range_hit if range_ok else None,
                "range_ok": range_ok,
                "from_ocr_ply": ply_ocr,
            }
        )

    if not hits:
        # Fallback: unique master row whose start-end matches a range candidate
        # (helps when ply OCR is wrong but bottom line recovered well)
        range_hits = []
        for ro in range_options:
            for key, row in by_ply.items():
                expected = normalize_range_str(f"{row.get('start')}-{row.get('end')}")
                if ranges_equal(ro, expected):
                    range_hits.append((key, row, expected, ro))
        # unique ply
        unique_plys = {h[0] for h in range_hits}
        if len(unique_plys) == 1:
            key, row, expected, ro = range_hits[0]
            return {
                "applied": True,
                "ply_corrected": key,
                "range_corrected": expected,
                "match": row,
                "range_ok": True,
                "expected_range": expected,
                "candidates_tried": list(ply_options),
                "rule": "OCR 4→try 9; OCR 9→try 7; accept only if in master_list",
                "note": (
                    f"Ply OCR '{ply_ocr}' not in master after remap; "
                    f"unique start-end match → ply '{key}' ({expected})"
                ),
            }
        return result

    # Prefer full ply+range match; else uniquely-closest OCR range among remaps in master
    # (disambiguates e.g. 44 → 94 vs 99 when both exist in master)
    def best_dist(h: dict) -> float:
        d = range_distance(range_ocr, h["expected_range"])
        for ro in range_options:
            d = min(d, range_distance(ro, h["expected_range"]))
        return d

    hits.sort(key=lambda h: (not h["range_ok"], best_dist(h)))
    best = hits[0]
    applied = best["ply"] != ply_ocr or best["range_ok"]
    soft_range_ok = False

    if not best["range_ok"] and len(hits) > 1:
        dists = sorted((best_dist(h), h) for h in hits)
        # Require clear winner (next at least 2m worse) to auto-accept ply remap without exact range
        if dists[0][0] >= 1e8 or (len(dists) > 1 and dists[1][0] - dists[0][0] < 2.0):
            return {
                "applied": False,
                "ply_corrected": None,
                "range_corrected": None,
                "rule": "OCR 4→try 9; OCR 9→try 7; accept only if in master_list",
                "candidates_tried": list(ply_options),
                "note": (
                    f"Ambiguous remaps {[h['ply'] for h in hits]} without start-end match; "
                    f"keeping OCR ply '{ply_ocr}'"
                ),
                "ambiguous_hits": [h["ply"] for h in hits],
            }
        best = dists[0][1]
        applied = best["ply"] != ply_ocr
        # Soft accept master start-end when OCR range is clearly nearest (typical 4/9/7 drift)
        if dists[0][0] <= 5.0:
            soft_range_ok = True

    elif not best["range_ok"] and len(hits) == 1 and best_dist(best) <= 5.0:
        soft_range_ok = True

    range_ok = best["range_ok"] or soft_range_ok
    result.update(
        {
            "applied": applied or soft_range_ok,
            "ply_corrected": best["ply"],
            "range_corrected": best["expected_range"] if range_ok else None,
            "match": best["row"],
            "range_ok": range_ok,
            "expected_range": best["expected_range"],
            "note": (
                f"OCR ply '{ply_ocr}' → '{best['ply']}' via 4→9 / 9→7; "
                + (
                    "start-end matched master"
                    if best["range_ok"]
                    else (
                        "start-end nearest master row (soft match)"
                        if soft_range_ok
                        else "ply in master but start-end not matched"
                    )
                )
            ),
        }
    )
    return result


def pick_best_reading(cands: list[dict], det_text: str, det_conf: float) -> dict:
    """
    Choose best EasyOCR reading for a crop — no master-list / confusion rewriting.
    Prefer detector text when preprocess invents longer garbage (e.g. 19.5 → 135).
    """
    pool = [{"variant": "detector", "text": sanitize(det_text), "confidence": det_conf}] + [
        {"variant": c["variant"], "text": sanitize(c["text"]), "confidence": float(c["confidence"])}
        for c in cands
        if sanitize(c.get("text", ""))
    ]
    if not pool:
        return {"candidate": sanitize(det_text), "confidence": det_conf, "source_variant": "detector"}

    def score(c: dict) -> float:
        t = c["text"]
        s = float(c["confidence"])
        # Prefer clean ply ints and decimals over smashed strings like 135 / 19-5
        if re.fullmatch(r"\d{1,4}", t):
            s += 0.25
        elif re.fullmatch(r"\d+\.\d+", t):
            s += 0.25
        elif re.fullmatch(r"\d+\.\d+-\d+\.\d+", t):
            s += 0.3
        elif re.fullmatch(r"\d+-\d+", t):
            s -= 0.05  # missing decimal
        # Penalize preprocess that drops the decimal (19.5 → 135)
        if c["variant"] in ("mild", "medium") and re.fullmatch(r"\d{3,}", t):
            s -= 0.8
        return s

    ranked = sorted(pool, key=score, reverse=True)
    best = ranked[0]
    return {
        "candidate": best["text"],
        "confidence": best["confidence"],
        "source_variant": best["variant"],
        "score": score(best),
    }


def classify_crop_role(text: str, box: tuple[int, int, int, int], all_boxes: list) -> str:
    """
    Pattern:
      top line  → ply_no (integer)
      bottom    → start / end (decimals or start-end)
    """
    t = sanitize(text)
    if re.fullmatch(r"\d{1,4}", t):
        return "ply_no"
    if re.fullmatch(r"\d+\.\d+\s*-\s*\d+\.\d+", t) or re.fullmatch(r"\d+\.\d+-\d+\.\d+", t):
        return "range"
    if re.fullmatch(r"\d+-\d+", t):
        return "range"
    if re.fullmatch(r"\d+\.\d+", t):
        return "length_frag"
    # Long digit soup from a merged start-end line
    digits = re.sub(r"\D", "", t)
    if len(digits) >= 5 and not re.fullmatch(r"\d{1,4}", t):
        return "range_soup"
    return "unknown"


def split_range_text(text: str) -> tuple[str, str] | None:
    """Split an OCR string into (start, end) if pattern allows."""
    t = sanitize(text).replace(" ", "")
    m = re.fullmatch(r"(\d+\.\d+)-(\d+\.\d+)", t)
    if m:
        return m.group(1), m.group(2)
    m = re.fullmatch(r"(\d+\.\d+)-(\d+)", t)
    if m:
        return m.group(1), m.group(2)
    m = re.fullmatch(r"(\d+)-(\d+\.\d+)", t)
    if m:
        return m.group(1), m.group(2)
    m = re.fullmatch(r"(\d+)-(\d+)", t)
    if m:
        return m.group(1), m.group(2)
    return None


def soup_to_range_candidates(text: str) -> list[str]:
    """
    Recover start-end candidates from a merged digit string like '474765455'
    → e.g. '47.4-65.455', '47.47-65.455', ...
    """
    digits = re.sub(r"\D", "", sanitize(text))
    if len(digits) < 5:
        return []
    out = []
    for split in range(2, len(digits) - 1):
        left, right = digits[:split], digits[split:]
        for lf in (1, 2, 3):
            if lf >= len(left):
                continue
            for rf in (1, 2, 3):
                if rf >= len(right):
                    continue
                a = f"{left[:-lf]}.{left[-lf:]}"
                b = f"{right[:-rf]}.{right[-rf:]}"
                try:
                    fa, fb = float(a), float(b)
                except ValueError:
                    continue
                # Plausible packing-list lengths (metres)
                if 0 < fa < 200 and 0 < fb < 200 and fa < fb:
                    out.append(f"{a}-{b}")
    # unique preserve order
    seen = set()
    return [x for x in out if not (x in seen or seen.add(x))]


def ocr_box_halves(
    reader: easyocr.Reader,
    bgr: np.ndarray,
    box: tuple[int, int, int, int],
) -> dict:
    """Force start/end by OCR on left and right halves of a wide bottom box."""
    x1, y1, x2, y2 = box
    w = x2 - x1
    if w < 40:
        return {"start": "", "end": "", "start_conf": 0.0, "end_conf": 0.0}
    mid = x1 + w // 2
    overlap = max(8, w // 20)
    left = bgr[y1:y2, x1 : min(x2, mid + overlap)]
    right = bgr[y1:y2, max(x1, mid - overlap) : x2]

    def read_half(img: np.ndarray) -> tuple[str, float]:
        if img.size == 0 or img.shape[0] < 2 or img.shape[1] < 2:
            return "", 0.0
        candidates = []
        for im in (img, preprocess_variant(img, "mild"), preprocess_variant(img, "raw_upscale")):
            t, c = read_best(reader, im)
            t = sanitize(t)
            if not t:
                continue
            bonus = 0.35 if re.fullmatch(r"\d+\.\d+", t) else 0.0
            candidates.append((t, c, c + bonus))
        if not candidates:
            return "", 0.0
        candidates.sort(key=lambda x: x[2], reverse=True)
        return candidates[0][0], candidates[0][1]

    st, sc = read_half(left)
    en, ec = read_half(right)
    range_str = None
    if re.fullmatch(r"\d+\.\d+", st) and re.fullmatch(r"\d+\.\d+", en):
        try:
            if float(st) <= float(en):
                range_str = f"{st}-{en}"
            else:
                range_str = f"{en}-{st}"
        except ValueError:
            range_str = f"{st}-{en}"
    return {
        "start": st,
        "end": en,
        "start_conf": sc,
        "end_conf": ec,
        "range": range_str,
        "left_box": [x1, y1, mid + overlap, y2],
        "right_box": [mid - overlap, y1, x2, y2],
    }


def expand_bottom_detections(
    reader: easyocr.Reader,
    bgr: np.ndarray,
    det_meta: list[dict],
) -> list[dict]:
    """
    For wide/bottom detections that look like merged start-end:
      1) left/right half OCR
      2) parse '-' / decimal patterns
      3) digit-soup recovery
    Returns extra synthetic detections (start frag, end frag) when found.
    """
    if not det_meta:
        return []
    extras = []
    # Identify likely bottom line: widest box, or below median y, or soup/range text
    by_y = sorted(det_meta, key=lambda d: d["box"][1])
    median_y = by_y[len(by_y) // 2]["box"][1]
    widest = max(det_meta, key=lambda d: d["box"][2] - d["box"][0])

    candidates = []
    for d in det_meta:
        x1, y1, x2, y2 = d["box"]
        wide = (x2 - x1) >= 0.45 * (widest["box"][2] - widest["box"][0]) or (x2 - x1) > 180
        below = y1 >= median_y - 5
        text = d["det_text"]
        role = classify_crop_role(text, d["box"], [x["box"] for x in det_meta])
        # Only expand merged bottom lines — not already-split decimals like 4.7 / 17.5
        if role in ("range", "range_soup"):
            candidates.append(d)
        elif wide and below and (
            "-" in text or role == "range_soup" or len(re.sub(r"\D", "", text)) >= 5
        ):
            candidates.append(d)

    if not candidates:
        # fallback: widest merged-looking non-ply box
        for d in sorted(det_meta, key=lambda x: -(x["box"][2] - x["box"][0])):
            t = d["det_text"]
            if re.fullmatch(r"\d{1,4}", t) or re.fullmatch(r"\d+\.\d+", t):
                continue
            candidates.append(d)
            break

    for d in candidates:
        print(f"  [bottom-split] expanding box {d['box']} text='{d['det_text']}'")
        halves = ocr_box_halves(reader, bgr, tuple(d["box"]))
        print(
            f"    halves: start='{halves['start']}' ({halves['start_conf']:.2f})  "
            f"end='{halves['end']}' ({halves['end_conf']:.2f})  range='{halves.get('range')}'"
        )
        d["half_ocr"] = halves

        if halves.get("range"):
            extras.append(
                {
                    "box": halves["left_box"],
                    "det_text": halves["start"],
                    "det_conf": halves["start_conf"],
                    "synthetic": True,
                    "from": "left_half",
                }
            )
            extras.append(
                {
                    "box": halves["right_box"],
                    "det_text": halves["end"],
                    "det_conf": halves["end_conf"],
                    "synthetic": True,
                    "from": "right_half",
                }
            )

        # Pattern split on detector text
        sp = split_range_text(d["det_text"])
        if sp:
            print(f"    pattern-split: {sp[0]}-{sp[1]}")
            d["pattern_split"] = f"{sp[0]}-{sp[1]}"

        # Soup recovery candidates
        soups = soup_to_range_candidates(d["det_text"])
        if soups:
            print(f"    soup candidates (first 5): {soups[:5]}")
            d["soup_candidates"] = soups[:20]

    return extras


def assemble_from_pattern(crop_results: list[dict]) -> dict:
    """
    Assemble ply_no + start-end using only the handwriting pattern.
    Uses: separate boxes, half-OCR, '-' splits, and digit-soup recovery.
    """
    boxes = [tuple(c["box"]) for c in crop_results]
    ply_cands = []
    length_frags = []
    range_cands = []
    soup_ranges = []

    for c in crop_results:
        det = sanitize(c["det_text"])
        picked = sanitize(c.get("picked", {}).get("candidate", ""))
        box = tuple(c["box"])
        texts = [(det, c["det_conf"], "det"), (picked, float(c.get("picked", {}).get("confidence") or 0), "picked")]

        # Include half-OCR if present on this crop
        half = c.get("half_ocr") or {}
        if half.get("start"):
            texts.append((sanitize(half["start"]), float(half.get("start_conf") or 0), "left_half"))
        if half.get("end"):
            texts.append((sanitize(half["end"]), float(half.get("end_conf") or 0), "right_half"))
        if half.get("range"):
            range_cands.append(
                {
                    "text": sanitize(half["range"]),
                    "confidence": (float(half.get("start_conf") or 0) + float(half.get("end_conf") or 0)) / 2,
                    "box": box,
                    "src": "halves",
                    "role": "range",
                }
            )
        if c.get("pattern_split"):
            range_cands.append(
                {
                    "text": sanitize(c["pattern_split"]),
                    "confidence": c["det_conf"],
                    "box": box,
                    "src": "pattern_split",
                    "role": "range",
                }
            )
        for s in c.get("soup_candidates") or []:
            soup_ranges.append(s)

        for text, conf, src in texts:
            if not text:
                continue
            # Left/right half OCR is always start/end, never ply_no
            if src in ("left_half", "right_half"):
                if re.fullmatch(r"\d+\.\d+", text):
                    length_frags.append(
                        {"text": text, "confidence": conf, "box": box, "src": src, "role": "length_frag"}
                    )
                continue
            role = classify_crop_role(text, box, boxes)
            item = {"text": text, "confidence": conf, "box": box, "src": src, "role": role}
            if role == "ply_no":
                ply_cands.append(item)
            elif role == "length_frag":
                length_frags.append(item)
            elif role == "range":
                range_cands.append(item)
            elif role == "range_soup":
                for s in soup_to_range_candidates(text):
                    soup_ranges.append(s)

    # Prefer short ply ints (2–4 digits typical); avoid using soup as ply
    ply_cands = [p for p in ply_cands if re.fullmatch(r"\d{1,4}", p["text"])]
    # Prefer multi-digit ply (1-digit is often mesh noise), then conf, then top-most
    ply_cands.sort(
        key=lambda x: (
            0 if len(x["text"]) >= 2 else 1,
            -x["confidence"],
            x["box"][1],
        )
    )
    ply_no = ply_cands[0]["text"] if ply_cands else None
    ply_conf = ply_cands[0]["confidence"] if ply_cands else 0.0

    range_str = None
    # 1) two decimal fragments from separate boxes / halves
    if len(length_frags) >= 2:
        frags = sorted(length_frags, key=lambda x: (x["box"][0], -x["confidence"]))
        texts = []
        for f in frags:
            if re.fullmatch(r"\d+\.\d+", f["text"]) and f["text"] not in texts:
                texts.append(f["text"])
        if len(texts) >= 2:
            a, b = texts[0], texts[1]
            try:
                fa, fb = float(a), float(b)
                lo, hi = (a, b) if fa <= fb else (b, a)
                range_str = f"{lo}-{hi}"
            except ValueError:
                range_str = f"{a}-{b}"

    # 2) explicit range with TWO decimals (halves / clean pattern)
    if range_str is None and range_cands:
        good = [
            r
            for r in range_cands
            if re.fullmatch(r"\d+\.\d+-\d+\.\d+", sanitize(r["text"]).replace(" ", ""))
        ]
        if good:
            good.sort(key=lambda x: -x["confidence"])
            range_str = sanitize(good[0]["text"]).replace(" ", "")

    # 3) digit-soup recovery — prefer well-formed start-end
    soup_ranges = list(dict.fromkeys(soup_ranges))
    good_soups = [s for s in soup_ranges if re.fullmatch(r"\d+\.\d+-\d+\.\d+", s)]
    if range_str is None and good_soups:
        range_str = good_soups[0]
    elif range_str is None and range_cands:
        range_cands.sort(key=lambda x: -x["confidence"])
        range_str = sanitize(range_cands[0]["text"]).replace(" ", "")
    # If current range lacks proper decimals but soup has better, upgrade
    if range_str and not re.fullmatch(r"\d+\.\d+-\d+\.\d+", range_str) and good_soups:
        range_str = good_soups[0]

    return {
        "ply_no_ocr": ply_no,
        "ply_no_confidence": ply_conf,
        "start_end_ocr": range_str,
        "soup_range_candidates": soup_ranges[:30],
        "ply_candidates": ply_cands,
        "length_fragments": length_frags,
        "range_candidates": range_cands,
    }


# ---------------------------------------------------------------------------
# Draw / run
# ---------------------------------------------------------------------------


def draw(bgr, roll_box, dets, title: str):
    out = bgr.copy()
    if roll_box:
        x1, y1, x2, y2 = roll_box
        cv2.rectangle(out, (x1, y1), (x2, y2), (255, 180, 0), 2)
        cv2.putText(out, "sheet roll only", (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 180, 0), 2)
    for box, label in dets:
        x1, y1, x2, y2 = box
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 200, 0), 2)
        cv2.putText(out, label, (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 0), 2)
    cv2.putText(out, title, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3)
    cv2.putText(out, title, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 140, 255), 2)
    return out


def run_ocr(
    image_path: Path,
    output_dir: Path,
    master_path: Path,
    min_det_conf: float = 0.2,
    gpu: bool = False,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    crops_dir = output_dir / "crops"
    crops_dir.mkdir(exist_ok=True)

    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise FileNotFoundError(image_path)
    h, w = bgr.shape[:2]

    master = load_master_list(master_path)
    print(f"Master list: {master_path} ({len(master)} rows)")

    roll_mask, roll_box = detect_sheet_roll_mask(bgr)
    if roll_box is None:
        raise RuntimeError("Sheet roll not found")
    write_bgr, write_box = handwriting_roi_inside_roll(bgr, roll_mask, roll_box)
    wx1, wy1, _, _ = write_box

    cv2.imwrite(str(output_dir / "01_roll_mask.png"), roll_mask)
    cv2.imwrite(str(output_dir / "02_roll_only.png"), mask_outside_roll(bgr, roll_mask))
    cv2.imwrite(str(output_dir / "03_writing_crop.png"), write_bgr)

    print("Loading EasyOCR only...")
    model_dir = output_dir.parent / "easyocr_models"
    user_dir = output_dir.parent / "easyocr_user_network"
    model_dir.mkdir(parents=True, exist_ok=True)
    user_dir.mkdir(parents=True, exist_ok=True)
    reader = easyocr.Reader(
        ["en"],
        gpu=gpu,
        verbose=False,
        model_storage_directory=str(model_dir),
        user_network_directory=str(user_dir),
        download_enabled=True,
    )

    # Dual detect: word-level (split start/end) + slightly merged (recover hyphens)
    print("Detecting (word-level + merge passes)...")
    boxes, scores, det_meta = [], [], []
    for width_ths, height_ths, label in ((0.35, 0.35, "word"), (0.55, 0.4, "merge")):
        det = reader.readtext(
            write_bgr,
            detail=1,
            paragraph=False,
            allowlist=DIGIT_ALLOWLIST,
            width_ths=width_ths,
            height_ths=height_ths,
            mag_ratio=1.5,
        )
        for bbox, text, conf in det:
            if conf < min_det_conf:
                continue
            abs_quad = [[float(p[0] + wx1), float(p[1] + wy1)] for p in bbox]
            if not bbox_center_inside_mask(abs_quad, roll_mask):
                continue
            box = clamp_xyxy(quad_to_xyxy(abs_quad), w, h)
            boxes.append(box)
            scores.append(float(conf))
            det_meta.append(
                {
                    "box": box,
                    "det_text": sanitize(text),
                    "det_conf": float(conf),
                    "det_pass": label,
                }
            )
            print(f"  [det/{label}] {box}  '{text}' @ {conf:.2f}")

    keep = nms_boxes(boxes, scores)
    det_meta = [det_meta[i] for i in keep]
    det_meta.sort(key=lambda d: (d["box"][1] // 20, d["box"][0]))

    # Split merged bottom start-end into halves / pattern / soup
    extras = expand_bottom_detections(reader, bgr, det_meta)
    for e in extras:
        # avoid dupes
        if not any(box_iou(tuple(e["box"]), tuple(d["box"])) > 0.7 for d in det_meta):
            det_meta.append(e)
            print(f"  [det-extra] {e['box']}  '{e['det_text']}' from {e.get('from')}")

    crop_results = []
    draw_dets = []
    for i, meta in enumerate(det_meta):
        x1, y1, x2, y2 = [int(v) for v in meta["box"]]
        crop = bgr[y1:y2, x1:x2].copy()
        if crop.size == 0:
            continue
        prefix = f"crop_{i:02d}"
        cv2.imwrite(str(crops_dir / f"{prefix}_raw.jpg"), crop)

        if meta.get("synthetic"):
            best = {
                "candidate": meta["det_text"],
                "confidence": meta["det_conf"],
                "source_variant": meta.get("from", "synthetic"),
                "score": meta["det_conf"],
            }
            variants = []
            print(f"\n  [crop {i} synthetic/{meta.get('from')}] text='{meta['det_text']}'")
        else:
            for mode in ("mild", "medium"):
                cv2.imwrite(str(crops_dir / f"{prefix}_{mode}.jpg"), preprocess_variant(crop, mode))
            variants = read_all_variants(reader, crop)
            best = pick_best_reading(variants, meta["det_text"], meta["det_conf"])
            print(f"\n  [crop {i}] det='{meta['det_text']}'")
            for v in variants:
                print(f"    variant={v['variant']:12s}  text='{v['text']}'  conf={v['confidence']:.3f}")
            print(
                f"    → picked '{best['candidate']}' "
                f"(from {best.get('source_variant')}, score={best.get('score', 0):.2f})"
            )

        crop_results.append(
            {
                "box": list(meta["box"]),
                "det_text": meta["det_text"],
                "det_conf": meta["det_conf"],
                "variants": variants,
                "picked": best,
                "half_ocr": meta.get("half_ocr"),
                "pattern_split": meta.get("pattern_split"),
                "soup_candidates": meta.get("soup_candidates"),
                "synthetic": bool(meta.get("synthetic")),
            }
        )
        draw_dets.append((meta["box"], f"{best['candidate']}"))

    # Pattern assembly: top int = ply_no, decimals = start-end (raw OCR)
    assembled = assemble_from_pattern(crop_results)
    ply_ocr = assembled["ply_no_ocr"]
    range_ocr = assembled["start_end_ocr"]

    # Confusion check: 4→9, 9→7 — only accept if master_list validates
    confusion = apply_confusion_master_check(
        ply_ocr,
        range_ocr,
        assembled.get("length_fragments") or [],
        master,
        assembled.get("soup_range_candidates") or [],
    )

    # If OCR range missing but soup matched master via confusion, surface that
    if not range_ocr and assembled.get("soup_range_candidates"):
        # Prefer a soup candidate that matches a master row for this ply (raw or confused)
        by_ply = master_index(master)
        for ply in digit_confusion_candidates(ply_ocr or ""):
            row = by_ply.get(ply)
            if not row:
                continue
            expected = normalize_range_str(f"{row.get('start')}-{row.get('end')}")
            for s in assembled["soup_range_candidates"]:
                if ranges_equal(s, expected):
                    range_ocr = s
                    assembled["start_end_ocr"] = s
                    break
            if range_ocr:
                break
        # fallback: first plausible soup
        if not range_ocr and assembled["soup_range_candidates"]:
            range_ocr = assembled["soup_range_candidates"][0]
            assembled["start_end_ocr"] = range_ocr

    ply_final = confusion["ply_corrected"] if confusion.get("applied") else ply_ocr
    range_final = (
        confusion["range_corrected"]
        if confusion.get("applied") and confusion.get("range_corrected")
        else range_ocr
    )

    master_check = {
        "status": "ply_not_in_master",
        "match": None,
        "confusion": confusion,
    }
    if confusion.get("applied") and confusion.get("match"):
        master_check = {
            "status": "corrected_via_4to9_9to7_and_master",
            "match": confusion["match"],
            "expected_start_end": confusion.get("expected_range"),
            "range_matches": confusion.get("range_ok"),
            "note": confusion.get("note"),
            "confusion": confusion,
        }
    elif ply_ocr:
        row = master_index(master).get(ply_ocr)
        if row:
            expected = f"{row.get('start')}-{row.get('end')}"
            master_check = {
                "status": "ply_found_exact_ocr",
                "match": row,
                "expected_start_end": expected,
                "range_matches": ranges_equal(range_ocr, expected),
                "note": "OCR ply matched master without digit remap",
                "confusion": confusion,
            }
        else:
            master_check = {
                "status": "ply_not_in_master",
                "match": None,
                "note": (
                    f"OCR ply '{ply_ocr}' not in master; "
                    f"tried remaps {confusion.get('candidates_tried')}"
                ),
                "confusion": confusion,
            }

    conf = assembled["ply_no_confidence"]
    if master_check.get("status") in ("corrected_via_4to9_9to7_and_master", "ply_found_exact_ocr"):
        action = "auto" if conf >= 0.35 or confusion.get("applied") else "confirm"
    elif ply_ocr and conf >= 0.35:
        action = "confirm"
    else:
        action = "manual"

    final = {
        "ply_no_ocr": ply_ocr,
        "start_end_ocr": range_ocr,
        "ply_no_final": ply_final,
        "start_end_final": range_final,
        "master_validation": {
            "status": master_check.get("status"),
            "note": master_check.get("note"),
            "expected_start_end": master_check.get("expected_start_end"),
            "range_matches": master_check.get("range_matches"),
            "no_of_ply": (master_check.get("match") or {}).get("no_of_ply"),
            "start": (master_check.get("match") or {}).get("start"),
            "end": (master_check.get("match") or {}).get("end"),
            "confusion_rule": "if OCR has 4 try as 9; if OCR has 9 try as 7; keep only if in master_list",
            "confusion": {
                "applied": confusion.get("applied"),
                "ply_corrected": confusion.get("ply_corrected"),
                "range_corrected": confusion.get("range_corrected"),
                "candidates_tried": confusion.get("candidates_tried"),
                "note": confusion.get("note"),
            },
        },
        "operator_action": action,
        "engine": "EasyOCR-only",
        "handwriting_pattern": "top=ply_no; bottom=start-end",
    }

    print("\n===== RESULT =====")
    print(f"  Pattern:           top ply_no / bottom start-end")
    print(f"  Ply No (OCR):      {final['ply_no_ocr']}")
    print(f"  Start-End (OCR):   {final['start_end_ocr']}")
    print(f"  Ply No (final):    {final['ply_no_final']}")
    print(f"  Start-End (final): {final['start_end_final']}")
    print(f"  Master check:      {final['master_validation']['status']}")
    if confusion.get("applied"):
        print(f"  Confusion:         {confusion.get('note')}")
    print(f"  Operator action:   {final['operator_action']}")

    title = f"ply={final['ply_no_final']} | {final['start_end_final']} | {action}"
    annotated = draw(bgr, roll_box, draw_dets, title)
    ann_path = output_dir / "06_annotated.jpg"
    cv2.imwrite(str(ann_path), annotated)
    cv2.imwrite(str(output_dir / "04_annotated.jpg"), annotated)

    summary = {
        "image": str(image_path),
        "master_list": str(master_path),
        "roll_box": list(roll_box),
        "crops": crop_results,
        "assembled": assembled,
        "final": final,
        "annotated": str(ann_path),
    }
    json_path = output_dir / "ocr_result.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {ann_path}")
    print(f"Wrote {json_path}")
    return summary


def main():
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description="EasyOCR writing-station OCR (pattern: ply_no + start-end)")
    p.add_argument("--image", type=Path, default=root / "data" / r"C:\Users\Tanvi\Documents\Controlone\data\data\writing_station_single_roll.jpg")
    p.add_argument("--output-dir", type=Path, default=root / "output" / "single_roll_ocr")
    p.add_argument("--master-list", type=Path, default=root / "data" / "master_list.csv")
    p.add_argument("--min-det-conf", type=float, default=0.2)
    p.add_argument("--gpu", action="store_true")
    args = p.parse_args()
    run_ocr(args.image, args.output_dir, args.master_list, args.min_det_conf, args.gpu)


if __name__ == "__main__":
    main()