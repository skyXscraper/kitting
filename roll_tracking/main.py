#!/usr/bin/env python3
"""
Two-camera sheet-roll tracker with a persistent cross-camera roll id.

Method
------
Rather than trying to geometrically hand off a track between two camera
views (which needs calibration data about where camera 1's exit maps to
camera 2's entry, and breaks the moment the rig is moved), identity here is
anchored to the thing that's actually unique about each roll: the ply number
handwritten on it, which this project already has a full OCR pipeline for
(see ../scripts/scripts/ocr_using_paddleocr.py).

Pipeline, per camera, run in its own thread:
  1. Detect roll-shaped blobs in the frame (moving-object background
     subtraction by default; a color-mask detector adapted from
     ocr_using_paddleocr.detect_sheet_roll_mask is also available).
  2. A lightweight greedy centroid tracker keeps each blob's box stable
     frame-to-frame under a short-lived, per-camera-only "local" id, so we
     don't have to run OCR on every single frame.
  3. Every few frames, any local track without a resolved identity yet gets
     OCR'd (digit-only, same allowlist/sanitizing/4-9-7 confusion-correction
     helpers as the single-roll script) looking for its top-line ply number.
  4. A resolved ply number is looked up in a shared RollRegistry. If that
     number (or a known OCR-confusion variant of it) was seen recently by
     *either* camera, the existing global id is reused; otherwise a new one
     is created from the number itself.

Because identity is the number on the roll rather than a position or an
appearance embedding, a roll keeps its id when it moves from camera 1 to
camera 2 (or back) with no geometric calibration between the two views and
no exit/entry timing heuristics required. The honest limitation: a track
that never becomes legible to either camera (bad angle, motion blur, OCR
just fails) never gets more than a temporary "reading..." local id — this
trades a rare "no id yet" for never silently merging two different physical
rolls.

Usage
-----
    python main.py --camera1 0 --show                        # single camera, no --camera2
    python main.py --camera1 0 --camera2 1 --show             # both cameras
    python main.py --camera1 "rtsp://user:pass@192.168.1.50:554/stream1" \\
                    --camera2 "rtsp://user:pass@192.168.1.51:554/stream1"

Press 'q' (then Enter if not using --show) or Ctrl+C to stop.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Reuse the existing OCR engine, digit-confusion helpers, and the
# already-hardened video-source/quit-key helpers from the single-roll
# writing-station script instead of re-implementing any of it here.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "scripts"))
import ocr_using_paddleocr as ocrp  # noqa: E402


# ---------------------------------------------------------------------------
# Detection: find candidate roll blobs in a frame
# ---------------------------------------------------------------------------


def apply_frame_crop(frame: np.ndarray, crop: str) -> np.ndarray:
    """
    Take one lens of a dual-lens camera.

    A stereo USB camera presents both lenses as a single double-wide frame
    (3840x1080 is two 1920x1080 views side by side), which is why the ffplay
    equivalent needs `-vf crop=iw/2:ih:0:0`. Left undone, every roll appears
    twice in the frame and the tracker opens a second track for the copy.
    """
    if crop == "none":
        return frame
    half = frame.shape[1] // 2
    return frame[:, :half] if crop == "left" else frame[:, half:]


def detect_rolls_motion(bg_subtractor, bgr: np.ndarray, min_area_frac: float) -> list[tuple[int, int, int, int]]:
    """
    Generic moving-blob detector via MOG2 background subtraction. Works for
    any two-camera rig regardless of roll color/lighting, at the cost of
    needing a mostly-static camera and a short warm-up, and of a roll that
    sits still long enough being absorbed back into the background model.
    """
    fg = bg_subtractor.apply(bgr)
    fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)[1]  # drop MOG2's gray shadow pixels (~127)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel, iterations=1)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel, iterations=2)

    h, w = bgr.shape[:2]
    min_area = min_area_frac * h * w
    contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        if cv2.contourArea(c) < min_area:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        boxes.append((x, y, x + bw, y + bh))
    return boxes


def detect_rolls_color(bgr: np.ndarray, min_area_frac: float) -> list[tuple[int, int, int, int]]:
    """
    Multi-instance sibling of ocr_using_paddleocr.detect_sheet_roll_mask:
    same bright/near-white, non-yellow pixel classification for a plain
    paper sheet roll on a darker background, but returns every qualifying
    blob in the full frame instead of picking the single most central one.
    Good fallback/companion for a roll that has stopped moving.
    """
    h, w = bgr.shape[:2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    bright = cv2.inRange(hsv, np.array([0, 0, 135]), np.array([180, 95, 255]))
    yellow = cv2.inRange(hsv, np.array([12, 50, 70]), np.array([40, 255, 255]))
    bright[yellow > 0] = 0
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    bright[gray < 90] = 0

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    min_area = min_area_frac * h * w
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        if cv2.contourArea(c) < min_area:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        aspect = bw / max(bh, 1)
        if aspect < 0.5 or aspect > 2.2:  # rolls are roughly round, not thin slivers
            continue
        boxes.append((x, y, x + bw, y + bh))
    return boxes


def writing_ink_mask(bgr: np.ndarray) -> np.ndarray:
    """
    Isolate the red marker writing, and nothing else.

    Deliberately NOT ocr_using_paddleocr.detect_red_mask: that one ORs in a
    LAB a-channel test, which also lights up on skin -- with an operator
    holding the roll, that put detections on hands and faces instead of the
    writing. Measured on real frames from this rig:

        writing   hue 8 / 175   sat  71-123   Cb 114-122
        skin      hue 16-19     sat 168-255   Cb  33-88

    So saturation is useless here (skin is *more* saturated than faded marker
    seen through the plastic wrap). The separators that do work are a tight
    true-red hue band, which excludes skin's orange, and Cb: red ink sits
    near neutral while skin sits far below it.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    red = cv2.inRange(hsv, np.array([0, 45, 55]), np.array([10, 255, 255])) | cv2.inRange(
        hsv, np.array([170, 45, 55]), np.array([180, 255, 255])
    )
    cb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)[:, :, 2]
    red[cb < 100] = 0
    return cv2.morphologyEx(red, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))


def boost_writing_ink(bgr: np.ndarray, thicken: int = 1) -> np.ndarray:
    """
    Repaint the marker strokes in a strong, uniform red, leaving the rest of
    the image untouched.

    PP-OCR's *text detector* frequently finds nothing at all on faded marker
    seen through plastic wrap over a woven mesh -- not because the digits are
    unreadable, but because they never get proposed as text regions in the
    first place. Boosting the strokes recovers most of those. It is not a
    strict improvement, though: on crops already tight around the writing it
    can lose a read that worked on the raw image, so callers should try the
    raw crop as well and keep whichever produces a usable result. The strokes
    stay red rather than becoming black-on-white, so the downstream red-ink
    ROI logic in ocr_using_paddleocr still works on the result.
    """
    mask = writing_ink_mask(bgr)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    if thicken:
        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=thicken)
    out = bgr.copy()
    out[mask > 0] = (40, 40, 205)
    return out


def detect_rolls_writing(bgr: np.ndarray, min_area_frac: float, min_strokes: int = 3) -> list[tuple[int, int, int, int]]:
    """
    Find rolls by the red handwriting on them, and return a box per roll.

    The writing is the one thing in a working environment that is unique to a
    roll. "Bright" matches every white wall, ceiling light and desk; "moving"
    matches the operator and any camera shake. Individual ink strokes are
    grouped into one cluster per roll, and a cluster must contain at least
    `min_strokes` separate strokes -- real writing is several digit groups,
    while an isolated red speck is not.
    """
    h, w = bgr.shape[:2]
    mask = writing_ink_mask(bgr)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    strokes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) >= 0.00002 * h * w]
    if not strokes:
        return []

    # Grow each stroke until the digit groups of one roll run together, so a
    # roll's several lines of writing become a single cluster.
    merged = np.zeros((h, w), np.uint8)
    for x, y, bw, bh in strokes:
        cv2.rectangle(merged, (x, y), (x + bw, y + bh), 255, -1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(25, w // 8), max(15, h // 10)))
    merged = cv2.dilate(merged, kernel)

    boxes = []
    clusters, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in clusters:
        cx, cy, cw, ch = cv2.boundingRect(c)
        inside = [s for s in strokes if cx <= s[0] + s[2] / 2 <= cx + cw and cy <= s[1] + s[3] / 2 <= cy + ch]
        if len(inside) < min_strokes:
            continue
        # Tighten back onto the actual ink, then pad out towards the roll body.
        x1 = min(s[0] for s in inside)
        y1 = min(s[1] for s in inside)
        x2 = max(s[0] + s[2] for s in inside)
        y2 = max(s[1] + s[3] for s in inside)
        pad_x = int((x2 - x1) * 0.20) + 8
        pad_y = int((y2 - y1) * 0.20) + 8
        box = (max(0, x1 - pad_x), max(0, y1 - pad_y), min(w, x2 + pad_x), min(h, y2 + pad_y))
        if (box[2] - box[0]) * (box[3] - box[1]) < min_area_frac * h * w:
            continue
        boxes.append(box)
    return boxes


def detect_rolls(
    method: str, state: dict, bgr: np.ndarray, min_area_frac: float
) -> list[tuple[int, int, int, int]]:
    """Dispatch to one or more detectors and de-duplicate overlapping boxes."""
    boxes: list[tuple[int, int, int, int]] = []
    if method in ("writing", "hybrid"):
        boxes += detect_rolls_writing(bgr, min_area_frac)
    if method in ("motion", "hybrid"):
        boxes += detect_rolls_motion(state["bg_subtractor"], bgr, min_area_frac)
    if method in ("color", "hybrid"):
        boxes += detect_rolls_color(bgr, min_area_frac)
    if not boxes:
        return []
    scores = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]  # bigger blob wins an overlap
    keep = ocrp.nms_boxes(boxes, scores, iou_thresh=0.35)
    return [boxes[i] for i in keep]


# ---------------------------------------------------------------------------
# Per-camera tracking: greedy nearest-centroid association
# ---------------------------------------------------------------------------


def _centroid(box: tuple[int, int, int, int]) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def associate_detections_to_tracks(
    tracks: dict[int, dict],
    detections: list[tuple[int, int, int, int]],
    max_dist: float,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """
    Greedy nearest-centroid matching -- adequate for the handful of rolls
    typically visible at once. Swap for scipy.optimize.linear_sum_assignment
    if you ever need to track many overlapping rolls precisely.
    """
    pairs = []
    for tid, tr in tracks.items():
        tc = _centroid(tr["bbox"])
        for di, box in enumerate(detections):
            dc = _centroid(box)
            dist = math.hypot(tc[0] - dc[0], tc[1] - dc[1])
            if dist <= max_dist:
                pairs.append((dist, tid, di))
    pairs.sort(key=lambda p: p[0])

    matched_tracks: set[int] = set()
    matched_dets: set[int] = set()
    matches: list[tuple[int, int]] = []
    for _, tid, di in pairs:
        if tid in matched_tracks or di in matched_dets:
            continue
        matches.append((tid, di))
        matched_tracks.add(tid)
        matched_dets.add(di)

    unmatched_tracks = [tid for tid in tracks if tid not in matched_tracks]
    unmatched_dets = [di for di in range(len(detections)) if di not in matched_dets]
    return matches, unmatched_tracks, unmatched_dets


# ---------------------------------------------------------------------------
# Identity: read the ply_no off a tracked roll, for use as its global id
# ---------------------------------------------------------------------------


def is_plausible_ply(candidate: str | None, master_index_map: dict) -> bool:
    """
    Guard against a partial read becoming a roll id.

    OCR on a badly-framed crop happily returns a confident single digit --
    live runs produced ids '1', '2', '3', '4' and '9' from fragments of the
    real writing, each registering as its own separate roll. When a master
    list is loaded, only a ply number that actually exists in it (directly or
    via a known 4/9/7 confusion variant) is accepted; without one, fall back
    to requiring at least two digits.
    """
    if not candidate:
        return False
    if master_index_map:
        if candidate in master_index_map:
            return True
        return any(alt in master_index_map for alt in ocrp.digit_confusion_candidates(candidate))
    return len(candidate) >= 2


def read_ply_no_from_crop(reader, ocr_lock: threading.Lock, crop_bgr: np.ndarray, master_index_map: dict) -> tuple[str | None, float]:
    """
    Try to read the ply_no written on a tracked roll. Reuses the same
    digit-only OCR + sanitize helpers as the single-roll pipeline, but skips
    its full start-end assembly -- here we only need the top-line integer
    ("Line 1 (top): <ply_no>" per the handwriting pattern this whole project
    is built around), not the whole ply_no + start-end record.
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return None, 0.0
    candidates = []
    # Raw first, then with the marker strokes boosted -- the text detector
    # often proposes no regions at all on faded marker, and boosting recovers
    # those without hurting the crops that already worked.
    for image in (crop_bgr, boost_writing_ink(crop_bgr)):
        with ocr_lock:
            results = reader.readtext(image, detail=1, paragraph=False, allowlist=ocrp.DIGIT_ALLOWLIST, mag_ratio=1.5)
        for box, text, conf in results:
            text = ocrp.sanitize(text)
            if re.fullmatch(r"\d{1,4}", text):
                y_top = min(p[1] for p in box)
                candidates.append((y_top, conf, text))
        if candidates:
            break
    if not candidates:
        return None, 0.0
    candidates.sort(key=lambda c: (c[0], -c[1]))  # topmost first, then most confident
    _, conf, text = candidates[0]
    if master_index_map and text not in master_index_map:
        for alt in ocrp.digit_confusion_candidates(text):
            if alt in master_index_map:
                text = alt
                break
    return text, conf


class RollRegistry:
    """
    Shared cross-camera identity store, keyed by the ply_no read off each
    roll. This is what implements "id#1 stays id#1 moving from camera 1 to
    camera 2": identity is anchored to the number written on the physical
    roll, not to position or appearance, so whichever camera currently sees
    a roll independently arrives at the same id by reading the same (or a
    known confusion-equivalent) number -- no geometric calibration or
    exit/entry timing between the two camera views is needed.
    """

    def __init__(self, handoff_ttl_s: float = 10.0):
        self._lock = threading.Lock()
        self._rolls: dict[str, dict] = {}
        self.handoff_ttl_s = handoff_ttl_s

    def resolve(self, candidate: str, now: float) -> str:
        """Return the global id a freshly-OCR'd candidate string should use."""
        with self._lock:
            if candidate in self._rolls:
                return candidate
            for alt in ocrp.digit_confusion_candidates(candidate):
                row = self._rolls.get(alt)
                if row and (now - row["last_seen"]) <= self.handoff_ttl_s:
                    return alt  # same still-active roll, read slightly differently this time
            self._rolls[candidate] = {
                "camera": None,
                "bbox": None,
                "first_seen": now,
                "last_seen": now,
                "sightings": 0,
            }
            return candidate

    def update(self, global_id: str, camera: str, bbox, now: float) -> None:
        with self._lock:
            row = self._rolls.setdefault(global_id, {"first_seen": now, "sightings": 0})
            row["camera"] = camera
            row["bbox"] = list(bbox)
            row["last_seen"] = now
            row["sightings"] += 1

    def snapshot(self, now: float) -> list[dict]:
        with self._lock:
            out = []
            for gid, row in self._rolls.items():
                last_seen = row.get("last_seen", now)
                out.append(
                    {
                        "id": gid,
                        "camera": row.get("camera"),
                        "bbox": row.get("bbox"),
                        "age_s": round(now - row.get("first_seen", now), 1),
                        "last_seen_s_ago": round(now - last_seen, 1),
                        "sightings": row.get("sightings", 0),
                        "active": (now - last_seen) <= self.handoff_ttl_s,
                    }
                )
            return sorted(out, key=lambda r: r["id"])

    def cleanup(self, now: float, forget_after_s: float) -> None:
        """Drop rolls not seen by either camera for a long time, to bound memory."""
        with self._lock:
            stale = [gid for gid, row in self._rolls.items() if now - row.get("last_seen", 0) > forget_after_s]
            for gid in stale:
                del self._rolls[gid]


# ---------------------------------------------------------------------------
# Per-camera worker thread
# ---------------------------------------------------------------------------


def draw_annotations(frame: np.ndarray, tracks: dict[int, dict]) -> np.ndarray:
    """
    Label each box with its roll id.

    This lightweight path only reads the ply number; use tracking_and_ocr.py
    for the full read including the start-end range. It deliberately does not
    fill the range in from the master list -- a value that was not read off
    the roll must not be displayed as though it had been.
    """
    out = frame.copy()
    for tid, tr in tracks.items():
        x1, y1, x2, y2 = [int(v) for v in tr["bbox"]]
        resolved = tr["global_id"] is not None
        label = f"id {tr['global_id']}" if resolved else f"reading... (#{tid})"
        color = (0, 200, 0) if resolved else (0, 165, 255)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(out, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return out


def camera_worker(
    name: str,
    source: str,
    registry: RollRegistry,
    reader,
    ocr_lock: threading.Lock,
    master_index_map: dict,
    stop_event: threading.Event,
    shared_frames: dict[str, np.ndarray],
    frames_lock: threading.Lock,
    log_path: Path | None,
    writer_holder: dict,
    args: argparse.Namespace,
) -> None:
    cap = ocrp._open_video_capture(
        source, width=args.width, height=args.height, fps=args.fps, fourcc=args.fourcc
    )
    if not cap.isOpened():
        print(f"[{name}] could not open video source {source!r}. "
              "If this is a USB camera, check `v4l2-ctl --list-devices` and `ffplay <device>`.")
        stop_event.set()
        return
    print(f"[{name}] opened {source!r} (backend={cap.getBackendName()}, "
          f"mode={ocrp.describe_capture_mode(cap)})")
    if args.width or args.height or args.fourcc or args.fps:
        print(f"[{name}] requested {args.width or '-'}x{args.height or '-'}"
              f"@{args.fps or '-'}fps {args.fourcc or 'MJPG'} -- cameras negotiate, "
              "so compare that against the mode above.")

    bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=32, detectShadows=True)
    detector_state = {"bg_subtractor": bg_subtractor}

    tracks: dict[int, dict] = {}
    recently_dropped: list[dict] = []  # ids whose track was lost, still re-acquirable
    next_local_id = 1
    frame_idx = 0
    frame_diag: float | None = None
    writer: cv2.VideoWriter | None = None
    writer_size: tuple[int, int] | None = None  # (w, h) the writer was opened with
    writer_fps: float | None = None
    next_write_due: float | None = None  # wall-clock time the next output frame-tick is due
    recording_attempted = False
    write_error_reported = False
    RECORD_WARMUP_FRAMES = 3  # let the camera/pipeline settle before locking in a frame size
    RECORD_MAX_STALL_SECONDS = 10.0  # cap duplicate-frame catch-up after a long stall

    try:
        while not stop_event.is_set():
            ok, frame = cap.read()
            if not ok:
                print(f"[{name}] end of stream / read failure.")
                break
            frame = apply_frame_crop(frame, args.crop)
            h, w = frame.shape[:2]
            if frame_diag is None:
                frame_diag = math.hypot(w, h)
            now = time.time()

            detections = detect_rolls(args.detector, detector_state, frame, args.min_area_frac)
            max_dist = frame_diag * args.max_track_dist_frac
            matches, unmatched_tracks, unmatched_dets = associate_detections_to_tracks(tracks, detections, max_dist)

            for tid, di in matches:
                tracks[tid]["bbox"] = detections[di]
                tracks[tid]["misses"] = 0
            for tid in unmatched_tracks:
                tracks[tid]["misses"] += 1
            for di in unmatched_dets:
                box = detections[di]
                # A roll that blinked out of detection for a moment and came back in
                # roughly the same place is the same roll -- inherit its id instead of
                # starting over at "reading...", which is what made ids churn.
                inherited = None
                for gone in recently_dropped:
                    if frame_idx - gone["frame"] <= args.reacquire_frames and ocrp.box_iou(tuple(box), tuple(gone["bbox"])) >= 0.3:
                        inherited = gone["global_id"]
                        break
                tracks[next_local_id] = {
                    "bbox": box,
                    "misses": 0,
                    "global_id": inherited,
                    "last_ocr_frame": -10**9,
                    "ocr_attempts": 0,
                    "ocr_box": None,
                }
                if inherited is not None:
                    print(f"[{name}] re-acquired roll id '{inherited}' as local track #{next_local_id}")
                next_local_id += 1

            for tid in [t for t, tr in tracks.items() if tr["misses"] > args.max_missed_frames]:
                if tracks[tid]["global_id"] is not None:
                    recently_dropped.append(
                        {"bbox": tracks[tid]["bbox"], "global_id": tracks[tid]["global_id"], "frame": frame_idx}
                    )
                del tracks[tid]
            recently_dropped = [g for g in recently_dropped if frame_idx - g["frame"] <= args.reacquire_frames]

            for tr in tracks.values():
                if tr["global_id"] is not None:
                    continue
                # A materially different view of the roll deserves a fresh budget --
                # earlier attempts may simply have been looking at a bad angle.
                if tr["ocr_box"] is not None and ocrp.box_iou(tuple(tr["bbox"]), tuple(tr["ocr_box"])) < 0.5:
                    tr["ocr_attempts"] = 0
                # Once the budget is spent, back off rather than giving up forever.
                exhausted = tr["ocr_attempts"] >= args.max_ocr_attempts
                interval = args.ocr_retry_interval * (5 if exhausted else 1)
                if frame_idx - tr["last_ocr_frame"] < interval:
                    continue
                x1, y1, x2, y2 = [max(0, int(v)) for v in tr["bbox"]]
                candidate, conf = read_ply_no_from_crop(reader, ocr_lock, frame[y1:y2, x1:x2], master_index_map)
                tr["last_ocr_frame"] = frame_idx
                tr["ocr_attempts"] += 1
                tr["ocr_box"] = tr["bbox"]
                if candidate and conf >= args.min_ocr_conf and is_plausible_ply(candidate, master_index_map):
                    global_id = registry.resolve(candidate, now)
                    tr["global_id"] = global_id
                    print(f"[{name}] resolved local track to roll id '{global_id}' (read '{candidate}' @ {conf:.2f})")
                elif candidate and conf >= args.min_ocr_conf:
                    print(f"[{name}] ignoring implausible ply read '{candidate}' @ {conf:.2f} (likely a partial read)")

            for tr in tracks.values():
                if tr["global_id"] is not None:
                    registry.update(tr["global_id"], name, tr["bbox"], now)

            annotated = draw_annotations(frame, tracks)
            with frames_lock:
                shared_frames[name] = annotated

            if args.record and not recording_attempted and log_path is not None and frame_idx >= RECORD_WARMUP_FRAMES:
                recording_attempted = True
                fps = cap.get(cv2.CAP_PROP_FPS)
                if not fps or fps <= 1 or fps > 120:
                    fps = 15.0  # just a pacing clock now -- doesn't need to match real capture rate
                rec_path = log_path.parent / f"{name}_recording.mp4"
                candidate_writer = cv2.VideoWriter(str(rec_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                if candidate_writer.isOpened():
                    writer = candidate_writer
                    writer_size = (w, h)
                    writer_fps = fps
                    next_write_due = now
                    # Publish it so the main thread can finalise the file even if this
                    # worker is still busy when shutdown comes.
                    with writer_holder["lock"]:
                        writer_holder["writer"] = writer
                    print(f"[{name}] recording annotated video to {rec_path} at {w}x{h}@{fps:.0f}fps "
                          "(frame-paced to real elapsed time -- a slow OCR pass holds/duplicates a frame "
                          "instead of the file quietly running fast)")
                else:
                    print(f"[{name}] could not open video writer for {rec_path} -- recording disabled")

            if writer is not None:
                to_write = annotated
                if (w, h) != writer_size:
                    # A frame that doesn't match the size the writer was opened with will
                    # silently corrupt an mp4 container instead of erroring -- resize to
                    # match rather than feed it a mismatched frame.
                    to_write = cv2.resize(annotated, writer_size)
                # Real-time pacing: write as many output-frame ticks as have actually
                # elapsed in wall-clock time since the last write (normally 1; more if
                # this iteration was slow, e.g. a full OCR pass), so the recording's
                # timeline tracks real time regardless of how bursty processing is.
                # Never fewer than 1 -- a real captured frame is never dropped, only
                # held longer during a stall.
                ticks_due = max(1, int((now - next_write_due) * writer_fps) + 1)
                ticks_due = min(ticks_due, int(writer_fps * RECORD_MAX_STALL_SECONDS))
                try:
                    # Under the lock, and only while the holder still owns the writer --
                    # the main thread may have finalised it during shutdown.
                    with writer_holder["lock"]:
                        if writer_holder["writer"] is not None:
                            for _ in range(ticks_due):
                                writer_holder["writer"].write(to_write)
                        else:
                            writer = None
                    next_write_due += ticks_due / writer_fps
                except cv2.error as err:
                    if not write_error_reported:
                        write_error_reported = True
                        print(f"[{name}] video writer error, further writes suppressed: {err}")

            if log_path is not None and frame_idx % max(1, args.log_stride) == 0:
                with log_path.open("a", encoding="utf-8") as f:
                    for tr in tracks.values():
                        f.write(
                            json.dumps(
                                {
                                    "ts": now,
                                    "camera": name,
                                    "frame": frame_idx,
                                    "global_id": tr["global_id"],
                                    "bbox": list(tr["bbox"]),
                                }
                            )
                            + "\n"
                        )

            frame_idx += 1
    finally:
        cap.release()
        # Finalise the recording here if the main thread hasn't already. Without a
        # release() the muxer never writes the mp4 moov atom and the file is
        # unplayable, so this must happen on every exit path.
        with writer_holder["lock"]:
            if writer_holder["writer"] is not None:
                writer_holder["writer"].release()
                writer_holder["writer"] = None
                print(f"[{name}] recording finalised.")
        print(f"[{name}] stopped.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description="Two-camera sheet-roll tracker with a persistent cross-camera roll id.")
    p.add_argument("--camera1", default="0", help="Video source for camera 1 (index, /dev/videoN, file, or URL).")
    p.add_argument(
        "--camera2",
        default=None,
        help="Video source for camera 2. Omit to run single-camera (no cross-camera "
             "handoff to test yet, but detection/tracking/OCR-id all still run).",
    )
    p.add_argument(
        "--detector",
        choices=("writing", "motion", "color", "hybrid"),
        default="writing",
        help="writing=cluster the red handwriting (default; the only cue unique to a roll "
             "in a normal working environment); motion=background subtraction (also catches "
             "the operator and any camera shake); color=bright-roll mask (also catches white "
             "walls, ceilings and desks); hybrid=all three, de-duplicated.",
    )
    p.add_argument("--backend", choices=("auto", "paddle", "rapidocr"), default="auto")
    p.add_argument("--width", type=int, default=None, help="Requested capture width, e.g. 3840.")
    p.add_argument("--height", type=int, default=None, help="Requested capture height, e.g. 1080.")
    p.add_argument("--fps", type=float, default=None, help="Requested capture frame rate, e.g. 30.")
    p.add_argument("--fourcc", type=str, default=None,
                    help="Requested pixel format as a 4-character V4L2 code, e.g. YUYV or MJPG. "
                         "Defaults to MJPG, which most USB cameras need for their higher modes.")
    p.add_argument("--crop", choices=("none", "left", "right"), default="none",
                    help="Take one lens of a dual-lens camera. A stereo USB camera sends both "
                         "lenses as one double-wide frame (3840x1080 = two 1920x1080 views), and "
                         "without cropping every roll is detected twice.")
    p.add_argument("--gpu", action="store_true")
    p.add_argument("--master-list", type=Path, default=PROJECT_ROOT / "data" / "data" / "master_list.csv")
    p.add_argument("--min-area-frac", type=float, default=0.004, help="Min blob area as a fraction of frame area.")
    p.add_argument("--max-track-dist-frac", type=float, default=0.12,
                    help="Max centroid jump (as a fraction of the frame diagonal) to still count as the same track.")
    p.add_argument("--max-missed-frames", type=int, default=45, help="Frames a track may go undetected before it's dropped from that camera.")
    p.add_argument("--reacquire-frames", type=int, default=90,
                    help="After a track is dropped, how long (in frames) a new detection in roughly "
                         "the same place may still inherit its roll id instead of starting fresh.")
    p.add_argument("--ocr-retry-interval", type=int, default=10, help="Frames between OCR attempts per unresolved track.")
    p.add_argument("--max-ocr-attempts", type=int, default=12, help="Give up OCR-resolving a track after this many tries.")
    p.add_argument("--min-ocr-conf", type=float, default=0.4)
    p.add_argument("--handoff-ttl", type=float, default=10.0,
                    help="Seconds a roll stays eligible for cross-camera id reuse after its last sighting.")
    p.add_argument("--log-stride", type=int, default=15, help="Write to the sightings log every Nth frame.")
    p.add_argument("--status-interval", type=float, default=2.0, help="Seconds between printed status snapshots.")
    p.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "output")
    p.add_argument("--show", action="store_true", help="Display annotated windows for both cameras.")
    p.add_argument("--record", action="store_true",
                    help="Save the annotated video (with detection/tracking boxes) for each camera to "
                         "<output-dir>/<camera>_recording.mp4.")
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "roll_sightings.jsonl"

    master = ocrp.load_master_list(args.master_list)
    master_index_map = ocrp.master_index(master) if master else {}
    if master:
        print(f"Master list: {args.master_list} ({len(master)} rows)")
    else:
        print(f"No master list at {args.master_list} -- OCR readings won't be cross-checked against known ply numbers.")

    model_dir = args.output_dir / "ppocr_models"
    model_dir.mkdir(parents=True, exist_ok=True)
    reader = ocrp.PPOCRReader(backend=args.backend, device="gpu" if args.gpu else "cpu", lang="en", model_dir=model_dir, verbose=True)
    print(f"Loading PaddleOCR ({reader.backend})...")
    reader.warmup()
    ocr_lock = threading.Lock()  # PP-OCR inference is serialized across both camera threads

    registry = RollRegistry(handoff_ttl_s=args.handoff_ttl)
    stop_event = threading.Event()
    shared_frames: dict[str, np.ndarray] = {}
    frames_lock = threading.Lock()

    cameras = [("camera1", args.camera1)]
    if args.camera2 is not None:
        cameras.append(("camera2", args.camera2))
    else:
        print("Running single-camera (no --camera2) -- cross-camera id handoff has nothing to test yet.")
    # One holder per camera, so the main thread can finalise a recording even if
    # that camera's worker is still busy when shutdown comes.
    writer_holders = {name: {"writer": None, "lock": threading.Lock()} for name, _ in cameras}
    threads = [
        threading.Thread(
            target=camera_worker,
            args=(name, source, registry, reader, ocr_lock, master_index_map, stop_event, shared_frames,
                  frames_lock, log_path, writer_holders[name], args),
            daemon=True,
        )
        for name, source in cameras
    ]
    for t in threads:
        t.start()

    print("Tracking started. Press 'q' (then Enter, if no --show window) or Ctrl+C to stop.")
    last_status = 0.0
    try:
        while not stop_event.is_set() and any(t.is_alive() for t in threads):
            now = time.time()
            if args.show:
                with frames_lock:
                    frames = dict(shared_frames)
                try:
                    for name, frame in frames.items():
                        cv2.imshow(name, frame)
                    quit_via_window = frames and (cv2.waitKey(30) & 0xFF) == ord("q")
                except cv2.error as err:
                    print(f"--show disabled: this OpenCV build has no GUI backend ({err}). "
                          "Continuing headless -- press 'q'+Enter or Ctrl+C to stop.")
                    args.show = False
                    quit_via_window = False
                if quit_via_window:
                    print("Stopped by user ('q').")
                    stop_event.set()
                    break
            else:
                if ocrp._quit_key_pressed():
                    print("Stopped by user ('q').")
                    stop_event.set()
                    break
                time.sleep(0.2)

            if now - last_status >= args.status_interval:
                last_status = now
                registry.cleanup(now, forget_after_s=args.handoff_ttl * 6)
                snap = registry.snapshot(now)
                if snap:
                    print("\n--- active rolls ---")
                    for row in snap:
                        state = "active" if row["active"] else "recently departed"
                        print(
                            f"  id={row['id']:>4}  camera={str(row['camera']):<8}  "
                            f"bbox={row['bbox']}  seen {row['sightings']}x  "
                            f"last seen {row['last_seen_s_ago']}s ago  [{state}]"
                        )
    except KeyboardInterrupt:
        print("\nStopped by Ctrl+C.")
        stop_event.set()
    finally:
        stop_event.set()
        for t, (cam_name, _) in zip(threads, cameras):
            t.join(timeout=20.0)
            # Backstop: threads are daemons, so if one is still stuck the interpreter
            # would kill it on exit without ever calling release() -- which leaves an
            # mp4 with no moov atom and no way to play it. Finalise it from here.
            holder = writer_holders.get(cam_name)
            if holder is not None:
                with holder["lock"]:
                    if holder["writer"] is not None:
                        holder["writer"].release()
                        holder["writer"] = None
                        print(f"[{cam_name}] worker still busy -- recording finalised from the main thread.")
        if args.show:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass  # no GUI backend to tear down
    print(f"Sightings log: {log_path}")


if __name__ == "__main__":
    main()
