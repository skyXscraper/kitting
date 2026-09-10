#!/usr/bin/env python3
"""
Integrates roll_tracking/main.py's multi-camera tracker with the full
single-roll OCR pipeline in scripts/scripts/ocr_using_paddleocr.py.

main.py's own OCR step is deliberately lightweight (one digit-only read of
the top line) so a track can be resolved to an id quickly, without paying
for the heavier multi-pass pipeline on every attempt. This file trades that
speed for accuracy: once a track is due for an OCR attempt, it crops a
padded region around it and runs the *actual* single-roll pipeline --
precise roll/handwriting-ROI detection (red-ink extraction), multiple
preprocessing variants per detected line, and 4<->9<->7 confusion-correction
against the master list -- the same one the standalone writing-station
script uses. That gets you, per tracked roll, not just an identity (the
ply_no) but the full reading: the start-end range, master-list validation
status, and the auto/confirm/manual operator action -- all still
cross-camera-consistent through the same ply-number-keyed RollRegistry
approach as main.py.

Usage mirrors main.py:
    python tracking_and_ocr.py --camera1 0 --show
    python tracking_and_ocr.py --camera1 0 --camera2 1 --record

Because each OCR attempt is heavier here (and writes its own debug crops
under <output-dir>/ocr_debug/, which will grow over a long run),
--ocr-retry-interval and --max-ocr-attempts default lower than in main.py.

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

ROLL_TRACKING_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ROLL_TRACKING_DIR.parent
sys.path.insert(0, str(ROLL_TRACKING_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "scripts"))
import main as track  # noqa: E402 - roll_tracking/main.py: detection, tracking, registry
import ocr_using_paddleocr as ocrp  # noqa: E402 - the full single-roll OCR pipeline


# ---------------------------------------------------------------------------
# Bridge: hand a tracked roll's region to the full single-roll OCR pipeline
# ---------------------------------------------------------------------------


def padded_crop(frame: np.ndarray, bbox: tuple[float, float, float, float], pad_frac: float = 0.6, min_size: int = 160) -> np.ndarray:
    """
    ocr_using_paddleocr.detect_sheet_roll_mask expects a frame with visible
    background margin around the roll -- it only searches a central band of
    whatever image it's given -- not a razor-tight bounding box. Pad the
    tracker's bbox out generously (and up to a minimum size) before handing
    it to the full pipeline, rather than the exact detector box.
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    pad_x = max(bw * pad_frac, (min_size - bw) / 2, 0)
    pad_y = max(bh * pad_frac, (min_size - bh) / 2, 0)
    x1 = int(max(0, x1 - pad_x))
    y1 = int(max(0, y1 - pad_y))
    x2 = int(min(w, x2 + pad_x))
    y2 = int(min(h, y2 + pad_y))
    return frame[y1:y2, x1:x2].copy()


def run_full_ocr_on_track(
    reader,
    ocr_lock: threading.Lock,
    master: list[dict],
    ocr_output_dir: Path,
    frame: np.ndarray,
    bbox: tuple[float, float, float, float],
    label: str,
    min_det_conf: float,
) -> dict | None:
    """
    Run ocr_using_paddleocr.run_ocr_on_frame on a padded crop around one
    tracked roll. Returns its full summary dict (ply_no + start-end +
    master validation + operator action, plus debug artifact paths), or
    None if no roll/legible text was found in the crop this attempt.
    """
    crop = padded_crop(frame, bbox)
    if crop.size == 0 or min(crop.shape[:2]) < 40:
        return None

    def attempt(img, suffix: str):
        try:
            with ocr_lock:  # PP-OCR inference is serialized across both camera threads
                return ocrp.run_ocr_on_frame(
                    img, label + suffix, ocr_output_dir, master, reader, min_det_conf, prefix=f"{label}{suffix}_"
                )
        except Exception as err:  # noqa: BLE001 - a bad crop must never take down the tracker
            print(f"  [{label}{suffix}] full OCR pipeline error: {err}")
            return None

    # Raw first, then with the marker strokes boosted. Neither wins alone:
    # boosting rescues the many crops where the text detector proposes no
    # regions at all, but it can also lose a read that the raw crop got.
    summary = attempt(crop, "")
    if summary is not None and summary["final"].get("ply_no_final"):
        return summary
    boosted = attempt(track.boost_writing_ink(crop), "_ink")
    if boosted is not None and boosted["final"].get("ply_no_final"):
        return boosted
    return summary or boosted


def ink_row_bands(bgr: np.ndarray) -> list[tuple[int, int]]:
    """
    Split the writing into its horizontal lines using the ink mask.

    The handwriting pattern this project is built around is fixed: the ply
    number on the top line, the start-end range on the bottom one. Finding
    the lines directly off the ink is far more reliable than hoping the text
    detector proposes a box per line, which is what it fails to do here.
    """
    mask = writing_ink_mask_of(bgr)
    h = mask.shape[0]
    profile = (mask > 0).sum(axis=1)
    if profile.max() == 0:
        return []
    threshold = max(1, int(profile.max() * 0.08))
    bands, start = [], None
    for y in range(h):
        if profile[y] >= threshold and start is None:
            start = y
        elif profile[y] < threshold and start is not None:
            if y - start >= 4:
                bands.append([start, y])
            start = None
    if start is not None and h - start >= 4:
        bands.append([start, h])
    merged: list[list[int]] = []
    for band in bands:
        if merged and band[0] - merged[-1][1] < max(3, int(h * 0.02)):
            merged[-1][1] = band[1]
        else:
            merged.append(band)
    # Drop specks: an isolated red mark elsewhere on the roll forms its own
    # thin band, and taking that as "the bottom line" reads nothing at all.
    tallest = max((b - a) for a, b in merged) if merged else 0
    return [(a, b) for a, b in merged if (b - a) >= max(6, tallest * 0.35)]


def writing_ink_mask_of(bgr: np.ndarray) -> np.ndarray:
    return track.writing_ink_mask(bgr)


def read_range_from_bottom_line(reader, ocr_lock: threading.Lock, crop: np.ndarray) -> str | None:
    """
    Read the start-end off the bottom line of the writing, from the image.

    The whole-crop detector regularly returns the range as broken fragments
    ('4.7.', '12', '7-') that no assembly path accepts. This instead locates
    the bottom line of ink and hands that band to the project's own
    ocr_box_halves(), which reads the left and right halves separately with
    preprocessing variants -- exactly the case it was written for. Returns a
    range only when both halves come back as proper decimals, so a partial
    read is reported as unread rather than guessed at.
    """
    if crop is None or crop.size == 0:
        return None
    for image in (crop, track.boost_writing_ink(crop)):
        bands = ink_row_bands(image)
        if not bands:
            continue
        h, w = image.shape[:2]
        # Bottom line first, per the handwriting pattern, but fall back up the
        # lines rather than giving up if the lowest one doesn't read.
        for y1, y2 in reversed(bands):
            pad = int((y2 - y1) * 0.35) + 4
            box = (0, max(0, y1 - pad), w, min(h, y2 + pad))
            with ocr_lock:
                halves = ocrp.ocr_box_halves(reader, image, box)
            rng = halves.get("range")
            if rng and re.fullmatch(r"\d+\.\d+-\d+\.\d+", rng):
                return rng
    return None


def resolve_start_end(final: dict) -> str | None:
    """
    The roll's start-end as read from the roll itself.

    Never falls back to the master list: this station exists to check what is
    handwritten on a roll against that list, so filling the field in from the
    list would make the comparison circular and verify nothing.
    """
    return final.get("start_end_final")


def harvest_decimals(summary: dict) -> list[str]:
    """Every clean xx.x decimal this attempt actually read off the roll."""
    assembled = summary.get("assembled") or {}
    seen = []
    for frag in assembled.get("length_fragments") or []:
        text = ocrp.sanitize(frag.get("text", ""))
        if re.fullmatch(r"\d+\.\d+", text):
            seen.append(text)
    for crop in summary.get("crops") or []:
        for text in (crop.get("det_text"), (crop.get("picked") or {}).get("candidate")):
            text = ocrp.sanitize(text or "")
            if re.fullmatch(r"\d+\.\d+", text):
                seen.append(text)
    return seen


def vote_range(decimal_votes: dict[str, int], min_votes: int = 2) -> str | None:
    """
    Build a start-end from decimals accumulated over successive frames.

    A single frame usually yields only one of the two numbers -- the other
    misreads -- but a live feed offers many looks at the same roll, so the
    pair can be assembled from separate frames. Both values still come from
    the roll itself; nothing is taken from the master list. A decimal has to
    turn up more than once before it counts, so one bad frame can't invent a
    reading, and the smaller of the two is taken as the start.
    """
    repeated = sorted(
        (d for d, n in decimal_votes.items() if n >= min_votes),
        key=lambda d: (-decimal_votes[d], d),
    )
    if len(repeated) < 2:
        return None
    try:
        pair = sorted((float(repeated[0]), float(repeated[1])))
    except ValueError:
        return None
    lookup = {float(d): d for d in repeated[:2]}
    return f"{lookup[pair[0]]}-{lookup[pair[1]]}"


def draw_annotations_rich(frame: np.ndarray, tracks: dict[int, dict], roll_details: dict[str, dict], details_lock: threading.Lock) -> np.ndarray:
    """
    Labels each box with both values read off the roll: the ply number and
    the start-end range, drawn on two lines so neither gets clipped on a
    narrow box. A range that could not be read shows as "unread" rather than
    being filled in from the master list.
    """
    out = frame.copy()
    for tid, tr in tracks.items():
        x1, y1, x2, y2 = [int(v) for v in tr["bbox"]]
        gid = tr["global_id"]
        if gid is not None:
            with details_lock:
                details = roll_details.get(gid, {})
            lines = [f"id: {gid}", f"start_end_ocr: {details.get('start_end') or 'unread'}"]
            color = (0, 200, 0)
        else:
            lines = [f"reading... (#{tid})"]
            color = (0, 165, 255)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        # Stack the labels upward from the top edge of the box.
        for i, text in enumerate(reversed(lines)):
            y = max(18 + i * 20, y1 - 8 - i * 20)
            cv2.putText(out, text, (x1, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
            cv2.putText(out, text, (x1, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return out


# ---------------------------------------------------------------------------
# Per-camera worker thread (adapted from main.camera_worker)
# ---------------------------------------------------------------------------


def camera_worker(
    name: str,
    source: str,
    registry: "track.RollRegistry",
    reader,
    ocr_lock: threading.Lock,
    master: list[dict],
    roll_details: dict[str, dict],
    details_lock: threading.Lock,
    ocr_output_dir: Path,
    stop_event: threading.Event,
    shared_frames: dict[str, np.ndarray],
    frames_lock: threading.Lock,
    log_path: Path | None,
    writer_holder: dict,
    args: argparse.Namespace,
) -> None:
    cap = ocrp._open_video_capture(source)
    if not cap.isOpened():
        print(f"[{name}] could not open video source {source!r}. "
              "If this is a USB camera, check `v4l2-ctl --list-devices` and `ffplay <device>`.")
        stop_event.set()
        return
    print(f"[{name}] opened {source!r} (backend={cap.getBackendName()})")

    bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=32, detectShadows=True)
    detector_state = {"bg_subtractor": bg_subtractor}

    master_index_map = ocrp.master_index(master) if master else {}
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
            h, w = frame.shape[:2]
            if frame_diag is None:
                frame_diag = math.hypot(w, h)
            now = time.time()

            detections = track.detect_rolls(args.detector, detector_state, frame, args.min_area_frac)
            max_dist = frame_diag * args.max_track_dist_frac
            matches, unmatched_tracks, unmatched_dets = track.associate_detections_to_tracks(tracks, detections, max_dist)

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

            # Identity + reading: the full pipeline, not the lightweight one in main.py.
            # A single pass can take seconds, so bail out the moment a stop is
            # requested -- otherwise shutdown waits on it, and the recording gets
            # cut off before it can be finalised.
            for tid, tr in tracks.items():
                if stop_event.is_set():
                    break
                if tr["global_id"] is not None:
                    # Identified, but keep looking until the range has been read too:
                    # a single frame rarely yields both decimals.
                    with details_lock:
                        known = (roll_details.get(tr["global_id"], {}) or {}).get("start_end")
                    if known:
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
                tr["last_ocr_frame"] = frame_idx
                tr["ocr_attempts"] += 1
                tr["ocr_box"] = tr["bbox"]
                label = f"{name}_t{tid}_f{frame_idx:06d}"
                summary = run_full_ocr_on_track(reader, ocr_lock, master, ocr_output_dir, frame, tr["bbox"], label, args.min_det_conf)
                if summary is None:
                    continue
                final = summary["final"]
                candidate = final.get("ply_no_final")
                conf = summary.get("assembled", {}).get("ply_no_confidence") or 0.0
                if candidate and conf >= args.min_ocr_conf and not track.is_plausible_ply(candidate, master_index_map):
                    print(f"[{name}] ignoring implausible ply read '{candidate}' @ {conf:.2f} (likely a partial read)")
                elif candidate and conf >= args.min_ocr_conf:
                    global_id = registry.resolve(candidate, now)
                    newly_identified = tr["global_id"] is None
                    tr["global_id"] = global_id

                    # Read the range, in order of directness: whole-crop assembly,
                    # then the bottom writing line on its own, then decimals voted
                    # across earlier frames. Every one of these reads the roll.
                    start_end = resolve_start_end(final)
                    if not start_end:
                        start_end = read_range_from_bottom_line(
                            reader, ocr_lock, padded_crop(frame, tr["bbox"])
                        )
                    with details_lock:
                        entry = roll_details.setdefault(global_id, {"decimal_votes": {}})
                        votes = entry.setdefault("decimal_votes", {})
                        for decimal in harvest_decimals(summary):
                            votes[decimal] = votes.get(decimal, 0) + 1
                        if not start_end:
                            start_end = vote_range(votes)
                        entry.update(
                            {
                                "ply_no": candidate,
                                "start_end": start_end or entry.get("start_end"),
                                "master_status": final["master_validation"]["status"],
                                "operator_action": final["operator_action"],
                                "updated": now,
                            }
                        )
                        resolved_range = entry.get("start_end")
                        vote_summary = dict(sorted(votes.items(), key=lambda kv: -kv[1])[:4])
                    if newly_identified or resolved_range:
                        print(
                            f"[{name}] track #{tid} -> roll id '{global_id}', "
                            f"start-end={resolved_range or 'unread'} "
                            f"(decimals seen: {vote_summary}, action={final['operator_action']})"
                        )

            for tr in tracks.values():
                if tr["global_id"] is not None:
                    registry.update(tr["global_id"], name, tr["bbox"], now)

            annotated = draw_annotations_rich(frame, tracks, roll_details, details_lock)
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
                    # worker is still mid-OCR when shutdown comes.
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
                        with details_lock:
                            details = roll_details.get(tr["global_id"], {}) if tr["global_id"] else {}
                        f.write(
                            json.dumps(
                                {
                                    "ts": now,
                                    "camera": name,
                                    "frame": frame_idx,
                                    "global_id": tr["global_id"],
                                    "ply_no_ocr": details.get("ply_no"),
                                    "start_end_ocr": details.get("start_end"),
                                    "bbox": list(tr["bbox"]),
                                    "master_status": details.get("master_status"),
                                    "operator_action": details.get("operator_action"),
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
    p = argparse.ArgumentParser(
        description="Two-camera sheet-roll tracker integrated with the full single-roll OCR pipeline "
                     "(ply_no + start-end + master-list confusion-correction) for both identity and readings."
    )
    p.add_argument("--camera1", default="0", help="Video source for camera 1 (index, /dev/videoN, file, or URL).")
    p.add_argument(
        "--camera2",
        default=None,
        help="Video source for camera 2. Omit to run single-camera (no cross-camera "
             "handoff to test yet, but detection/tracking/full-OCR all still run).",
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
    p.add_argument("--gpu", action="store_true")
    p.add_argument("--master-list", type=Path, default=PROJECT_ROOT / "data" / "data" / "master_list.csv")
    p.add_argument("--min-area-frac", type=float, default=0.004, help="Min blob area as a fraction of frame area.")
    p.add_argument("--max-track-dist-frac", type=float, default=0.12,
                    help="Max centroid jump (as a fraction of the frame diagonal) to still count as the same track.")
    p.add_argument("--max-missed-frames", type=int, default=45, help="Frames a track may go undetected before it's dropped from that camera.")
    p.add_argument("--reacquire-frames", type=int, default=90,
                    help="After a track is dropped, how long (in frames) a new detection in roughly "
                         "the same place may still inherit its roll id instead of starting fresh.")
    p.add_argument("--ocr-retry-interval", type=int, default=20,
                    help="Frames between full-OCR attempts per unresolved track. Higher than main.py's default "
                         "since this runs the full multi-pass pipeline, not a single quick read.")
    p.add_argument("--max-ocr-attempts", type=int, default=10, help="Give up OCR-resolving a track after this many tries.")
    p.add_argument("--min-det-conf", type=float, default=0.2, help="Digit-box detection threshold inside the full OCR pipeline.")
    p.add_argument("--min-ocr-conf", type=float, default=0.35, help="Minimum ply_no recognition confidence to accept an id.")
    p.add_argument("--handoff-ttl", type=float, default=10.0,
                    help="Seconds a roll stays eligible for cross-camera id reuse after its last sighting.")
    p.add_argument("--log-stride", type=int, default=15, help="Write to the sightings log every Nth frame.")
    p.add_argument("--status-interval", type=float, default=2.0, help="Seconds between printed status snapshots.")
    p.add_argument("--output-dir", type=Path, default=ROLL_TRACKING_DIR / "output")
    p.add_argument("--show", action="store_true", help="Display annotated windows for both cameras.")
    p.add_argument("--record", action="store_true",
                    help="Save the annotated video (with detection/tracking boxes) for each camera to "
                         "<output-dir>/<camera>_recording.mp4.")
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "roll_sightings.jsonl"
    ocr_output_dir = args.output_dir / "ocr_debug"
    ocr_output_dir.mkdir(parents=True, exist_ok=True)

    master = ocrp.load_master_list(args.master_list)
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

    registry = track.RollRegistry(handoff_ttl_s=args.handoff_ttl)
    roll_details: dict[str, dict] = {}
    details_lock = threading.Lock()
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
            args=(
                name, source, registry, reader, ocr_lock, master, roll_details, details_lock,
                ocr_output_dir, stop_event, shared_frames, frames_lock, log_path,
                writer_holders[name], args,
            ),
            daemon=True,
        )
        for name, source in cameras
    ]
    for t in threads:
        t.start()

    print("Tracking + full OCR started. Press 'q' (then Enter, if no --show window) or Ctrl+C to stop.")
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
                        with details_lock:
                            details = roll_details.get(row["id"], {})
                        start_end_ocr = details.get("start_end") or "unread"
                        print(
                            f"  id={row['id']:>4}  start_end_ocr: {start_end_ocr}  "
                            f"camera={str(row['camera']):<8}  bbox={row['bbox']}  "
                            f"seen {row['sightings']}x  "
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
    print(f"Per-reading OCR debug artifacts: {ocr_output_dir}")


if __name__ == "__main__":
    main()
