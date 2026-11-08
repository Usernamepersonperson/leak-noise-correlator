"""Compact LoRaWAN payloads for a noise logger.

Port 1 (uplink, 10 bytes): night noise statistics
  byte 0     bits 7-6 version (=1), bit 5 leak_suspect, bit 4 snippet_ready, bit 3 low_battery,
             bits 2-0 frame counter mod 8
  byte 1     min   level, uint8, dB = raw * 0.5 - 100   (dB re full scale)
  byte 2     mean  level, same scale
  byte 3     max   level, same scale
  byte 4     spread (std of window levels), uint8, dB = raw * 0.1
  byte 5     dominant frequency, uint8, Hz = raw * 8
  byte 6     battery, uint8, mV = 2000 + raw * 10
  byte 7     temperature, int8, deg C
  bytes 8-9  minute of night (0..1439) at which the minimum occurred, uint16 big-endian

Port 2 (downlink): 0x01 <snippet_id> <seconds>   request a raw snippet.

Port 3 (uplink, snippet frame): raw samples as 4-bit IMA-ADPCM, each frame independently
decodable so a lost frame only loses its own samples:
  byte 0 snippet_id, byte 1 frame index, byte 2 frame count,
  bytes 3-4 predictor (int16 big-endian), byte 5 step index (0..88), bytes 6.. nibbles
  (low nibble = first sample of each byte).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

VERSION = 1
PORT_STATS, PORT_CMD, PORT_SNIPPET = 1, 2, 3

STEP_TABLE = [
    7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 19, 21, 23, 25, 28, 31, 34, 37, 41, 45, 50, 55, 60, 66,
    73, 80, 88, 97, 107, 118, 130, 143, 157, 173, 190, 209, 230, 253, 279, 307, 337, 371, 408,
    449, 494, 544, 598, 658, 724, 796, 876, 963, 1060, 1166, 1282, 1411, 1552, 1707, 1878, 2066,
    2272, 2499, 2749, 3024, 3327, 3660, 4026, 4428, 4871, 5358, 5894, 6484, 7132, 7845, 8630,
    9493, 10442, 11487, 12635, 13899, 15289, 16818, 18500, 20350, 22385, 24623, 27086, 29794,
    32767,
]
INDEX_TABLE = [-1, -1, -1, -1, 2, 4, 6, 8]


@dataclass
class Stats:
    min_db: float
    mean_db: float
    max_db: float
    spread_db: float
    dom_hz: float
    battery_mv: int
    temp_c: int
    min_minute: int
    leak_suspect: bool = False
    snippet_ready: bool = False
    low_battery: bool = False


def _lvl(db: float) -> int:
    return int(min(max(round((db + 100.0) / 0.5), 0), 255))


def encode_stats(s: Stats, counter: int = 0) -> bytes:
    if not 0 <= s.min_minute <= 1439:
        raise ValueError("min_minute must be 0..1439")
    if not -128 <= s.temp_c <= 127:
        raise ValueError("temp_c out of int8 range")
    head = (
        (VERSION << 6)
        | (int(s.leak_suspect) << 5)
        | (int(s.snippet_ready) << 4)
        | (int(s.low_battery) << 3)
        | (counter & 7)
    )
    return bytes(
        [
            head,
            _lvl(s.min_db),
            _lvl(s.mean_db),
            _lvl(s.max_db),
            int(min(max(round(s.spread_db / 0.1), 0), 255)),
            int(min(max(round(s.dom_hz / 8.0), 0), 255)),
            int(min(max(round((s.battery_mv - 2000) / 10.0), 0), 255)),
            s.temp_c & 0xFF,
            (s.min_minute >> 8) & 0xFF,
            s.min_minute & 0xFF,
        ]
    )


def decode_stats(b: bytes) -> dict:
    if len(b) != 10:
        raise ValueError(f"stats payload must be 10 bytes, got {len(b)}")
    if b[0] >> 6 != VERSION:
        raise ValueError("unsupported payload version")
    return {
        "version": b[0] >> 6,
        "leak_suspect": bool(b[0] >> 5 & 1),
        "snippet_ready": bool(b[0] >> 4 & 1),
        "low_battery": bool(b[0] >> 3 & 1),
        "counter": b[0] & 7,
        "min_db": b[1] * 0.5 - 100.0,
        "mean_db": b[2] * 0.5 - 100.0,
        "max_db": b[3] * 0.5 - 100.0,
        "spread_db": b[4] / 10,
        "dom_hz": b[5] * 8,
        "battery_mv": 2000 + b[6] * 10,
        "temp_c": b[7] - 256 if b[7] > 127 else b[7],
        "min_minute": (b[8] << 8) | b[9],
    }


def encode_snippet_request(snippet_id: int, seconds: int) -> bytes:
    if not (0 <= snippet_id <= 255 and 1 <= seconds <= 255):
        raise ValueError("snippet_id 0..255, seconds 1..255")
    return bytes([0x01, snippet_id, seconds])


# ---------- IMA ADPCM ----------

def _adpcm_step(pred: int, idx: int, nib: int):
    step = STEP_TABLE[idx]
    diff = step >> 3
    if nib & 1:
        diff += step >> 2
    if nib & 2:
        diff += step >> 1
    if nib & 4:
        diff += step
    pred = pred - diff if nib & 8 else pred + diff
    pred = max(-32768, min(32767, pred))
    idx = max(0, min(88, idx + INDEX_TABLE[nib & 7]))
    return pred, idx


def _adpcm_encode_sample(sample: int, pred: int, idx: int):
    step = STEP_TABLE[idx]
    diff = sample - pred
    nib = 0
    if diff < 0:
        nib = 8
        diff = -diff
    if diff >= step:
        nib |= 4
        diff -= step
    if diff >= step >> 1:
        nib |= 2
        diff -= step >> 1
    if diff >= step >> 2:
        nib |= 1
    pred, idx = _adpcm_step(pred, idx, nib)
    return nib, pred, idx


def encode_snippet(samples, snippet_id: int = 0, frame_bytes: int = 48) -> list[bytes]:
    """ADPCM-encode int16 samples into independent frames of at most `frame_bytes` bytes."""
    x = np.clip(np.round(np.asarray(samples, float)), -32768, 32767).astype(int)
    if frame_bytes < 8:
        raise ValueError("frame_bytes too small")
    per = (frame_bytes - 6) * 2  # samples per frame
    n_frames = math.ceil(len(x) / per)
    if n_frames > 255:
        raise ValueError("snippet needs more than 255 frames")
    pred, idx = int(x[0]) if len(x) else 0, 0
    frames = []
    for fi in range(n_frames):
        chunk = x[fi * per : (fi + 1) * per]
        head = bytes([snippet_id & 255, fi, n_frames]) + int(pred).to_bytes(2, "big", signed=True)
        head += bytes([idx])
        nibs = []
        for v in chunk:
            nib, pred, idx = _adpcm_encode_sample(int(v), pred, idx)
            nibs.append(nib)
        if len(nibs) % 2:
            nibs.append(0)
        body = bytes(nibs[i] | (nibs[i + 1] << 4) for i in range(0, len(nibs), 2))
        frames.append(head + body)
    return frames


def decode_frame(frame: bytes) -> dict:
    if len(frame) < 7:
        raise ValueError("snippet frame too short")
    pred = int.from_bytes(frame[3:5], "big", signed=True)
    idx = frame[5]
    if idx > 88:
        raise ValueError("bad step index")
    out = []
    for byte in frame[6:]:
        for nib in (byte & 15, byte >> 4):
            pred, idx = _adpcm_step(pred, idx, nib)
            out.append(pred)
    return {"snippet_id": frame[0], "index": frame[1], "total": frame[2], "samples": out}


def decode_snippet(frames: list[bytes], n_samples: int | None = None) -> np.ndarray:
    """Reassemble frames (any order, missing frames become zeros)."""
    parts = {}
    total = None
    per = None
    for fr in frames:
        d = decode_frame(fr)
        total = d["total"]
        parts[d["index"]] = d["samples"]
        per = max(per or 0, len(d["samples"]))
    if total is None:
        raise ValueError("no frames")
    out = []
    for i in range(total):
        out.extend(parts.get(i, [0] * per))
    arr = np.array(out, float)
    return arr[:n_samples] if n_samples is not None else arr


# ---------- airtime ----------

MAX_APP_PAYLOAD_EU868 = {12: 51, 11: 51, 10: 51, 9: 115, 8: 222, 7: 222}


def airtime_s(app_payload: int, sf: int, bw_hz: int = 125_000, cr: int = 1, preamble: int = 8) -> float:
    """Time on air (s) of an uplink: standard LoRa formula, explicit header, CRC on.
    The 13 bytes of LoRaWAN MAC overhead are added to the application payload."""
    pl = app_payload + 13
    t_sym = (2**sf) / bw_hz
    de = 1 if t_sym > 0.016 else 0
    n_pay = 8 + max(math.ceil((8 * pl - 4 * sf + 28 + 16) / (4 * (sf - 2 * de))) * (cr + 4), 0)
    return (preamble + 4.25 + n_pay) * t_sym


def snippet_plan(n_samples: int, sf: int = 9, frame_bytes: int = 48, duty: float = 0.01) -> dict:
    """Frames, total airtime and the wall-clock time to send them under a duty-cycle limit."""
    if frame_bytes > MAX_APP_PAYLOAD_EU868[sf]:
        raise ValueError("frame larger than the maximum payload at this SF")
    per = (frame_bytes - 6) * 2
    frames = math.ceil(n_samples / per)
    ta = airtime_s(frame_bytes, sf)
    return {
        "frames": frames,
        "airtime_per_frame_s": ta,
        "airtime_total_s": frames * ta,
        "wall_clock_s": frames * ta / duty,
        "bytes_total": frames * frame_bytes,
    }
