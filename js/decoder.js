// Payload decoder for the leak noise logger (LoRaWAN).
// Works as a payload formatter for ChirpStack v4 and The Things Stack v3 (decodeUplink /
// encodeDownlink). Layout is documented in leakcorr/lorawan.py and the README.
"use strict";

const STEP_TABLE = [
  7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 19, 21, 23, 25, 28, 31, 34, 37, 41, 45, 50, 55, 60, 66,
  73, 80, 88, 97, 107, 118, 130, 143, 157, 173, 190, 209, 230, 253, 279, 307, 337, 371, 408,
  449, 494, 544, 598, 658, 724, 796, 876, 963, 1060, 1166, 1282, 1411, 1552, 1707, 1878, 2066,
  2272, 2499, 2749, 3024, 3327, 3660, 4026, 4428, 4871, 5358, 5894, 6484, 7132, 7845, 8630,
  9493, 10442, 11487, 12635, 13899, 15289, 16818, 18500, 20350, 22385, 24623, 27086, 29794,
  32767,
];
const INDEX_TABLE = [-1, -1, -1, -1, 2, 4, 6, 8];

function decodeStats(b) {
  if (b.length !== 10) return { errors: ["stats payload must be 10 bytes, got " + b.length] };
  if (b[0] >> 6 !== 1) return { errors: ["unsupported payload version " + (b[0] >> 6)] };
  return {
    data: {
      version: b[0] >> 6,
      leak_suspect: !!((b[0] >> 5) & 1),
      snippet_ready: !!((b[0] >> 4) & 1),
      low_battery: !!((b[0] >> 3) & 1),
      counter: b[0] & 7,
      min_db: b[1] * 0.5 - 100,
      mean_db: b[2] * 0.5 - 100,
      max_db: b[3] * 0.5 - 100,
      spread_db: b[4] / 10,
      dom_hz: b[5] * 8,
      battery_mv: 2000 + b[6] * 10,
      temp_c: b[7] > 127 ? b[7] - 256 : b[7],
      min_minute: (b[8] << 8) | b[9],
    },
  };
}

function decodeSnippetFrame(b) {
  if (b.length < 7) return { errors: ["snippet frame too short"] };
  let pred = (b[3] << 8) | b[4];
  if (pred > 32767) pred -= 65536;
  let idx = b[5];
  if (idx > 88) return { errors: ["bad step index"] };
  const samples = [];
  for (let i = 6; i < b.length; i++) {
    for (const nib of [b[i] & 15, b[i] >> 4]) {
      const step = STEP_TABLE[idx];
      let diff = step >> 3;
      if (nib & 1) diff += step >> 2;
      if (nib & 2) diff += step >> 1;
      if (nib & 4) diff += step;
      pred = nib & 8 ? pred - diff : pred + diff;
      pred = Math.max(-32768, Math.min(32767, pred));
      idx = Math.max(0, Math.min(88, idx + INDEX_TABLE[nib & 7]));
      samples.push(pred);
    }
  }
  return { data: { snippet_id: b[0], index: b[1], total: b[2], samples } };
}

function decodeUplink(input) {
  const b = input.bytes;
  if (input.fPort === 1) return decodeStats(b);
  if (input.fPort === 3) return decodeSnippetFrame(b);
  return { errors: ["unknown fPort " + input.fPort] };
}

function encodeDownlink(input) {
  const r = input.data && input.data.requestSnippet;
  if (!r || !(r.id >= 0 && r.id <= 255) || !(r.seconds >= 1 && r.seconds <= 255)) {
    return { errors: ["expected data.requestSnippet {id 0..255, seconds 1..255}"] };
  }
  return { bytes: [0x01, r.id, r.seconds], fPort: 2 };
}

if (typeof module !== "undefined") module.exports = { decodeUplink, encodeDownlink };
