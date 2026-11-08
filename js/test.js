// Checks the JS decoder against golden vectors produced by the Python encoder.
// Run: node js/test.js   (exit code 1 on any mismatch)
"use strict";
const fs = require("fs");
const path = require("path");
const { decodeUplink, encodeDownlink } = require("./decoder.js");

const vec = JSON.parse(fs.readFileSync(path.join(__dirname, "test-vectors.json"), "utf8"));
let fails = 0;
function eq(a, b, label) {
  if (JSON.stringify(a) !== JSON.stringify(b)) {
    fails++;
    console.error("FAIL", label, "\n  got     ", JSON.stringify(a), "\n  expected", JSON.stringify(b));
  }
}
const bytes = (h) => Array.from(Buffer.from(h, "hex"));

vec.stats.forEach((v, i) => eq(decodeUplink({ fPort: 1, bytes: bytes(v.hex) }).data, v.decoded, "stats " + i));
vec.snippet_frames.forEach((v, i) => eq(decodeUplink({ fPort: 3, bytes: bytes(v.hex) }).data, v.decoded, "frame " + i));
eq(encodeDownlink({ data: { requestSnippet: { id: vec.downlink.snippet_id, seconds: vec.downlink.seconds } } }),
  { bytes: bytes(vec.downlink.hex), fPort: 2 }, "downlink");
eq(!!decodeUplink({ fPort: 1, bytes: [1, 2, 3] }).errors, true, "short stats rejected");
eq(!!decodeUplink({ fPort: 9, bytes: [] }).errors, true, "unknown port rejected");
eq(!!decodeUplink({ fPort: 3, bytes: [0, 0, 1, 0, 0, 200, 0] }).errors, true, "bad step index rejected");
eq(!!encodeDownlink({ data: {} }).errors, true, "bad downlink rejected");

if (fails) process.exit(1);
console.log("js decoder ok:", vec.stats.length, "stats,", vec.snippet_frames.length, "frames");
