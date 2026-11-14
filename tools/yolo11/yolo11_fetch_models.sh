#!/usr/bin/env bash
# Downloads the YOLO11n weights (Ultralytics, COCO, 80 classes) into models/yolo11n/ and exports
# them for TinyYolo: the float parameters, the int8 weights and scales, the board blob
# (yolo11n_board.bin) and the operation table (yolo11n_graph.inc). The export needs the ultralytics
# package: set PYTHON to a Python that has it (default: python3). CALIBRATION_IMAGE names the image
# for the int8 calibration (default: the test photo of the examples). Run from anywhere.
# The weights are AGPL-3.0 (Ultralytics); the exported files are derived from them.
set -euo pipefail
here="$(cd "$(dirname "$0")/../.." && pwd)"
python="${PYTHON:-python3}"
dir="$here/models/yolo11n"
mkdir -p "$dir/export"

pt="$dir/yolo11n.pt"
sha="0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1"
if [ -f "$pt" ] && echo "$sha  $pt" | sha256sum -c --status; then
  echo "up to date: $pt"
else
  echo "downloading $pt"
  curl -sfL "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt" -o "$pt"
  echo "$sha  $pt" | sha256sum -c --status || { echo "checksum mismatch: $pt" >&2; exit 1; }
fi

# The int8 activation scales are calibrated on one image. The default is the test photo of the
# examples (raw RGB in examples/Yolo11Graph/photo.rgb); CALIBRATION_IMAGE can name another image.
calib="${CALIBRATION_IMAGE:-$dir/export/calibration.ppm}"
if [ -z "${CALIBRATION_IMAGE:-}" ]; then
  { printf 'P6\n375 250\n255\n'; cat "$here/examples/Yolo11Graph/photo.rgb"; } > "$calib"
fi

"$python" "$here/tools/yolo/yolo_export.py" "$pt" "$dir/export" "$calib"
cp "$dir/export/yolo11n.inc" "$dir/yolo11n_graph.inc"
echo "done: $dir/export (blob: yolo11n_board.bin, table: $dir/yolo11n_graph.inc)"
