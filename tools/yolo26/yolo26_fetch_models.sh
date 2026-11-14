#!/usr/bin/env bash
# Downloads the YOLO26n weights (Ultralytics, COCO, 80 classes) into models/yolo26n/ and exports
# them for TinyYolo: the float parameters, the int8 weights and scales, the board blob
# (yolo26n_board.bin) and the operation table (yolo26n_graph.inc). The export needs the ultralytics
# package: set PYTHON to a Python that has it (default: python3). CALIBRATION_IMAGE names the image
# for the int8 calibration (default: the test photo of the examples). Run from anywhere.
# The weights are AGPL-3.0 (Ultralytics); the exported files are derived from them.
set -euo pipefail
here="$(cd "$(dirname "$0")/../.." && pwd)"
python="${PYTHON:-python3}"
dir="$here/models/yolo26n"
mkdir -p "$dir/export"

pt="$dir/yolo26n.pt"
sha="9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef"
if [ -f "$pt" ] && echo "$sha  $pt" | sha256sum -c --status; then
  echo "up to date: $pt"
else
  echo "downloading $pt"
  curl -sfL "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt" -o "$pt"
  echo "$sha  $pt" | sha256sum -c --status || { echo "checksum mismatch: $pt" >&2; exit 1; }
fi

# The int8 activation scales are calibrated on one image (see models/yolo11n/README.md).
calib="${CALIBRATION_IMAGE:-$dir/export/calibration.ppm}"
if [ ! -f "$calib" ]; then
  { printf 'P6\n375 250\n255\n'; cat "$here/examples/Yolo11Graph/photo.rgb"; } > "$calib"
fi

"$python" "$here/tools/yolo/yolo_export.py" "$pt" "$dir/export" "$calib"
cp "$dir/export/yolo26n.inc" "$dir/yolo26n_graph.inc"
echo "done: $dir/export (blob: yolo26n_board.bin, table: $dir/yolo26n_graph.inc)"
