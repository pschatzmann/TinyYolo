#!/usr/bin/env bash
# Downloads the trained YOLOv4-tiny weights into models/yolov4-tiny/ (they are too large for the
# repository). The .cfg and .names files are already there. Run from anywhere.
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"

fetch() {  # url, destination, sha256
  local url="$1" dest="$2" sha="$3"
  if [ -f "$dest" ] && echo "$sha  $dest" | sha256sum -c --status; then
    echo "up to date: $dest"; return
  fi
  echo "downloading $dest"
  curl -sfL "$url" -o "$dest"
  echo "$sha  $dest" | sha256sum -c --status || { echo "checksum mismatch: $dest" >&2; exit 1; }
}

# YOLOv4-tiny, trained on COCO (80 classes). Darknet is public domain; the COCO dataset is
# CC BY 4.0 (https://cocodataset.org).
fetch "https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4-tiny.weights" \
      "$here/models/yolov4-tiny/yolov4-tiny.weights" "cf9fbfd0f6d4869b35762f56100f50ed05268084078805f0e7989efe5bb8ca87"
