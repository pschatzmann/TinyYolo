#!/usr/bin/env python3
"""Downloads the trained models into models/ (checked by SHA-256) and, for the YOLO models, exports them.

usage: fetch_models.py <yolo11n | yolo26n | yolov4-tiny | all> [--python PATH] [--calibration IMAGE]

  yolo11n, yolo26n  download the Ultralytics checkpoint (<name>.pt), then run tools/yolo/yolo_export.py,
                    which writes models/<name>/export and the single-file model models/<name>/<name>.tym.
                    The export needs the ultralytics package (see docs/Setup.md); --python selects the
                    interpreter that has it (default: this one).
  yolov4-tiny       downloads the Darknet weights (the .cfg and .names files are in the repository).

The int8 activation scales of the YOLO exports are calibrated on one image: by default the test photo of
the examples, or the image given with --calibration (or the CALIBRATION_IMAGE environment variable).
Works on Windows, Linux and macOS; it needs only the Python standard library and the export's packages.
"""
import argparse
import hashlib
import os
import subprocess
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODELS = {
    "yolo11n": {
        "url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt",
        "sha256": "0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1",
        "file": "models/yolo11n/yolo11n.pt",
        "export": True,
    },
    "yolo26n": {
        "url": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt",
        "sha256": "9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef",
        "file": "models/yolo26n/yolo26n.pt",
        "export": True,
    },
    # YOLOv4-tiny, trained on COCO (80 classes). Darknet is public domain; the COCO dataset is
    # CC BY 4.0 (https://cocodataset.org).
    "yolov4-tiny": {
        "url": "https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4-tiny.weights",
        "sha256": "cf9fbfd0f6d4869b35762f56100f50ed05268084078805f0e7989efe5bb8ca87",
        "file": "models/yolov4-tiny/yolov4-tiny.weights",
        "export": False,
    },
}


def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(name, spec):
    """Downloads the file unless it is already there with the right checksum."""
    dest = os.path.join(ROOT, spec["file"])
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.isfile(dest) and sha256_of(dest) == spec["sha256"]:
        print("up to date: %s" % dest)
        return dest
    print("downloading %s" % dest)
    tmp = dest + ".part"
    with urllib.request.urlopen(spec["url"], timeout=120) as response, open(tmp, "wb") as out:
        while True:
            block = response.read(1 << 20)
            if not block:
                break
            out.write(block)
    if sha256_of(tmp) != spec["sha256"]:
        os.remove(tmp)
        raise SystemExit("checksum mismatch: %s" % dest)
    os.replace(tmp, dest)
    return dest


def calibration_image(name, path):
    """The default calibration picture: the test photo of the examples as a binary PPM."""
    if path is not None:
        return path
    base = os.path.join(ROOT, "models", name, "export", "calibration.ppm")
    os.makedirs(os.path.dirname(base), exist_ok=True)
    with open(os.path.join(ROOT, "examples", "Yolo11Graph", "photo.rgb"), "rb") as photo:
        raw = photo.read()
    with open(base, "wb") as out:
        out.write(b"P6\n375 250\n255\n" + raw)
    return base


def export(name, checkpoint, python, calibration):
    export_dir = os.path.join(ROOT, "models", name, "export")
    script = os.path.join(ROOT, "tools", "yolo", "yolo_export.py")
    subprocess.run([python, script, checkpoint, export_dir, calibration], check=True)
    print("done: %s and %s (the single-file model)" % (export_dir, os.path.join(ROOT, "models", name, name + ".tym")))


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", choices=list(MODELS) + ["all"])
    parser.add_argument("--python", default=sys.executable, help="Python with ultralytics for the export")
    parser.add_argument("--calibration", default=os.environ.get("CALIBRATION_IMAGE"),
                        help="image for the int8 calibration (default: the test photo)")
    args = parser.parse_args(argv)
    names = list(MODELS) if args.model == "all" else [args.model]
    for name in names:
        spec = MODELS[name]
        checkpoint = fetch(name, spec)
        if spec["export"]:
            calibration = calibration_image(name, args.calibration)
            export(name, checkpoint, args.python, calibration)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
