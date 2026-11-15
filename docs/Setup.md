# Setup

What to install for the desktop tools, the model export and the boards. Written for someone setting up
the repository on a new machine.

## Desktop build and tests

- A C++17 compiler (GCC or Clang), CMake and Make (or Ninja).
- The library, the tests and the command-line tools (`tinyyolo --check`, `--tym`, `--model`) need nothing else.
- The window build (`-DTINYYOLO_BUILD_CAMERA_VIEW=ON`) also needs:
  - SDL2 development files, found through `pkg-config`,
  - the sibling folders `../TinyCamera`, `../TinyGPU`, `../TinyMaterialDesign` and `../TinyJPEG`
    (or set `TINYCAMERA_DIR`, `TINYGPU_DIR`, `TINYMD_DIR` and `TINYJPEG_DIR`),
  - V4L2 for a webcam on Linux.

```sh
cmake -S . -B build
cmake --build build
ctest --test-dir build
```

## Python export (regenerates the YOLO models from the `.pt` checkpoints)

The export needs Python 3 with the `ultralytics` package, which installs PyTorch with it. Install into a
virtual environment:

```sh
python3 -m venv ~/venvs/tinyyolo
source ~/venvs/tinyyolo/bin/activate
pip install --upgrade pip
pip install ultralytics
```

For a CPU-only machine, install the CPU build of PyTorch first (see pytorch.org for the command), then the
`ultralytics` line. Check the install:

```sh
python -c "import ultralytics, torch; print(ultralytics.__version__, torch.__version__)"
```

To run the fetch script with this environment, pass its Python:

```sh
python tools/fetch_models.py yolo11n --python ~/venvs/tinyyolo/bin/python
```

On Windows, use the same commands with the Python launcher, for example `py tools\fetch_models.py yolo11n --python C:\path\to\venv\Scripts\python.exe`.
`tools/fetch_models.py` replaces the old shell scripts and runs the same way on Windows, Linux and macOS. It
accepts `yolo11n`, `yolo26n`, `yolov4-tiny` or `all`.

The checkpoints `models/yolo11n/yolo11n.pt` and `models/yolo26n/yolo26n.pt` are in the repository, and the
fetch script downloads them again if they are missing. Verified: with `ultralytics` 8.4.214 and PyTorch 2.15.0
the export of YOLO11n and YOLO26n reproduces `models/*/*.tym` byte for byte, and the layer checks give the same
results. Other versions are not tested; a different release can change the exported files.

## Boards (Arduino)

- `arduino-cli` with the `esp32:esp32` core. The examples were built with version 3.3.12.
- The Arduino libraries the sketches use: TinyYolo, TinyCamera, TinyGPU, TinyMaterialDesign and TinyJPEG.
- `esptool` for flashing. The system package may not have the stub for the ESP32-S3; the copy in the Arduino
  core works (`~/.arduino15/packages/esp32/tools/esptool_py/<version>/esptool`).
- The ESP32-P4 examples also need the ESP-NN library linked in (see the example READMEs).

Use the model file in the examples: `models/yolo11n/yolo11n.tym` (see [Model format](ModelFormat.md)).
