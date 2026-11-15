# Yolo26Graph

YOLO26n on the ESP32-P4 with the TinyYolo graph path (`Graph` + `Esp32NnBackend`). The int8
convolutions run on the P4 assembly kernels; the rest of the graph runs in this library. On the
test photo (`photo.rgb`, 375x250, letterboxed to 640x640). The sketch uses `tinyyolo::GraphYolo11`
(one call per frame) and `tinyyolo::ModelFilePartition` (the model file in the `model` partition).
On the board, one frame takes 6.9 s with the P4 assembly kernels (`kP4Kernels` true) and 16.3 s with
ESP-NN for the convolutions (`kP4Kernels` false). The attention blocks take 2.4 s in both builds.

## Requirements

- Board: ESP32-P4 with PSRAM. Select `PSRAM=enabled`, `ChipVariant=postv3` and
  `PartitionScheme=custom` (`esp32:esp32:esp32p4`).
- The P4 ESP-NN library (for the fallback kernels), linked with:
  `--build-property "compiler.c.elf.extra_flags=-L<esp32p4-libs>/lib -lespressif__esp-nn"`.
- Python with `ultralytics` for the export (`tools/yolo/yolo_export.py`).

## Create the weights and the table

Quick way: `python tools/fetch_models.py yolo26n` (see `models/yolo26n/README.md`).
The manual steps are:

```sh
python3 tools/yolo/yolo_export.py yolo26n.pt out/ photo.jpg
```

The export writes `out/` and `yolo26n.tym` next to it: the single-file model for the board.

## Flash the blob

The blob goes to the start of the "model" partition (absolute offset 0x310000, see
`partitions.csv`):

```sh
esptool --chip esp32p4 --port /dev/ttyACM0 write_flash 0x310000 yolo26n.tym
```

The blob is 13104288 bytes. The sketch's sizes must match the export: `kFloatCount` 2408932 floats,
`kInt8Count` 2397856 bytes and `kScaleCount` 8516 floats for the nano model. Change them if you
export another model.

## Run

Open `Yolo26Graph.ino`, upload it, and read the serial output at 115200 baud. The sketch runs three
frames and prints, for each, the graph time, the head time and the detections. Expected output for
the test photo. On the desktop (int8, shared-scale table) the result is one detection, class 17,
score 0.881.

Choose the table in `setup()`: `kQuant` (default, shared scales, fast) or `kQuantPerTensor` (one
scale per tensor, more accurate, slower; the speed has not been measured on this code).
