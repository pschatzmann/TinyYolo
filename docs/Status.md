# Status

Current state, as measured in this repository:

- **Working on every board:** the YOLO11n and YOLO26n int8 graphs find the same object (class 17,
  the horse) on the sample photo, in all four targets: the ESP32-P4, the ESP32-S3, the RP2350B and the
  desktop. Frame times are in the [Performance](Performance.md) table.
- **Accuracy:** the int8 scores on the photo are 0.818 to 0.905 against the reference (0.905 with the
  portable kernels). The SiLU activations are applied after the rounding, so the ESP-NN and CMSIS-NN
  paths differ from the reference by one step on up to about 4 % of the values.
- **Not supported on these boards:** YOLOv4-tiny (its float Darknet weights do not fit in the
  S3's or the RP2350B's flash or PSRAM at 320x320), and the classic ESP32 (4 MB flash).
- **Not measured yet:** the float32 graph on the ESP32-P4 (the 78.7 s in the table is from an earlier
  build), and the float32 graph on the RP2350B. The RP2350B has no example sketch in the repository yet;
  its test sketches are outside it.
- **Not committed:** the RP2350B support and the desktop `--size` option are in the working tree.
