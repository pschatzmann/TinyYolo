# Performance

Total time per picture on the ESP32-P4 (Waveshare ESP32-P4-WIFI6), the ESP32-S3, the RP2350B and the desktop (the host PC, `desktop/yolo/`, mean of three runs), with every model at the same
network input of 320x320. The photo (375x250) is letterboxed to 320x320. Each row is the mean of two
or three frames; the frame times varied by less than 3 %. The graph rows include the head.

| Model | Path | ESP32-P4 time | ESP32-S3 time | RP2350B time | Desktop time | Detections on the photo (class 17 = horse) |
|---|---|---:|---:|---:|---:|---|
| YOLO11n | graph, int8 (P4 assembly kernels; ESP-NN and exact 1x1 on the S3; CMSIS-NN on the RP2350B) | 1.5 s (head 7 ms) | 6.6 s (head 16 ms) | 18.9 s | 1.0 s | P4: score 0.818; S3: score 0.852; RP2350B: score 0.852; desktop: score 0.905 |
| YOLO11n | graph, float32 | 78.7 s (head 7 ms) | 63 s (head 16 ms) | not run | 2.2 s | score 0.833 |
| YOLO26n | graph, int8 (P4 assembly kernels; ESP-NN and exact 1x1 on the S3; CMSIS-NN on the RP2350B) | 1.9 s (head 6 ms) | 7.1 s (head 15 ms) | 15.8 s | 1.0 s | P4: score 0.818; S3: score 0.881; RP2350B: score 0.818; desktop: score 0.881 |
| YOLO26n | graph, float32 | not run | not run | not run | 2.0 s | score 0.891 |
| YOLOv4-tiny | Darknet model, int8 (weights only) | 13.8 s (head included) | does not fit (8 MB PSRAM) | does not fit (16 MB flash) | 10.2 s | P4: score 0.989; desktop: score 0.981 |
| YOLOv4-tiny | Darknet model, int8 with int8 activations (`--calibrate`) | not run | not run | not run | 3.3 s | desktop: score 0.985 |
| YOLOv4-tiny | Darknet model, float32 | not run | not run | not run | 7.5 s | score 0.979 |

The int8 convolutions on the S3 use the ESP-NN kernels, vendored in the library
(ESP-NN v1.5.0 as headers, so no extra flags are needed). ESP-NN rounds the requantization once, as the
reference does. Its activations are applied after the rounding, so the SiLU layers differ from the
reference by at most one step, on up to about 4 % of the values, and the linear 1x1 convolutions (the detection heads and the
neck projections) run on our own exact kernel, which decides the final scores. The YOLO11n S3 graph
finds the horse (class 17) at score 0.852, against 0.881 for the reference, in 6.6 s. The S3 numbers
are at 320x320 and come from three frames.

The RP2350B (SpotPear RP2350B-MINI-A, 8 MB PSRAM on GPIO 19, 16 MB flash) runs at 300 MHz with
`-Ofast` and PSRAM at 133 MHz. Its int8 convolutions use CMSIS-NN, which is the default for the RP2040
and RP2350. The weights of YOLO11n and YOLO26n are read from flash, and their int8 copies are in PSRAM.
Each RP2350B row is the mean of three frames.

The desktop tools run on the host PC with `DotProductBackend`. On x86-64 the int8 convolutions use AVX2 with exact int32 sums (the same result as the portable kernel). The Darknet models use weights-only int8 by default; `--calibrate` also quantizes the activations, which is the faster path. The desktop times were measured on a machine with other load (load average 1 to 5), so they vary by up to about 10 %. The desktop float32 and int8 scores are the reference values.

The P4 times use the P4 assembly kernels for all convolutions and depthwise convolutions. YOLOv4-tiny uses its Darknet model at 320x320 (the 192 config with the input size changed).

Espressif publishes faster INT8 YOLO11n timings for the ESP32-P4. We have not reproduced them: the
table above is measured on our board only.
