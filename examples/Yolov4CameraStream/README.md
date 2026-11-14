# Yolov4CameraStream

YOLOv4-tiny (COCO, 80 classes) on an ESP32-S3 camera board, with the weights in flash.

Files:
- `Yolov4CameraStream.ino`: the sketch.
- `Yolov4Model.h`: the network description (`yolov4-tiny-192.cfg`) and class names, as arrays.
- `partitions.csv`: an 8 MB flash layout with a 6.75 MB data partition `model`.

## Steps

1. Export the weights to a model blob (on the desktop, from the repository root):
   ```bash
   tools/yolov4_fetch_models.sh
   build/desktop/tinyyolo_export_blob examples/Yolov4CameraStream/yolov4-tiny-192.cfg \
       models/yolov4-tiny/yolov4-tiny.weights yolov4-tiny.blob
   ```
2. Compile and upload the sketch for the ESP32-S3-EYE. The sketch uses the ESP-NN kernels, which
   are part of the library (see [Quantization](../../docs/Quantization.md#esp-nn-on-the-esp32-s3)), so no
   extra flags are needed:
   ```bash
   arduino-cli compile -b esp32:esp32:esp32s3:PSRAM=opi,FlashSize=8M,PartitionScheme=custom \
       examples/Yolov4CameraStream
   arduino-cli upload  -b esp32:esp32:esp32s3:PSRAM=opi,FlashSize=8M,PartitionScheme=custom \
       -p /dev/ttyACM0 examples/Yolov4CameraStream
   ```
3. Write the weights into the `model` partition (offset 0x140000 in `partitions.csv`):
   ```bash
   esptool.py --chip esp32s3 --port /dev/ttyACM0 write_flash 0x140000 yolov4-tiny.blob
   ```
4. Start the desktop viewer: `camera_view --serial /dev/ttyACM0 --rotate 180`.

## Limits

- Weights are int8, convolutions that feed a YOLO head stay float32 (see the Quantization doc).
- The model runs at 192x192 (`yolov4-tiny-192.cfg`). For 256x256, use a 256 config and
  regenerate `Yolov4Model.h`; the blob is the same. The activations need about 2.4 MB of PSRAM at 192.
- The detector runs on every third frame (`kDetectEvery`); the other frames show the last boxes.
- Start-up: the sketch calibrates the int8 activations on the first camera frame, which takes
  about 82 s (`calibrated in ... ms` on the serial port). The camera is idle meanwhile.
- Speed (ESP32-S3-EYE, 192x192, int8 weights and int8 activations, `Esp32NnBackend`): see
  [complete run times](#complete-run-times) below. The slowest layers are the 3x3 layers (up to
  3 s each) and the two float32 head layers take about 1.3 s together. The per-layer table is in
  [Quantization](../../docs/Quantization.md#full-network-on-the-esp32-s3).
- The sketch prints the free memory after start-up (`free after setup`) and the time of every
  layer once, after the first detection (`layer N ... us`). `kProfileLayers` turns the layer
  times off.
- Buffers are placed in internal RAM first, and in PSRAM when internal RAM is too small. Build
  with `-DTINYYOLO_PSRAM_FIRST` to try PSRAM first; the speed is the same in this network.
- The kernels are compiled with `-O2` by a pragma in the library, even when the sketch uses the
  core's `-Os`, so the default build runs at the speed of an `-O2` build.

## Complete run times

Measured on an ESP32-S3-EYE with `Esp32NnBackend`, YOLOv4-tiny at 192x192 with int8 weights and
int8 activations. The default build (`-Os` with the kernel pragma) unless noted.

| Stage | Time |
|---|---:|
| Calibration on the first camera frame | 82 s |
| First detection frame | 16.4 s |
| Calibration and first detection together | about 100 s |
| Detection frame, every third camera frame (over 4 runs and 4 builds) | 16.0 to 16.9 s |
| Frame without detection (stream only) | 21 to 22 ms |

Per detection cycle, the camera delivers three frames: one detection frame of about 16.5 s
followed by two frames of about 22 ms. So the picture on the desktop is updated about three times
per 16.5 s, and the boxes are refreshed once per cycle.

Free memory after start-up: internal RAM 81 KB, PSRAM 3.77 MB (default build).

Build variants (calibration and detection frame, measured on the same board):

| Build | Calibration | Detection frame |
|---|---:|---:|
| `-Os` with the kernel pragma (default) | 82 s | 16.4 s |
| `-O2` | 83 s | 16.2 s |
| `-O3` | 83 s | 16.2 s |
| `-Os`, without the kernel pragma | 110 s | 16.8 s |
| `-Os` with `-DTINYYOLO_INT8_HEADS` (two head layers in int8) | 82 s | 16.0 s |

## ESP32-P4 (Waveshare ESP32-P4-WIFI6)

The same sketch builds for the P4. The camera pins and the RGB565 byte order follow the chip, and
the weights are copied into PSRAM. Build with FQBN `esp32:esp32:esp32p4:PSRAM=enabled,FlashSize=32M,ChipVariant=postv3,PartitionScheme=custom`
and the P4 ESP-NN link flag `-L <esp32p4-libs>/lib -lespressif__esp-nn`. Write the blob with
`esptool --chip esp32p4 write_flash 0x140000 yolov4-tiny.blob`.

Measured with the camera running (one boot, `calibrated in` and `T` lines):

| Stage | Time |
|---|---:|
| Calibration on the first camera frame | 239 s |
| Detection frame | 30.5 s (76 s in another run) |
| Frame without detection (stream only) | 5 to 6 ms |

These are much slower than the same model on a stored photo on the P4 (calibration 22 s, detection
5.4 s). The camera's own tasks compete with the inference for CPU and memory (internal RAM free
after setup is 39 KB, against 96 KB without the camera). The cause is not yet confirmed.
