# Yolo11CameraStream

YOLO11n on the Waveshare ESP32-P4-WIFI6 camera, streamed to a desktop computer. Each QVGA frame is
letterboxed to 640x640 and detected with the TinyYolo graph (`tinyyolo::GraphYolo11`). The viewer shows every frame with the boxes of the last detection. The stream and the viewer
are the same as in `Yolov4CameraStream`.

## Files

- `Yolo11CameraStream.ino`: the sketch.
- `yolo11n_graph.inc`: the operation table (copy of `models/yolo11n/yolo11n_graph.inc`).
- `partitions.csv`: 32 MB flash; the `model` partition at 0x310000 is about 30 MB (0x1CF0000 bytes).

## Steps

1. Get the weights and export them (needs Python with `ultralytics`):
   ```bash
   PYTHON=python3 tools/yolo11/yolo11_fetch_models.sh      # writes models/yolo11n/export/yolo11n_board.bin
   ```
2. Flash the blob to the `model` partition (the sketch reads it from there):
   ```bash
   esptool --chip esp32p4 --port /dev/ttyACM0 write_flash 0x310000 models/yolo11n/export/yolo11n_board.bin
   ```
3. Compile and upload for the ESP32-P4 (`esp32:esp32:esp32p4`, `PSRAM=enabled`, `ChipVariant=postv3`,
   `PartitionScheme=custom`). The int8 convolutions use the ESP-NN fallback kernels, so the build links
   the P4 ESP-NN library:
   ```bash
   LIB=~/.arduino15/packages/esp32/tools/esp32p4-libs/3.3.12/lib
   arduino-cli compile -b esp32:esp32:esp32p4:PSRAM=enabled,FlashSize=32M,ChipVariant=postv3,PartitionScheme=custom \
       --build-property "compiler.c.elf.extra_flags=-L$LIB -lespressif__esp-nn" examples/Yolo11CameraStream
   arduino-cli upload  -b esp32:esp32:esp32p4:PSRAM=enabled,FlashSize=32M,ChipVariant=postv3,PartitionScheme=custom \
       -p /dev/ttyACM0 examples/Yolo11CameraStream
   ```
4. Start the desktop viewer (built with the `tinyyolo` target):
   `tinyyolo --serial /dev/ttyACM0 --rotate 180`.

## Notes

- The boxes are sent in the camera picture's coordinates (320x240), which is the viewer's space.
- Detection runs on every frame, so the stream runs at the detection rate (about one frame per 6 s).
  The `kDetectEvery` constant skips frames if you want the stream to run faster.
- The `T` line of the serial output gives the detection time and the frame time.
