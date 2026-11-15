# CameraStream

YOLO11n (int8, graph path) on an ESP32-S3-EYE camera board. The sketch runs the detector on each
camera frame and streams the picture and the detections to a desktop computer over the USB-UART
link. The desktop tool shows the stream: `build/desktop/tinyyolo --serial /dev/ttyACM0`. YOLO11n is
the fastest model on the S3 (about 6.6 s per frame, measured in the table in the main README).

## Files

- `CameraStream.ino`: the sketch.
- `partitions.csv`: the 16 MB flash layout. The `model` partition starts at 0x310000 and has size
  0xCF0000, which ends exactly at the end of the flash.

## Steps

1. Make the model file from the YOLO11n export, and flash it to the `model` partition at 0x310000. The
   file is one `.tym` model (docs/ModelFormat.md), 13,124,885 bytes:

   ```sh
   (the export writes models/yolo11n/yolo11n.tym)
   esptool --chip esp32s3 --port /dev/ttyACM0 --baud 921600 write_flash 0x310000 yolo11n.tym
   ```

   The Arduino ESP32 core ships an `esptool` that works with the S3 stub (the system package may not).
   The write takes about three minutes.

2. Compile and upload the sketch (the custom partition table is written with it):

   ```sh
   arduino-cli compile -b esp32:esp32:esp32s3:PSRAM=opi,FlashSize=16M,PartitionScheme=custom examples/CameraStream
   arduino-cli upload  -b esp32:esp32:esp32s3:PSRAM=opi,FlashSize=16M,PartitionScheme=custom -p /dev/ttyACM0 examples/CameraStream
   ```

3. Start the desktop tool with the stream (the class names come from the model file):

   ```sh
   build/desktop/tinyyolo --serial /dev/ttyACM0 --rotate 0
   ```

The window shows `YOLO11n int8` on the top card, the picture, and each detection with its class name.

## Stream protocol

Binary-safe, 921600 baud, 8N1. Before each frame the sketch sends:

- `MODEL <name>` (the model name shown in the window),
- `JPEG <w> <h> <nbytes> <ndet>` (the picture size, 320x240, the byte count of the JPEG and the number of detections),
- one line per detection, `x y width height score class`, in the picture coordinates,
- the camera's JPEG picture, as the camera delivered it (about 4.5 KB per frame).

The desktop tool decodes the JPEG with TinyJPEG. The older `FRAME <w> <h> <nbytes> <ndet>` header
(a raw RGB565 picture) is still understood by the tool.

## Notes

- The camera delivers JPEG frames, and each frame is decoded to RGB565 in PSRAM for the detector. A QVGA
  RGB565 capture does not initialize on the S3-EYE (`cam_dma_config failed`), while JPEG does.
- The sketch uses the S3-EYE pin preset of TinyCamera. The graph, the scales and the class names all come
from the model file, so the sketch needs no table of its own. Logging is off, because log lines would corrupt the stream.
- After a flash, the board sometimes stays in the ROM's download mode (`waiting for download`). Press
  RESET once, with BOOT not held, and the sketch starts.
- The stream is read at 921600 baud. A serial monitor at 115200 shows only the boot messages.
