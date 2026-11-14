# Memory requirements

Measured on the ESP32-S3 (`PSRAM=opi`, 8 MB, `FlashSize=16M`) and the ESP32-P4 (`PSRAM=enabled`,
`FlashSize=32M`, `ChipVariant=postv3`) with the YOLO11n graph at 320x320. The S3 numbers are from the
graph example after three frames; the P4 numbers are the free counts after three frames. The P4
PSRAM total was not read from the board, so only its free count is given.

## Flash

| Item | Size |
|---|---|
| Sketch, YOLO11n graph example (S3) | 658 KB |
| Sketch, YOLO11n graph example (P4) | 749 KB |
| YOLO11n weight blob (`yolo11n_board.bin`) | 12.5 MB |
| YOLO26n weight blob (`yolo26n_board.bin`) | 11.5 MB |
| `model` partition, S3 (`partitions.csv`) | 12.9 MB (0xCF0000) |
| `model` partition, P4 (`partitions.csv`) | 28.9 MB (0x1CF0000) |

Both blobs fit their partition. The blob holds the float parameters first (2,616,232 floats,
about 10 MB), then the int8 weights (2.5 MB), then their scales (8,136 floats, 32 KB).

There is no PROGMEM handling. The sketch code runs from flash, and the blob is mapped
with `esp_partition_mmap` and read in place. The float parameters stay in flash; reading them is slower
than from RAM, but they take no RAM.

## PSRAM

- **Int8 weights and scales:** copied from the blob at startup (`GraphYolo11Partition`), because the
  flash cache reads them slowly: 2.5 MB of weights and 32 KB of scales.
- **Activations:** the graph's int8 and float tensors. The large ones are placed in PSRAM first (the
  quantized input and the convolution outputs); the rest use internal RAM when it has room.
- **ESP-NN scratch:** each convolution gets a scratch buffer of at most 512 KB. Large feature maps are
  computed in row bands so that the scratch stays under that limit. The scratch is placed like the
  other buffers (internal RAM first).
- **Measured:** on the S3, 2.8 MB of PSRAM were free after a frame (about 5.2 MB in use of the 8 MB).
  On the P4, 24.8 MB were free.

## Internal RAM (S3)

- Global variables of the graph example: 22.6 KB of the 327 KB available.
- Free after a frame: 37 KB. This is the margin that the ESP-NN and the task stacks share.
- The S3-only multicore worker (`esp_nn_dual_core_enable`) is not started by the library, so it adds no
  stack unless the sketch enables it.

## RP2350B (8 MB PSRAM)

Measured on the SpotPear RP2350B-MINI-A (16 MB flash, 8 MB PSRAM on GPIO 19), with the test sketches
(outside the repository) at 320x320:

- **Program:** the YOLO11n sketch uses 13.5 MB of the 16 MB flash (80 %), the YOLO26n sketch 12.5 MB
  (74 %). The weight blob is linked into the program with an assembly `.incbin`, so the flash holds it
  directly (no partition).
- **PSRAM:** the int8 weights (2.5 MB for YOLO11n) and their scales are copied from the blob at
  startup, and the activations are placed in PSRAM when the internal RAM cannot hold them. The
  float parameters stay in flash.
- **Internal RAM:** the program's globals use about 15 KB of the 512 KB RAM.

## Choosing a board

- **YOLO11n or YOLO26n:** the S3 with 8 MB PSRAM works, at 6.6 s per frame (see the README). The P4 is
  faster (1.5 s) and has more PSRAM.
- **YOLOv4-tiny (Darknet, int8):** does not fit on the S3 at 320x320 (see the README).
- **RP2350B with 8 MB PSRAM:** YOLO11n and YOLO26n (CMSIS-NN, 16 to 19 s per frame).
- **Classic ESP32:** not supported (4 MB flash cannot hold the blob).

The numbers above are for the graph example as built in this repository. Changing the input size,
the model or the camera buffers changes them; check the free counts with `heap_caps_get_free_size`.
