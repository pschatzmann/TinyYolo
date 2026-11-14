# Training a model

TinyYolo runs models; it does not train them. Train with Darknet, which writes the
`.cfg`, `.names` and `.weights` files that TinyYolo loads (see [Models](Models.md)).

## What you need

- Darknet (the AlexeyAB fork is the usual one), built with CUDA if you have a GPU.
- Images of your objects, with bounding boxes in Darknet format (one `.txt` per image:
  `class x_center y_center width height`, all relative to the image size). Label tools
  such as CVAT or LabelImg can export this format.
- A pretrained YOLO-tiny `.weights` file for your starting point (optional but much faster
  to train).

## Steps

1. **Choose a tiny architecture.** Use `yolov3-tiny.cfg` or `yolov4-tiny.cfg` from Darknet's
   `cfg/` folder. They use only supported sections.
2. **Set the input size for your board.** In `[net]`, use `width=` and `height=` as a
   multiple of 32, for example 96 or 128. Smaller inputs need less RAM and run faster.
3. **Set your classes.** In each `[yolo]` section, set `classes=N`. In the `[convolutional]`
   layer just before each `[yolo]`, set `filters=(N + 5) * 3`. For three anchors per head
   that is 24 for three classes.
4. **Write the data files.** `obj.names` lists the class names, one per line. `obj.data`
   points to the names, the training and validation image lists, and a backup folder.
5. **Train.**
   ```bash
   ./darknet detector train obj.data custom-yolov3-tiny.cfg yolov3-tiny.conv.15
   ```
   Use the pretrained `.conv` weights for the first run. Darknet writes checkpoints to the
   backup folder; `custom-yolov3-tiny_final.weights` is the last one.
6. **Check the result** with `./darknet detector test obj.data custom.cfg custom_final.weights`
   on a few images you didn't train on.

## Using the trained model

Copy `custom.cfg`, `obj.names` and `custom_final.weights` to the board. The `.cfg` is the
same file Darknet trained with, so it can be used as is. The `.names` file lists the labels
in the same order as the class ids.

Then load them as in [Models](Models.md). Two checks catch most mistakes:

- A `Head expects N channels` error means `filters` in the convolution before a `[yolo]`
  section does not match `(classes + 5) * masks`.
- A `weight file has unread data` warning means the `.cfg` and `.weights` do not match
  (for example the `.weights` came from a different architecture).

## Making the model fit the board

- **Input size:** smaller is the biggest saving. Activation memory scales with the input
  area.
- **Fewer filters:** reduce `filters` in the convolution layers and retrain.
- **Int8 weights:** set `weight_precision = WeightPrecision::Int8` to store weights in
  about a quarter of the memory (see [Quantization](Quantization.md)). Check the detections
  afterwards, because the quantization error depends on the trained weights.
- **PSRAM:** boards with PSRAM (for example the ESP32-S3 and ESP32-P4 modules) can hold
  larger models. Large buffers are placed there automatically.

## Notes

- TinyYolo's box decoding follows Darknet's YOLOv3 style for `[yolo]` sections, so a model
  trained with Darknet decodes correctly without changes.
- `tinyyolo_make_weights` (see [Desktop](Desktop.md)) writes pseudo-random weights for a `.cfg`,
  which is useful for testing the pipeline. Replace them with your trained model before judging
  detection quality.
