# Edge TPU Integration

## Installation

Install runtime and compiler first:

```
$ curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo gpg --dearmor -o /usr/share/keyrings/google-cloud.gpg
$ echo "deb [signed-by=/usr/share/keyrings/google-cloud.gpg] https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
$ sudo apt update
$ sudo apt install libedgetpu1-std edgetpu-compiler
```

Make sure you have `/usr/lib/udev/rules.d/60-libedgetpu1-std.rules` after installation. If not, create `/etc/udev/rules.d/99-edgetpu-accelerator.rules` manually with the following content:

```
SUBSYSTEM=="usb",ATTRS{idVendor}=="1a6e",GROUP="plugdev"
SUBSYSTEM=="usb",ATTRS{idVendor}=="18d1",GROUP="plugdev"
```

Reboot and confirm the TPU device is recognized.

```
$ lsusb
Bus 002 Device 003: ID 18d1:9302 Google Inc.
```

Next, install pip dependencies with `requirements_edgetpu.txt` ([@oberluz](https://github.com/oberluz) is kind enough to share PyCoral binary for Python 3.10, which is not supported officially):

```
$ cd ~/ros2_ws/src/ultralytics_ros
$ pip install -r requirements_edgetpu.txt
```

You can test installation by running a demo inference:

```
$ git clone https://github.com/google-coral/pycoral.git
$ cd pycoral
$ examples/install_requirements.sh classify_image.py
$ python examples/classify_image.py \
--model test_data/mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite \
--labels test_data/inat_bird_labels.txt \
--input test_data/parrot.jpg
----INFERENCE TIME----
Note: The first inference on Edge TPU is slow because it includes loading the model into Edge TPU memory.
11.8ms
3.0ms
2.8ms
2.9ms
2.9ms
-------RESULTS--------
Ara macao (Scarlet Macaw): 0.75781
```

## Prediction with TPU acceleration

Convert your pretrained model to TF Edge TPU format:

```
$ yolo export model=<path to model> format=edgetpu imgsz=<size of model's input image>
```

You will get a file with the suffix `_edgetpu.tflite`. Try prediction using the command below:

```
$ yolo predict model=<path to converted model> task=detect source="https://ultralytics.com/images/bus.jpg" imgsz=<size of model's input image>
```

If everything is going well, you can see the result in `run/detect/predict` directory.
