# ultralytics_ros

ROS2 package for using [Ultralytics](https://github.com/ultralytics/ultralytics).

## Requirements

- ROS 2 (Tested on Humble, would work on any distributions with Python 3.8+ required by `ultralytics`)

Other dependencies are listed in `package.xml`, `requirements.txt` (and optionally `requirements_edgetpu.txt`).

## Usage

### Setup

```
$ cd ~/ros2_ws/src
$ git clone https://github.com/mitukou1109/ultralytics_ros.git
$ cd ultralytics_ros
$ rosdep install -iyr --from-paths .
$ pip install -r requirements.txt
$ cd ~/ros2_ws
$ colcon build --symlink-install
$ source install/local_setup.bash
```

### Prediction

You can try prediction using a default model (`yolov8n.pt`) with the following commands if you have a webcam connected to your PC:

```
$ sudo apt install ros-humble-v4l2-camera
$ ros2 launch ultralytics_ros demo.launch.py
```

For those who unfortunately don't have access to GPU, Google's Edge TPU is a good option. To use Ultralytics with Edge TPU, follow [this instruction](edge_tpu_integration.md) or see [official docs](https://coral.ai/docs/accelerator/get-started/) for details.

## Topics

### `detector`

Subscribed topics:

| Name         | Type                | Description |
| :----------- | :------------------ | :---------- |
| `/image_raw` | `sensor_msgs/Image` | Input image |

Published topics:

| Name             | Type                           | Description                                                              |
| :--------------- | :----------------------------- | :----------------------------------------------------------------------- |
| `~/detections`   | `vision_msgs/Detection2DArray` | Sequence of bounding box, label and confidence for each detected objects |
| `~/result_image` | `sensor_msgs/Image`            | Input image with bounding boxes and labels drawn                         |

## Parameters

### `detector`

| Name                   | Type   | Description                                                                                                 | Default value  |
| :--------------------- | :----- | :---------------------------------------------------------------------------------------------------------- | :------------- |
| `yolo_model`           | string | path to model file (will be directly passed to `YOLO` class constructor)                                    | `"yolov8n.pt"` |
| `divide_source_image`  | bool   | whether to divide source image into small square sub-images which could help detect tiny objects            | `False`        |
| `source_subimage_size` | string | size of sub-image (int or tuple (e.g. `"(640, 480)"`), will be ignored if `divide_source_image` is `False`) | `"640"`        |
| `model_conf_threshold` | double | confidence threshold for detection                                                                          | `0.25`         |
| `model_iou_threshold`  | double | IoU (Intersection Over Union) threshold for NMS                                                             | `0.7`          |
| `model_image_size`     | string | size of model's input image (int or tuple)                                                                  | `"640"`        |
