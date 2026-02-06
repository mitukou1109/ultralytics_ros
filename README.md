# ultralytics_ros

🤖 ROS 2 wrapper for [Ultralytics](https://github.com/ultralytics/ultralytics)

## 📋 Prerequisites

- CUDA-compatible GPU
- [uv](https://docs.astral.sh/uv/getting-started/installation) (Python package and project manager)
- ROS 2 Humble+

## 🚀 Installation

### 1. Clone

```bash
cd ~/ros2_ws/src
git clone https://github.com/mitukou1109/ultralytics_ros.git
```

### 2. Install dependencies

```bash
cd ~/ros2_ws/src
rosdep install -iyr --from-paths .
```

### 3. Build

```bash
cd ~/ros2_ws
colcon build --symlink-install
```

## 💻 Usage

### SAM 3

> [!IMPORTANT]
> SAM 3 checkpoints must be downloaded manually. Follow the steps in the [SAM 3 documentation](https://docs.ultralytics.com/models/sam-3/#installation):
>
> 1. Request access on the [SAM 3 Hugging Face repo](https://huggingface.co/facebook/sam3)
> 2. Download `sam3.pt` from [here](https://huggingface.co/facebook/sam3/resolve/main/sam3.pt?download=true)
> 3. Place the file in your desired location (or in your working directory to use the default `model_file` parameter)

```bash
source ~/ros2_ws/install/local_setup.bash
ros2 run ultralytics_ros sam3_segmentation_node \
  --ros-args \
  -p model_file:="/path/to/sam3.pt" \
  -p text_prompt:="['person', 'car']"
```

### YOLO Segmentation

Standard segmentation:

```bash
source ~/ros2_ws/install/local_setup.bash
ros2 run ultralytics_ros yolo_segmentation_node \
  --ros-args \
  -p model_file:="yolo26l-seg.pt"
```

Open-vocabulary segmentation with YOLOE:

```bash
source ~/ros2_ws/install/local_setup.bash
ros2 run ultralytics_ros yolo_segmentation_node \
  --ros-args \
  -p model_file:="yoloe-26l-seg.pt" \
  -p text_prompt:="['dog', 'cat']"
```

## 📦 Nodes

### Segmentation Nodes

Available nodes:

- `sam3_segmentation_node` - SAM 3
- `yolo_segmentation_node` - YOLO (supports open-vocabulary segmentation with [YOLOE](https://docs.ultralytics.com/models/yoloe) models)

#### Subscribed Topics

| Topic               | Type                              | Description            |
| ------------------- | --------------------------------- | ---------------------- |
| `/image`            | `sensor_msgs/msg/Image`           | Raw input image        |
| `/image/compressed` | `sensor_msgs/msg/CompressedImage` | Compressed input image |

#### Published Topics

| Topic                       | Type                               | Description                                       |
| --------------------------- | ---------------------------------- | ------------------------------------------------- |
| `~/mask_image`              | `sensor_msgs/msg/Image`            | Mask image (pixel value corresponds to mask ID)   |
| `~/mask_image/compressed`   | `sensor_msgs/msg/CompressedImage`  | Compressed mask image                             |
| `~/detections`              | `vision_msgs/msg/Detection2DArray` | Bounding boxes of detected masks                  |
| `~/result_image`            | `sensor_msgs/msg/Image`            | Result image with colored masks for visualization |
| `~/result_image/compressed` | `sensor_msgs/msg/CompressedImage`  | Compressed result image                           |

#### Parameters

| Parameter              | Type       | Default                      | Description                                                                                        |
| ---------------------- | ---------- | ---------------------------- | -------------------------------------------------------------------------------------------------- |
| `model_file`           | `string`   | `sam3.pt`                    | Path of model file                                                                                 |
| `input_size`           | `int`      | `1008` (SAM 3), `640` (YOLO) | Input size for the model (pixels)                                                                  |
| `confidence_threshold` | `float`    | `0.5`                        | Minimum confidence threshold for mask filtering                                                    |
| `use_half_precision`   | `bool`     | `True`                       | Use half precision for faster inference                                                            |
| `text_prompt`          | `[string]` | `[]`                         | Text prompt for concept segmentation (SAM 3 and open-vocabulary models like YOLOE)                 |
| `publish_result_image` | `bool`     | `True`                       | If `false`, disables publishing `~/result_image` topics to save bandwidth and increase output rate |

### Detection Nodes

TODO
