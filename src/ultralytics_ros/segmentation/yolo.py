import queue
import threading
import time
import typing

import cv2
import cv_bridge
import geometry_msgs.msg
import numpy as np
import numpy.typing as npt
import rclpy.node
import rclpy.parameter
import rclpy.qos
import sensor_msgs.msg
import std_msgs.msg
import torch
import ultralytics.engine.results
import ultralytics.models.yolo
import vision_msgs.msg


class YOLO(rclpy.node.Node):
    class Result:
        def __init__(
            self,
            header: std_msgs.msg.Header,
            results: ultralytics.engine.results.Results,
        ) -> None:
            self.header = header
            self.results = results

    def __init__(self) -> None:
        super().__init__("segmentation")

        model_file = (
            self.declare_parameter("model_file", "").get_parameter_value().string_value
        )
        input_size = (
            self.declare_parameter("input_size", 640)
            .get_parameter_value()
            .integer_value
        )
        confidence_threshold = (
            self.declare_parameter("confidence_threshold", 0.5)
            .get_parameter_value()
            .double_value
        )
        use_half_precision = (
            self.declare_parameter("use_half_precision", True)
            .get_parameter_value()
            .bool_value
        )
        self.publish_result_image = (
            self.declare_parameter("publish_result_image", True)
            .get_parameter_value()
            .bool_value
        )

        self.add_on_set_parameters_callback(self._on_set_parameters_callback)

        assert model_file != "", "Parameter 'model_file' is not set"
        self.model = ultralytics.models.yolo.YOLO(model_file, task="segment")
        self.model.overrides["imgsz"] = input_size
        self.model.overrides["conf"] = confidence_threshold
        self.model.overrides["half"] = use_half_precision
        self.model.overrides["verbose"] = False

        if isinstance(self.model, ultralytics.models.yolo.YOLOE):
            self.text_prompts: list[str] = list(
                self.declare_parameter(
                    "text_prompts",
                    rclpy.Parameter.Type.STRING_ARRAY,
                    descriptor=rclpy.node.ParameterDescriptor(
                        type=rclpy.node.ParameterType.PARAMETER_STRING_ARRAY
                    ),
                )
                .get_parameter_value()
                .string_array_value
            )
            assert len(self.text_prompts) > 0, "Parameter 'text_prompts' is not set"
            self.model.set_classes(self.text_prompts)

        self.result_queue: queue.Queue[YOLO.Result] = queue.Queue()
        self.build_and_publish_result_thread = threading.Thread(
            target=self._build_and_publish_result, daemon=True
        )
        self.build_and_publish_result_thread.start()

        self.cv_bridge = cv_bridge.CvBridge()

        self.mask_image_pub = self.create_publisher(
            sensor_msgs.msg.Image, "~/mask_image", 1
        )
        self.mask_image_compressed_pub = self.create_publisher(
            sensor_msgs.msg.CompressedImage, "~/mask_image/compressed", 1
        )
        self.detections_pub = self.create_publisher(
            vision_msgs.msg.Detection2DArray, "~/detections", 1
        )

        self.image_sub = self.create_subscription(
            sensor_msgs.msg.Image,
            "image",
            self._image_callback,
            rclpy.qos.qos_profile_sensor_data,
        )
        self.image_compressed_sub = self.create_subscription(
            sensor_msgs.msg.CompressedImage,
            "image/compressed",
            self._image_compressed_callback,
            rclpy.qos.qos_profile_sensor_data,
        )

        self.result_image_pub = self.create_publisher(
            sensor_msgs.msg.Image, "~/result_image", 1
        )
        self.result_image_compressed_pub = self.create_publisher(
            sensor_msgs.msg.CompressedImage, "~/result_image/compressed", 1
        )

    def _on_set_parameters_callback(
        self, params: list[rclpy.parameter.Parameter]
    ) -> rclpy.node.SetParametersResult:
        for param in params:
            if param.name == "input_size":
                self.model.overrides["imgsz"] = (
                    param.get_parameter_value().integer_value
                )
            elif param.name == "confidence_threshold":
                self.model.overrides["conf"] = param.get_parameter_value().double_value
            elif param.name == "use_half_precision":
                self.model.overrides["half"] = param.get_parameter_value().bool_value
            elif param.name == "publish_result_image":
                self.publish_result_image = param.get_parameter_value().bool_value
            elif param.name == "text_prompts" and (
                isinstance(self.model, ultralytics.models.yolo.YOLOE)
            ):
                self.text_prompts = list(param.get_parameter_value().string_array_value)
                self.model.set_classes(self.text_prompts)
        return rclpy.node.SetParametersResult(successful=True)

    def _image_compressed_callback(self, msg: sensor_msgs.msg.CompressedImage) -> None:
        image = self.cv_bridge.compressed_imgmsg_to_cv2(msg).astype(np.uint8)
        self._run_inference(image, msg.header)

    def _image_callback(self, msg: sensor_msgs.msg.Image) -> None:
        image = self.cv_bridge.imgmsg_to_cv2(msg)
        self._run_inference(image, msg.header)

    def _run_inference(
        self, image: npt.NDArray[np.uint8], header: std_msgs.msg.Header
    ) -> None:
        start = time.time_ns()

        results = self.model.predict(image)[0]

        self.get_logger().debug(
            f"Inference time: {(time.time_ns() - start) / 1e6:.3f} ms"
        )

        self.result_queue.put(YOLO.Result(header, results))

    def _build_and_publish_result(self) -> None:
        while rclpy.ok():
            result = self.result_queue.get()
            start = time.time_ns()

            detections = vision_msgs.msg.Detection2DArray(header=result.header)

            if result.results.masks is None or result.results.boxes is None:
                self._publish_result(
                    detections,
                    np.zeros(result.results.orig_shape, dtype=np.uint8),
                    result.results,
                )
                continue

            detections.detections = self._build_detections(
                result.results.boxes,
                (
                    self.text_prompts
                    if isinstance(self.model, ultralytics.models.yolo.YOLOE)
                    else result.results.names
                ),
            )

            masks = typing.cast(torch.Tensor, result.results.masks.data)
            mask_image = (
                (
                    (masks > 0.5).unsqueeze(3)
                    * (result.results.boxes.cls[:, None, None, None] + 1)
                )
                .max(0)
                .values
            )

            self._publish_result(
                detections, mask_image.cpu().numpy().astype(np.uint8), result.results
            )

            self.get_logger().debug(
                f"Publish time: {(time.time_ns() - start) / 1e6:.3f} ms"
            )

    def _build_detections(
        self,
        boxes: ultralytics.engine.results.Boxes,
        class_names: dict[int, str] | list[str],
    ) -> list[vision_msgs.msg.Detection2D]:
        detections = []

        for xyxy, cls, conf in zip(boxes.xyxy, boxes.cls, boxes.conf):
            x1, y1, x2, y2 = xyxy.float().tolist()
            detection = vision_msgs.msg.Detection2D()
            detection.bbox.size_x = x2 - x1
            detection.bbox.size_y = y2 - y1
            detection.bbox.center.position.x = (x1 + x2) / 2
            detection.bbox.center.position.y = (y1 + y2) / 2
            typing.cast(list, detection.results).append(
                vision_msgs.msg.ObjectHypothesisWithPose(
                    hypothesis=vision_msgs.msg.ObjectHypothesis(
                        class_id=class_names[int(cls)],
                        score=conf.item(),
                    ),
                    pose=geometry_msgs.msg.PoseWithCovariance(
                        pose=geometry_msgs.msg.Pose(
                            position=geometry_msgs.msg.Point(
                                x=detection.bbox.center.position.x,
                                y=detection.bbox.center.position.y,
                                z=0.0,
                            )
                        ),
                    ),
                )
            )
            detections.append(detection)

        return detections

    def _publish_result(
        self,
        detections: vision_msgs.msg.Detection2DArray,
        mask_image: npt.NDArray[np.uint8],
        results: ultralytics.engine.results.Results,
    ) -> None:
        self.detections_pub.publish(detections)

        mask_image_msg = self.cv_bridge.cv2_to_imgmsg(
            mask_image, header=detections.header
        )
        self.mask_image_pub.publish(mask_image_msg)

        mask_image_compressed_msg = self.cv_bridge.cv2_to_compressed_imgmsg(
            mask_image, "png"
        )
        mask_image_compressed_msg.header = detections.header
        self.mask_image_compressed_pub.publish(mask_image_compressed_msg)

        if not self.publish_result_image:
            return

        resize_ratio = 640 / max(results.orig_shape)
        result_image = cv2.resize(
            results.plot(), dsize=None, fx=resize_ratio, fy=resize_ratio
        )

        result_image_msg = self.cv_bridge.cv2_to_imgmsg(
            result_image, header=detections.header
        )
        self.result_image_pub.publish(result_image_msg)

        result_image_compressed_msg = self.cv_bridge.cv2_to_compressed_imgmsg(
            result_image, "jpg"
        )
        result_image_compressed_msg.header = detections.header
        self.result_image_compressed_pub.publish(result_image_compressed_msg)


def main(args=None):
    rclpy.init(args=args)
    yolo = YOLO()
    rclpy.spin(yolo)
    yolo.destroy_node()
    rclpy.shutdown()
