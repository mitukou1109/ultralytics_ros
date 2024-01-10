import cv_bridge
import numpy as np
import rclpy
import rclpy.node
import rclpy.publisher
import rclpy.qos
import rclpy.subscription
import sensor_msgs.msg
import ultralytics
import ultralytics.engine.results
import vision_msgs.msg
import functools


class Detector(rclpy.node.Node):
    def __init__(self) -> None:
        super().__init__("detector")

        self.declare_parameter("yolo_model", "yolov8n.pt")
        self.declare_parameter("image_topics", ["/image_raw"])
        self.declare_parameter("divide_source_image", False)
        self.declare_parameter("source_subimage_size", "640")
        self.declare_parameter("model_conf_threshold", 0.25)
        self.declare_parameter("model_iou_threshold", 0.7)
        self.declare_parameter("model_image_size", "640")

        self.model = ultralytics.YOLO(
            self.get_parameter("yolo_model").get_parameter_value().string_value,
            task="detect",
        )

        self.cv_bridge = cv_bridge.CvBridge()

        image_topics = (
            self.get_parameter("image_topics").get_parameter_value().string_array_value
        )
        if image_topics:
            self.detections_pubs: list[rclpy.publisher.Publisher] = []
            self.result_image_pubs: list[rclpy.publisher.Publisher] = []
            self.source_image_subs: list[rclpy.subscription.Subscription] = []
            for i, image_topic in enumerate(image_topics):
                self.detections_pubs.append(
                    self.create_publisher(
                        vision_msgs.msg.Detection2DArray,
                        f"~/detections_{i}",
                        5,
                    )
                )
                self.result_image_pubs.append(
                    self.create_publisher(
                        sensor_msgs.msg.Image,
                        f"~/result_image_{i}",
                        5,
                    )
                )
                source_image_sub_qos = rclpy.qos.qos_profile_sensor_data
                source_image_sub_qos.depth = 1
                self.source_image_subs.append(
                    self.create_subscription(
                        sensor_msgs.msg.Image,
                        image_topic,
                        functools.partial(self.source_image_callback, source_id=i),
                        source_image_sub_qos,
                    )
                )
        else:
            self.get_logger().error("No image topics are given, exiting...")
            self.destroy_node()
            exit(1)

    def source_image_callback(self, msg: sensor_msgs.msg.Image, source_id: int) -> None:
        divide_source_image = (
            self.get_parameter("divide_source_image").get_parameter_value().bool_value
        )
        if divide_source_image:
            source_subimage_size: int | tuple = eval(
                self.get_parameter("source_subimage_size")
                .get_parameter_value()
                .string_value
            )
            if type(source_subimage_size) is int:
                source_subimage_size = (source_subimage_size, source_subimage_size)
        else:
            source_subimage_size = (msg.height, msg.width)

        model_conf_threshold = (
            self.get_parameter("model_conf_threshold")
            .get_parameter_value()
            .double_value
        )
        model_iou_threshold = (
            self.get_parameter("model_iou_threshold").get_parameter_value().double_value
        )

        model_image_size: int | tuple = eval(
            self.get_parameter("model_image_size").get_parameter_value().string_value
        )
        if type(model_image_size) is int:
            model_image_size = (model_image_size, model_image_size)

        source_image = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")

        detections_msg = vision_msgs.msg.Detection2DArray()
        result_image_chunks = []
        row_images: list[np.ndarray] = np.array_split(
            source_image,
            max(1, source_image.shape[0] // source_subimage_size[0]),
            axis=0,
        )
        for row, row_image in enumerate(row_images):
            result_row_image_chunks = []
            chunks: list[np.ndarray] = np.array_split(
                row_image,
                max(1, row_image.shape[1] // source_subimage_size[1]),
                axis=1,
            )
            for col, chunk in enumerate(chunks):
                result: ultralytics.engine.results.Results = self.model.predict(
                    chunk,
                    conf=model_conf_threshold,
                    iou=model_iou_threshold,
                    imgsz=model_image_size,
                    verbose=False,
                )[0]
                for conf, cls, xywh in zip(
                    result.boxes.conf, result.boxes.cls, result.boxes.xywh
                ):
                    class_id = result.names[int(cls)]
                    hypothesis = vision_msgs.msg.ObjectHypothesis(
                        class_id=class_id,
                        score=float(conf),
                    )
                    object_center = vision_msgs.msg.Point2D(
                        x=sum(map(lambda c: c.shape[1], chunks[:col])) + float(xywh[0]),
                        y=sum(map(lambda c: c.shape[0], row_images[:row]))
                        + float(xywh[1]),
                    )
                    bbox = vision_msgs.msg.BoundingBox2D(
                        center=vision_msgs.msg.Pose2D(position=object_center),
                        size_x=float(xywh[2]),
                        size_y=float(xywh[3]),
                    )
                    detections_msg.detections.append(
                        vision_msgs.msg.Detection2D(
                            results=[
                                vision_msgs.msg.ObjectHypothesisWithPose(
                                    hypothesis=hypothesis
                                )
                            ],
                            bbox=bbox,
                            id=class_id,
                        )
                    )
                result_row_image_chunks.append(result.plot())
            result_image_chunks.append(np.hstack(result_row_image_chunks))
        result_image = np.vstack(result_image_chunks)

        detections_msg.header.stamp = self.get_clock().now().to_msg()
        detections_msg.header.frame_id = msg.header.frame_id
        self.detections_pubs[source_id].publish(detections_msg)

        result_image_msg = self.cv_bridge.cv2_to_imgmsg(result_image, "bgr8")
        result_image_msg.header.stamp = self.get_clock().now().to_msg()
        result_image_msg.header.frame_id = msg.header.frame_id
        self.result_image_pubs[source_id].publish(result_image_msg)


def main(args: list[str] = None):
    rclpy.init(args=args)
    detector = Detector()
    rclpy.spin(detector)
    detector.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
