import cv_bridge
import numpy as np
import rclpy
import rclpy.node
import rclpy.qos
import sensor_msgs.msg
import ultralytics
import ultralytics.engine.results


class Detector(rclpy.node.Node):
    def __init__(self):
        super().__init__("detector")

        self.declare_parameter("yolo_model", "yolov8n.pt")
        self.declare_parameter("source_subimage_size", 640)
        self.declare_parameter("model_conf_threshold", 0.25)
        self.declare_parameter("model_iou_threshold", 0.7)
        self.declare_parameter("model_image_size", 640)

        self.source_image_sub = self.create_subscription(
            sensor_msgs.msg.Image,
            "/image_raw",
            self.source_image_callback,
            rclpy.qos.qos_profile_sensor_data,
        )
        self.result_image_pub = self.create_publisher(
            sensor_msgs.msg.Image,
            "~/result/image_raw",
            5,
        )

        self.cv_bridge = cv_bridge.CvBridge()

        self.model = ultralytics.YOLO(
            self.get_parameter("yolo_model").get_parameter_value().string_value,
            task="detect",
        )

    def source_image_callback(self, msg: sensor_msgs.msg.Image):
        source_subimage_size = (
            self.get_parameter("source_subimage_size")
            .get_parameter_value()
            .integer_value
        )
        model_conf_threshold = (
            self.get_parameter("model_conf_threshold")
            .get_parameter_value()
            .double_value
        )
        model_iou_threshold = (
            self.get_parameter("model_iou_threshold").get_parameter_value().double_value
        )
        model_image_size = (
            self.get_parameter("model_image_size").get_parameter_value().integer_value
        )

        source_image = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")

        result_image_chunks = []
        for row_image in np.array_split(
            source_image, max(1, source_image.shape[0] // source_subimage_size), axis=0
        ):
            result_row_image_chunks = []
            for chunk in np.array_split(
                row_image, max(1, row_image.shape[1] // source_subimage_size), axis=1
            ):
                result: ultralytics.engine.results.Results = self.model.predict(
                    chunk,
                    conf=model_conf_threshold,
                    iou=model_iou_threshold,
                    imgsz=model_image_size,
                    verbose=False,
                )[0]
                result_row_image_chunks.append(result.plot())
            result_image_chunks.append(np.hstack(result_row_image_chunks))
        result_image = np.vstack(result_image_chunks)

        result_image_msg = self.cv_bridge.cv2_to_imgmsg(result_image, "bgr8")
        result_image_msg.header.stamp = self.get_clock().now().to_msg()
        result_image_msg.header.frame_id = msg.header.frame_id
        self.result_image_pub.publish(result_image_msg)


def main(args: list[str] = None):
    rclpy.init(args=args)
    detector = Detector()
    rclpy.spin(detector)
    detector.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
