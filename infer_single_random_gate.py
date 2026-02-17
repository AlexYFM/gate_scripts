#!/usr/bin/env python3
import argparse
import json
import math
import os
import random
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from ultralytics import YOLO


class SingleGateInference(Node):
    def __init__(self, weights_path: str, output_dir: str):
        super().__init__('single_gate_inference')
        self.bridge = CvBridge()
        self.latest_msg = None

        with open('world_config.json', 'r') as config_file:
            self.config = json.load(config_file)

        self.output_dir = Path(output_dir).expanduser()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.model = YOLO(weights_path)
        self.get_logger().info(f"Loaded model: {weights_path}")

        self.create_subscription(Image, '/X3/camera/image_raw', self.img_cb, 10)

    def img_cb(self, msg):
        self.latest_msg = msg

    def teleport_to_random_view_of_gate(self, gate_pose):
        gx, gy, _, _, _, gyaw = gate_pose

        dist = random.uniform(4.0, 8.5)
        angle_offset = random.uniform(-math.radians(40), math.radians(40))
        total_yaw = gyaw + angle_offset
        tz = random.uniform(0.5, 3.0)

        tx = gx - dist * math.cos(total_yaw)
        ty = gy - dist * math.sin(total_yaw)

        look_at_yaw = total_yaw + random.uniform(-math.radians(5), math.radians(5))
        qz = math.sin(look_at_yaw / 2)
        qw = math.cos(look_at_yaw / 2)

        req_data = f'name:"X3", position:{{x:{tx}, y:{ty}, z:{tz}}}, orientation:{{z:{qz}, w:{qw}}}'
        cmd = [
            'ign', 'service', '-s', '/world/a2rl_track_ign/set_pose',
            '--reqtype', 'ignition.msgs.Pose',
            '--reptype', 'ignition.msgs.Boolean',
            '--timeout', '1000',
            '--req', req_data
        ]

        self.get_logger().info(
            f"Teleporting near gate to ({tx:.2f}, {ty:.2f}, {tz:.2f}), yaw={math.degrees(look_at_yaw):.1f}°"
        )

        for attempt in range(2):
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                return True
            self.get_logger().warn(f"Teleport attempt {attempt + 1} failed: {result.stderr.strip()}")
            time.sleep(0.5)

        return False

    def capture_fresh_frame(self, timeout_sec: float = 3.0):
        self.latest_msg = None

        for _ in range(10):
            rclpy.spin_once(self, timeout_sec=0.01)

        end_time = time.time() + timeout_sec
        while rclpy.ok() and time.time() < end_time:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.latest_msg is not None:
                return self.bridge.imgmsg_to_cv2(self.latest_msg, 'bgr8')

        return None

    @staticmethod
    def draw_predictions(image, result):
        output = image.copy()

        if result.boxes is not None:
            for box_idx, box in enumerate(result.boxes.xyxy.cpu().numpy()):
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(output, (x1, y1), (x2, y2), (255, 255, 0), 2)
                cv2.putText(
                    output,
                    f"det {box_idx}",
                    (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 0),
                    1,
                    cv2.LINE_AA,
                )

        if result.keypoints is not None and result.keypoints.xy is not None:
            keypoints_xy = result.keypoints.xy.cpu().numpy()
            confidences = None
            if result.keypoints.conf is not None:
                confidences = result.keypoints.conf.cpu().numpy()

            for det_idx, points in enumerate(keypoints_xy):
                for kp_idx, point in enumerate(points):
                    x, y = int(point[0]), int(point[1])

                    if confidences is not None and confidences[det_idx][kp_idx] < 0.2:
                        continue

                    cv2.circle(output, (x, y), 4, (0, 255, 0), -1)
                    cv2.putText(
                        output,
                        str(kp_idx),
                        (x + 5, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (0, 255, 0),
                        1,
                        cv2.LINE_AA,
                    )

        return output

    def run_once(self, conf=0.25):
        gate = random.choice(self.config['world_layout'])
        self.get_logger().info(f"Selected gate: {gate['name']} ({gate['type']})")

        if not self.teleport_to_random_view_of_gate(gate['pose']):
            self.get_logger().error('Teleport failed. Aborting.')
            return False

        time.sleep(0.6)
        image = self.capture_fresh_frame()
        if image is None:
            self.get_logger().error('No camera frame received after teleport.')
            return False

        results = self.model.predict(source=image, conf=conf, verbose=False)
        if not results:
            self.get_logger().warn('Model returned no results object.')
            return False

        result = results[0]
        drawn = self.draw_predictions(image, result)

        timestamp = int(time.time() * 1000)
        raw_path = self.output_dir / f'raw_{timestamp}.png'
        pred_path = self.output_dir / f'pred_{timestamp}.png'
        cv2.imwrite(str(raw_path), image)
        cv2.imwrite(str(pred_path), drawn)

        det_count = int(len(result.boxes)) if result.boxes is not None else 0
        self.get_logger().info(f"Detections: {det_count}")

        if result.keypoints is not None and result.keypoints.xy is not None:
            keypoints_xy = result.keypoints.xy.cpu().numpy()
            keypoints_conf = result.keypoints.conf.cpu().numpy() if result.keypoints.conf is not None else None

            for det_idx, points in enumerate(keypoints_xy):
                self.get_logger().info(f"Detection {det_idx} keypoints:")
                for kp_idx, point in enumerate(points):
                    kp_conf_str = ''
                    if keypoints_conf is not None:
                        kp_conf_str = f", conf={keypoints_conf[det_idx][kp_idx]:.3f}"
                    self.get_logger().info(
                        f"  kp{kp_idx}: x={point[0]:.1f}, y={point[1]:.1f}{kp_conf_str}"
                    )

        self.get_logger().info(f"Saved raw image: {raw_path}")
        self.get_logger().info(f"Saved prediction image: {pred_path}")
        return True


def resolve_weights_path(user_path: str | None) -> str:
    if user_path:
        candidate = Path(user_path).expanduser()
        if candidate.exists():
            return str(candidate)
        raise FileNotFoundError(f"Provided weights not found: {candidate}")

    preferred = Path('runs/pose/gate_training/v2_training/weights/best.pt')
    if preferred.exists():
        return str(preferred)

    fallback = Path('runs/pose/gate_training/v2_initial_run/weights/best.pt')
    if fallback.exists():
        return str(fallback)

    best_candidates = sorted(Path('runs/pose').glob('**/weights/best.pt'))
    if best_candidates:
        return str(best_candidates[-1])

    raise FileNotFoundError('No best.pt found under runs/pose. Pass --weights explicitly.')


def main():
    parser = argparse.ArgumentParser(description='Teleport to one random gate, run YOLO pose inference, and draw predicted waypoints.')
    parser.add_argument('--weights', type=str, default=None, help='Path to trained weights (.pt).')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold for detection.')
    parser.add_argument('--output-dir', type=str, default='~/mav_sim/test_projections', help='Where to save raw and predicted images.')
    args = parser.parse_args()

    weights_path = resolve_weights_path(args.weights)

    rclpy.init()
    node = SingleGateInference(weights_path=weights_path, output_dir=args.output_dir)
    try:
        node.run_once(conf=args.conf)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
