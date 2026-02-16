import rclpy
from rclpy.node import Node
import cv2
import json
import math
import os
import subprocess
import time
import re
import random
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from scipy.spatial.transform import Rotation as R

class GateVisualTester(Node):
    def __init__(self):
        super().__init__('gate_visual_tester')
        self.bridge = CvBridge()
        self.latest_msg = None

        # --- Section: Load Config ---
        with open('world_config.json', 'r') as f:
            self.config = json.load(f)

        cam_cfg = self.config['camera']
        f_len = cam_cfg['width'] / (2 * np.tan(cam_cfg['fov'] / 2))
        self.K = np.array([
            [f_len, 0, cam_cfg['width'] / 2],
            [0, f_len, cam_cfg['height'] / 2],
            [0, 0, 1]
        ], dtype=np.float32)

        self.img_dir = os.path.expanduser('~/mav_sim/test_projections')
        os.makedirs(self.img_dir, exist_ok=True)

        self.create_subscription(Image, '/X3/camera/image_raw', self.img_cb, 10)

    def img_cb(self, msg):
        self.latest_msg = msg

    # def teleport_and_verify(self, target_pose):
    #     """
    #     Section: Reliable Teleport
    #     Ensures the service call succeeds and the drone physically arrives.
    #     """
    #     gx, gy, gz, _, _, gyaw = target_pose
    #     dist = random.uniform(4.0, 7.0)
    #     angle_offset = random.uniform(-math.radians(20), math.radians(20))
    #     total_yaw = gyaw + angle_offset
    #     tx, ty, tz = gx - dist * math.cos(total_yaw), gy - dist * math.sin(total_yaw), random.uniform(1.0, 2.0)
        
    #     qz, qw = math.sin(total_yaw / 2), math.cos(total_yaw / 2)

    #     req = f'name:"X3", position:{{x:{tx}, y:{ty}, z:{tz}}}, orientation:{{z:{qz}, w:{qw}}}'
    #     cmd = ["ign", "service", "-s", "/world/a2rl_track_ign/set_pose", 
    #            "--reqtype", "ignition.msgs.Pose", "--reptype", "ignition.msgs.Boolean", 
    #            "--timeout", "2000", "--req", req]

    #     # Retry service call if it fails (common on first gate)
    #     for i in range(3):
    #         self.get_logger().info(f"--- Section: Teleporting (Attempt {i+1}) ---")
    #         res = subprocess.run(cmd, capture_output=True, text=True)
    #         if res.returncode == 0: break
    #         time.sleep(1.0)

    #     # Verification polling
    #     start = time.time()
    #     while (time.time() - start) < 8.0:
    #         check = subprocess.run(['ign', 'topic', '-e', '-t', '/model/X3/pose', '-n', '1'], capture_output=True, text=True)
    #         if check.returncode == 0 and check.stdout:
    #             m = re.search(r'position\s*\{([\s\S]*?)\}', check.stdout)
    #             if m:
    #                 block = m.group(1)
    #                 cx = float(re.search(r'x:\s*([-eE0-9.+]+)', block).group(1))
    #                 cy = float(re.search(r'y:\s*([-eE0-9.+]+)', block).group(1))
    #                 if math.sqrt((cx-tx)**2 + (cy-ty)**2) < 0.5:
    #                     self.get_logger().info("Verified!")
    #                     return True
    #         time.sleep(0.5)
    #     return False

    # def get_current_pose(self):
    #     res = subprocess.run(['ign', 'topic', '-e', '-t', '/model/X3/pose', '-n', '1'], capture_output=True, text=True)
    #     out = res.stdout
    #     p_b = re.search(r'position\s*\{([\s\S]*?)\}', out).group(1)
    #     o_b = re.search(r'orientation\s*\{([\s\S]*?)\}', out).group(1)
        
    #     pos = {k: float(re.search(fr'{k}:\s*([-eE0-9.+]+)', p_b).group(1)) for k in ['x', 'y', 'z']}
    #     ori = {k: float(re.search(fr'{k}:\s*([-eE0-9.+]+)', o_b).group(1)) for k in ['x', 'y', 'z', 'w']}
    #     return pos, ori
    def teleport_and_verify(self, target_pose):
        """
        Section: High-Variance Randomized Teleport
        Samples a wide 3D space in front of the gate to ensure dataset diversity.
        """
        gx, gy, gz, _, _, gyaw = target_pose
        
        # 1. Randomize Distance: 4.0m (close up) to 8.5m (far back)
        dist = random.uniform(4.0, 8.5)
        
        # 2. Randomize Orbital Angle: +/- 40 degrees off-center
        # This creates those skewed, side-on perspectives.
        angle_offset = random.uniform(-math.radians(40), math.radians(40))
        total_yaw = gyaw + angle_offset
        
        # 3. Randomize Altitude: 0.5m to 3.0m 
        # (X3 camera is pitched down 30 deg, so higher alt sees more ground)
        tz = random.uniform(0.5, 3.0)
        
        # Calculate X, Y based on the random orbit around the gate
        tx = gx - dist * math.cos(total_yaw)
        ty = gy - dist * math.sin(total_yaw)
        
        # 4. Camera Look-at: Face the gate, but add +/- 5 degrees of "yaw jitter"
        # This ensures the gate isn't always perfectly centered in the pixels.
        look_at_yaw = total_yaw + random.uniform(-math.radians(5), math.radians(5))
        
        qz = math.sin(look_at_yaw / 2)
        qw = math.cos(look_at_yaw / 2)

        req_data = f'name:"X3", position:{{x:{tx}, y:{ty}, z:{tz}}}, orientation:{{z:{qz}, w:{qw}}}'
        cmd = ["ign", "service", "-s", "/world/a2rl_track_ign/set_pose",
               "--reqtype", "ignition.msgs.Pose", "--reptype", "ignition.msgs.Boolean",
               "--timeout", "1000", "--req", req_data]

        self.get_logger().info(f"--- Section: Randomized View (Dist: {dist:.2f}m, Ang: {math.degrees(angle_offset):.1f}°) ---")
        
        # Service Call with basic retry
        for i in range(2):
            res = subprocess.run(cmd, capture_output=True)
            if res.returncode == 0: break
            time.sleep(0.5)

        # Verification polling
        start = time.time()
        while (time.time() - start) < 8.0:
            check = subprocess.run(['ign', 'topic', '-e', '-t', '/model/X3/pose', '-n', '1'], capture_output=True, text=True)
            if check.returncode == 0 and check.stdout:
                m = re.search(r'x:\s*([-eE0-9.+]+)[\s\S]*?y:\s*([-eE0-9.+]+)', check.stdout)
                if m and math.sqrt((float(m.group(1))-tx)**2 + (float(m.group(2))-ty)**2) < 0.3:
                    return True
            time.sleep(0.2)
        return False

    def get_current_pose(self, retries=5, delay=0.1):
        """
        Poll /model/X3/pose via ign topic CLI and return position & orientation.
        Parses the protobuf text output instead of JSON.
        """
        import time
        import re

        for _ in range(retries):
            cmd = ["ign", "topic", "-e", "-t", "/model/X3/pose", "-n", "1"]
            result = subprocess.run(cmd, capture_output=True, text=True)
            output = result.stdout.strip()

            if not output:
                time.sleep(delay)
                continue

            # Extract position
            pos_match = re.search(
                r"position\s*{\s*x:\s*([-eE0-9.+]+)\s*y:\s*([-eE0-9.+]+)\s*z:\s*([-eE0-9.+]+)\s*}", 
                output
            )
            # Extract orientation
            ori_match = re.search(
                r"orientation\s*{\s*x:\s*([-eE0-9.+]+)\s*y:\s*([-eE0-9.+]+)\s*z:\s*([-eE0-9.+]+)\s*w:\s*([-eE0-9.+]+)\s*}", 
                output
            )

            if pos_match and ori_match:
                position = {
                    "x": float(pos_match.group(1)),
                    "y": float(pos_match.group(2)),
                    "z": float(pos_match.group(3))
                }
                orientation = {
                    "x": float(ori_match.group(1)),
                    "y": float(ori_match.group(2)),
                    "z": float(ori_match.group(3)),
                    "w": float(ori_match.group(4))
                }
                return position, orientation

            time.sleep(delay)

        raise RuntimeError("Failed to get pose from Ignition CLI after retries")
    
    def run(self):
        for i, gate in enumerate(self.config['world_layout']):
            self.get_logger().info(f"=== Section: Processing Gate {i+1} ===")
            gx, gy, gz, _, _, gyaw = gate['pose']

            # --- 1. Teleport and verify target via CLI ---
            if not self.teleport_and_verify(gate['pose']):
                self.get_logger().error(f"--- Example: Teleport Failed for Gate {i+1} ---")
                continue

            # --- 2. Ensure drone is fully settled at target ---
            target_pos = np.array([gx, gy, gz])
            for _ in range(10):
                pos, quat = self.get_current_pose()
                curr_pos = np.array([pos['x'], pos['y'], pos['z']])
                if np.linalg.norm(curr_pos - target_pos) < 0.02:  # 2cm tolerance
                    break
                time.sleep(0.05)

            # --- 3. Flush old ROS camera frames ---
            self.latest_msg = None
            for _ in range(10):
                rclpy.spin_once(self, timeout_sec=0.01)

            # --- 4. Wait for fresh frame ---
            while rclpy.ok() and self.latest_msg is None:
                rclpy.spin_once(self, timeout_sec=0.1)

            img = self.bridge.imgmsg_to_cv2(self.latest_msg, 'bgr8')

            # --- 5. Immediately grab CLI pose after image ---
            pos, quat = self.get_current_pose()

            # --- 6. Projection math (same as before) ---
            t_wb = np.array([pos['x'], pos['y'], pos['z']])
            R_wb = R.from_quat([quat['x'], quat['y'], quat['z'], quat['w']]).as_matrix()

            t_bc = np.array([-0.0015, 0.00604, 0.0623])
            R_bc = R.from_euler('xyz', [0.0, -0.523599, 0.0]).as_matrix()

            R_wc = R_wb @ R_bc
            t_wc = t_wb + R_wb @ t_bc

            R_gz_to_rdf = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])
            R_cw = R_gz_to_rdf @ R_wc.T
            t_cw = -R_cw @ t_wc

            rvec, _ = cv2.Rodrigues(R_cw)
            tvec = t_cw.reshape(3, 1)

            h, w = img.shape[:2]

            for g in self.config['world_layout']:
                is_target = (g['name'] == gate['name'])
                gtype = self.config['gate_types'][g['type']]
                local_pts = np.array(gtype['local_keypoints'], dtype=np.float32)
                r_gate_mat = R.from_euler('z', g['pose'][5]).as_matrix()
                world_pts = (r_gate_mat @ local_pts.T).T + np.array(g['pose'][:3])

                pts_in_cam = (R_cw @ world_pts.T).T + t_cw.flatten()
                if np.any(pts_in_cam[:, 2] < 0.2):
                    continue

                img_pts, _ = cv2.projectPoints(world_pts, rvec, tvec, self.K, np.zeros(5))
                img_pts = img_pts.reshape(-1, 2)
                pixels = [(int(round(pt[0])), int(round(pt[1]))) for pt in img_pts]

                if is_target:
                    color = (0, 255, 0)
                    for j in range(len(pixels)):
                        cv2.line(img, pixels[j], pixels[(j + 1) % len(pixels)], color, 2)
                        cv2.circle(img, pixels[j], 3, color, -1)
                else:
                    if any(0 <= p[0] < w and 0 <= p[1] < h for p in pixels):
                        color = (0, 0, 255)
                        for j in range(len(pixels)):
                            cv2.line(img, pixels[j], pixels[(j + 1) % len(pixels)], color, 1)

            cv2.imwrite(os.path.join(self.img_dir, f'randgate{i+1}.png'), img)
            self.get_logger().info(f"--- Example: Saved randgate{i+1}.png ---")


def main():
    rclpy.init()
    node = GateVisualTester()
    try:
        node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()