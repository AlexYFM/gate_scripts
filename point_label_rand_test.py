import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
import cv2
import json
import math
import os
import subprocess
import time
import random
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from scipy.spatial.transform import Rotation as R

class GateVisualTester(Node):
    def __init__(self):
        super().__init__('gate_visual_tester')
        self.bridge = CvBridge()
        self.latest_msg = None
        self.latest_odom = None

        # --- Section: Load Config ---
        with open('world_config.json', 'r') as f:
            self.config = json.load(f)

        cam_cfg: dict = self.config['camera']
        hfov = cam_cfg.get('horizontal_fov', cam_cfg.get('fov'))
        fx = cam_cfg['width'] / (2 * np.tan(hfov / 2))
        fy = fx

        # self.K = np.array([
        #     [fx, 0, cam_cfg['width'] / 2],
        #     [0, fy, cam_cfg['height'] / 2],
        #     [0, 0, 1]
        # ], dtype=np.float32)
        self.K = np.array([
            [834.0615, 0, 640],
            [0, 834.0616, 360],
            [0, 0, 1]
        ], dtype=np.float32)

        self.img_dir = os.path.expanduser('~/mav_sim/test_projections')
        os.makedirs(self.img_dir, exist_ok=True)

        # Use BEST_EFFORT QoS with depth=1 for latest odometry only
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.create_subscription(Image, '/X3/camera/image_raw', self.img_cb, 10)
        self.create_subscription(Odometry, '/X3/odom', self.odom_cb, qos_profile)

    def img_cb(self, msg):
        self.latest_msg = msg

    def odom_cb(self, msg):
        """Store latest odometry data with timestamp"""
        self.latest_odom = {
            'pos': {
                'x': msg.pose.pose.position.x,
                'y': msg.pose.pose.position.y,
                'z': msg.pose.pose.position.z
            },
            'ori': {
                'x': msg.pose.pose.orientation.x,
                'y': msg.pose.pose.orientation.y,
                'z': msg.pose.pose.orientation.z,
                'w': msg.pose.pose.orientation.w
            },
            'receive_time': time.time()
        }

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
    #     cmd = ["ign", "service", "-s", "/world/x3_illini_warehouse/set_pose", 
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
    def teleport_and_verify(self, target_pose, gate_type: str):
        """CLI-based teleport with validation (aligned with training script)"""
        gx, gy, gz, _, _, gyaw = target_pose

        # Calculate random viewpoint
        dist = random.uniform(4.0, 8.5)
        # dist = random.uniform(5.5, 12)
        angle_offset = random.uniform(-math.radians(40), math.radians(40))
        total_yaw = gyaw + angle_offset

        if gate_type == "single_gate":
            tz = random.uniform(0.1, 1)
        else:
            tz = random.uniform(0.1, 2)
        # tz = random.uniform(0.5, 3.0)
        # tz = random.uniform(0.25, 1.25)

        tx = gx - dist * math.cos(total_yaw)
        ty = gy - dist * math.sin(total_yaw)

        look_at_yaw = total_yaw + random.uniform(-math.radians(5), math.radians(5))
        qz = math.sin(look_at_yaw / 2)
        qw = math.cos(look_at_yaw / 2)

        # CLEAR OLD DATA BEFORE TELEPORT
        self.latest_odom = None
        self.latest_msg = None

        # Build CLI command
        req_data = f'name:"X3", position:{{x:{tx}, y:{ty}, z:{tz}}}, orientation:{{z:{qz}, w:{qw}}}'
        cmd = [
            "ign", "service", "-s", "/world/x3_illini_warehouse/set_pose",
            "--reqtype", "ignition.msgs.Pose",
            "--reptype", "ignition.msgs.Boolean",
            "--timeout", "2000",
            "--req", req_data
        ]

        self.get_logger().info(f"Teleporting to ({tx:.2f}, {ty:.2f}, {tz:.2f})")

        # Execute CLI teleport
        res = subprocess.run(cmd, capture_output=True, text=True)

        if res.returncode != 0:
            self.get_logger().error(f"CLI teleport failed: {res.stderr}")
            return False, None

        # Wait for teleport to complete
        time.sleep(0.5)

        # Flush old messages aggressively
        for _ in range(50):
            rclpy.spin_once(self, timeout_sec=0.02)

        # VERIFICATION: Wait for fresh synchronized data
        target_xy = np.array([tx, ty])
        timeout = time.time() + 3.0

        while time.time() < timeout:
            rclpy.spin_once(self, timeout_sec=0.05)

            # Need both image and odom
            if self.latest_msg is None or self.latest_odom is None:
                continue

            # Check data freshness
            data_age = time.time() - self.latest_odom['receive_time']
            if data_age > 0.5:
                continue

            # Verify position
            curr_xy = np.array([
                self.latest_odom['pos']['x'],
                self.latest_odom['pos']['y'],
            ])
            error = np.linalg.norm(curr_xy - target_xy)

            if error < 0.5:  # Within 50cm is good enough
                self.get_logger().info(f"✓ Teleport verified (error: {error:.3f}m)")
                return True, (self.latest_odom['pos'], self.latest_odom['ori'])

        self.get_logger().warn("Teleport verification timeout")
        return False, None
    
    def run(self):
        for i, gate in enumerate(self.config['world_layout']):
            self.get_logger().info(f"=== Section: Processing Gate {i+1} ===")
            gx, gy, gz, _, _, gyaw = gate['pose']

            # --- 1. Teleport and verify target via CLI ---
            success, pose_data = self.teleport_and_verify(gate['pose'], gate['type'])
            if not success:
                self.get_logger().error(f"--- Example: Teleport Failed for Gate {i+1} ---")
                continue
            pos, quat = pose_data

            # --- 3. Flush old ROS camera frames ---
            self.latest_msg = None
            for _ in range(10):
                rclpy.spin_once(self, timeout_sec=0.01)

            # --- 4. Wait for fresh frame ---
            while rclpy.ok() and self.latest_msg is None:
                rclpy.spin_once(self, timeout_sec=0.1)

            target_img_msg = self.latest_msg
            img = self.bridge.imgmsg_to_cv2(target_img_msg, 'bgr8')

            # --- 6. Projection math (same as before) ---
            t_wb = np.array([pos['x'], pos['y'], pos['z']])
            R_wb = R.from_quat([quat['x'], quat['y'], quat['z'], quat['w']]).as_matrix()

            t_bc = np.array([-0.0015, 0.00604, 0.0623])
            # t_bc = np.array([
            #     -0.0015 ,       # Cam X - Base X
            #     0.00604,      # Cam Y - Base Y
            #     0.0623 - 0.053302   # Cam Z - Base Z (The culprit!)
            # ])
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
                
                distortion = np.zeros(5)
                # distortion = np.array([-0.25390268, 0.06240127, -0.00113674, 0.00213326, 0.0], dtype=np.float32)
                img_pts, _ = cv2.projectPoints(world_pts, rvec, tvec, self.K, distortion)
                img_pts = img_pts.reshape(-1, 2)
                pixels = [(int(round(pt[0])), int(round(pt[1]))) for pt in img_pts]

                # if is_target:
                #     color = (0, 255, 0)
                #     for j in range(len(pixels)):
                #         cv2.line(img, pixels[j], pixels[(j + 1) % len(pixels)], color, 2)
                #         cv2.circle(img, pixels[j], 3, color, -1)
                # else:
                #     if any(0 <= p[0] < w and 0 <= p[1] < h for p in pixels):
                #         color = (0, 0, 255)
                #         for j in range(len(pixels)):
                #             cv2.line(img, pixels[j], pixels[(j + 1) % len(pixels)], color, 1)
                if is_target:
                    color = (0, 255, 0)
                    thickness = 2
                    # Get loops from config: e.g., [[0,1,2,3], [4,5,6,7]]
                    loops = gtype.get('loops', [range(len(pixels))]) 
                    
                    for loop in loops:
                        # Convert loop indices to a list of pixel tuples
                        loop_pts = [pixels[idx] for idx in loop]
                        
                        for j in range(len(loop_pts)):
                            pt1 = loop_pts[j]
                            pt2 = loop_pts[(j + 1) % len(loop_pts)] # Closes the loop
                            cv2.line(img, pt1, pt2, color, thickness)
                            cv2.circle(img, pt1, 3, color, -1)
                else:
                    # Non-target gates (red)
                    if any(0 <= p[0] < w and 0 <= p[1] < h for p in pixels):
                        color = (0, 0, 255)
                        loops = gtype.get('loops', [range(len(pixels))])
                        for loop in loops:
                            loop_pts = [pixels[idx] for idx in loop]
                            for j in range(len(loop_pts)):
                                cv2.line(img, loop_pts[j], loop_pts[(j + 1) % len(loop_pts)], color, 1)

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