import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
import cv2
import json
import math
import os
import time
import random
import subprocess
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

        cam_cfg = self.config['camera']
        f_len = cam_cfg['width'] / (2 * np.tan(cam_cfg['fov'] / 2))
        self.K = np.array([
            [f_len, 0, cam_cfg['width'] / 2],
            [0, f_len, cam_cfg['height'] / 2],
            [0, 0, 1]
        ], dtype=np.float32)

        # Output Directories (YOLO Standard)
        self.base_dir = os.path.expanduser('~/mav_sim/dataset')
        self.split_ratio = 0.8
        
        for split in ['train', 'val']:
            os.makedirs(os.path.join(self.base_dir, 'images', split), exist_ok=True)
            os.makedirs(os.path.join(self.base_dir, 'labels', split), exist_ok=True)

        # Use BEST_EFFORT QoS with depth=1 for latest odometry only
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

        # ROS2 Subscriptions
        self.create_subscription(Image, '/X3/camera/image_raw', self.img_cb, 10)
        self.create_subscription(Odometry, '/X3/odom', self.odom_cb, qos_profile)
        
        self.get_logger().info("✓ Dataset generator ready (using CLI teleport)")

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

    def teleport_and_verify(self, target_pose):
        """CLI-based teleport with validation (from test_teleport.py)"""
        gx, gy, gz, _, _, gyaw = target_pose

        # Calculate random viewpoint
        dist = random.uniform(4.0, 8.5)
        angle_offset = random.uniform(-math.radians(40), math.radians(40))
        total_yaw = gyaw + angle_offset
        tz = random.uniform(0.5, 3.0)
        
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
            "ign", "service", "-s", "/world/a2rl_track_ign/set_pose",
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
                self.latest_odom['pos']['y']
            ])
            error = np.linalg.norm(curr_xy - target_xy)
            
            if error < 0.5:  # Within 50cm is good enough
                self.get_logger().info(f"✓ Teleport verified (error: {error:.3f}m)")
                return True, (self.latest_odom['pos'], self.latest_odom['ori'])
        
        self.get_logger().warn("Teleport verification timeout")
        return False, None

    def run(self, total_goal=5000):
        type_to_idx = {name: i for i, name in enumerate(self.config['gate_types'].keys())}
        max_kpts = max(len(t['local_keypoints']) for t in self.config['gate_types'].values())
        
        num_gates = len(self.config['world_layout'])
        num_passes = (total_goal // num_gates) + 1
        count = 0
        failures = 0
        consecutive_failures = 0

        for p in range(num_passes):
            layout = self.config['world_layout'].copy()
            random.shuffle(layout)
            
            for i, gate in enumerate(layout):
                if count >= total_goal: 
                    break

                # Safety: pause if too many consecutive failures
                if consecutive_failures >= 5:
                    self.get_logger().warn("5 consecutive failures - pausing for recovery")
                    time.sleep(3.0)
                    consecutive_failures = 0

                try:
                    self.get_logger().info(f"=== Pass {p+1} | Gate {i+1}/{len(layout)} ===")
                    
                    # 1. Teleport with verification
                    success, pose_data = self.teleport_and_verify(gate['pose'])
                    if not success:
                        failures += 1
                        consecutive_failures += 1
                        self.get_logger().warn(f"Teleport failed (total: {failures}, consecutive: {consecutive_failures})")
                        time.sleep(1.0)
                        continue
                    
                    consecutive_failures = 0
                    pos, quat = pose_data
                    
                    # 2. Capture locked image
                    target_img_msg = self.latest_msg
                    img = self.bridge.imgmsg_to_cv2(target_img_msg, 'bgr8')
                    h, w = img.shape[:2]

                    # 3. Build projection matrix
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

                    # 4. Generate annotations
                    zero_dist = np.zeros(5) 
                    annotation_lines = []

                    for g in self.config['world_layout']:
                        is_target = (g['name'] == gate['name'])
                        gtype_name = g['type']
                        gtype_cfg = self.config['gate_types'][gtype_name]
                        local_pts = np.array(gtype_cfg['local_keypoints'], dtype=np.float32)
                        
                        r_gate = R.from_euler('z', g['pose'][5])
                        world_pts = r_gate.apply(local_pts) + np.array(g['pose'][:3])
                        pts_in_cam = (R_cw @ world_pts.T).T + t_cw.flatten()

                        if np.all(pts_in_cam[:, 2] < 0.1): continue
                        front_pts_idx = pts_in_cam[:, 2] > 0.1
                        if not np.any(front_pts_idx): continue

                        img_pts, _ = cv2.projectPoints(world_pts, rvec, tvec, self.K, zero_dist)
                        img_pts = img_pts.reshape(-1, 2)
                        
                        visible_count = sum(1 for k, pt in enumerate(img_pts) 
                                       if pts_in_cam[k, 2] > 0.1 and (0 <= pt[0] < w and 0 <= pt[1] < h))
                        if visible_count < (1 if is_target else 2): continue

                        front_img_pts = img_pts[front_pts_idx]
                        x_min_c, x_max_c = np.clip([np.min(front_img_pts[:, 0]), np.max(front_img_pts[:, 0])], 0, w)
                        y_min_c, y_max_c = np.clip([np.min(front_img_pts[:, 1]), np.max(front_img_pts[:, 1])], 0, h)
                        
                        bw, bh = (x_max_c - x_min_c) / w, (y_max_c - y_min_c) / h
                        bx = (x_min_c + (x_max_c - x_min_c) / 2.0) / w
                        by = (y_min_c + (y_max_c - y_min_c) / 2.0) / h
                        if bw < 0.005 or bh < 0.005: continue

                        kpt_entries = []
                        for k, pt in enumerate(img_pts):
                            knx, kny = pt[0] / w, pt[1] / h
                            v = 2.0 if (0 <= knx <= 1 and 0 <= kny <= 1 and pts_in_cam[k, 2] > 0.1) else 1.0 if pts_in_cam[k, 2] > 0.1 else 0.0
                            knx_s, kny_s = np.clip([knx, kny], 0.0, 1.0)
                            kpt_entries.append(f"{knx_s:.6f} {kny_s:.6f} {v}")
                        
                        while len(kpt_entries) < max_kpts:
                            kpt_entries.append("0.000000 0.000000 0.0")

                        class_idx = type_to_idx[gtype_name]
                        kpt_str = " " + " ".join(kpt_entries)
                        annotation_lines.append(f"{class_idx} {bx:.6f} {by:.6f} {bw:.6f} {bh:.6f}{kpt_str}")

                    if annotation_lines:
                        split = 'train' if random.random() < self.split_ratio else 'val'
                        img_save_path = os.path.join(self.base_dir, 'images', split)
                        lbl_save_path = os.path.join(self.base_dir, 'labels', split)

                        file_stem = f"frame_{count}_{int(time.time()*1000)}"
                        
                        cv2.imwrite(os.path.join(img_save_path, f"{file_stem}.jpg"), img)
                        with open(os.path.join(lbl_save_path, f"{file_stem}.txt"), 'w') as f:
                            f.write("\n".join(annotation_lines))
                        
                        count += 1
                        if count % 50 == 0:
                            success_rate = (count / (count + failures)) * 100
                            self.get_logger().info(f"✓ Progress: {count}/{total_goal} ({success_rate:.1f}% success)")
            
                except Exception as e:
                    failures += 1
                    consecutive_failures += 1
                    self.get_logger().error(f"Error on gate {i}: {e}")
                    import traceback
                    self.get_logger().error(traceback.format_exc())
                    time.sleep(1.0)
                    continue

        self.get_logger().info(f"=== COMPLETE: {count} images, {failures} failures ===")

def main():
    rclpy.init()
    node = GateVisualTester()
    try:
        node.run(5000)
        # node.run(10)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
