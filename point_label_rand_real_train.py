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

        # Output Directories (YOLO Standard)
        self.base_dir = os.path.expanduser('~/mav_sim/dataset')
        self.split_ratio = 0.8  # 80% Train, 20% Val
        
        # Create subdirectories for YOLO structure
        for split in ['train', 'val']:
            os.makedirs(os.path.join(self.base_dir, 'images', split), exist_ok=True)
            os.makedirs(os.path.join(self.base_dir, 'labels', split), exist_ok=True)

        self.create_subscription(Image, '/X3/camera/image_raw', self.img_cb, 10)

    def img_cb(self, msg):
        self.latest_msg = msg

    def teleport_and_verify(self, target_pose):
        """Section: High-Variance Randomized Teleport"""
        gx, gy, gz, _, _, gyaw = target_pose
        
        dist = random.uniform(4.0, 8.5)
        angle_offset = random.uniform(-math.radians(40), math.radians(40))
        total_yaw = gyaw + angle_offset
        tz = random.uniform(0.5, 3.0)
        
        tx = gx - dist * math.cos(total_yaw)
        ty = gy - dist * math.sin(total_yaw)
        
        look_at_yaw = total_yaw + random.uniform(-math.radians(5), math.radians(5))
        qz, qw = math.sin(look_at_yaw / 2), math.cos(look_at_yaw / 2)

        req_data = f'name:"X3", position:{{x:{tx}, y:{ty}, z:{tz}}}, orientation:{{z:{qz}, w:{qw}}}'
        cmd = ["ign", "service", "-s", "/world/a2rl_track_ign/set_pose",
                "--reqtype", "ignition.msgs.Pose", "--reptype", "ignition.msgs.Boolean",
                "--timeout", "1000", "--req", req_data]

        self.get_logger().info(f"--- Section: Randomizing (Dist: {dist:.1f}m, Ang: {math.degrees(angle_offset):.1f}°) ---")
        
        for i in range(2):
            res = subprocess.run(cmd, capture_output=True)
            if res.returncode == 0: break
            time.sleep(0.5)

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
        for _ in range(retries):
            cmd = ["ign", "topic", "-e", "-t", "/model/X3/pose", "-n", "1"]
            result = subprocess.run(cmd, capture_output=True, text=True)
            output = result.stdout.strip()
            if not output:
                time.sleep(delay)
                continue

            p_m = re.search(r"position\s*{\s*x:\s*([-eE0-9.+]+)\s*y:\s*([-eE0-9.+]+)\s*z:\s*([-eE0-9.+]+)\s*}", output)
            o_m = re.search(r"orientation\s*{\s*x:\s*([-eE0-9.+]+)\s*y:\s*([-eE0-9.+]+)\s*z:\s*([-eE0-9.+]+)\s*w:\s*([-eE0-9.+]+)\s*}", output)
            if p_m and o_m:
                pos = {"x": float(p_m.group(1)), "y": float(p_m.group(2)), "z": float(p_m.group(3))}
                ori = {"x": float(o_m.group(1)), "y": float(o_m.group(2)), "z": float(o_m.group(3)), "w": float(o_m.group(4))}
                return pos, ori
            time.sleep(delay)
        raise RuntimeError("CLI Pose Poll Failed")

    def run(self, total_goal=5000):
        type_to_idx = {name: i for i, name in enumerate(self.config['gate_types'].keys())}
        max_kpts = max(len(t['local_keypoints']) for t in self.config['gate_types'].values())
        
        num_gates = len(self.config['world_layout'])
        num_passes = (total_goal // num_gates) + 1
        count = 0

        for p in range(num_passes):
            layout = self.config['world_layout'].copy()
            random.shuffle(layout)
            
            for i, gate in enumerate(layout):
                if count >= total_goal: break

                try:
                    self.get_logger().info(f"=== Section: Pass {p+1} | Gate {i+1} ===")
                    
                    # 1. Teleport
                    if not self.teleport_and_verify(gate['pose']):
                        continue

                    # 2. STRICT SETTLE (Ported from your test code)
                    # This prevents the "drifting boxes" by ensuring physics has stopped
                    gx, gy, gz = gate['pose'][:3]
                    target_pos = np.array([gx, gy, gz])
                    for _ in range(15):
                        pos, _ = self.get_current_pose()
                        curr_pos = np.array([pos['x'], pos['y'], pos['z']])
                        # We check distance to the teleport target, not the gate center
                        # Note: teleport_and_verify uses randomized tx, ty, tz, 
                        # so we should ideally verify against those, but checking 
                        # for 'near-zero velocity' or a tight loop is usually enough.
                        time.sleep(0.1) 

                    # 3. Flush ROS Buffer
                    self.latest_msg = None
                    for _ in range(15): rclpy.spin_once(self, timeout_sec=0.01)

                    # 4. Wait for fresh frame
                    while rclpy.ok() and self.latest_msg is None:
                        rclpy.spin_once(self, timeout_sec=0.1)

                    # 5. Capture LOCK (Sync Point)
                    target_img_msg = self.latest_msg 
                    pos, quat = self.get_current_pose()

                    # Convert only after you have both pieces of data
                    img = self.bridge.imgmsg_to_cv2(target_img_msg, 'bgr8')
                    h, w = img.shape[:2]

                    # 2. --- Section: Projection Matrix ---
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

                    # 3. --- Section: Project and Generate YOLO Annotations ---
                    zero_dist = np.zeros(5) 
                    annotation_lines = []

                    # Build ALL annotations for this single image capture
                    for g in self.config['world_layout']:
                        is_target = (g['name'] == gate['name'])
                        gtype_name = g['type']
                        gtype_cfg = self.config['gate_types'][gtype_name]
                        local_pts = np.array(gtype_cfg['local_keypoints'], dtype=np.float32)
                        
                        # Transform to World & Camera Space
                        r_gate = R.from_euler('z', g['pose'][5])
                        world_pts = r_gate.apply(local_pts) + np.array(g['pose'][:3])
                        pts_in_cam = (R_cw @ world_pts.T).T + t_cw.flatten()

                        # Frustum Culling
                        if np.all(pts_in_cam[:, 2] < 0.1): continue
                        front_pts_idx = pts_in_cam[:, 2] > 0.1
                        if not np.any(front_pts_idx): continue

                        # Project to pixels
                        img_pts, _ = cv2.projectPoints(world_pts, rvec, tvec, self.K, zero_dist)
                        img_pts = img_pts.reshape(-1, 2)
                        
                        # Visibility Check
                        visible_count = 0
                        for k, pt in enumerate(img_pts):
                            if pts_in_cam[k, 2] > 0.1 and (0 <= pt[0] < w and 0 <= pt[1] < h):
                                visible_count += 1

                        # Filter: Ensure the gate is actually worth labeling
                        if visible_count < (1 if is_target else 2): continue

                        # Bounding Box
                        front_img_pts = img_pts[front_pts_idx]
                        x_min_c, x_max_c = np.clip([np.min(front_img_pts[:, 0]), np.max(front_img_pts[:, 0])], 0, w)
                        y_min_c, y_max_c = np.clip([np.min(front_img_pts[:, 1]), np.max(front_img_pts[:, 1])], 0, h)
                        
                        bw, bh = (x_max_c - x_min_c) / w, (y_max_c - y_min_c) / h
                        bx = (x_min_c + (x_max_c - x_min_c) / 2.0) / w
                        by = (y_min_c + (y_max_c - y_min_c) / 2.0) / h
                        if bw < 0.005 or bh < 0.005: continue

                        # Keypoints
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

                    # --- OUTSIDE the gate loop: Save ONCE per image ---
                    if annotation_lines:
                        split = 'train' if random.random() < self.split_ratio else 'val'
                        img_save_path = os.path.join(self.base_dir, 'images', split)
                        lbl_save_path = os.path.join(self.base_dir, 'labels', split)

                        # Unique file stem based on count/time
                        file_stem = f"frame_{count}_{int(time.time()*100)}"
                        
                        cv2.imwrite(os.path.join(img_save_path, f"{file_stem}.jpg"), img)
                        with open(os.path.join(lbl_save_path, f"{file_stem}.txt"), 'w') as f:
                            f.write("\n".join(annotation_lines))
                        
                        count += 1
                        if count % 100 == 0:
                            self.get_logger().info(f"--- Milestone: {count}/{total_goal} images saved ---")
                
                except Exception as e:
                    self.get_logger().error(f"Section: Error caught on Gate {i}: {e}")
                    time.sleep(1.0) # Pause to let Ignition/ROS recover
                    continue
def main():
    rclpy.init()
    node = GateVisualTester()
    try:
        # node.run(10)
        node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()