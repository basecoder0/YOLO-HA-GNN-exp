# utils/lidar_utils.py
import numpy as np
import json
import os


class LidarProcessor:
    def __init__(self, calib_file_path):
        # 1. 初始化默认值
        self.T_lidar_to_rgb = np.eye(4)
        self.K_rgb = np.eye(3)

        if os.path.exists(calib_file_path):
            self._load_calibration(calib_file_path)
        else:
            print(f"❌ 错误: 找不到标定文件 {calib_file_path}")

    def _load_calibration(self, json_path):
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)

            # ==========================================
            # 🎯 核心修复：根据你的日志精准提取
            # ==========================================

            # === 1. 解析外参 (LiDAR -> RGB) ===
            # 日志显示: lidar_to_rgb 是字典，里面有 'extrinsic_matrix'
            found_extrinsic = False
            if 'lidar_to_rgb' in data:
                l2c_data = data['lidar_to_rgb']
                # 检查是否为字典且包含 extrinsic_matrix
                if isinstance(l2c_data, dict) and 'extrinsic_matrix' in l2c_data:
                    self.T_lidar_to_rgb = np.array(l2c_data['extrinsic_matrix'])
                    found_extrinsic = True
                # 兼容旧逻辑：如果直接是列表
                elif isinstance(l2c_data, list):
                    self.T_lidar_to_rgb = np.array(l2c_data)
                    found_extrinsic = True

            if found_extrinsic:
                print("✅ 外参矩阵 (Extrinsic) 加载成功")
            else:
                print("⚠️ 警告: 未能解析 'lidar_to_rgb'，将使用单位矩阵！")

            # === 2. 解析内参 (Camera Intrinsic) ===
            # 日志显示: rgb_camera 里面有 'intrinsic_matrix'
            found_intrinsic = False
            if 'rgb_camera' in data:
                cam_data = data['rgb_camera']
                if 'intrinsic_matrix' in cam_data:
                    self.K_rgb = np.array(cam_data['intrinsic_matrix'])
                    found_intrinsic = True

            if found_intrinsic:
                print("✅ 内参矩阵 (Intrinsic) 加载成功")
            else:
                print("⚠️ 警告: 未能解析 'intrinsic_matrix'，投影将失效！")

        except Exception as e:
            print(f"❌ 解析标定文件崩溃: {e}")

    def project_lidar_to_image(self, points_3d, img_size=(1280, 720)):
        if len(points_3d) == 0: return np.zeros((0, 2)), [], []

        n = points_3d.shape[0]
        pts_hom = np.hstack((points_3d[:, :3], np.ones((n, 1))))

        # 矩阵乘法
        try:
            pts_cam = (self.T_lidar_to_rgb @ pts_hom.T).T
        except Exception:
            return np.zeros((0, 2)), [], []

        depth = pts_cam[:, 2]
        mask_valid = depth > 0.1  # 只保留前方

        pts_cam = pts_cam[mask_valid]
        depth = depth[mask_valid]

        if len(pts_cam) == 0: return np.zeros((0, 2)), [], []

        # 归一化平面
        x_norm = pts_cam[:, 0] / depth
        y_norm = pts_cam[:, 1] / depth
        pts_2d_hom = np.stack([x_norm, y_norm, np.ones_like(x_norm)], axis=1)

        # 投影到像素
        pts_img = (self.K_rgb @ pts_2d_hom.T).T

        u, v = pts_img[:, 0], pts_img[:, 1]

        h, w = img_size
        mask_in_img = (u >= 0) & (u < w) & (v >= 0) & (v < h)

        return np.stack([u[mask_in_img], v[mask_in_img]], axis=1), depth[mask_in_img], mask_in_img

    def load_bin_file(self, file_path):
        try:
            scan = np.fromfile(file_path, dtype=np.float32)
            scan = scan.reshape((-1, 4))
            return scan
        except Exception:
            return np.zeros((0, 4))

    def get_object_depth_and_speed(self, points_3d, bbox_2d, previous_center=None, time_delta=0.1):
        pts_2d, depth_vals, _ = self.project_lidar_to_image(points_3d)

        # 如果投影失败或没有点
        if len(pts_2d) == 0: return 999.0, 0.0, None

        x1, y1, x2, y2 = bbox_2d
        in_box_mask = (pts_2d[:, 0] >= x1) & (pts_2d[:, 0] <= x2) & \
                      (pts_2d[:, 1] >= y1) & (pts_2d[:, 1] <= y2)

        target_depths = depth_vals[in_box_mask]
        if len(target_depths) == 0: return 999.0, 0.0, None

        # 计算深度 (取前20%均值)
        sorted_d = np.sort(target_depths)
        est_depth = np.mean(sorted_d[:max(1, int(len(sorted_d) * 0.2))])

        current_center = np.array([0, 0, est_depth])
        speed = 0.0
        if previous_center is not None:
            dist = np.linalg.norm(current_center - previous_center)
            speed = dist / time_delta

        return est_depth, speed, current_center