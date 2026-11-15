# utils/lidar_utils.py
import numpy as np
import json
import os


class LidarProcessor:
    def __init__(self, calib_file_path):
        # 1. initializationdefaultvalue
        self.T_lidar_to_rgb = np.eye(4)
        self.K_rgb = np.eye(3)

        if os.path.exists(calib_file_path):
            self._load_calibration(calib_file_path)
        else:
            print(f"❌ error: findnottotargetdefinefile {calib_file_path}")

    def _load_calibration(self, json_path):
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)

            # ==========================================
            # 🎯 core fix：rootaccordingyouoflog精准extract
            # ==========================================

            # === 1. parseextraparameter (LiDAR -> RGB) ===
            # logshow: lidar_to_rgb isdictionary，infacehave 'extrinsic_matrix'
            found_extrinsic = False
            if 'lidar_to_rgb' in data:
                l2c_data = data['lidar_to_rgb']
                # checkwhetherfor/asdictionaryandcontains extrinsic_matrix
                if isinstance(l2c_data, dict) and 'extrinsic_matrix' in l2c_data:
                    self.T_lidar_to_rgb = np.array(l2c_data['extrinsic_matrix'])
                    found_extrinsic = True
                # compatibleoldlogic：ifdirectlyislist
                elif isinstance(l2c_data, list):
                    self.T_lidar_to_rgb = np.array(l2c_data)
                    found_extrinsic = True

            if found_extrinsic:
                print("✅ extraparametermatrix (Extrinsic) loadingsuccess")
            else:
                print("⚠️ warning: notability/canparse 'lidar_to_rgb'，willusingsingledigitmatrix！")

            # === 2. parseinsideparameter (Camera Intrinsic) ===
            # logshow: rgb_camera infacehave 'intrinsic_matrix'
            found_intrinsic = False
            if 'rgb_camera' in data:
                cam_data = data['rgb_camera']
                if 'intrinsic_matrix' in cam_data:
                    self.K_rgb = np.array(cam_data['intrinsic_matrix'])
                    found_intrinsic = True

            if found_intrinsic:
                print("✅ insideparametermatrix (Intrinsic) loadingsuccess")
            else:
                print("⚠️ warning: notability/canparse 'intrinsic_matrix'，projectionwillloseeffect！")

        except Exception as e:
            print(f"❌ parsetargetdefinefile崩溃: {e}")

    def project_lidar_to_image(self, points_3d, img_size=(1280, 720)):
        if len(points_3d) == 0: return np.zeros((0, 2)), [], []

        n = points_3d.shape[0]
        pts_hom = np.hstack((points_3d[:, :3], np.ones((n, 1))))

        # matrix乘method
        try:
            pts_cam = (self.T_lidar_to_rgb @ pts_hom.T).T
        except Exception:
            return np.zeros((0, 2)), [], []

        depth = pts_cam[:, 2]
        mask_valid = depth > 0.1  # onlykeepkeepfirstmethod

        pts_cam = pts_cam[mask_valid]
        depth = depth[mask_valid]

        if len(pts_cam) == 0: return np.zeros((0, 2)), [], []

        # normalizeflatface
        x_norm = pts_cam[:, 0] / depth
        y_norm = pts_cam[:, 1] / depth
        pts_2d_hom = np.stack([x_norm, y_norm, np.ones_like(x_norm)], axis=1)

        # projectiontopixel
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

        # ifprojectionfailedornothavepoint
        if len(pts_2d) == 0: return 999.0, 0.0, None

        x1, y1, x2, y2 = bbox_2d
        in_box_mask = (pts_2d[:, 0] >= x1) & (pts_2d[:, 0] <= x2) & \
                      (pts_2d[:, 1] >= y1) & (pts_2d[:, 1] <= y2)

        target_depths = depth_vals[in_box_mask]
        if len(target_depths) == 0: return 999.0, 0.0, None

        # calculationdepth (getfirst20%mean)
        sorted_d = np.sort(target_depths)
        est_depth = np.mean(sorted_d[:max(1, int(len(sorted_d) * 0.2))])

        current_center = np.array([0, 0, est_depth])
        speed = 0.0
        if previous_center is not None:
            dist = np.linalg.norm(current_center - previous_center)
            speed = dist / time_delta

        return est_depth, speed, current_center