# models/risk_assessment/risk_calculator.py
import numpy as np
import torch


class RiskCalculator:
    def __init__(self):
        # 1. 类别索引映射 (根据 R-LiViT 数据集)
        self.classes = [
            'car', 'person', 'bicycle', 'motorcycle',
            'bus', 'truck', 'tramway', 'escooter'
        ]

        # 2. 风险参数配置表 (交通场景定制版)
        # W_cls: 固有风险权重 (0-1)
        # alpha: 速度敏感系数 (速度对风险的放大倍率)
        # beta: 距离敏感度 (Sigmoid 的陡峭程度)
        # d_safe: 安全距离阈值 (米)
        self.risk_params = {
            'person': {'W_cls': 1.00, 'alpha': 0.1, 'beta': 2.5, 'd_safe': 10.0},  # 行人：最脆弱，最高优先级
            'escooter': {'W_cls': 0.95, 'alpha': 0.2, 'beta': 2.0, 'd_safe': 12.0},  # 也就是 e-scooter
            'bicycle': {'W_cls': 0.90, 'alpha': 0.2, 'beta': 2.0, 'd_safe': 12.0},
            'motorcycle': {'W_cls': 0.85, 'alpha': 0.3, 'beta': 1.8, 'd_safe': 15.0},  # 速度快
            'car': {'W_cls': 0.60, 'alpha': 0.3, 'beta': 1.5, 'd_safe': 15.0},  # 常规目标
            'truck': {'W_cls': 0.80, 'alpha': 0.2, 'beta': 1.2, 'd_safe': 20.0},  # 大型车：动量大，需更远距离
            'bus': {'W_cls': 0.80, 'alpha': 0.2, 'beta': 1.2, 'd_safe': 20.0},
            'tramway': {'W_cls': 0.70, 'alpha': 0.1, 'beta': 1.5, 'd_safe': 25.0},  # 轨迹固定，但制动难
        }

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def calculate_risk(self, cls_id, conf, depth, speed=0.0):
        """
        计算单个目标的风险分数 R_final
        :param cls_id: 类别 ID (int)
        :param conf: 检测置信度 (0-1)
        :param depth: LiDAR 测得的距离 (m)
        :param speed: 相对速度 (m/s)
        :return: risk_score, risk_details (dict)
        """
        # 获取类别名
        if cls_id >= len(self.classes):
            cls_name = 'car'  # 默认回退
        else:
            cls_name = self.classes[int(cls_id)]

        params = self.risk_params.get(cls_name, self.risk_params['car'])

        # 1. 基础风险 R_base = W_cls * conf
        r_base = params['W_cls'] * conf

        # 2. 运动风险 R_motion = 1 + alpha * v_rel
        # 交通场景下，速度越大风险越大
        r_motion = 1.0 + params['alpha'] * abs(speed)

        # 3. 距离风险 R_proximity = Sigmoid(beta * (d_safe - d))
        # 当 distance < d_safe 时，(d_safe - d) > 0，Sigmoid > 0.5 (风险急剧上升)
        dist_diff = params['d_safe'] - depth
        r_proximity = self.sigmoid(params['beta'] * dist_diff)

        # 4. 最终风险
        r_final = r_base * r_motion * r_proximity

        # 5. 归一化/截断 (可选，防止分数过高)
        # 这里不做硬截断，但在输出时可以限制

        details = {
            'class': cls_name,
            'depth': round(depth, 2),
            'speed': round(speed, 2),
            'components': {
                'r_base': round(r_base, 3),
                'r_motion': round(r_motion, 3),
                'r_proximity': round(r_proximity, 3)
            }
        }

        return r_final, details

    def get_risk_level(self, score):
        """根据分数返回风险等级字符串"""
        if score > 0.85: return "🔴 EMERGENCY"
        if score > 0.6: return "🟠 HIGH"
        if score > 0.3: return "🟡 MEDIUM"
        return "🟢 LOW"