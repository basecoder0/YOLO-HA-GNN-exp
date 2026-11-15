# models/risk_assessment/risk_calculator.py
import numpy as np
import torch


class RiskCalculator:
    def __init__(self):
        # 1. categoryindexmapping (rootaccording R-LiViT dataset)
        self.classes = [
            'car', 'person', 'bicycle', 'motorcycle',
            'bus', 'truck', 'tramway', 'escooter'
        ]

        # 2. riskparameterconfigurationtable (interactpassscenedefinecontrolversion)
        # W_cls: 固haveriskweight (0-1)
        # alpha: velocity敏感coefficient (velocitypairriskofscalelarge倍rate)
        # beta: distancesensitivity (Sigmoid of陡峭programdegree)
        # d_safe: safety/fulldistancethreshold (米)
        self.risk_params = {
            'person': {'W_cls': 1.00, 'alpha': 0.1, 'beta': 2.5, 'd_safe': 10.0},  # lineperson：most脆weak，mosthighbetterfirstlevel
            'escooter': {'W_cls': 0.95, 'alpha': 0.2, 'beta': 2.0, 'd_safe': 12.0},  # also就is e-scooter
            'bicycle': {'W_cls': 0.90, 'alpha': 0.2, 'beta': 2.0, 'd_safe': 12.0},
            'motorcycle': {'W_cls': 0.85, 'alpha': 0.3, 'beta': 1.8, 'd_safe': 15.0},  # velocityfast
            'car': {'W_cls': 0.60, 'alpha': 0.3, 'beta': 1.5, 'd_safe': 15.0},  # oftenrule/conventiontarget
            'truck': {'W_cls': 0.80, 'alpha': 0.2, 'beta': 1.2, 'd_safe': 20.0},  # largetypevehicle：motionamountlarge，needmore远distance
            'bus': {'W_cls': 0.80, 'alpha': 0.2, 'beta': 1.2, 'd_safe': 20.0},
            'tramway': {'W_cls': 0.70, 'alpha': 0.1, 'beta': 1.5, 'd_safe': 25.0},  # 轨迹fixed，butcontrolmotion难
        }

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def calculate_risk(self, cls_id, conf, depth, speed=0.0):
        """
        calculationsingleatargetofriskscore R_final
        :param cls_id: category ID (int)
        :param conf: detectionconfidence (0-1)
        :param depth: LiDAR 测getofdistance (m)
        :param speed: relativevelocity (m/s)
        :return: risk_score, risk_details (dict)
        """
        # getcategoryname
        if cls_id >= len(self.classes):
            cls_name = 'car'  # defaultfallbackexit
        else:
            cls_name = self.classes[int(cls_id)]

        params = self.risk_params.get(cls_name, self.risk_params['car'])

        # 1. basicrisk R_base = W_cls * conf
        r_base = params['W_cls'] * conf

        # 2. motionrisk R_motion = 1 + alpha * v_rel
        # interactpassscenebelow，velocitymore/greaterlargeriskmore/greaterlarge
        r_motion = 1.0 + params['alpha'] * abs(speed)

        # 3. distancerisk R_proximity = Sigmoid(beta * (d_safe - d))
        # when distance < d_safe when，(d_safe - d) > 0，Sigmoid > 0.5 (risk急剧above升)
        dist_diff = params['d_safe'] - depth
        r_proximity = self.sigmoid(params['beta'] * dist_diff)

        # 4. finalrisk
        r_final = r_base * r_motion * r_proximity

        # 5. normalize/cutbreak (optional，preventscoretoohigh)
        # herenotdo硬cutbreak，butat/inoutputwhenoptionalby/withlimit

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
        """rootaccordingscorereturnriskwaitlevelcharactersymbol串"""
        if score > 0.85: return "🔴 EMERGENCY"
        if score > 0.6: return "🟠 HIGH"
        if score > 0.3: return "🟡 MEDIUM"
        return "🟢 LOW"