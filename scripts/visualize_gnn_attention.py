import torch
import cv2
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from models.gnn.hierarchical_gnn import HierarchicalGNN
from torch_geometric.nn import GATConv


# ==========================================
# 1. define义oneawith Hook of GNN packinstalldevice
# ==========================================
class VisualizableGNN(HierarchicalGNN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attn_weights = {}  # existstorageeachonelayerofnoteability

    def forward_with_attention(self, node_features, bboxes, scores):
        # this is a simplified version forward，special门forextract attention
        # ouronlyprocessing batch size = 1 ofsituationsituationperformvisualization

        # 1. predict/preprocessing (Filter & Build Graph)
        feat, box, keep_idx = self._filter_nodes(node_features[0], bboxes[0], scores[0])
        if feat.size(0) < 2: return None, None, None  # nodetooleast，notmethoddraw image

        edge_index = self._build_graph_for_sample(feat, box)

        # 2. encoding
        x = self.node_encoder(feat)

        # 3. extract Appearance Attention
        # temporaryreplacechange GATConv of forward linefor/as
        x_app, (edge_index_app, alpha_app) = self.appearance_attention.attention(x, edge_index,
                                                                                 return_attention_weights=True)
        self.attn_weights['appearance'] = (edge_index_app, alpha_app)

        # 4. extract Spatial Attention
        sp_weight = self.compute_spatial_weights(box, device=x.device)
        # Spatial Layer havepoint特殊，itat/in forward indopast tense marker乘method，ourheremanualsimulateonebelow
        x_spa_raw, (edge_index_spa, alpha_spa) = self.spatial_attention.attention(x, edge_index,
                                                                                  return_attention_weights=True)
        self.attn_weights['spatial'] = (edge_index_spa, alpha_spa)

        return box, self.attn_weights, keep_idx


# ==========================================
# 2. visualizationdrawimagefunction
# ==========================================
def draw_attention_map(img, boxes, edge_index, attn_scores, title, save_path):
    """
    at/inimagepieceabovedrawoutnodeofspaceof连line，lineof粗fine/color深浅representnoteabilityweight
    """
    vis_img = img.copy()
    h, w = vis_img.shape[:2]

    # normalizenoteabilityscoreby/withconvenientdrawimage
    attn_scores = attn_scores.mean(dim=1).detach().cpu().numpy()  # Average over heads
    attn_scores = (attn_scores - attn_scores.min()) / (attn_scores.max() - attn_scores.min() + 1e-6)

    edge_index = edge_index.cpu().numpy()

    # draw bounding box
    centers = []
    for box in boxes:
        # box is [x, y, w, h] (normalized) or [x1, y1, x2, y2]
        # assumeisnormalize [cx, cy, w, h] -> restoretopixel
        cx, cy, bw, bh = box
        px, py = int(cx * 1280), int(cy * 1280)  # assume 1280 size
        centers.append((px, py))

        x1, y1 = int((cx - bw / 2) * 1280), int((cy - bh / 2) * 1280)
        x2, y2 = int((cx + bw / 2) * 1280), int((cy + bh / 2) * 1280)
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # drawline
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[0, i], edge_index[1, i]
        score = attn_scores[i]

        if score < 0.2: continue  # ignoreweak连connect

        pt1 = centers[src]
        pt2 = centers[dst]

        # color：red represents strong correlation，蓝colorrepresentweakcorrelation
        color_val = int(255 * score)
        color = (255 - color_val, 0, color_val)  # BGR
        thickness = max(1, int(score * 3))

        cv2.line(vis_img, pt1, pt2, color, thickness)

    cv2.putText(vis_img, title, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imwrite(save_path, vis_img)
    print(f"🎨 Saved attention map: {save_path}")


# ==========================================
# 3. major流program
# ==========================================
def run_visualization(ckpt_path, sample_pt_path, img_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # loadmodel (usingourof Wrapper)
    model = VisualizableGNN(node_feature_dim=256, hidden_dim=256).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    if 'model' in ckpt:
        model.load_state_dict(ckpt['model'], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)
    model.eval()

    # loaddata
    data = torch.load(sample_pt_path, map_location=device)
    feat = [data['features'].float()]
    boxes = [data['boxes'].float()]
    scores = [data['scores'][:, 0]]

    # extractnoteability
    boxes_filtered, attn_weights, _ = model.forward_with_attention(feat, boxes, scores)

    if boxes_filtered is None:
        print("❌ nodecountamountnot足，unable tovisualization")
        return

    # loadoriginal image
    img = cv2.imread(img_path)
    img = cv2.resize(img, (1280, 1280))  # make suresizematch

    # draw Appearance Attention
    edge_idx, alpha = attn_weights['appearance']
    draw_attention_map(img, boxes_filtered, edge_idx, alpha, "Appearance Attention", "vis_attn_app.jpg")

    # draw Spatial Attention
    edge_idx, alpha = attn_weights['spatial']
    draw_attention_map(img, boxes_filtered, edge_idx, alpha, "Spatial Attention", "vis_attn_spa.jpg")


if __name__ == "__main__":
    # fillactualpath
    ckpt = "runs/stage2_gnn_1280/epoch_100.pth"
    pt_file = "stage2_gnn_dataset_1280/processed/sample_0.pt"  # needfinda specific sample
    img_file = "raw_data/images/sample_0.png"

    if os.path.exists(ckpt) and os.path.exists(pt_file):
        run_visualization(ckpt, pt_file, img_file)
    else:
        print("pleasecheckfilepathconfiguration")