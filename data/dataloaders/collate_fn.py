import torch


def fusion_collate_fn(batch):
    """
    自定义 Collate Function
    将 list of dicts 转换为 dict of tensors
    并正确处理 labels 的 batch_idx
    """
    rgb_list = []
    thermal_list = []
    labels_list = []

    for i, item in enumerate(batch):
        rgb_list.append(item['rgb'])
        thermal_list.append(item['thermal'])

        # 更新 labels 中的 batch_idx
        lbl = item['labels']
        if lbl.shape[0] > 0:
            lbl[:, 0] = i  # 设置当前 batch index
        labels_list.append(lbl)

    return {
        'rgb': torch.stack(rgb_list, 0),
        'thermal': torch.stack(thermal_list, 0),
        'bboxes': torch.cat(labels_list, 0)[:, 2:] if len(labels_list) > 0 else torch.zeros(0, 4),  # xywh
        'cls': torch.cat(labels_list, 0)[:, 1] if len(labels_list) > 0 else torch.zeros(0),  # class
        'batch_idx': torch.cat(labels_list, 0)[:, 0] if len(labels_list) > 0 else torch.zeros(0)
    }