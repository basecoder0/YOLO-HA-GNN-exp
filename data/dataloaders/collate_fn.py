import torch


def fusion_collate_fn(batch):
    """
    custom Collate Function
    will list of dicts convertfor/as dict of tensors
    andpositivesureprocessing labels of batch_idx
    """
    rgb_list = []
    thermal_list = []
    labels_list = []

    for i, item in enumerate(batch):
        rgb_list.append(item['rgb'])
        thermal_list.append(item['thermal'])

        # update labels inof batch_idx
        lbl = item['labels']
        if lbl.shape[0] > 0:
            lbl[:, 0] = i  # setwhenfirst batch index
        labels_list.append(lbl)

    return {
        'rgb': torch.stack(rgb_list, 0),
        'thermal': torch.stack(thermal_list, 0),
        'bboxes': torch.cat(labels_list, 0)[:, 2:] if len(labels_list) > 0 else torch.zeros(0, 4),  # xywh
        'cls': torch.cat(labels_list, 0)[:, 1] if len(labels_list) > 0 else torch.zeros(0),  # class
        'batch_idx': torch.cat(labels_list, 0)[:, 0] if len(labels_list) > 0 else torch.zeros(0)
    }