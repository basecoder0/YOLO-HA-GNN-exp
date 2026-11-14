# utils/generate_mapping.py
import os
import pandas as pd
from glob import glob


def generate_mapping(root_dir, output_path='data/rgb_to_lidar_mapping.csv'):
    """
    遍历 R-LiViT 数据集结构，生成 RGB 与 LiDAR 的对应关系 CSV。
    假设结构：
    RGB: R-LiViT_RGB-T/rgb/<seq_id>/<filename>.png
    LiDAR: R-LiViT_LiDAR/point_clouds/<seq_id>/<filename>.bin
    """
    print(f"正在扫描数据集: {root_dir}")

    rgb_root = os.path.join(root_dir, 'R-LiViT_RGB-T', 'rgb')
    lidar_root = os.path.join(root_dir, 'R-LiViT_LiDAR', 'point_clouds')

    mapping_data = []

    # 获取所有序列文件夹
    if not os.path.exists(rgb_root):
        raise FileNotFoundError(f"找不到 RGB 路径: {rgb_root}")

    sequences = sorted(os.listdir(rgb_root))
    print(f"发现 {len(sequences)} 个序列")

    for seq in sequences:
        rgb_seq_dir = os.path.join(rgb_root, seq)
        # R-LiViT 的 LiDAR 序列名通常与 RGB 一致，或者需要补零 (例如 rgb '01' -> lidar '0001')
        # 这里假设文件夹名称完全一致，如果不一致需在此修改
        lidar_seq_dir = os.path.join(lidar_root, seq)

        if not os.path.exists(lidar_seq_dir):
            # 尝试补零匹配 (例如 rgb 00 -> lidar 0000)
            if len(seq) < 4:
                padded_seq = seq.zfill(4)
                lidar_seq_dir = os.path.join(lidar_root, padded_seq)

            if not os.path.exists(lidar_seq_dir):
                print(f"⚠️ 警告: 序列 {seq} 没有对应的 LiDAR 文件夹，跳过。")
                continue

        # 获取该序列下的所有图片
        rgb_files = sorted(glob(os.path.join(rgb_seq_dir, '*.png')) + glob(os.path.join(rgb_seq_dir, '*.jpg')))
        lidar_files = sorted(glob(os.path.join(lidar_seq_dir, '*.bin')))

        # 简单帧对齐：假设 RGB 和 LiDAR 已经过同步或只取共同的帧数
        # 如果文件名包含时间戳，这里应该解析时间戳进行最近邻匹配
        # 针对 R-LiViT，通常文件名是索引 (00000.bin)

        min_len = min(len(rgb_files), len(lidar_files))
        for i in range(min_len):
            rgb_path = rgb_files[i]
            lidar_path = lidar_files[i]

            # 提取相对路径用于存储
            rgb_rel = os.path.relpath(rgb_path, root_dir)
            lidar_rel = os.path.relpath(lidar_path, root_dir)

            mapping_data.append({
                'seq_id': seq,
                'rgb_path': rgb_rel,
                'lidar_path': lidar_rel,
                'frame_idx': i
            })

    df = pd.DataFrame(mapping_data)
    df.to_csv(output_path, index=False)
    print(f"✅ 映射文件已生成: {output_path}, 共包含 {len(df)} 帧数据")


if __name__ == '__main__':
    # 修改为你的数据集实际根目录
    DATA_ROOT = r"D:\BaiduNetdiskDownload\R-LiViT\R-LiViT"
    generate_mapping(DATA_ROOT)