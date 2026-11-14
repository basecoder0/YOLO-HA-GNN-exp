import os


def convert_txt(input_path, output_path):
    print(f"🔄 正在转换 {input_path} ...")

    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    valid_lines = []
    for line in lines:
        line = line.strip()
        if not line: continue

        # 你的 train.txt 每一行可能是这样的：
        # "path/to/rgb.jpg path/to/thermal.jpg path/to/xml.xml"
        # 我们只需要第一部分 (RGB路径)
        parts = line.split()
        # 或者如果是逗号分隔：parts = line.split(',')

        rgb_path = parts[0]  # 假设第一列是 RGB

        # 确保是绝对路径
        if not os.path.isabs(rgb_path):
            # 如果是相对路径，需要拼上根目录
            # 根据你的报错信息，根目录似乎是 D:/BaiduNetdiskDownload/R-LiViT/R-LiViT/R-LiViT_RGB-T
            root = "D:/BaiduNetdiskDownload/R-LiViT/R-LiViT/R-LiViT_RGB-T"
            rgb_path = os.path.join(root, rgb_path)

        # 替换反斜杠
        rgb_path = rgb_path.replace('\\', '/')

        if os.path.exists(rgb_path):
            valid_lines.append(rgb_path)
        else:
            print(f"⚠️ 警告: 找不到文件 {rgb_path}")

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(valid_lines))

    print(f"✅ 生成完毕: {output_path} (共 {len(valid_lines)} 张图片)")


if __name__ == '__main__':
    # 请根据你实际的文件位置修改这里
    base_dir = "D:/BaiduNetdiskDownload/R-LiViT/R-LiViT/R-LiViT_RGB-T"

    convert_txt(
        input_path=os.path.join(base_dir, "train.txt"),
        output_path=os.path.join(base_dir, "train_rgb_official.txt")
    )

    convert_txt(
        input_path=os.path.join(base_dir, "test.txt"),
        output_path=os.path.join(base_dir, "test_rgb_official.txt")
    )