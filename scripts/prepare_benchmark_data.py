import os


def convert_txt(input_path, output_path):
    print(f"🔄 positiveat/inconvert {input_path} ...")

    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    valid_lines = []
    for line in lines:
        line = line.strip()
        if not line: continue

        # youof train.txt each line might look like this：
        # "path/to/rgb.jpg path/to/thermal.jpg path/to/xml.xml"
        # ouronlyneedfirstpartial (RGBpath)
        parts = line.split()
        # oroneifis逗号score/divide隔：parts = line.split(',')

        rgb_path = parts[0]  # assumefirstlistis RGB

        # make sureisabsolutepath
        if not os.path.isabs(rgb_path):
            # ifisrelativepath，needcombineaboverootdirectory
            # rootaccordingyouoferrorinformation，rootdirectoryseemis D:/BaiduNetdiskDownload/R-LiViT/R-LiViT/R-LiViT_RGB-T
            root = "D:/BaiduNetdiskDownload/R-LiViT/R-LiViT/R-LiViT_RGB-T"
            rgb_path = os.path.join(root, rgb_path)

        # replacechange反斜杠
        rgb_path = rgb_path.replace('\\', '/')

        if os.path.exists(rgb_path):
            valid_lines.append(rgb_path)
        else:
            print(f"⚠️ warning: findnottofile {rgb_path}")

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(valid_lines))

    print(f"✅ generate完毕: {output_path} (total {len(valid_lines)} tensorimagepiece)")


if __name__ == '__main__':
    # pleaserootaccordingyouactualoffiledigitplacemodifyhere
    base_dir = "D:/BaiduNetdiskDownload/R-LiViT/R-LiViT/R-LiViT_RGB-T"

    convert_txt(
        input_path=os.path.join(base_dir, "train.txt"),
        output_path=os.path.join(base_dir, "train_rgb_official.txt")
    )

    convert_txt(
        input_path=os.path.join(base_dir, "test.txt"),
        output_path=os.path.join(base_dir, "test_rgb_official.txt")
    )