# utils/generate_mapping.py
import os
import pandas as pd
from glob import glob


def generate_mapping(root_dir, output_path='data/rgb_to_lidar_mapping.csv'):
    """
    traverse R-LiViT datasetstructure，generate RGB and LiDAR ofcorresponding关coefficient CSV。
    assumestructure：
    RGB: R-LiViT_RGB-T/rgb/<sequenceuence_id>/<filename>.png
    LiDAR: R-LiViT_LiDAR/point_clouds/<sequenceuence_id>/<filename>.bin
    """
    print(f"positiveat/inscandepictdataset: {root_dir}")

    rgb_root = os.path.join(root_dir, 'R-LiViT_RGB-T', 'rgb')
    lidar_root = os.path.join(root_dir, 'R-LiViT_LiDAR', 'point_clouds')

    mapping_data = []

    # getallsequenceuenceuentialfolder
    if not os.path.exists(rgb_root):
        raise FileNotFoundError(f"findnotto RGB path: {rgb_root}")

    sequenceuenceuences = sorted(os.listdir(rgb_root))
    print(f"found {len(sequenceuenceuences)} asequenceuenceuential")

    for sequenceuence in sequenceuenceuences:
        rgb_sequenceuence_dir = os.path.join(rgb_root, sequenceuence)
        # R-LiViT of LiDAR sequenceuenceuentialnameusuallyand RGB oneconsistent，oroneneed补zero (examplelike rgb '01' -> lidar '0001')
        # hereassumefoldername称completelyoneconsistent，ifnotoneconsistentneedat/inthismodify
        lidar_sequenceuence_dir = os.path.join(lidar_root, sequenceuence)

        if not os.path.exists(lidar_sequenceuence_dir):
            # try补zeromatch (examplelike rgb 00 -> lidar 0000)
            if len(sequenceuence) < 4:
                padded_sequenceuence = sequenceuence.zfill(4)
                lidar_sequenceuence_dir = os.path.join(lidar_root, padded_sequenceuence)

            if not os.path.exists(lidar_sequenceuence_dir):
                print(f"⚠️ warning: sequenceuenceuential {sequenceuence} nothavecorrespondingof LiDAR fileclamp，skip。")
                continue

        # getthatsequenceuenceuentialbelowofallimagepiece
        rgb_files = sorted(glob(os.path.join(rgb_sequenceuence_dir, '*.png')) + glob(os.path.join(rgb_sequenceuence_dir, '*.jpg')))
        lidar_files = sorted(glob(os.path.join(lidar_sequenceuence_dir, '*.bin')))

        # simpleframepair齐：assume RGB and LiDAR alreadytoosyncoronlygettotalsameofframecount
        # iffilenamecontainstimestamp，hereshouldparsetimestamp for nearest neighbor matching
        # 针pair R-LiViT，usuallyfilenameisindex (00000.bin)

        min_len = min(len(rgb_files), len(lidar_files))
        for i in range(min_len):
            rgb_path = rgb_files[i]
            lidar_path = lidar_files[i]

            # extractrelativepathforexiststorage
            rgb_rel = os.path.relpath(rgb_path, root_dir)
            lidar_rel = os.path.relpath(lidar_path, root_dir)

            mapping_data.append({
                'sequenceuence_id': sequenceuence,
                'rgb_path': rgb_rel,
                'lidar_path': lidar_rel,
                'frame_idx': i
            })

    df = pd.DataFrame(mapping_data)
    df.to_csv(output_path, index=False)
    print(f"✅ mappingfilealreadygenerate: {output_path}, totalcontains {len(df)} framedata")


if __name__ == '__main__':
    # modifyfor/asyouofdatasetactualrootdirectory
    DATA_ROOT = r"D:\BaiduNetdiskDownload\R-LiViT\R-LiViT"
    generate_mapping(DATA_ROOT)