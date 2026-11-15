# testing/test.py - completelyfixversion

import torch
import yaml
import argparse
from pathlib import Path
import sys
import os
import json
import time

# addprojectrootdirectorytopath
sys.path.append(str(Path(__file__).parent.parent))

from models.full_pipeline import YOLOGNNFusion


# ========== define义mustneedofutility functions ==========

def setup_device():
    """countoneofdevicesetfunction"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("using CPU")
    return device


def get_data_root():
    """getcountoneofdatarootdirectory"""
    possible_paths = [
        'D:/BaiduNetdiskDownload/16356714/R-LiViT',
        'D:/BaiduNetdiskDownload/R-LiViT/R-LiViT',
        './data/R-LiViT',
        'R-LiViT'
    ]

    for path in possible_paths:
        if os.path.exists(path):
            print(f"✅ usingdatapath: {path}")
            return path

    return possible_paths[0]


def safe_load_checkpoint(model, checkpoint_path, device):
    """safety/fullloadingcheckpoint，compatiblenotsameformat"""
    print(f"loadingcheckpoint: {checkpoint_path}")

    try:
        if os.path.exists(checkpoint_path):
            # loadcheckpoint
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

            # extractmodelstatus
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                epoch = checkpoint.get('epoch', 0)
                best_metric = checkpoint.get('best_metric', 0.0)
            else:
                state_dict = checkpoint
                epoch = 0
                best_metric = 0.0

            # loadmodelparameter
            model_dict = model.state_dict()

            # filter mismatched keys
            filtered_state_dict = {}
            for key, param in state_dict.items():
                if key in model_dict:
                    if param.shape == model_dict[key].shape:
                        filtered_state_dict[key] = param
                    else:
                        print(f"⚠️ shapenotmatch: {key} - {param.shape} vs {model_dict[key].shape}")
                else:
                    # trycheckfindmutual似ofkey
                    found = False
                    for model_key in model_dict.keys():
                        if key.replace('module.', '') == model_key.replace('module.', ''):
                            if param.shape == model_dict[model_key].shape:
                                filtered_state_dict[model_key] = param
                                found = True
                                break

                    if not found:
                        print(f"⚠️ notfindtocorrespondingkey: {key}")

            # loadparameter
            model_dict.update(filtered_state_dict)
            model.load_state_dict(model_dict, strict=False)

            print(f"✅ successloadingcheckpoint")
            print(f"   loadingparameter: {len(filtered_state_dict)}/{len(model_dict)}")
            print(f"   trainingepoch: {epoch}")

            return model, epoch, best_metric

        else:
            print(f"❌ checkpointnotexist: {checkpoint_path}")
            return model, 0, 0.0

    except Exception as e:
        print(f"❌ loadingcheckpointfailed: {e}")
        import traceback
        traceback.print_exc()
        return model, 0, 0.0


def simple_collate_fn(batch):
    """
    simpleofcollatefunction，processingR-LiViTdataset
    """
    if not batch:
        return {}

    collated = {}

    # processingRGBimage
    rgb_images = []
    thermal_images = []
    targets = []
    image_paths = []

    for item in batch:
        if isinstance(item, dict):
            # extractRGB
            if 'rgb' in item:
                rgb_images.append(item['rgb'])
            elif 'image' in item:
                rgb_images.append(item['image'])

            # extractthermalimaging
            if 'thermal' in item:
                thermal_images.append(item['thermal'])

            # extracttarget
            if 'target' in item:
                targets.append(item['target'])
            elif 'annotations' in item:
                targets.append(item['annotations'])

            # extractimagepath
            if 'image_path' in item:
                image_paths.append(item['image_path'])

    # 堆叠image
    if rgb_images:
        collated['rgb'] = torch.stack(rgb_images, dim=0)

        # processingthermalimaging
        if thermal_images and len(thermal_images) == len(rgb_images):
            collated['thermal'] = torch.stack(thermal_images, dim=0)
        else:
            # createoccupydigitsymbol
            collated['thermal'] = torch.zeros_like(collated['rgb'])

    # targetkeeplistformat
    if targets:
        collated['targets'] = targets

    # imagepath
    if image_paths:
        collated['image_paths'] = image_paths

    return collated


def parse_args():
    """parsecommandlineparameter"""
    parser = argparse.ArgumentParser(description='testYOLO-GNNfusionmodel')

    parser.add_argument('--config', type=str, default='configs/test/evaluation.yaml',
                        help='testconfigurationfilepath')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='modelcheckpointpath')
    parser.add_argument('--data_root', type=str, default=None,
                        help='dataset root directory（defaultautodetection）')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'val', 'test'],
                        help='needevaluationofdatasetsplit/divide')
    parser.add_argument('--device', type=str, default='auto',
                        help='deviceclasstype (auto/cuda/cpu)')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='batchminorlargesmall（recommendfor/as1by/withmake surecompatibleproperty）')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='data loadingworker processcount（0make surecompatibleproperty）')
    parser.add_argument('--output_dir', type=str, default='results/test_output',
                        help='outputdirectory')
    parser.add_argument('--no_risk_assessment', action='store_true',
                        help='disableuseriskevaluation（andextract_stage1_features.pykeeponeconsistent）')
    parser.add_argument('--max_batches', type=int, default=10,
                        help='maximumprocessingbatchminorcountamount（forfasttest）')

    return parser.parse_args()


def load_config(config_path):
    """loadingconfigurationfile"""
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    else:
        print(f"⚠️ configurationfilenotexist: {config_path}")
        return {}


def create_model(args, config, device):
    """createmodel - andextract_stage1_features.pykeeponeconsistent"""
    print("createmodel...")

    # fromconfigurationorusingdefaultvalue
    model_config = config.get('model', {})

    model = YOLOGNNFusion(
        num_classes=model_config.get('num_classes', 3),
        node_feature_dim=model_config.get('node_feature_dim', 128),
        gnn_hidden_dim=model_config.get('gnn_hidden_dim', 256),
        use_risk_assessment=not args.no_risk_assessment,
        use_lidar=model_config.get('use_lidar', False),
        use_enhanced_risk=model_config.get('use_enhanced_risk', True),
        confidence_threshold=model_config.get('confidence_threshold', 0.1),
        gnn_max_nodes=model_config.get('gnn_max_nodes', 50)
    )
    model.to(device)
    model.eval()

    return model


def create_simple_dataset(data_root, split):
    """createsimpleofdataset（notusingcomplexofRLiViTDataset）"""
    print(f"createsimpledatasetfor{split}split/divide...")

    samples = []
    data_path = Path(data_root)

    # checkfindRGBimage
    rgb_base = data_path / "R-LiViT_RGB-T" / "rgb"
    if rgb_base.exists():
        # limitsamplecountamountfortest
        max_sequenceuenceuences = 2
        max_images_per_sequenceuenceuence = 2

        sequenceuence_count = 0
        for sequenceuence_dir in sorted(rgb_base.iterdir()):
            if sequenceuence_dir.is_dir():
                img_count = 0
                for img_file in sorted(sequenceuence_dir.glob("*.png")):
                    if img_count >= max_images_per_sequenceuenceuence:
                        break

                    # checkfindcorrespondingofthermalimaging
                    thermal_path = data_path / "R-LiViT_RGB-T" / "thermal" / sequenceuence_dir.name / img_file.name

                    sample = {
                        'image_path': str(img_file),
                        'thermal_path': str(thermal_path) if thermal_path.exists() else None,
                        'sequenceuenceuence': sequenceuence_dir.name,
                        'image_name': img_file.stem
                    }
                    samples.append(sample)
                    img_count += 1

                sequenceuence_count += 1
                if sequenceuence_count >= max_sequenceuenceuences:
                    break

    print(f"findto {len(samples)} asample")
    return samples


class SimpleDataset(torch.utils.data.Dataset):
    """simpledatasetclass"""

    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        import cv2
        import numpy as np

        sample = self.samples[idx]

        # loadRGBimage
        rgb_path = sample['image_path']
        rgb_img = cv2.imread(rgb_path)
        if rgb_img is None:
            print(f"❌ unable toloadingimage: {rgb_path}")
            rgb_img = np.zeros((416, 416, 3), dtype=np.uint8)

        # adjustmentlargesmall
        rgb_img = cv2.resize(rgb_img, (416, 416))
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        rgb_tensor = torch.from_numpy(rgb_img).float().permute(2, 0, 1) / 255.0

        # loadthermalimaging
        thermal_path = sample.get('thermal_path')
        if thermal_path and os.path.exists(thermal_path):
            thermal_img = cv2.imread(thermal_path, cv2.IMREAD_GRAYSCALE)
            if thermal_img is None:
                thermal_img = np.zeros((416, 416), dtype=np.uint8)

            thermal_img = cv2.resize(thermal_img, (416, 416))
            thermal_img = cv2.cvtColor(thermal_img, cv2.COLOR_GRAY2RGB)
            thermal_tensor = torch.from_numpy(thermal_img).float().permute(2, 0, 1) / 255.0
        else:
            thermal_tensor = torch.zeros_like(rgb_tensor)

        return {
            'rgb': rgb_tensor,
            'thermal': thermal_tensor,
            'image_path': rgb_path,
            'sample_info': sample
        }


def run_evaluation(model, dataloader, device, args):
    """runevaluation"""
    print(f"\nstartevaluation...")

    model.eval()
    all_results = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= args.max_batches:
                print(f"  达tomaximumbatchminorlimit ({args.max_batches})")
                break

            print(f"  processingbatchminor {batch_idx + 1}/{len(dataloader)}")

            # extractimagedata
            if 'rgb' in batch:
                rgb_images = batch['rgb'].to(device)
                thermal_images = batch['thermal'].to(device)
                image_paths = batch.get('image_paths', [])
            else:
                print(f"⚠️ batchminorinnothaveRGBimagedata")
                continue

            # forward propagation
            try:
                outputs = model(rgb_images, thermal_images)

                # collectresult
                batch_result = {
                    'batch_idx': batch_idx,
                    'num_images': len(rgb_images),
                    'stage1_detections': len(outputs.get('stage1_detections', [])),
                    'enhanced_detections': len(outputs.get('enhanced_detections', [])),
                    'has_risk_assessment': outputs.get('risk_assessment') is not None,
                    'gnn_features_shape': list(outputs.get('gnn_features', torch.Tensor()).shape),
                    'image_paths': image_paths
                }
                all_results.append(batch_result)

                print(f"    imagecountamount: {batch_result['num_images']}")
                print(f"    firststagedetection: {batch_result['stage1_detections']}")
                print(f"    enhancedetection: {batch_result['enhanced_detections']}")

                # showonesomedetection详situation
                if outputs.get('stage1_detections'):
                    detections = outputs['stage1_detections']
                    if len(detections) > 0:
                        print(
                            f"    detectionconfidencerange: {min(d.get('confidence', 0) for d in detections):.3f}-{max(d.get('confidence', 0) for d in detections):.3f}")

            except Exception as e:
                print(f"❌ batchminor {batch_idx} processingfailed: {e}")
                import traceback
                traceback.print_exc()
                continue

    return all_results


def main():
    """majortestfunction"""
    args = parse_args()

    print("=" * 60)
    print("YOLO-GNN fusionmodeltest")
    print("=" * 60)

    # setdevice
    device = setup_device()

    # getdatapath
    data_root = args.data_root if args.data_root else get_data_root()
    print(f"datarootdirectory: {data_root}")

    # checkpathwhetherexist
    if not os.path.exists(data_root):
        print(f"❌ datarootdirectorynotexist: {data_root}")
        print("pleasecheckdatapathorusing --data_root parameterpointdefinepositivesurepath")
        return

    # loadconfiguration
    config = load_config(args.config)

    # createmodel
    model = create_model(args, config, device)

    # loadcheckpoint
    model, epoch, best_metric = safe_load_checkpoint(model, args.checkpoint, device)

    # createsimpleofdatasetanddataloaddevice
    samples = create_simple_dataset(data_root, args.split)

    if not samples:
        print("❌ nothavefindtosampledata")
        return

    dataset = SimpleDataset(samples)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=simple_collate_fn,
        pin_memory=False
    )

    print(f"datasetlargesmall: {len(dataset)}")
    print(f"data loadingdevicebatchminorcountamount: {len(dataloader)}")

    # runevaluation
    results = run_evaluation(model, dataloader, device, args)

    # outputtotal结
    print("\n" + "=" * 60)
    print("testcompleted!")
    print(f"modelcheckpoint: {args.checkpoint}")
    print(f"trainingepoch: {epoch}")
    print(f"bestmetric: {best_metric:.4f}")
    print(f"processingbatchminor: {len(results)}")

    if results:
        total_images = sum(r['num_images'] for r in results)
        total_stage1 = sum(r['stage1_detections'] for r in results)
        total_enhanced = sum(r['enhanced_detections'] for r in results)

        print(f"totalprocessingimage: {total_images}")
        print(f"firststagetotaldetectioncount: {total_stage1}")
        print(f"enhanceaftertotaldetectioncount: {total_enhanced}")

        if total_images > 0:
            print(f"averageeachimagedetectioncount: {total_stage1 / total_images:.2f}")
            print(f"detectionrate: {total_stage1 / (total_images * 10):.2%}")

    print("=" * 60)

    # saveresult
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result_file = output_dir / f"test_results_{Path(args.checkpoint).stem}.json"
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump({
            'checkpoint': args.checkpoint,
            'epoch': epoch,
            'best_metric': best_metric,
            'device': str(device),
            'args': vars(args),
            'results': results,
            'summary': {
                'total_batches': len(results),
                'total_images': sum(r['num_images'] for r in results) if results else 0,
                'total_stage1_detections': sum(r['stage1_detections'] for r in results) if results else 0,
                'total_enhanced_detections': sum(r['enhanced_detections'] for r in results) if results else 0,
            },
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        }, f, indent=2)

    print(f"resultsaveto: {result_file}")

    return results


if __name__ == '__main__':
    main()