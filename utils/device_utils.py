import torch
import os


def setup_device():
    """设置最优设备"""
    # 强制使用NVIDIA GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)

        # 打印GPU信息
        print("🎯 GPU Configuration:")
        print(f"   Device: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")
        print(f"   CUDA Version: {torch.version.cuda}")

        # 设置优化选项
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

        return device
    else:
        print("❌ No CUDA device available, using CPU")
        return torch.device('cpu')


def print_gpu_usage():
    """打印GPU使用情况"""
    if torch.cuda.is_available():
        print(f"📊 GPU Memory - Allocated: {torch.cuda.memory_allocated() / 1024 ** 3:.2f}GB, "
              f"Cached: {torch.cuda.memory_reserved() / 1024 ** 3:.2f}GB")