import torch, six, types
if not hasattr(torch, "_six"):
    torch._six = types.SimpleNamespace(string_classes=six.string_types)


from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class CIFAR10Train(Dataset):
    def __init__(self, size=32):
        self.base = CIFAR10(root='./data', train=True, download=True,
                             transform=transforms.Resize((size, size)))

    def __getitem__(self, index):
        img_pil, _ = self.base[index]
        img = np.array(img_pil).astype(np.float32) / 127.5 - 1.0
        return {"image": img}

    def __len__(self):
        return len(self.base)

class CIFAR10Validation(Dataset):
    def __init__(self, size=32, **kwargs):
        self.base = CIFAR10(root='./data', train=False, download=True,
                             transform=transforms.Resize((size, size)))

    def __getitem__(self, index):
        img_pil, _ = self.base[index]
        img = np.array(img_pil).astype(np.float32) / 127.5 - 1.0
        return {"image": img}

    def __len__(self):
        return len(self.base)
