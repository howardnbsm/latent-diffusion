import torch, six, types
if not hasattr(torch, "_six"):
    torch._six = types.SimpleNamespace(string_classes=six.string_types)
    
import os
import random
import numpy as np
import torch
from PIL import Image
from torchvision.datasets import CIFAR10
from torchvision.utils import save_image, make_grid
from torchvision import transforms
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from torchvision.transforms import ToTensor, Normalize, Resize, Compose
import math

# from torchmetrics.image.fid import FrechetInceptionDistance

#1 construct path
CONFIG_PATH = "configs/autoencoder/autoencoder_cifar10.yaml"
CKPT_PATH   = "logs/2025-07-25T14-09-05_autoencoder_cifar10/checkpoints/last.ckpt"

#2 load autoencoder
cfg = OmegaConf.load(CONFIG_PATH)
ae  = instantiate_from_config(cfg.model).cuda().eval()
ckpt = torch.load(CKPT_PATH, map_location="cpu")
ae.load_state_dict(ckpt["state_dict"], strict=False)

#3 construct CIFAR-10 dataset
dataset = CIFAR10(
    root="./data", train=False, download=True,
    transform=transforms.Resize((32, 32))
)

indices = random.sample(range(len(dataset)), 16)
imgs_pil = [dataset[i][0] for i in indices]

#4 input image
orig_tensors = []
to_tensor = transforms.ToTensor()
for img in imgs_pil:
    t = to_tensor(img)
    orig_tensors.append(t)
orig_batch = torch.stack(orig_tensors, dim=0)

grid_in = make_grid(orig_batch, nrow=8)
save_image(grid_in, "ae_input.png")
print("Saved input grid → ae_input.png")

#5 Reconstruction
imgs_np = np.stack([np.array(img).astype(np.float32) for img in imgs_pil], 0)
imgs_np = imgs_np / 127.5 - 1.0

imgs_t = torch.from_numpy(imgs_np)

batch = {"image": imgs_t}
x = ae.get_input(batch, "image").cuda()

with torch.no_grad():
    rec, _ = ae(x)
rec = torch.clamp(rec, -1, 1)
rec = (rec + 1.0) * 0.5

#diff = torch.abs(orig_batch - rec.cpu())
#print("max diff =", diff.max().item(), "mean diff =", diff.mean().item())
grid_rec = make_grid(rec.cpu(), nrow=8)

save_image(grid_rec, "ae_reconstruction.png")
print("Saved reconstruction grid → ae_reconstruction.png")

diff = orig_batch - rec.cpu()
mse = diff.pow(2).mean().item()
mae = diff.abs().mean().item()
psnr = 10 * math.log10(1.0 / mse)
print(f"MSE: {mse:.6f}, MAE: {mae:.6f}, PSNR: {psnr:.2f} dB")

# fid = FrechetInceptionDistance(feature=64).cuda()  # feature=64 for CIFAR-10

# # 將原始和重建影像都加到 FID 計算
# fid.update(orig_batch.cuda(), real=True)
# fid.update(rec.cuda(), real=False)

# fid_score = fid.compute().item()
# print(f"FID: {fid_score:.2f}")