import torch, six, types
if not hasattr(torch, "_six"):
    torch._six = types.SimpleNamespace(string_classes=six.string_types)

import os
import random
import numpy as np
# import torch
from PIL import Image
from torchvision.datasets import CIFAR10
from torchvision.utils import save_image, make_grid
from torchvision import transforms
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
import math

# Clean‑FID
from cleanfid import fid

from torch.utils.data import DataLoader
from datetime import datetime 

# Constants
BATCH_SIZE = 64
NUM_WORKERS = 4
N_IMAGES = 10000  # CIFAR-10 test set 大小

BASE_REAL    = "cleanfid_real"
BASE_FAKE    = "cleanfid_fake"

ts           = datetime.now().strftime("%Y%m%d_%H%M%S")
REAL_DIR     = os.path.join(BASE_REAL, ts)
FAKE_DIR     = os.path.join(BASE_FAKE, ts)
LOG_DIR  = os.path.join("cleanfid_logs", ts)


os.makedirs(REAL_DIR, exist_ok=True)
os.makedirs(FAKE_DIR, exist_ok=True)
os.makedirs(LOG_DIR,  exist_ok=True)


CONFIG_PATH = "configs/autoencoder/autoencoder_cifar10.yaml"
CKPT_PATH   = "logs/2025-07-25T14-09-05_autoencoder_cifar10/checkpoints/last.ckpt"
cfg = OmegaConf.load(CONFIG_PATH)
ae  = instantiate_from_config(cfg.model).cuda().eval()
ckpt = torch.load(CKPT_PATH, map_location="cpu")
ae.load_state_dict(ckpt["state_dict"], strict=False)

# 3. 準備 DataLoader
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])
dataset = CIFAR10(root="./data", train=False, download=True, transform=transform)
loader  = DataLoader(dataset, batch_size=BATCH_SIZE,
                     shuffle=False, num_workers=NUM_WORKERS)

# 4. 批次重建並累積
orig_batches = []
rec_batches  = []

for i, (imgs, _) in enumerate(loader):
    start = i * BATCH_SIZE
    if start >= N_IMAGES: break
    if start + imgs.size(0) > N_IMAGES:
        imgs = imgs[: N_IMAGES - start]

    imgs = imgs.cuda()
    imgs_norm = imgs * 2.0 - 1.0

    # with torch.no_grad():
    #     batch = {"image": imgs_norm}
    #     x = ae.get_input(batch, "image").cuda()
    #     rec, _ = ae(x)

    with torch.no_grad():
        rec, _ = ae(imgs_norm)

    rec = torch.clamp(rec, -1, 1)
    rec = (rec + 1.0) * 0.5
    rec = rec.cpu()

    orig_batches.append(imgs.cpu())
    rec_batches.append(rec)

orig_all = torch.cat(orig_batches, dim=0)[:N_IMAGES]
rec_all  = torch.cat(rec_batches,  dim=0)[:N_IMAGES]

# 計算 MSE、MAE、PSNR
diff = orig_all - rec_all
mse  = diff.pow(2).mean().item()
mae  = diff.abs().mean().item()
psnr = 10 * math.log10(1.0 / mse)
print(f"MSE:  {mse:.6f}")
print(f"MAE:  {mae:.6f}")
print(f"PSNR: {psnr:.2f} dB")

# 6. 存影像到 Clean‑FID 用的資料夾
for idx in range(N_IMAGES):
    save_image(orig_all[idx], os.path.join(REAL_DIR,  f"{idx:05d}.png"))
    save_image(rec_all[idx],  os.path.join(FAKE_DIR, f"{idx:05d}.png"))

# 7. 計算 Clean‑FID
cleanfid_score = fid.compute_fid(REAL_DIR, FAKE_DIR, mode="clean")
print(f"Clean‑FID: {cleanfid_score:.2f}")

# 8. 將四個指標寫入 log 檔
log_path = os.path.join(LOG_DIR, f"{ts}.txt")
with open(log_path, "w") as f:
    f.write(f"MSE:        {mse:.6f}\n")
    f.write(f"MAE:        {mae:.6f}\n")
    f.write(f"PSNR (dB):  {psnr:.2f}\n")
    f.write(f"Clean-FID:  {cleanfid_score:.2f}\n")

print(f"Log saved to: {log_path}")

# # 1. 載入模型
# CONFIG_PATH = "configs/autoencoder/autoencoder_cifar10.yaml"
# CKPT_PATH   = "logs/2025-07-25T14-09-05_autoencoder_cifar10/checkpoints/last.ckpt"
# cfg = OmegaConf.load(CONFIG_PATH)
# ae  = instantiate_from_config(cfg.model).cuda().eval()
# ckpt = torch.load(CKPT_PATH, map_location="cpu")
# ae.load_state_dict(ckpt["state_dict"], strict=False)

# # 2. 準備 CIFAR-10 測試影像
# dataset = CIFAR10(root="./data", train=False, download=True,
#                   transform=transforms.Resize((32, 32)))
# indices = random.sample(range(len(dataset)), 16)
# imgs_pil = [dataset[i][0] for i in indices]
# orig_tensors = [transforms.ToTensor()(img) for img in imgs_pil]
# orig_batch = torch.stack(orig_tensors, dim=0).cuda()  # 範圍 [0,1]

# # 3. 重建
# imgs_np = np.stack([np.array(img).astype(np.float32) for img in imgs_pil], 0)
# imgs_np = imgs_np / 127.5 - 1.0
# imgs_t = torch.from_numpy(imgs_np).cuda()
# batch = {"image": imgs_t}
# with torch.no_grad():
#     rec, _ = ae(ae.get_input(batch, "image"))
# rec = torch.clamp(rec, -1, 1)
# rec = (rec + 1.0) * 0.5  # scale 到 [0,1]
# rec = rec.cpu()

# # 4. 計算 MSE、MAE、PSNR
# diff = orig_batch.cpu() - rec
# mse = diff.pow(2).mean().item()
# mae = diff.abs().mean().item()
# psnr = 10 * math.log10(1.0 / mse)
# print(f"MSE:  {mse:.6f}")
# print(f"MAE:  {mae:.6f}")
# print(f"PSNR: {psnr:.2f} dB")

# # 5. （可選）存對比圖
# save_image(make_grid(orig_batch.cpu(), nrow=8), "ae_input_test.png")
# save_image(make_grid(rec,      nrow=8), "ae_reconstruction_test.png")

# # 6. 使用 Clean‑FID：先把影像存到兩個資料夾
# real_dir = "cleanfid_real"
# fake_dir = "cleanfid_fake"
# os.makedirs(real_dir, exist_ok=True)
# os.makedirs(fake_dir, exist_ok=True)
# for i, img in enumerate(orig_batch.cpu()):
#     save_image(img, os.path.join(real_dir, f"{i:04d}.png"))
# for i, img in enumerate(rec):
#     save_image(img, os.path.join(fake_dir, f"{i:04d}.png"))

# # 7. 計算 Clean‑FID
# cleanfid_score = fid.compute_fid(real_dir, fake_dir, mode="clean")
# print(f"Clean‑FID: {cleanfid_score:.2f}")
