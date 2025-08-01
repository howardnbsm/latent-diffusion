from cleanfid import fid

# 假圖資料夾
fake_dir = "/raid/chunhualin/project/latent-diffusion/logs/2025-07-29T02-17-09_cifar10-ldm/checkpoints/samples/00294000/2025-07-30-00-06-35/img"

# 直接用內建的 CIFAR-10 test 集，不用自己下載
score = fid.compute_fid(
    fake_dir,
    dataset_name="cifar10",
    dataset_split="test",
    dataset_res=32,
    mode="clean",
)

print(f"Clean-FID: {score:.4f}")