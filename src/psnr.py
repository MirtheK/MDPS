import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchmetrics.functional import peak_signal_noise_ratio as psnr
from dataset import SHOMRI

def compute_psnr_over_diffusion(
    normal_images,     
    anomalous_images, 
    beta_start,
    beta_end,
    diffusion_steps,
    device,
    skip=1,
    n_pairs=50,         # how many pairs to average over
):
    # Build alpha schedule
    betas = np.linspace(beta_start, beta_end, diffusion_steps, dtype=np.float32)
    betas = torch.tensor(betas)
    beta = torch.cat([torch.zeros(1), betas], dim=0).to(device)
    alphas_cumprod = (1 - beta).cumprod(dim=0)

    normal_images = normal_images.to(device)
    anomalous_images = anomalous_images.to(device)

    timesteps = list(range(0, diffusion_steps, skip))
    psnr_normal_list = []
    psnr_anomalous_list = []

    for t in timesteps:
        at = alphas_cumprod[t + 1].view(-1, 1, 1, 1, 1)  # 5D for 3D volumes

        # Average over multiple noise samples for stability
        psnr_normal_t = []
        psnr_anomalous_t = []
        for _ in range(3):  # average over 3 noise samples per timestep
            noise_n = torch.randn_like(normal_images)
            noisy_normal = at.sqrt() * normal_images + (1 - at).sqrt() * noise_n

            noise_a = torch.randn_like(anomalous_images)
            noisy_anomalous = at.sqrt() * anomalous_images + (1 - at).sqrt() * noise_a

            # Both compared against original normal (x0_HR equivalent)
            psnr_normal_t.append(psnr(noisy_normal, normal_images, data_range=1.0).item())
            psnr_anomalous_t.append(psnr(noisy_anomalous, normal_images, data_range=1.0).item())

        psnr_normal_list.append(np.mean(psnr_normal_t))
        psnr_anomalous_list.append(np.mean(psnr_anomalous_t))

        if t % 50 == 0:
            print(f"t={t:4d} | PSNR normal: {psnr_normal_list[-1]:.2f} | PSNR anomalous: {psnr_anomalous_list[-1]:.2f}")

    return timesteps, psnr_normal_list, psnr_anomalous_list


def find_convergence(timesteps, psnr_normal, psnr_anomalous, threshold=0.3):
    for i, t in enumerate(timesteps):
        if abs(psnr_normal[i] - psnr_anomalous[i]) < threshold:
            return t
    return timesteps[-1]


def plot_and_save(timesteps, psnr_normal, psnr_anomalous, convergence_t, save_path="/projects/prjs1633/repositories/MDPS/experiments/psnr_convergence_03.png"):
    plt.figure(figsize=(10, 5))
    plt.plot(timesteps, psnr_normal, label="Normal images", color="blue")
    plt.plot(timesteps, psnr_anomalous, label="Anomalous images", color="red")
    plt.axvline(x=convergence_t, color="green", linestyle="--", 
                label=f"Convergence at t={convergence_t}")
    plt.xlabel("Diffusion timestep t")
    plt.ylabel("PSNR (dB)")
    plt.title("PSNR Convergence: Normal vs Anomalous MRI Volumes")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    # plt.show()
    print(f"Saved to {save_path}")


if __name__ == "__main__":
    # Config — adjust to match your config object
    beta_start = 0.0001
    beta_end = 0.02
    diffusion_steps = 1000
    device = "cuda" if torch.cuda.is_available() else "cpu"
    skip = 10  # check every 10 steps to save time
    root_dir = "/projects/prjs1633/anomaly_detection/SHOMRI"

    # Load test set (has both normal and abnormal)
    test_dataset = SHOMRI(root_dir=root_dir, is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Collect a batch of normal and anomalous volumes
    normal_vols = []
    anomalous_vols = []
    n_collect = 16  # collect 5 of each for averaging

    for batch in test_loader:
        img = batch["image"]
        label = batch["label"][0]
        if label == "good" and len(normal_vols) < n_collect:
            normal_vols.append(img)
        elif label == "defective" and len(anomalous_vols) < n_collect:
            anomalous_vols.append(img)
        if len(normal_vols) == n_collect and len(anomalous_vols) == n_collect:
            break

    normal_images = torch.cat(normal_vols, dim=0)       # [N, 1, 128, 128, 128]
    anomalous_images = torch.cat(anomalous_vols, dim=0) # [N, 1, 128, 128, 128]

    timesteps, psnr_normal, psnr_anomalous = compute_psnr_over_diffusion(
        normal_images, anomalous_images,
        beta_start=beta_start,
        beta_end=beta_end,
        diffusion_steps=diffusion_steps,
        device=device,
        skip=skip,
    )

    convergence_t = find_convergence(timesteps, psnr_normal, psnr_anomalous, threshold=0.5)
    print(f"\nConvergence point: t={convergence_t}")
    print(f"Suggested mask_steps = {convergence_t}")

    plot_and_save(timesteps, psnr_normal, psnr_anomalous, convergence_t)