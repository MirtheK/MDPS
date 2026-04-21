import os
import argparse
from omegaconf import OmegaConf
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from collections import defaultdict

import nibabel as nib
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu

from src.models.unet import UNetModel
from src.models.resnet import Resnet
from src.dataset import SHOMRI
from src.metrics import metric
from src.compare import distance
from src.diffusion import sample, sample_mask, compute_alpha


def extract_patches(volume, patch_size, stride=None):
    """
    volume:     (C, D, H, W) tensor
    patch_size: (pd, ph, pw)
    Returns:
        patches:   (N, C, pd, ph, pw)
        locations: list of (d, h, w)
    """
    if stride is None:
        stride = [x // 2 for x in patch_size]
    C, D, H, W = volume.shape
    pd, ph, pw = patch_size
    sd, sh, sw = stride

    patches, locations = [], []
    for d in range(0, D - pd + 1, sd):
        for h in range(0, H - ph + 1, sh):
            for w in range(0, W - pw + 1, sw):
                patches.append(volume[:, d:d+pd, h:h+ph, w:w+pw])
                locations.append((d, h, w))

    return torch.stack(patches, dim=0), locations  # (N, C, pd, ph, pw)


def reconstruct_volume(patches, locations, volume_shape, patch_size):
    """
    patches:      (N, C, pd, ph, pw)
    locations:    list of (d, h, w)
    volume_shape: (C, D, H, W)
    """
    C, D, H, W = volume_shape
    pd, ph, pw = patch_size
    device = patches.device

    reconstructed = torch.zeros((C, D, H, W), device=device)
    counts        = torch.zeros((C, D, H, W), device=device)

    for idx, (d, h, w) in enumerate(locations):
        reconstructed[:, d:d+pd, h:h+ph, w:w+pw] += patches[idx]
        counts[:, d:d+pd, h:h+ph, w:w+pw]         += 1

    return reconstructed / counts.clamp(min=1)


def save_histogram(scores, labels, save_dir, threshold=None, tag="test"):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    normal_scores   = [s for s, l in zip(scores, labels) if l == 0]
    abnormal_scores = [s for s, l in zip(scores, labels) if l == 1]

    plt.figure(figsize=(10, 6))
    if normal_scores:
        plt.hist(normal_scores,   bins=30, alpha=0.5, label='Normal',
                 color='blue', edgecolor='black')
    if abnormal_scores:
        plt.hist(abnormal_scores, bins=30, alpha=0.5, label='Abnormal',
                 color='red',  edgecolor='black')
    if threshold is not None:
        plt.axvline(x=threshold, color='green', linestyle='--', linewidth=2,
                    label=f'Threshold: {threshold:.3f}')
    plt.xlabel('Anomaly Score', fontsize=12)
    plt.ylabel('Frequency',     fontsize=12)
    plt.title('Anomaly Score Distribution', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    save_path = save_dir / f'histogram_{tag}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Histogram saved → {save_path}")
    return save_path


def save_nifti_with_prediction(anomaly_map, reconstruction, original_data,
                                filename, save_dir, score, threshold, label):
    save_dir   = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    prediction = "DEFECTIVE" if score > threshold else "GOOD"
    true_label = label.upper() if isinstance(label, str) else ("NORMAL" if label == 0 else "ABNORMAL")
    correct    = "CORRECT" if prediction == true_label else "WRONG"
    base_name  = Path(filename).stem.replace('.nii', '')

    for tag, tensor in [
        ("anomaly",  anomaly_map),
        ("recon",    reconstruction),
        ("original", original_data),
    ]:
        out_path = save_dir / f"{base_name}_{true_label}_{prediction}_{correct}_{tag}.nii.gz"
        arr = tensor.squeeze().cpu().numpy()
        nii = nib.Nifti1Image(arr, np.eye(4))
        nib.save(nii, str(out_path))



def batch_iter(tensor, batch_size):
    """Yield mini-batches from a (N, ...) tensor."""
    for i in range(0, tensor.shape[0], batch_size):
        yield tensor[i:i+batch_size]


def run_inference_on_patches(patches, unet, resnet, config):
    """
    patches: (N, C, pd, ph, pw) on CPU
    Returns anomaly_patches, recon_patches both (N, C, pd, ph, pw) on CPU
    """
    anomaly_list, recon_list = [], []

    for batch in batch_iter(patches, config.data.batch_size):
        batch = batch.to(config.model.device)

        if config.model.mask_steps == 0:
            test_steps = torch.tensor([config.model.test_steps], dtype=torch.int64).to(config.model.device)
            at  = compute_alpha(test_steps, config)
            seq = range(0, config.model.test_steps, config.model.skip)

            anomaly_batch, recon_accum = [], None
            for _ in range(config.model.test_repeat):
                noisy              = at.sqrt() * batch + (1 - at).sqrt() * torch.randn_like(batch)
                recon              = sample(batch, noisy, seq, unet, config, w=config.model.w)
                anomaly            = distance(recon, batch, resnet, config) / 2
                anomaly_batch.append(anomaly.unsqueeze(0))
                recon_accum        = recon if recon_accum is None else recon_accum + recon

            anomaly_list.append(torch.mean(torch.cat(anomaly_batch, dim=0), dim=0).cpu())
            recon_list.append((recon_accum / config.model.test_repeat).cpu())

        else:
            # coarse mask pass
            mask_steps = torch.tensor([config.model.mask_steps], dtype=torch.int64).to(config.model.device)
            at  = compute_alpha(mask_steps, config)
            seq_mask = range(0, config.model.mask_steps, config.model.skip_mask)

            coarse_batch = []
            for _ in range(config.model.mask_repeat):
                noisy  = at.sqrt() * batch + (1 - at).sqrt() * torch.randn_like(batch)
                recon  = sample(batch, noisy, seq_mask, unet, config, w=config.model.w_mask)
                anomaly = distance(recon, batch, resnet, config) / 2
                coarse_batch.append(anomaly.unsqueeze(0))

            coarse = torch.mean(torch.cat(coarse_batch, dim=0), dim=0)
            pixel_min, pixel_max = coarse.min(), coarse.max()
            mask = (coarse > pixel_min + config.model.mask0_thresholds * (pixel_max - pixel_min)).float()

            # masked sampling pass
            seq = range(0, config.model.test_steps, config.model.skip)
            anomaly_batch, recon_accum = [], None
            for _ in range(config.model.mask_repeat):
                recon   = sample_mask(batch, mask, seq, unet, config, w=config.model.w)
                anomaly = distance(recon, batch, resnet, config) / 2
                anomaly_batch.append(anomaly.unsqueeze(0))
                recon_accum = recon if recon_accum is None else recon_accum + recon

            anomaly_list.append(torch.mean(torch.cat(anomaly_batch, dim=0), dim=0).cpu())
            recon_list.append((recon_accum / config.model.mask_repeat).cpu())

    return torch.cat(anomaly_list, dim=0), torch.cat(recon_list, dim=0)


def mdps(args):
    config = OmegaConf.load(args.config)
    patch_size = tuple(config.data.patch_size)

    save_dir = Path(config.data.save_dir) / config.data.category
    save_dir.mkdir(parents=True, exist_ok=True)

    unet = UNetModel(64, 64, dropout=0.0, n_heads=4, in_channels=config.data.imput_channel)
    checkpoint = torch.load(os.path.join(os.getcwd(), config.model.checkpoint_dir,
                                         config.data.ckpt_category, str(config.model.ckpt))) #['model_state_dict']
    unet.load_state_dict(checkpoint)
    unet = torch.nn.DataParallel(unet)
    unet.to(config.model.device)
    unet.eval()

    resnet = Resnet(config).to(config.model.device)
    resnet.eval()

    test_dataset = SHOMRI(
        root_dir=config.data.data_dir,
        patch_size=patch_size,
        is_train=False,
        cache_rate=0.5,
    )

    labels_list, predictions, vol_anomaly_maps = [], [], []

    with torch.no_grad():
        for idx in range(len(test_dataset)):
            sample_dict = test_dataset[idx]
            volume   = sample_dict["image"]        # (C, 128, 128, 128)
            label    = sample_dict["label"]
            filename = sample_dict["filename"]

            patches, locations = extract_patches(volume, patch_size)
            # patches: (N, C, pd, ph, pw) on CPU

            anomaly_patches, recon_patches = run_inference_on_patches(
                patches, unet, resnet, config
            )

            anomaly_vol = reconstruct_volume(anomaly_patches, locations, volume.shape, patch_size)
            recon_vol   = reconstruct_volume(recon_patches,   locations, volume.shape, patch_size)

            # score
            pred_flat = anomaly_vol.reshape(1, -1)
            # pred_soft = F.softmax(pred_flat, dim=1)
            k         = 500 #min(500, pred_flat.numel())
            k_max, _  = pred_flat.topk(k, largest=True)
            score     = k_max.sum().item()

            int_label = 0 if label == 'good' else 1
            labels_list.append(int_label)
            predictions.append(score)
            vol_anomaly_maps.append(anomaly_vol.unsqueeze(0))

            # print(f"[{idx+1}/{len(test_dataset)}] {filename}  score={score:.4f}  label={label}")

    threshold, _ = metric(labels_list, predictions, vol_anomaly_maps, [])
    print(threshold)
    if threshold is None:
        threshold = threshold_otsu(np.array(predictions))

    save_histogram(predictions, labels_list, save_dir, threshold=threshold)

    # re-run to save (or store during loop above if memory allows)
    with torch.no_grad():
        for idx in range(len(test_dataset)):
            sample_dict = test_dataset[idx]
            volume   = sample_dict["image"]
            label    = sample_dict["label"]
            filename = sample_dict["filename"]

            patches, locations = extract_patches(volume, patch_size)
            anomaly_patches, recon_patches = run_inference_on_patches(patches, unet, resnet, config)
            anomaly_vol = reconstruct_volume(anomaly_patches, locations, volume.shape, patch_size)
            recon_vol   = reconstruct_volume(recon_patches,   locations, volume.shape, patch_size)

            save_nifti_with_prediction(
                anomaly_map    = anomaly_vol,
                reconstruction = recon_vol,
                original_data  = volume,
                filename       = filename,
                save_dir       = save_dir,
                score          = predictions[idx],
                threshold      = threshold,
                label          = label,
            )

    print("Done")


def parse_args():
    parser = argparse.ArgumentParser('MDPS')
    parser.add_argument('-cfg', '--config', help='config file')
    args, unknowns = parser.parse_known_args()
    return args


if __name__ == "__main__":
    seed = 42
    torch.cuda.empty_cache()
    args = parse_args()
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print('*************')
    mdps(args)