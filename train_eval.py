import argparse
from omegaconf import OmegaConf
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import nibabel as nib
import csv
from skimage.filters import threshold_otsu
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

from src.dataset import SHOMRI, SHOMRIGridPatches
from src.diffusion import diffusion_loss, sample, sample_mask, compute_alpha
from src.models.unet import UNetModel
from src.models.resnet import Resnet
from src.compare import distance

import time


class CSVLogger:
    """
    Buffered CSV logger — accumulates rows in memory and flushes to disk
    periodically. Avoids opening/closing the file on every training step.
    """
    def __init__(self, path, flush_every=50):
        self.path            = Path(path)
        self.flush_every     = flush_every
        self._buffer         = []
        self._header_written = self.path.exists()

    def append(self, row: dict):
        self._buffer.append(row)
        if len(self._buffer) >= self.flush_every:
            self.flush()

    def flush(self):
        if not self._buffer:
            return
        write_header = not self._header_written
        with open(self.path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(self._buffer[0].keys()))
            if write_header:
                writer.writeheader()
                self._header_written = True
            writer.writerows(self._buffer)
        self._buffer.clear()

    def __del__(self):
        self.flush()


def write_csv(path, rows: list):
    """Write a list of dicts to a CSV file (overwrites)."""
    if not rows:
        return
    path = Path(path)
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

def append_csv(path, row: dict):
    """Single-row append (used for eval summary — infrequent, no buffering needed)."""
    path = Path(path)
    write_header = not path.exists()
    with open(path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def extract_patches_3d(volume, patch_size, stride=None):
    """
    Extract patches from a 3D volume **one volume at a time**.

    Args:
        volume: tensor of shape (1, C, D, H, W)  — must be a single volume.
    Returns:
        patches:   (num_locs, C, pd, ph, pw)
        locations: list of (d, h, w) tuples
    """
    if stride is None:
        stride = patch_size

    if volume.shape[0] != 1:
        raise ValueError(
            f"extract_patches_3d expects a single volume (B=1), got B={volume.shape[0]}. "
            "Call it inside a loop over the batch dimension."
        )

    # Strip MONAI MetaTensor to plain Tensor to avoid dispatch issues
    if type(volume) is not torch.Tensor:
        volume = volume.as_tensor()

    _, C, D, H, W = volume.shape
    pd, ph, pw    = patch_size
    sd, sh, sw    = stride

    patches   = []
    locations = []

    for d in range(0, D - pd + 1, sd):
        for h in range(0, H - ph + 1, sh):
            for w in range(0, W - pw + 1, sw):
                # squeeze out batch dim → (C, pd, ph, pw), then unsqueeze back → (1, C, ...)
                patches.append(volume[0:1, :, d:d+pd, h:h+ph, w:w+pw])
                locations.append((d, h, w))

    # (num_locs, C, pd, ph, pw)
    return torch.cat(patches, dim=0), locations


def reconstruct_from_patches_3d(patches, locations, volume_shape, patch_size, stride=None):
    """
    Reconstruct a single volume from patches.
    patches:      (num_locs, C, pd, ph, pw)
    volume_shape: (1, C, D, H, W)
    Returns:      (1, C, D, H, W) plain torch.Tensor
    """
    if stride is None:
        stride = patch_size

    if type(patches) is not torch.Tensor:
        patches = patches.as_tensor()

    _, C, D, H, W = volume_shape
    pd, ph, pw     = patch_size
    device         = patches.device

    reconstructed = torch.zeros((1, C, D, H, W), device=device)
    counts        = torch.zeros((1, C, D, H, W), device=device)

    for loc_idx, (d, h, w) in enumerate(locations):
        reconstructed[0, :, d:d+pd, h:h+ph, w:w+pw] += patches[loc_idx]
        counts[0, :, d:d+pd, h:h+ph, w:w+pw]         += 1

    return reconstructed / counts.clamp(min=1)

def save_nifti_with_prediction(anomaly_map, reconstruction, original_data,
                                filename, save_dir, affine, score, threshold, label):
    save_dir   = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    prediction = "DEFECTIVE" if score > threshold else "GOOD"
    true_label = label.upper() if isinstance(label, str) else ("NORMAL" if label == 0 else "ABNORMAL")
    correct    = "CORRECT" if prediction == true_label else "WRONG"
    base_name  = Path(filename).stem.replace('.nii', '')

    for tag, data_tensor in [
        ("anomaly",  anomaly_map),
        ("recon",    reconstruction),
        ("original", original_data),
    ]:
        out_path = save_dir / f"{base_name}_{true_label}_{prediction}_{correct}_{tag}.nii.gz"
        arr = data_tensor.squeeze().cpu().numpy()
        nii = nib.Nifti1Image(arr, affine if affine is not None else np.eye(4))
        nib.save(nii, str(out_path))

def save_histogram(scores, labels, epoch, save_dir, threshold=None):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    normal_scores   = [s for s, l in zip(scores, labels) if l == 0]
    abnormal_scores = [s for s, l in zip(scores, labels) if l == 1]

    plt.figure(figsize=(10, 6))
    if normal_scores:
        plt.hist(normal_scores,   bins=30, alpha=0.5, label='Normal',   color='blue', edgecolor='black')
    if abnormal_scores:
        plt.hist(abnormal_scores, bins=30, alpha=0.5, label='Abnormal', color='red',  edgecolor='black')
    if threshold is not None:
        plt.axvline(x=threshold, color='green', linestyle='--', linewidth=2,
                    label=f'Threshold: {threshold:.3f}')
    plt.xlabel('Anomaly Score', fontsize=12)
    plt.ylabel('Frequency',     fontsize=12)
    plt.title(f'Anomaly Score Distribution - Epoch {epoch}', fontsize=14)
    plt.legend(); plt.grid(True, alpha=0.3)
    save_path = save_dir / f'histogram_epoch_{epoch}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    return save_path


def save_roc_curve(labels_list, predictions, epoch, save_dir):
    save_dir    = Path(save_dir)
    fpr, tpr, _ = roc_curve(labels_list, predictions)
    auroc       = roc_auc_score(labels_list, predictions)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {auroc:.3f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    ax.set_xlim([0.0, 1.0]); ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curve - Epoch {epoch}')
    ax.legend(loc='lower right'); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_path = save_dir / f'roc_epoch_{epoch}.png'
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return save_path


def save_training_curves(log_csv_path, save_dir):
    save_dir = Path(save_dir)
    steps, losses, grads = [], [], []
    with open(log_csv_path, newline='') as f:
        for row in csv.DictReader(f):
            steps.append(int(row['global_step']))
            losses.append(float(row['loss']))
            grads.append(float(row['grad_norm']))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(steps, losses, linewidth=0.8)
    axes[0].set_xlabel('Global Step'); axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss'); axes[0].grid(True, alpha=0.3)
    axes[1].plot(steps, grads, linewidth=0.8, color='orange')
    axes[1].set_xlabel('Global Step'); axes[1].set_ylabel('Gradient Norm')
    axes[1].set_title('Gradient Norm'); axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    out = save_dir / 'training_curves.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return out


def save_eval_curves(eval_csv_path, save_dir):
    save_dir = Path(save_dir)
    epochs, accs, aurocs, auprcs = [], [], [], []
    with open(eval_csv_path, newline='') as f:
        for row in csv.DictReader(f):
            epochs.append(int(row['epoch']))
            accs.append(float(row['accuracy']))
            aurocs.append(float(row['auroc']) if row['auroc'] != 'nan' else float('nan'))
            auprcs.append(float(row['auprc']) if row['auprc'] != 'nan' else float('nan'))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, vals, title, color in zip(
        axes, [accs, aurocs, auprcs],
        ['Accuracy', 'AUROC', 'AUPRC'],
        ['steelblue', 'darkorange', 'green'],
    ):
        ax.plot(epochs, vals, marker='o', color=color)
        ax.set_xlabel('Epoch'); ax.set_ylabel(title)
        ax.set_title(title);    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = save_dir / 'eval_curves.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return out


def cohen_d(group1, group2):
    if len(group1) == 0 or len(group2) == 0:
        return 0.0
    m1, m2 = np.mean(group1), np.mean(group2)
    s1, s2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    n1, n2 = len(group1), len(group2)
    pooled = np.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2) / (n1+n2-2))
    return (m2 - m1) / (pooled + 1e-8)


def evaluate_model(model, resnet, test_loader, config, epoch, save_dir, eval_csv_path):
    model.eval()
    resnet.eval()
    torch.cuda.empty_cache()

    patch_size       = tuple(config.data.get('patch_size'))
    stride           = config.data.get('inference_stride')
    patch_batch_size = config.data.get('inference_patch_batch_size')
    mask_steps       = config.model.get('mask_steps')

    times = {'data_load': 0.0, 'stage1_mask': 0.0, 'stage2_sample': 0.0,
             'resnet': 0.0, 'reconstruct': 0.0}
    t_loop_start = time.perf_counter()

    print(f"\n{'='*60}\nRunning Evaluation - Epoch {epoch}\n{'='*60}")

    mask_init = None
    if mask_steps > 0:
        all_coarse_maps = []
        seq_mask = range(0, mask_steps, config.model.skip_mask)
        at_mask  = compute_alpha(
            torch.tensor([mask_steps], device=config.model.device), config
        )

        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                for stage1_batch in tqdm(test_loader):
                    s1_data = stage1_batch['image'].to(config.model.device, non_blocking=True)
                    if s1_data.dim() == 4:
                        s1_data = s1_data.unsqueeze(1)
                    B1 = s1_data.shape[0]

                    for vi in range(B1):
                        vol = s1_data[vi : vi + 1]          # (1, C, D, H, W)
                        patches, locations = extract_patches_3d(vol, patch_size, stride)

                        all_recon = []
                        for ps in range(0, len(locations), patch_batch_size):
                            pb    = patches[ps : ps + patch_batch_size]
                            noisy = at_mask.sqrt() * pb + (1 - at_mask).sqrt() * torch.randn_like(pb)
                            recon_result = sample(pb, noisy, seq_mask, model, config,
                                                  w=config.model.get('w_mask'))
                            if isinstance(recon_result, torch.Tensor):
                                recon_tensor = recon_result
                            else:
                                recon_tensor = recon_result[-1]
                            all_recon.append(recon_tensor)

                        all_recon_tensor = torch.cat(all_recon, dim=0)   # (num_locs, C, pd, ph, pw)
                        recon_vol  = reconstruct_from_patches_3d(
                            all_recon_tensor, locations, vol.shape, patch_size, stride
                        )                                                  # (1, C, D, H, W)
                        coarse_map = distance(recon_vol, vol, resnet, config) / 2
                        all_coarse_maps.append(coarse_map)                # (1, ...)

        # Global threshold across entire test set — all_coarse_maps[i] is volume i
        all_coarse_tensor = torch.cat(all_coarse_maps, dim=0)             # (N, ...)
        pixel_min      = all_coarse_tensor.min()
        pixel_max      = all_coarse_tensor.max()
        lam            = config.model.get('mask0_thresholds', 0.15)
        threshold_mask = pixel_min + lam * (pixel_max - pixel_min)
        mask_init      = (all_coarse_tensor > threshold_mask).float()     # (N, ...)
        print(f"Global mask threshold: {threshold_mask:.4f} | "
              f"Masked voxels: {mask_init.mean():.3%}")

        times['stage1_mask'] = time.perf_counter() - t_loop_start

    labels_list  = []
    predictions  = []
    save_data    = []
    seq          = range(0, config.model.test_steps, config.model.skip)
    vol_idx      = 0   # global volume counter to index into mask_init

    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            for batch_data in tqdm(test_loader, desc="Evaluating"):

                torch.cuda.synchronize()
                t0 = time.perf_counter()

                data = batch_data['image'].to(config.model.device, non_blocking=True)
                if data.dim() == 4:
                    data = data.unsqueeze(1)

                labels    = (batch_data['label']
                             if isinstance(batch_data['label'], (list, tuple))
                             else [batch_data['label']])
                filenames = batch_data['filename']
                affine    = batch_data.get('affine', None)
                B, C, D, H, W = data.shape

                torch.cuda.synchronize()
                t1 = time.perf_counter()
                times['data_load'] += t1 - t0

                # Process each volume in the batch individually
                for vi in range(B):
                    vol   = data[vi : vi + 1]               # (1, C, D, H, W)
                    label = labels[vi]
                    fname = filenames[vi]
                    cur_affine = (affine[vi].cpu().numpy()
                                  if affine is not None and vi < len(affine) else None)

                    patches, locations = extract_patches_3d(vol, patch_size, stride)
                    num_locs = len(locations)

                    if mask_steps > 0 and mask_init is not None:
                        mask         = mask_init[vol_idx : vol_idx + 1].to(config.model.device)
                        mask_patches, _ = extract_patches_3d(mask, patch_size, stride)

                    torch.cuda.synchronize()
                    t1b = time.perf_counter()

                    all_recon_patches = []
                    for _ in range(config.model.test_repeat):
                        repeat_recon = []
                        for ps in range(0, num_locs, patch_batch_size):
                            pb = patches[ps : ps + patch_batch_size]

                            if mask_steps > 0 and mask_init is not None:
                                mb           = mask_patches[ps : ps + patch_batch_size]
                                recon_result = sample_mask(pb, mb, seq, model, config,
                                                           w=config.model.w)
                                if isinstance(recon_result, torch.Tensor):
                                    recon_tensor = recon_result
                                else:
                                    recon_tensor = recon_result[-1]
                                repeat_recon.append(recon_tensor)
                            else:
                                at    = compute_alpha(
                                    torch.tensor([config.model.test_steps],
                                                 device=config.model.device), config
                                )
                                noisy = at.sqrt() * pb + (1 - at).sqrt() * torch.randn_like(pb)
                                recon_result = sample(pb, noisy, seq, model, config,
                                                      w=config.model.w)
                                if isinstance(recon_result, torch.Tensor):
                                    recon_tensor = recon_result
                                else:
                                    recon_tensor = recon_result[-1]
                                repeat_recon.append(recon_tensor)

                        all_recon_patches.append(torch.cat(repeat_recon, dim=0))

                    torch.cuda.synchronize()
                    t2 = time.perf_counter()
                    times['stage2_sample'] += t2 - t1b

                    # Average across repeats: (test_repeat, num_locs, C, pd, ph, pw)
                    averaged_patches = torch.stack(all_recon_patches, dim=0).mean(dim=0)

                    recon_vol = reconstruct_from_patches_3d(
                        averaged_patches, locations, vol.shape, patch_size, stride
                    )                                                      # (1, C, D, H, W)
                    recon_vol = recon_vol[:, :, :D, :H, :W]

                    torch.cuda.synchronize()
                    t3 = time.perf_counter()
                    times['reconstruct'] += t3 - t2

                    anomaly_map_vol = distance(recon_vol, vol, resnet, config) / 2

                    torch.cuda.synchronize()
                    t4 = time.perf_counter()
                    times['resnet'] += t4 - t3

                    # Score: softmax → top-k sum  (matches test.py exactly)
                    pred_flat = anomaly_map_vol.reshape(1, -1)
                    pred_soft = F.softmax(pred_flat, dim=1)
                    k         = min(500, pred_soft.numel())
                    k_max, _  = pred_soft.topk(k, largest=True)
                    score     = k_max.sum().item()

                    labels_list.append(0 if label == 'good' else 1)
                    predictions.append(score)
                    save_data.append({
                        'anomaly_map':          anomaly_map_vol[0],
                        'reconstructed_volume': recon_vol[0],
                        'original_data':        vol[0],
                        'filename':             fname,
                        'affine':               cur_affine,
                        'score':                score,
                        'label':                label,
                    })

                    vol_idx += 1

    # Print timing summary
    total = sum(times.values())
    print(f"\n{'='*60}")
    print(f"Evaluation Timing Breakdown ({len(labels_list)} volumes)")
    print(f"{'='*60}")
    for name, t in times.items():
        print(f"  {name:<20s}: {t:6.1f}s  ({100*t/total:5.1f}%)")
    print(f"  {'TOTAL':<20s}: {total:6.1f}s")
    print(f"  {'Wall clock':<20s}: {time.perf_counter()-t_loop_start:6.1f}s")
    print(f"{'='*60}\n")

    # Metrics
    normal_scores   = [s for s, l in zip(predictions, labels_list) if l == 0]
    abnormal_scores = [s for s, l in zip(predictions, labels_list) if l == 1]
    threshold       = threshold_otsu(np.array(predictions))

    correct_predictions = sum(
        1 for score, label in zip(predictions, labels_list)
        if (score > threshold) == (label == 1)
    )
    accuracy  = correct_predictions / len(labels_list) if labels_list else 0
    has_both  = len(set(labels_list)) > 1
    auroc     = roc_auc_score(labels_list, predictions)           if has_both else float('nan')
    auprc     = average_precision_score(labels_list, predictions) if has_both else float('nan')
    d_score   = cohen_d(normal_scores, abnormal_scores)

    print(f"\nEvaluation Results - Epoch {epoch}")
    print(f"{'='*60}")
    print(f"Total / Normal / Abnormal: {len(labels_list)} / {labels_list.count(0)} / {labels_list.count(1)}")
    print(f"Threshold:  {threshold:.4f}")
    print(f"Accuracy:   {accuracy:.4f} ({correct_predictions}/{len(labels_list)})")
    print(f"AUROC:      {auroc:.4f}" if has_both else "AUROC:      N/A")
    print(f"AUPRC:      {auprc:.4f}" if has_both else "AUPRC:      N/A")
    print(f"Cohen's d:  {d_score:.4f}")
    if normal_scores:
        print(f"Normal score range:   [{min(normal_scores):.4f}, {max(normal_scores):.4f}]")
    if abnormal_scores:
        print(f"Abnormal score range: [{min(abnormal_scores):.4f}, {max(abnormal_scores):.4f}]")
    print(f"{'='*60}\n")

    # Save outputs
    epoch_save_dir = Path(save_dir) / f"epoch_{epoch}"
    epoch_save_dir.mkdir(parents=True, exist_ok=True)

    save_histogram(predictions, labels_list, epoch, epoch_save_dir, threshold)
    if has_both:
        save_roc_curve(labels_list, predictions, epoch, epoch_save_dir)

    write_csv(epoch_save_dir / 'sample_scores.csv', [
        {
            'filename':   d['filename'],
            'label':      d['label'],
            'score':      f"{d['score']:.6f}",
            'threshold':  f"{threshold:.6f}",
            'prediction': 'DEFECTIVE' if d['score'] > threshold else 'GOOD',
        }
        for d in save_data
    ])

    append_csv(eval_csv_path, {
        'epoch':         epoch,
        'accuracy':      f"{accuracy:.6f}",
        'auroc':         f"{auroc:.6f}",
        'auprc':         f"{auprc:.6f}",
        'cohens_d':      f"{d_score:.6f}",
        'threshold':     f"{threshold:.6f}",
        'n_normal':      labels_list.count(0),
        'n_abnormal':    labels_list.count(1),
        'mean_normal':   f"{np.mean(normal_scores):.6f}"   if normal_scores   else 'nan',
        'mean_abnormal': f"{np.mean(abnormal_scores):.6f}" if abnormal_scores else 'nan',
    })

    print(f"Saving {len(save_data)} volumes...")
    for d in tqdm(save_data, desc="Saving volumes"):
        save_nifti_with_prediction(
            d['anomaly_map'], d['reconstructed_volume'], d['original_data'],
            d['filename'], epoch_save_dir, d['affine'],
            d['score'], threshold, d['label'],
        )

    print(f"All results saved to: {epoch_save_dir}\n")
    model.train()
    return accuracy, threshold, predictions, labels_list, auroc, auprc


def trainer(args):
    config     = OmegaConf.load(args.config)
    patch_size = config.data.get('patch_size')
    print(f"Training with patch size: {patch_size}")

    # Model
    model = UNetModel(
        patch_size[0], 32,
        dropout=0.0, n_heads=4,
        in_channels=config.data.imput_channel,
    )
    print("Num params:", sum(p.numel() for p in model.parameters()))
    model = model.to(config.model.device).float()
    model.train()

    resnet = Resnet(config).to(config.model.device)
    resnet.eval()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.model.learning_rate,
        weight_decay=config.model.weight_decay,
    )

    # Datasets
    if config.data.name == 'SHOMRI':
        patches_per_volume = config.data.get('patches_per_volume', 4)
        train_dataset = SHOMRI(
            root_dir=config.data.data_dir,
            patch_size=patch_size,
            patches_per_volume=patches_per_volume,
            is_train=True,
            cache_rate=1.0,
        )
        test_dataset = SHOMRI(
            root_dir=config.data.data_dir,
            patch_size=patch_size,
            patches_per_volume=1,
            is_train=False,
            cache_rate=0.5,
        )

    trainloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.model.num_workers,
        drop_last=True,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )
    testloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=2,
        drop_last=False,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    print(f"Training dataset size: {len(train_dataset)} patches")
    print(f"Test dataset size:     {len(test_dataset)} volumes")
    print(f"Batches per epoch:     {len(trainloader)}")

    # Paths
    checkpoint_dir = Path(config.model.checkpoint_dir)
    model_save_dir = checkpoint_dir / config.data.category
    model_save_dir.mkdir(parents=True, exist_ok=True)

    log_dir = Path('/projects/prjs1633/anomaly_detection/MDPS_traininglog') / config.data.category
    log_dir.mkdir(parents=True, exist_ok=True)

    step_csv_path = log_dir / 'train_steps.csv'
    eval_csv_path = log_dir / 'eval_summary.csv'

    step_logger = CSVLogger(step_csv_path, flush_every=50)

    # Training loop
    scaler        = torch.amp.GradScaler('cuda')
    best_accuracy = 0.0
    global_step   = 0

    for epoch in range(config.model.epochs):
        model.train()
        epoch_loss  = 0.0
        num_batches = 0

        pbar = tqdm(trainloader, desc=f"Epoch {epoch}")
        for step, batch in enumerate(pbar):
            data = batch[0].to(config.model.device, non_blocking=True)

            t = torch.randint(
                1, config.model.diffusion_steps,
                (data.shape[0],), device=config.model.device,
            ).long()
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda'):
                loss = diffusion_loss(model, data, t, config)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=float('inf')
            )

            scaler.step(optimizer)
            scaler.update()

            loss_val     = loss.item()
            grad_val     = grad_norm.item()
            epoch_loss  += loss_val
            num_batches += 1
            global_step += 1

            step_logger.append({
                'global_step': global_step,
                'epoch':       epoch,
                'step':        step,
                'loss':        f"{loss_val:.6f}",
                'grad_norm':   f"{grad_val:.6f}",
                'lr':          f"{optimizer.param_groups[0]['lr']:.8f}",
                'gpu_mem_GB':  f"{torch.cuda.max_memory_allocated() / 1e9:.3f}",
            })

            pbar.set_postfix({'loss': f'{loss_val:.4f}', 'grad': f'{grad_val:.3f}'})

        step_logger.flush()

        avg_loss = epoch_loss / num_batches
        print(f"Epoch {epoch} completed | Average Loss: {avg_loss:.4f}")

        # Evaluation
        eval_frequency = config.model.get('eval_frequency')
        if epoch % eval_frequency == 0 or epoch == config.model.epochs - 1:
            accuracy, threshold, preds, labels, auroc, auprc = evaluate_model(
                model, resnet, testloader, config, epoch, log_dir, eval_csv_path
            )
            save_eval_curves(eval_csv_path, log_dir)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                torch.save({
                    'epoch':                epoch,
                    'model_state_dict':     model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss':                 avg_loss,
                    'accuracy':             accuracy,
                    'auroc':                auroc,
                    'auprc':                auprc,
                    'threshold':            threshold,
                }, model_save_dir / 'best_model.pt')
                print(f"✓ New best model saved! Accuracy: {accuracy:.4f} | AUROC: {auroc:.4f}")

        # Periodic checkpoint
        if epoch % config.model.epochs_checkpoint == 0:
            torch.save({
                'epoch':                epoch,
                'model_state_dict':     model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss':                 avg_loss,
            }, model_save_dir / f"epoch_{epoch}.pt")
            print(f"Checkpoint saved: epoch_{epoch}.pt")
            torch.cuda.empty_cache()

    # Final training curve plot
    step_logger.flush()
    curves_path = save_training_curves(step_csv_path, log_dir)
    print(f"Training curves saved to: {curves_path}")
    print(f"\nTraining complete! Best accuracy: {best_accuracy:.4f}")


def parse_args():
    parser = argparse.ArgumentParser('MDPS')
    parser.add_argument('-cfg', '--config', help='config file')
    args, _ = parser.parse_known_args()
    return args


if __name__ == "__main__":
    seed = 42
    torch.cuda.empty_cache()
    args = parse_args()
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    trainer(args)