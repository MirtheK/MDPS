# import os
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
# import argparse
# from omegaconf import OmegaConf
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from pathlib import Path
# import matplotlib.pyplot as plt
# from tqdm import tqdm
# import nibabel as nib

# from src.dataset import SHOMRI, SHOMRIGridPatches 
# from src.diffusion import diffusion_loss, sample, compute_alpha
# from src.models.unet import UNetModel
# from src.models.resnet import Resnet
# from src.compare import distance


# def extract_patches_3d(volume, patch_size, stride=None):
#     """Extract overlapping or non-overlapping patches from a 3D volume."""
#     if stride is None:
#         stride = patch_size
    
#     B, C, D, H, W = volume.shape
#     pd, ph, pw = patch_size
#     sd, sh, sw = stride
    
#     patches = []
#     locations = []
    
#     for d in range(0, D - pd + 1, sd):
#         for h in range(0, H - ph + 1, sh):
#             for w in range(0, W - pw + 1, sw):
#                 patch = volume[:, :, d:d+pd, h:h+ph, w:w+pw]
#                 patches.append(patch)
#                 locations.append((d, h, w))
    
#     patches = torch.cat(patches, dim=0)
#     return patches, locations


# def reconstruct_from_patches_3d(patches, locations, volume_shape, patch_size, stride=None):
#     """Reconstruct full volume from patches using averaging for overlapping regions."""
#     if stride is None:
#         stride = patch_size
    
#     B, C, D, H, W = volume_shape
#     pd, ph, pw = patch_size
    
#     device = patches.device
#     reconstructed = torch.zeros(volume_shape, device=device)
#     counts = torch.zeros(volume_shape, device=device)
    
#     num_patches_per_volume = len(locations)
    
#     for i, (d, h, w) in enumerate(locations):
#         for b in range(B):
#             patch_idx = b * num_patches_per_volume + i
#             patch = patches[patch_idx]
            
#             reconstructed[b, :, d:d+pd, h:h+ph, w:w+pw] += patch
#             counts[b, :, d:d+pd, h:h+ph, w:w+pw] += 1
    
#     reconstructed = reconstructed / counts.clamp(min=1)
#     return reconstructed


# def save_nifti_with_prediction(anomaly_map, reconstruction, original_data, filename, 
#                                 save_dir, affine, score, threshold, label):
#     """Save anomaly map and reconstruction with prediction outcome in filename."""
#     save_dir = Path(save_dir)
#     save_dir.mkdir(parents=True, exist_ok=True)
    
#     # Determine prediction outcome
#     prediction = "DEFECTIVE" if score > threshold else "GOOD"
#     true_label = label.upper() if isinstance(label, str) else ("NORMAL" if label == 0 else "ABNORMAL") #
#     correct = "CORRECT" if prediction == true_label else "WRONG"
    
#     # Create descriptive filename
#     base_name = Path(filename).stem.replace('.nii', '')
    
#     # Save anomaly map
#     anomaly_filename = f"{base_name}_{true_label}_{prediction}_{correct}_anomaly.nii.gz"
#     anomaly_path = save_dir / anomaly_filename
    
#     anomaly_np = anomaly_map.squeeze().cpu().numpy()
#     anomaly_nii = nib.Nifti1Image(anomaly_np, affine if affine is not None else np.eye(4))
#     nib.save(anomaly_nii, str(anomaly_path))
    
#     # Save reconstruction
#     recon_filename = f"{base_name}_{true_label}_{prediction}_{correct}_recon.nii.gz"
#     recon_path = save_dir / recon_filename
    
#     recon_np = reconstruction.squeeze().cpu().numpy()
#     recon_nii = nib.Nifti1Image(recon_np, affine if affine is not None else np.eye(4))
#     nib.save(recon_nii, str(recon_path))
    
#     # Save original for reference
#     orig_filename = f"{base_name}_{true_label}_{prediction}_{correct}_original.nii.gz"
#     orig_path = save_dir / orig_filename
    
#     orig_np = original_data.squeeze().cpu().numpy()
#     orig_nii = nib.Nifti1Image(orig_np, affine if affine is not None else np.eye(4))
#     nib.save(orig_nii, str(orig_path))
    
#     return anomaly_path, recon_path, orig_path


# def save_histogram(scores, labels, epoch, save_dir, threshold=None):
#     """Save histogram of anomaly scores separated by label."""
#     save_dir = Path(save_dir)
#     save_dir.mkdir(parents=True, exist_ok=True)
    
#     normal_scores = [s for s, l in zip(scores, labels) if l == 0]
#     abnormal_scores = [s for s, l in zip(scores, labels) if l == 1]
    
#     plt.figure(figsize=(10, 6))
    
#     if normal_scores:
#         plt.hist(normal_scores, bins=30, alpha=0.5, label='Normal', color='blue', edgecolor='black')
#     if abnormal_scores:
#         plt.hist(abnormal_scores, bins=30, alpha=0.5, label='Abnormal', color='red', edgecolor='black')
    
#     if threshold is not None:
#         plt.axvline(x=threshold, color='green', linestyle='--', linewidth=2, label=f'Threshold: {threshold:.3f}')
    
#     plt.xlabel('Anomaly Score', fontsize=12)
#     plt.ylabel('Frequency', fontsize=12)
#     plt.title(f'Anomaly Score Distribution - Epoch {epoch}', fontsize=14)
#     plt.legend()
#     plt.grid(True, alpha=0.3)
    
#     save_path = save_dir / f'histogram_epoch_{epoch}.png'
#     plt.savefig(save_path, dpi=150, bbox_inches='tight')
#     plt.close()
    
#     return save_path


# def evaluate_model(model, resnet, test_loader, config, epoch, save_dir):
#     """Run evaluation and save results."""
#     model.eval()
#     resnet.eval()
    
#     labels_list = []
#     predictions = []
#     anomaly_map_list = []
#     gt_list = []
#     save_data = []
    
#     patch_size = tuple(config.data.get('patch_size', [64, 64, 64]))
#     stride = config.data.get('inference_stride', patch_size)
    
#     print(f"\n{'='*60}")
#     print(f"Running Evaluation - Epoch {epoch}")
#     print(f"{'='*60}")
    
#     with torch.no_grad():
#         for batch_idx, batch_data in enumerate(tqdm(test_loader, desc="Evaluating")):
#             data_dict = batch_data
#             data = data_dict['image'].to(config.model.device)
            
#             # Ensure correct shape
#             if len(data.shape) == 4:
#                 data = data.unsqueeze(1)
            
#             labels = [data_dict['label']] if isinstance(data_dict['label'], str) else data_dict['label']
#             targets = data_dict.get('mask', torch.zeros_like(data))
#             filenames = data_dict["filename"]

#             # if hasattr(filenames, "meta"):
#             #     filenames = filenames.meta.get("filename_or_obj", "unknown")
#             affine = data_dict.get('affine', None)
            
#             # Extract patches
#             B, C, D, H, W = data.shape
#             patches, locations = extract_patches_3d(data, patch_size, stride)
            
#             # Process patches in batches
#             patch_batch_size = config.data.get('inference_patch_batch_size', 4)
#             num_patches = len(locations)
#             all_reconstructed_patches = []
            
#             for repeat_idx in range(config.model.test_repeat):
#                 for patch_start in range(0, num_patches, patch_batch_size):
#                     patch_end = min(patch_start + patch_batch_size, num_patches)
#                     patch_batch = patches[patch_start:patch_end]
                    
#                     # Run diffusion inference
#                     test_steps = torch.Tensor([config.model.test_steps]).type(torch.int64).to(config.model.device)
#                     at = compute_alpha(test_steps.long(), config)
                    
#                     noisy_image = at.sqrt() * patch_batch + (1 - at).sqrt() * torch.randn_like(patch_batch)
#                     seq = range(0, config.model.test_steps, config.model.skip)
#                     reconstructed = sample(patch_batch, noisy_image, seq, model, config, w=config.model.w)
#                     data_reconstructed = reconstructed[-1]
#                     all_reconstructed_patches.append(data_reconstructed)
            
#             # Concatenate and average across repeats
#             all_reconstructed_patches = torch.cat(all_reconstructed_patches, dim=0)
#             num_repeats = config.model.test_repeat
#             patches_per_repeat = len(locations)
            
#             averaged_patches = []
#             for loc_idx in range(patches_per_repeat):
#                 repeat_patches = []
#                 for repeat_idx in range(num_repeats):
#                     patch_idx = repeat_idx * patches_per_repeat + loc_idx
#                     repeat_patches.append(all_reconstructed_patches[patch_idx:patch_idx+1])
#                 averaged_patch = torch.mean(torch.cat(repeat_patches, dim=0), dim=0, keepdim=True)
#                 averaged_patches.append(averaged_patch)
            
#             averaged_patches = torch.cat(averaged_patches, dim=0)
            
#             # Reconstruct full volume
#             reconstructed_volume = reconstruct_from_patches_3d(
#                 averaged_patches, 
#                 locations, 
#                 data.shape, 
#                 patch_size, 
#                 stride
#             )
            
#             reconstructed_volume = reconstructed_volume[:, :, :D, :H, :W]
            
#             # Compute anomaly map
#             anomaly_map = distance(reconstructed_volume, data, resnet, config) / 2
            
#             # Store results
#             anomaly_map_list.append(anomaly_map)
#             gt_list.append(targets)
            
#             # Process predictions
#             for idx, (pred, label) in enumerate(zip(anomaly_map, labels)):
#                 labels_list.append(0 if label == 'good' else 1)
#                 k = 500
#                 pred_flat = pred.reshape(1, -1)
#                 # pred_soft = F.softmax(pred_flat, dim=1)
#                 k_max, _ = pred_flat.topk(k, largest=True)
#                 score = torch.sum(k_max).item()
#                 predictions.append(score)
                
#                 filename = filenames[idx]
#                 current_affine = affine[idx].cpu().numpy() if affine is not None and idx < len(affine) else None
                
#                 save_data.append({
#                     'anomaly_map': anomaly_map[idx],
#                     'reconstructed_volume': reconstructed_volume[idx],
#                     'original_data': data[idx],
#                     'filename': filename,
#                     'affine': current_affine,
#                     'score': score,
#                     'label': label
#                 })
    
#     # Compute threshold (simple percentile-based)
#     normal_scores = [s for s, l in zip(predictions, labels_list) if l == 0]
#     abnormal_scores = [s for s, l in zip(predictions, labels_list) if l == 1]
    
#     if len(normal_scores) > 0 and len(abnormal_scores) > 0:
#         threshold = np.percentile(normal_scores, 95)  # 95th percentile of normal scores
#     elif len(normal_scores) > 0:
#         threshold = np.percentile(normal_scores, 95)
#     else:
#         threshold = np.median(predictions)
    
#     # Calculate metrics
#     correct_predictions = sum([
#         1 for score, label in zip(predictions, labels_list)
#         if (score > threshold and label == 1) or (score <= threshold and label == 0)
#     ])
#     accuracy = correct_predictions / len(labels_list) if len(labels_list) > 0 else 0
    
#     print(f"\nEvaluation Results - Epoch {epoch}")
#     print(f"{'='*60}")
#     print(f"Total samples: {len(labels_list)}")
#     print(f"Normal samples: {labels_list.count(0)}")
#     print(f"Abnormal samples: {labels_list.count(1)}")
#     print(f"Threshold: {threshold:.4f}")
#     print(f"Accuracy: {accuracy:.4f} ({correct_predictions}/{len(labels_list)})")
#     print(f"Normal score range: [{min(normal_scores):.4f}, {max(normal_scores):.4f}]")
#     if abnormal_scores:
#         print(f"Abnormal score range: [{min(abnormal_scores):.4f}, {max(abnormal_scores):.4f}]")
#     print(f"{'='*60}\n")
    
#     # Create epoch-specific save directory
#     epoch_save_dir = Path(save_dir) / f"epoch_{epoch}"
#     epoch_save_dir.mkdir(parents=True, exist_ok=True)
    
#     # Save histogram
#     hist_path = save_histogram(predictions, labels_list, epoch, epoch_save_dir, threshold)
#     print(f"Saved histogram to: {hist_path}")
    
#     # Save anomaly maps and reconstructions
#     print(f"Saving {len(save_data)} volumes...")
#     for data in tqdm(save_data, desc="Saving volumes"):
#         save_nifti_with_prediction(
#             data['anomaly_map'],
#             data['reconstructed_volume'],
#             data['original_data'],
#             data['filename'],
#             epoch_save_dir,
#             data['affine'],
#             data['score'],
#             threshold,
#             data['label']
#         )
    
#     print(f"All results saved to: {epoch_save_dir}\n")
    
#     model.train()
#     return accuracy, threshold, predictions, labels_list


# def trainer(args):
#     config = OmegaConf.load(args.config)
    
#     # Setup
#     patch_size = config.data.get('patch_size', (64, 64, 64))
#     print(f"Training with patch size: {patch_size}")
    
#     # Model setup
#     model = UNetModel(
#         patch_size[0],
#         32, 
#         dropout=0.0, 
#         n_heads=4,
#         in_channels=config.data.imput_channel
#     )
#     print("Num params: ", sum(p.numel() for p in model.parameters()))
#     model = model.to(config.model.device)
#     model = model.float()
#     model.train()
    
#     # Resnet for anomaly scoring (used during evaluation)
#     resnet = Resnet(config).to(config.model.device)
#     resnet.eval()
    
#     optimizer = torch.optim.Adam(
#         model.parameters(), 
#         lr=config.model.learning_rate, 
#         weight_decay=config.model.weight_decay
#     )
    
#     # Training dataset
#     if config.data.name == 'SHOMRI':
#         patches_per_volume = config.data.get('patches_per_volume', 4)
        
#         train_dataset = SHOMRI(
#             root_dir=config.data.data_dir,
#             patch_size=patch_size,
#             patches_per_volume=patches_per_volume,
#             is_train=True,
#             cache_rate=1.0,
#         )
        
#         # Test dataset for evaluation
#         test_dataset = SHOMRI(
#             root_dir=config.data.data_dir,
#             patch_size=patch_size,
#             patches_per_volume=1,
#             is_train=False,
#             cache_rate=0.5,
#         )
    
#     trainloader = torch.utils.data.DataLoader(
#         train_dataset,
#         batch_size=config.data.batch_size,
#         shuffle=True,
#         num_workers=config.model.num_workers,
#         drop_last=True,
#         pin_memory=True,
#     )
    
#     testloader = torch.utils.data.DataLoader(
#         test_dataset,
#         batch_size=1,  # Process one volume at a time for evaluation
#         shuffle=False,
#         num_workers=2,
#         drop_last=False,
#     )
    
#     print(f"Training dataset size: {len(train_dataset)} patches")
#     print(f"Test dataset size: {len(test_dataset)} volumes")
#     print(f"Batches per epoch: {len(trainloader)}")
    
#     # Create directories
#     checkpoint_dir = Path(config.model.checkpoint_dir)
#     checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
#     model_save_dir = checkpoint_dir / config.data.category
#     model_save_dir.mkdir(parents=True, exist_ok=True)
    
#     eval_save_dir = Path('evaluation_results') / config.data.category
#     eval_save_dir.mkdir(parents=True, exist_ok=True)
    
#     # Training loop
#     scaler = torch.amp.GradScaler('cuda')
#     best_accuracy = 0.0
    
#     # Track metrics
#     training_losses = []
#     eval_accuracies = []
#     eval_epochs = []
    
#     for epoch in range(config.model.epochs):
#         model.train()
#         epoch_loss = 0.0
#         num_batches = 0
        
#         # Training
#         pbar = tqdm(trainloader, desc=f"Epoch {epoch}")
#         for step, batch in enumerate(pbar):
#             t = torch.randint(
#                 1, 
#                 config.model.diffusion_steps, 
#                 (batch[0].shape[0],), 
#                 device=config.model.device
#             ).long()
            
#             optimizer.zero_grad()
            
#             with torch.amp.autocast('cuda'):
#                 loss = diffusion_loss(model, batch[0], t, config)
            
#             scaler.scale(loss).backward()
#             scaler.step(optimizer)
#             scaler.update()
            
#             epoch_loss += loss.item()
#             num_batches += 1
            
#             pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
#         avg_loss = epoch_loss / num_batches
#         training_losses.append(avg_loss)
#         print(f"Epoch {epoch} completed | Average Loss: {avg_loss:.4f}")
        
#         # Evaluation (every N epochs)
#         eval_frequency = config.model.get('eval_frequency', 10)
#         if epoch % eval_frequency == 0 or epoch == config.model.epochs - 1:
#             accuracy, threshold, predictions, labels = evaluate_model(
#                 model, resnet, testloader, config, epoch, eval_save_dir
#             )
#             eval_accuracies.append(accuracy)
#             eval_epochs.append(epoch)
            
#             # Save best model
#             if accuracy > best_accuracy:
#                 best_accuracy = accuracy
#                 best_model_path = model_save_dir / 'best_model.pt'
#                 torch.save({
#                     'epoch': epoch,
#                     'model_state_dict': model.state_dict(),
#                     'optimizer_state_dict': optimizer.state_dict(),
#                     'loss': avg_loss,
#                     'accuracy': accuracy,
#                     'threshold': threshold,
#                 }, best_model_path)
#                 print(f"✓ New best model saved! Accuracy: {accuracy:.4f}")
        
#         # Save checkpoint
#         if epoch % config.model.epochs_checkpoint == 0:
#             checkpoint_path = model_save_dir / f"epoch_{epoch}.pt"
#             torch.save({
#                 'epoch': epoch,
#                 'model_state_dict': model.state_dict(),
#                 'optimizer_state_dict': optimizer.state_dict(),
#                 'loss': avg_loss,
#             }, checkpoint_path)
#             print(f"Checkpoint saved: {checkpoint_path}")
            
#             torch.cuda.empty_cache()
    
#     # Save training curves
#     plt.figure(figsize=(12, 5))
    
#     plt.subplot(1, 2, 1)
#     plt.plot(training_losses)
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.title('Training Loss')
#     plt.grid(True, alpha=0.3)
    
#     plt.subplot(1, 2, 2)
#     plt.plot(eval_epochs, eval_accuracies, marker='o')
#     plt.xlabel('Epoch')
#     plt.ylabel('Accuracy')
#     plt.title('Evaluation Accuracy')
#     plt.grid(True, alpha=0.3)
    
#     plt.tight_layout()
#     curves_path = eval_save_dir / 'training_curves.png'
#     plt.savefig(curves_path, dpi=150, bbox_inches='tight')
#     plt.close()
#     print(f"Training curves saved to: {curves_path}")
    
#     print(f"\nTraining complete!")
#     print(f"Best accuracy: {best_accuracy:.4f}")


# def parse_args():
#     parser = argparse.ArgumentParser('MDPS')    
#     parser.add_argument('-cfg', '--config', help='config file')
#     args, unknowns = parser.parse_known_args()
#     return args


# if __name__ == "__main__":
#     seed = 42
#     torch.cuda.empty_cache()
#     args = parse_args()
#     torch.manual_seed(seed)
#     np.random.seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(seed)
    
#     trainer(args)
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
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
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

from src.dataset import SHOMRI, SHOMRIGridPatches 
from src.diffusion import diffusion_loss, sample, compute_alpha
from src.models.unet import UNetModel
from src.models.resnet import Resnet
from src.compare import distance


def extract_patches_3d(volume, patch_size, stride=None):
    """Extract overlapping or non-overlapping patches from a 3D volume."""
    if stride is None:
        stride = patch_size
    
    B, C, D, H, W = volume.shape
    pd, ph, pw = patch_size
    sd, sh, sw = stride
    
    patches = []
    locations = []
    
    for d in range(0, D - pd + 1, sd):
        for h in range(0, H - ph + 1, sh):
            for w in range(0, W - pw + 1, sw):
                patch = volume[:, :, d:d+pd, h:h+ph, w:w+pw]
                patches.append(patch)
                locations.append((d, h, w))
    
    patches = torch.cat(patches, dim=0)
    return patches, locations


def reconstruct_from_patches_3d(patches, locations, volume_shape, patch_size, stride=None):
    """Reconstruct full volume from patches using averaging for overlapping regions."""
    if stride is None:
        stride = patch_size
    
    B, C, D, H, W = volume_shape
    pd, ph, pw = patch_size
    
    device = patches.device
    reconstructed = torch.zeros(volume_shape, device=device)
    counts = torch.zeros(volume_shape, device=device)
    
    num_patches_per_volume = len(locations)
    
    for i, (d, h, w) in enumerate(locations):
        for b in range(B):
            patch_idx = b * num_patches_per_volume + i
            patch = patches[patch_idx]
            
            reconstructed[b, :, d:d+pd, h:h+ph, w:w+pw] += patch
            counts[b, :, d:d+pd, h:h+ph, w:w+pw] += 1
    
    reconstructed = reconstructed / counts.clamp(min=1)
    return reconstructed


def save_nifti_with_prediction(anomaly_map, reconstruction, original_data, filename, 
                                save_dir, affine, score, threshold, label):
    """Save anomaly map and reconstruction with prediction outcome in filename."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    prediction = "DEFECTIVE" if score > threshold else "GOOD"
    true_label = label.upper() if isinstance(label, str) else ("NORMAL" if label == 0 else "ABNORMAL")
    correct = "CORRECT" if prediction == true_label else "WRONG"
    
    base_name = Path(filename).stem.replace('.nii', '')
    
    # Save anomaly map
    anomaly_filename = f"{base_name}_{true_label}_{prediction}_{correct}_anomaly.nii.gz"
    anomaly_path = save_dir / anomaly_filename
    anomaly_np = anomaly_map.squeeze().cpu().numpy()
    anomaly_nii = nib.Nifti1Image(anomaly_np, affine if affine is not None else np.eye(4))
    nib.save(anomaly_nii, str(anomaly_path))
    
    # Save reconstruction
    recon_filename = f"{base_name}_{true_label}_{prediction}_{correct}_recon.nii.gz"
    recon_path = save_dir / recon_filename
    recon_np = reconstruction.squeeze().cpu().numpy()
    recon_nii = nib.Nifti1Image(recon_np, affine if affine is not None else np.eye(4))
    nib.save(recon_nii, str(recon_path))
    
    # Save original
    orig_filename = f"{base_name}_{true_label}_{prediction}_{correct}_original.nii.gz"
    orig_path = save_dir / orig_filename
    orig_np = original_data.squeeze().cpu().numpy()
    orig_nii = nib.Nifti1Image(orig_np, affine if affine is not None else np.eye(4))
    nib.save(orig_nii, str(orig_path))
    
    return anomaly_path, recon_path, orig_path


def save_histogram(scores, labels, epoch, save_dir, threshold=None):
    """Save histogram of anomaly scores separated by label."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    normal_scores = [s for s, l in zip(scores, labels) if l == 0]
    abnormal_scores = [s for s, l in zip(scores, labels) if l == 1]
    
    plt.figure(figsize=(10, 6))
    
    if normal_scores:
        plt.hist(normal_scores, bins=30, alpha=0.5, label='Normal', color='blue', edgecolor='black')
    if abnormal_scores:
        plt.hist(abnormal_scores, bins=30, alpha=0.5, label='Abnormal', color='red', edgecolor='black')
    
    if threshold is not None:
        plt.axvline(x=threshold, color='green', linestyle='--', linewidth=2, label=f'Threshold: {threshold:.3f}')
    
    plt.xlabel('Anomaly Score', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title(f'Anomaly Score Distribution - Epoch {epoch}', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_path = save_dir / f'histogram_epoch_{epoch}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return save_path


def log_roc_curve(writer, labels_list, predictions, epoch):
    """Render ROC curve and log it to TensorBoard as an image."""
    fpr, tpr, _ = roc_curve(labels_list, predictions)
    auroc = roc_auc_score(labels_list, predictions)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {auroc:.3f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curve - Epoch {epoch}')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    writer.add_figure("Eval/ROC_curve", fig, epoch)
    plt.close(fig)


def log_anomaly_map_slices(writer, anomaly_map, original, epoch, tag_prefix="Sample"):
    """Log central axial slice of anomaly map and original to TensorBoard."""
    # anomaly_map: (C, D, H, W) or (D, H, W)
    am = anomaly_map.squeeze().cpu()
    orig = original.squeeze().cpu()

    if am.dim() == 3:
        mid = am.shape[0] // 2
        am_slice = am[mid]       # (H, W)
        orig_slice = orig[mid]
    else:
        am_slice = am
        orig_slice = orig

    # Normalize to [0, 1] for display
    def norm(x):
        mn, mx = x.min(), x.max()
        return (x - mn) / (mx - mn + 1e-8)

    am_slice = norm(am_slice).unsqueeze(0).unsqueeze(0)       # (1, 1, H, W)
    orig_slice = norm(orig_slice).unsqueeze(0).unsqueeze(0)

    combined = torch.cat([orig_slice, am_slice], dim=3)        # side by side
    writer.add_images(f"AnomalyMap/{tag_prefix}_orig|anomaly", combined, epoch)


def cohen_d(group1, group2):
    """Cohen's d score separation between two score distributions."""
    if len(group1) == 0 or len(group2) == 0:
        return 0.0
    m1, m2 = np.mean(group1), np.mean(group2)
    s1, s2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    n1, n2 = len(group1), len(group2)
    pooled_std = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    return (m2 - m1) / (pooled_std + 1e-8)


def _visualize_noisy_input(original_patch, noisy_patch, epoch, save_dir):
    save_dir = Path(save_dir) / "noise_check"
    save_dir.mkdir(parents=True, exist_ok=True)

    # Take first item in batch, squeeze channel dim
    orig = original_patch[0, 0].cpu().float()   # (D, H, W)
    noisy = noisy_patch[0, 0].cpu().float()

    affine = nib.load("/projects/prjs1633/anomaly_detection/SHOMRI/zero_mask.nii.gz").affine

    # Save original
    orig_filename = f"original.nii.gz"
    orig_path = save_dir / orig_filename
    orig_np = orig.squeeze().cpu().numpy()
    orig_nii = nib.Nifti1Image(orig_np, affine)
    nib.save(orig_nii, str(orig_filename))

    # Save noisy
    noisy_filename = f"noisy.nii.gz"
    noisy_path = save_dir / noisy_filename
    noisy_np = noisy.squeeze().cpu().numpy()
    noisy_nii = nib.Nifti1Image(noisy_np, affine)
    nib.save(noisy_nii, str(noisy_filename))

    # Print SNR to console so you don't even need to open the image
    signal = orig.std().item()
    noise = (noisy - orig).std().item()
    snr = signal / (noise + 1e-8)
    print(f"[Noise check] epoch={epoch} | signal_std={signal:.4f} | "
          f"noise_std={noise:.4f} | SNR={snr:.4f}")
    print(f"  → If SNR << 1.0, test_steps is too high (model sees near-pure noise)")



def evaluate_model(model, resnet, test_loader, config, epoch, save_dir, writer=None):
    """Run evaluation and save results."""
    model.eval()
    resnet.eval()
    
    labels_list = []
    predictions = []
    anomaly_map_list = []
    gt_list = []
    save_data = []
    
    patch_size = tuple(config.data.get('patch_size', [64, 64, 64]))
    stride = config.data.get('inference_stride', patch_size)
    
    print(f"\n{'='*60}")
    print(f"Running Evaluation - Epoch {epoch}")
    print(f"{'='*60}")
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(tqdm(test_loader, desc="Evaluating")):
            data_dict = batch_data
            data = data_dict['image'].to(config.model.device)
            
            if len(data.shape) == 4:
                data = data.unsqueeze(1)
            
            labels = [data_dict['label']] if isinstance(data_dict['label'], str) else data_dict['label']
            targets = data_dict.get('mask', torch.zeros_like(data))
            filenames = data_dict["filename"]
            affine = data_dict.get('affine', None)
            
            B, C, D, H, W = data.shape
            patches, locations = extract_patches_3d(data, patch_size, stride)
            
            patch_batch_size = config.data.get('inference_patch_batch_size', 4)
            num_patches = len(locations)
            all_reconstructed_patches = []
            
            for repeat_idx in range(config.model.test_repeat):
                for patch_start in range(0, num_patches, patch_batch_size):
                    patch_end = min(patch_start + patch_batch_size, num_patches)
                    patch_batch = patches[patch_start:patch_end]
                    
                    test_steps = torch.Tensor([config.model.test_steps]).type(torch.int64).to(config.model.device)
                    at = compute_alpha(test_steps.long(), config)
                    
                    noisy_image = at.sqrt() * patch_batch + (1 - at).sqrt() * torch.randn_like(patch_batch)

                    # Add check for trianing
                    if batch_idx == 0 and patch_start == 0 and repeat_idx == 0:
                        _visualize_noisy_input(patch_batch, noisy_image, epoch, save_dir)


                    seq = range(0, config.model.test_steps, config.model.skip)
                    reconstructed = sample(patch_batch, noisy_image, seq, model, config, w=config.model.w)
                    data_reconstructed = reconstructed[-1]
                    all_reconstructed_patches.append(data_reconstructed)
            
            all_reconstructed_patches = torch.cat(all_reconstructed_patches, dim=0)
            num_repeats = config.model.test_repeat
            patches_per_repeat = len(locations)
            
            averaged_patches = []
            for loc_idx in range(patches_per_repeat):
                repeat_patches = []
                for repeat_idx in range(num_repeats):
                    patch_idx = repeat_idx * patches_per_repeat + loc_idx
                    repeat_patches.append(all_reconstructed_patches[patch_idx:patch_idx+1])
                averaged_patch = torch.mean(torch.cat(repeat_patches, dim=0), dim=0, keepdim=True)
                averaged_patches.append(averaged_patch)
            
            averaged_patches = torch.cat(averaged_patches, dim=0)
            
            reconstructed_volume = reconstruct_from_patches_3d(
                averaged_patches, locations, data.shape, patch_size, stride
            )
            reconstructed_volume = reconstructed_volume[:, :, :D, :H, :W]
            
            anomaly_map = distance(reconstructed_volume, data, resnet, config) / 2
            
            anomaly_map_list.append(anomaly_map)
            gt_list.append(targets)
            
            for idx, (pred, label) in enumerate(zip(anomaly_map, labels)):
                labels_list.append(0 if label == 'good' else 1)
                # k = 500
                pred_flat = pred.reshape(1, -1)
                # k_max, _ = pred_flat.topk(k, largest=True)
                # score = torch.sum(k_max).item()
                # pred = (pred - pred.mean()) / (pred.std() + 1e-8)
                k = int(0.02 * pred_flat.numel())
                score = torch.mean(pred_flat.topk(k).values).item()
                

                predictions.append(score)
                
                filename = filenames[idx]
                current_affine = affine[idx].cpu().numpy() if affine is not None and idx < len(affine) else None
                
                save_data.append({
                    'anomaly_map': anomaly_map[idx],
                    'reconstructed_volume': reconstructed_volume[idx],
                    'original_data': data[idx],
                    'filename': filename,
                    'affine': current_affine,
                    'score': score,
                    'label': label
                })
    
    # Compute threshold
    normal_scores = [s for s, l in zip(predictions, labels_list) if l == 0]
    abnormal_scores = [s for s, l in zip(predictions, labels_list) if l == 1]
    
    if len(normal_scores) > 0 and len(abnormal_scores) > 0:
        threshold = np.percentile(normal_scores, 95)
    elif len(normal_scores) > 0:
        threshold = np.percentile(normal_scores, 95)
    else:
        threshold = np.median(predictions)
    
    correct_predictions = sum([
        1 for score, label in zip(predictions, labels_list)
        if (score > threshold and label == 1) or (score <= threshold and label == 0)
    ])
    accuracy = correct_predictions / len(labels_list) if len(labels_list) > 0 else 0
    
    # AUROC and AUPRC
    has_both_classes = len(set(labels_list)) > 1
    auroc = roc_auc_score(labels_list, predictions) if has_both_classes else float('nan')
    auprc = average_precision_score(labels_list, predictions) if has_both_classes else float('nan')
    d_score = cohen_d(normal_scores, abnormal_scores)

    print(f"\nEvaluation Results - Epoch {epoch}")
    print(f"{'='*60}")
    print(f"Total samples:    {len(labels_list)}")
    print(f"Normal samples:   {labels_list.count(0)}")
    print(f"Abnormal samples: {labels_list.count(1)}")
    print(f"Threshold:        {threshold:.4f}")
    print(f"Accuracy:         {accuracy:.4f} ({correct_predictions}/{len(labels_list)})")
    print(f"AUROC:            {auroc:.4f}" if has_both_classes else "AUROC:            N/A (single class)")
    print(f"AUPRC:            {auprc:.4f}" if has_both_classes else "AUPRC:            N/A")
    print(f"Cohen's d:        {d_score:.4f}")
    print(f"Normal score range:   [{min(normal_scores):.4f}, {max(normal_scores):.4f}]")
    if abnormal_scores:
        print(f"Abnormal score range: [{min(abnormal_scores):.4f}, {max(abnormal_scores):.4f}]")
    print(f"{'='*60}\n")
    
    # ── TensorBoard logging ────────────────────────────────────────────────
    if writer is not None:
        writer.add_scalar("Eval/accuracy",  accuracy,  epoch)
        writer.add_scalar("Eval/threshold", threshold, epoch)

        if has_both_classes:
            writer.add_scalar("Eval/AUROC",    auroc,   epoch)
            writer.add_scalar("Eval/AUPRC",    auprc,   epoch)
            writer.add_scalar("Eval/cohens_d", d_score, epoch)
            log_roc_curve(writer, labels_list, predictions, epoch)

        # Score distributions as histograms
        if normal_scores:
            writer.add_histogram("Scores/normal",   torch.tensor(normal_scores),   epoch)
        if abnormal_scores:
            writer.add_histogram("Scores/abnormal", torch.tensor(abnormal_scores), epoch)

        # Mean scores per class
        if normal_scores:
            writer.add_scalar("Scores/mean_normal",   np.mean(normal_scores),   epoch)
        if abnormal_scores:
            writer.add_scalar("Scores/mean_abnormal", np.mean(abnormal_scores), epoch)

        # Log one anomaly map slice per class (first found)
        logged_normal = logged_abnormal = False
        for d in save_data:
            lbl = 0 if d['label'] == 'good' else 1
            if lbl == 0 and not logged_normal:
                log_anomaly_map_slices(writer, d['anomaly_map'], d['original_data'],
                                       epoch, tag_prefix="Normal")
                logged_normal = True
            elif lbl == 1 and not logged_abnormal:
                log_anomaly_map_slices(writer, d['anomaly_map'], d['original_data'],
                                       epoch, tag_prefix="Abnormal")
                logged_abnormal = True
            if logged_normal and logged_abnormal:
                break
    # ──────────────────────────────────────────────────────────────────────

    # Create epoch-specific save directory
    epoch_save_dir = Path(save_dir) / f"epoch_{epoch}"
    epoch_save_dir.mkdir(parents=True, exist_ok=True)
    
    hist_path = save_histogram(predictions, labels_list, epoch, epoch_save_dir, threshold)
    print(f"Saved histogram to: {hist_path}")
    
    print(f"Saving {len(save_data)} volumes...")
    for d in tqdm(save_data, desc="Saving volumes"):
        save_nifti_with_prediction(
            d['anomaly_map'], d['reconstructed_volume'], d['original_data'],
            d['filename'], epoch_save_dir, d['affine'], d['score'], threshold, d['label']
        )
    
    print(f"All results saved to: {epoch_save_dir}\n")
    
    model.train()
    return accuracy, threshold, predictions, labels_list, auroc, auprc


def trainer(args):
    config = OmegaConf.load(args.config)
    
    patch_size = config.data.get('patch_size', (64, 64, 64))
    print(f"Training with patch size: {patch_size}")
    
    # ── TensorBoard writer ─────────────────────────────────────────────────
    writer = SummaryWriter(log_dir=f"evaluation_results/{config.data.category}")
    print(f"TensorBoard logs → evaluation_results/{config.data.category}")
    print(f"  Launch with:  tensorboard --logdir evaluation_results/")
    # ──────────────────────────────────────────────────────────────────────

    # Model setup
    model = UNetModel(
        patch_size[0],
        32, 
        dropout=0.0, 
        n_heads=4,
        in_channels=config.data.imput_channel
    )
    print("Num params: ", sum(p.numel() for p in model.parameters()))
    model = model.to(config.model.device)
    model = model.float()
    model.train()
    
    resnet = Resnet(config).to(config.model.device)
    resnet.eval()
    
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config.model.learning_rate, 
        weight_decay=config.model.weight_decay
    )
    
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
    )
    
    testloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        drop_last=False,
    )
    
    print(f"Training dataset size: {len(train_dataset)} patches")
    print(f"Test dataset size:     {len(test_dataset)} volumes")
    print(f"Batches per epoch:     {len(trainloader)}")
    
    checkpoint_dir = Path(config.model.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    model_save_dir = checkpoint_dir / config.data.category
    model_save_dir.mkdir(parents=True, exist_ok=True)
    
    eval_save_dir = Path('evaluation_results') / config.data.category
    eval_save_dir.mkdir(parents=True, exist_ok=True)
    
    scaler = torch.amp.GradScaler('cuda')
    best_accuracy = 0.0
    global_step = 0
    
    training_losses = []
    eval_accuracies = []
    eval_epochs = []
    
    for epoch in range(config.model.epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(trainloader, desc=f"Epoch {epoch}")
        for step, batch in enumerate(pbar):
            t = torch.randint(
                1, 
                config.model.diffusion_steps, 
                (batch[0].shape[0],), 
                device=config.model.device
            ).long()
            
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                loss = diffusion_loss(model, batch[0], t, config)
            
            scaler.scale(loss).backward()

            # ── Gradient norm (unscaled) ───────────────────────────────────
            scaler.unscale_(optimizer)
            grad_norm = sum(
                p.grad.norm() ** 2 for p in model.parameters() if p.grad is not None
            ) ** 0.5
            # ──────────────────────────────────────────────────────────────

            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            num_batches += 1
            global_step += 1

            # ── Per-step TB logging ────────────────────────────────────────
            writer.add_scalar("Loss/step",          loss.item(),  global_step)
            writer.add_scalar("Gradients/norm",     grad_norm,    global_step)
            writer.add_scalar("LR",                 optimizer.param_groups[0]['lr'], global_step)
            writer.add_scalar("GPU/memory_GB",
                              torch.cuda.max_memory_allocated() / 1e9, global_step)
            # ──────────────────────────────────────────────────────────────
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'grad': f'{grad_norm:.3f}'})
        
        avg_loss = epoch_loss / num_batches
        training_losses.append(avg_loss)

        # ── Per-epoch TB logging ───────────────────────────────────────────
        writer.add_scalar("Loss/epoch", avg_loss, epoch)
        # ──────────────────────────────────────────────────────────────────

        print(f"Epoch {epoch} completed | Average Loss: {avg_loss:.4f}")
        
        eval_frequency = config.model.get('eval_frequency', 10)
        if epoch % eval_frequency == 0 or epoch == config.model.epochs - 1:
            accuracy, threshold, preds, labels, auroc, auprc = evaluate_model(
                model, resnet, testloader, config, epoch, eval_save_dir, writer=writer
            )
            eval_accuracies.append(accuracy)
            eval_epochs.append(epoch)
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model_path = model_save_dir / 'best_model.pt'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_loss,
                    'accuracy': accuracy,
                    'auroc': auroc,
                    'auprc': auprc,
                    'threshold': threshold,
                }, best_model_path)
                print(f"✓ New best model saved! Accuracy: {accuracy:.4f} | AUROC: {auroc:.4f}")
        
        if epoch % config.model.epochs_checkpoint == 0:
            checkpoint_path = model_save_dir / f"epoch_{epoch}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")
            torch.cuda.empty_cache()
    
    writer.close()

    # Save training curves (also kept as files alongside TB)
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(training_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(eval_epochs, eval_accuracies, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Evaluation Accuracy')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    curves_path = eval_save_dir / 'training_curves.png'
    plt.savefig(curves_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved to: {curves_path}")
    
    print(f"\nTraining complete!")
    print(f"Best accuracy: {best_accuracy:.4f}")


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
    
    trainer(args)