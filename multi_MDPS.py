# import os
# import argparse
# from omegaconf import OmegaConf
# import numpy as np
# import torch
# import torch.nn.functional as F
# from torchvision import transforms
# from src.models.unet import UNetModel
# from src.models.resnet import Resnet
# from src.dataset import SHOMRI
# from src.metrics import metric
# from src.compare import distance
# import glob
# from src.diffusion import sample, sample_mask, compute_alpha

# os.environ['CUDA_VISIBLE_DEVICES'] = "0"


# def extract_patches_3d(volume, patch_size, stride=None):
#     """
#     Extract overlapping or non-overlapping patches from a 3D volume.
    
#     Args:
#         volume: (B, C, D, H, W) tensor
#         patch_size: tuple (pd, ph, pw)
#         stride: tuple (sd, sh, sw) or None for non-overlapping
    
#     Returns:
#         patches: tensor of shape (B, N, C, pd, ph, pw) where N is number of patches
#         locations: list of (d, h, w) positions for each patch
#     """
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
    
#     # Stack all patches: (B*N, C, pd, ph, pw)
#     patches = torch.cat(patches, dim=0)
#     return patches, locations


# def reconstruct_from_patches_3d(patches, locations, volume_shape, patch_size, stride=None):
#     """
#     Reconstruct full volume from patches using averaging for overlapping regions.
    
#     Args:
#         patches: (B*N, C, pd, ph, pw) tensor
#         locations: list of (d, h, w) positions
#         volume_shape: (B, C, D, H, W)
#         patch_size: tuple (pd, ph, pw)
#         stride: tuple or None
    
#     Returns:
#         reconstructed volume: (B, C, D, H, W)
#     """
#     if stride is None:
#         stride = patch_size
    
#     B, C, D, H, W = volume_shape
#     pd, ph, pw = patch_size
    
#     # Initialize accumulation tensors
#     device = patches.device
#     reconstructed = torch.zeros(volume_shape, device=device)
#     counts = torch.zeros(volume_shape, device=device)
    
#     num_patches_per_volume = len(locations)
    
#     for i, (d, h, w) in enumerate(locations):
#         # Get patches for all volumes in batch
#         for b in range(B):
#             patch_idx = b * num_patches_per_volume + i
#             patch = patches[patch_idx]
            
#             reconstructed[b, :, d:d+pd, h:h+ph, w:w+pw] += patch
#             counts[b, :, d:d+pd, h:h+ph, w:w+pw] += 1
    
#     # Average overlapping regions
#     reconstructed = reconstructed / counts.clamp(min=1)
#     return reconstructed


# def mdps(args):
#     config = OmegaConf.load(args.config)
#     print(config.data.category)
    
#     # Determine if we're working with 3D medical images
#     is_3d = config.data.name == 'SHOMRI'
    
#     # Load model with appropriate patch size
#     if is_3d:
#         model_input_size = config.data.get('patch_size', [32, 32, 32])[0]
#     else:
#         model_input_size = config.data.image_size
    
#     unet = UNetModel(
#         model_input_size, 
#         32, 
#         dropout=0.0, 
#         n_heads=4,
#         in_channels=config.data.imput_channel
#     )
    
#     checkpoint_path = os.path.join(
#         os.getcwd(), 
#         config.model.checkpoint_dir, 
#         config.data.category, 
#         str(config.model.ckpt)
#     )
#     checkpoint = torch.load(checkpoint_path, weights_only=False)
    
#     # Handle checkpoint format
#     if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
#         state_dict = checkpoint['model_state_dict']
#     else:
#         state_dict = checkpoint
    
#     # Remove 'module.' prefix if exists
#     if list(state_dict.keys())[0].startswith('module.'):
#         state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
#     unet.load_state_dict(state_dict)
#     unet.to(config.model.device)
#     unet.eval()

#     # Load appropriate dataset
#     if config.data.name == 'MVTec':
#         test_dataset = MVTec(
#             root=config.data.data_dir,
#             category=config.data.category,
#             config=config,
#             is_train=False,
#         )
#     elif config.data.name == 'BTAD':
#         test_dataset = BTAD(
#             root=config.data.data_dir,
#             category=config.data.category,
#             config=config,
#             is_train=False,
#         )
#     elif config.data.name == 'SHOMRI':
#         patch_size = config.data.get('patch_size', [64, 64, 64])
#         test_dataset = SHOMRI(
#             root_dir=config.data.data_dir,
#             patch_size=tuple(patch_size),
#             patches_per_volume=1,  
#             is_train=False,
#             cache_rate=0.5,  
#         )
#     else:
#         raise ValueError(f"Unknown dataset: {config.data.name}")
        
#     testloader = torch.utils.data.DataLoader(
#         test_dataset,
#         batch_size=1 if is_3d else config.data.batch_size,
#         shuffle=False,
#         num_workers=config.model.num_workers,
#         drop_last=False,
#     )
    
#     resnet = Resnet(config).to(config.model.device)
#     resnet.eval()
    
#     labels_list = []
#     predictions = []
#     anomaly_map_list = []
#     gt_list = []
    
#     # Get patch size for 3D processing
#     if is_3d:
#         patch_size = tuple(config.data.get('patch_size', [64, 64, 64]))
#         stride = config.data.get('inference_stride', patch_size)
#         print(f"Using patch size: {patch_size}, stride: {stride}")
#         print(f"\nStarting evaluation on {len(testloader)} batches...")
    
#     if config.model.mask_steps == 0:
#         with torch.no_grad():
#             for batch_idx, batch_data in enumerate(testloader):
#                 if is_3d:
#                     data_dict = batch_data
#                     data = data_dict['image'].to(config.model.device)

#                     if len(data.shape) == 4:  # (B, D, H, W) - missing channel
#                         data = data.unsqueeze(1)  # -> (B, 1, D, H, W)
#                     elif len(data.shape) == 5:  # (B, C, D, H, W) - correct
#                         pass
#                     else:
#                         raise ValueError(f"Unexpected data shape: {data.shape}")


#                     labels = [data_dict['label']] if isinstance(data_dict['label'], str) else data_dict['label']
#                     targets = data_dict.get('mask', torch.zeros_like(data))
                    
#                     print(f"Processing batch {batch_idx+1}/{len(testloader)}: {labels}")
                    
#                     # Extract patches from full volume
#                     B, C, D, H, W = data.shape
#                     patches, locations = extract_patches_3d(data, patch_size, stride)
                    
#                     # Process patches in mini-batches to avoid OOM
#                     patch_batch_size = config.data.get('inference_patch_batch_size', 4)
#                     num_patches = len(locations)

#                     all_reconstructed_patches = []

#                     for i in range(config.model.test_repeat):
#                         for patch_start in range(0, num_patches, patch_batch_size):
#                             patch_end = min(patch_start + patch_batch_size, num_patches)
#                             patch_batch = patches[patch_start:patch_end]
                            
#                             # Run diffusion inference on patch batch
#                             test_steps = torch.Tensor([config.model.test_steps]).type(torch.int64).to(config.model.device)
#                             at = compute_alpha(test_steps.long(), config)
                            
#                             noisy_image = at.sqrt() * patch_batch + (1 - at).sqrt() * torch.randn_like(patch_batch)
#                             seq = range(0, config.model.test_steps, config.model.skip)
#                             reconstructed = sample(patch_batch, noisy_image, seq, unet, config, w=config.model.w)
#                             data_reconstructed = reconstructed[-1]
#                             all_reconstructed_patches.append(data_reconstructed)

#                     # Concatenate all patches from all repeats
#                     all_reconstructed_patches = torch.cat(all_reconstructed_patches, dim=0)

#                     # Average across test repeats
#                     num_repeats = config.model.test_repeat
#                     patches_per_repeat = len(locations)

#                     averaged_patches = []
#                     for loc_idx in range(patches_per_repeat):
#                         repeat_patches = []
#                         for repeat_idx in range(num_repeats):
#                             patch_idx = repeat_idx * patches_per_repeat + loc_idx
#                             repeat_patches.append(all_reconstructed_patches[patch_idx:patch_idx+1])
#                         averaged_patch = torch.mean(torch.cat(repeat_patches, dim=0), dim=0, keepdim=True)
#                         averaged_patches.append(averaged_patch)

#                     averaged_patches = torch.cat(averaged_patches, dim=0)
                    
#                     # Reconstruct full volume
#                     reconstructed_volume = reconstruct_from_patches_3d(
#                         averaged_patches, 
#                         locations, 
#                         data.shape, 
#                         patch_size, 
#                         stride
#                     )
                    
#                     reconstructed_volume = reconstructed_volume[:, :, :data.shape[2], :data.shape[3], :data.shape[4]]
#                     # Compute anomaly map
#                     anomaly_map = distance(reconstructed_volume, data, resnet, config) / 2
                    
#                 else:
#                     data, targets, labels = batch_data
#                     anomaly_batch = []
#                     data = data.to(config.model.device)
#                     test_steps = torch.Tensor([config.model.test_steps]).type(torch.int64).to(config.model.device)
#                     at = compute_alpha(test_steps.long(), config)
                    
#                     seq = range(0, config.model.test_steps, config.model.skip)
#                     for i in range(config.model.test_repeat):
#                         noisy_image = at.sqrt() * data + (1 - at).sqrt() * torch.randn_like(data)
#                         reconstructed = sample(data, noisy_image, seq, unet, config, w=config.model.w)
#                         data_reconstructed = reconstructed[-1]
#                         anomaly_map = distance(data_reconstructed, data, resnet, config) / 2
#                         anomaly_batch.append(anomaly_map.unsqueeze(0))
                    
#                     anomaly_batch = torch.cat(anomaly_batch, dim=0)
#                     anomaly_map = torch.mean(anomaly_batch, dim=0)
                    
#                     # Apply center crop for 2D datasets
#                     transform = transforms.Compose([
#                         transforms.CenterCrop((224)),
#                     ])
                    
#                     if config.data.name == 'MVTec':
#                         anomaly_map = transform(anomaly_map)
#                         targets = transform(targets)

#                 # Store results
#                 anomaly_map_list.append(anomaly_map)
#                 gt_list.append(targets)
                
#                 # Process predictions and labels
#                 for pred, label in zip(anomaly_map, labels):
#                     labels_list.append(0 if label == 'good' else 1)
#                     k = 500
#                     pred = pred.reshape(1, -1)
#                     pred = F.softmax(pred, dim=1)
#                     k_max, idx = pred.topk(k, largest=True)
#                     score = torch.sum(k_max)
#                     predictions.append(score.item())

#     # Print summary before computing metrics
#     print(f"\nEvaluation complete!")
#     print(f"Total samples: {len(labels_list)}")
#     print(f"Normal samples: {labels_list.count(0)}")
#     print(f"Abnormal samples: {labels_list.count(1)}")

#     if len(set(labels_list)) < 2:
#         raise ValueError(f"Only found {set(labels_list)} class(es) in test set. Need both normal and abnormal samples!")

#     # Compute metrics
#     threshold, _, _ = metric(labels_list, predictions, anomaly_map_list, gt_list)
#     print(f"Detection threshold: {threshold}")

    
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
#     mdps(args)

import os
import argparse
from omegaconf import OmegaConf
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from src.models.unet import UNetModel
from src.models.resnet import Resnet
from src.dataset import SHOMRI
from src.metrics import metric
from src.compare import distance
import glob
from src.diffusion import sample, sample_mask, compute_alpha
import nibabel as nib

os.environ['CUDA_VISIBLE_DEVICES'] = "0"


def extract_patches_3d(volume, patch_size, stride=None):
    """
    Extract overlapping or non-overlapping patches from a 3D volume.
    
    Args:
        volume: (B, C, D, H, W) tensor
        patch_size: tuple (pd, ph, pw)
        stride: tuple (sd, sh, sw) or None for non-overlapping
    
    Returns:
        patches: tensor of shape (B, N, C, pd, ph, pw) where N is number of patches
        locations: list of (d, h, w) positions for each patch
    """
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
    
    # Stack all patches: (B*N, C, pd, ph, pw)
    patches = torch.cat(patches, dim=0)
    return patches, locations


def reconstruct_from_patches_3d(patches, locations, volume_shape, patch_size, stride=None):
    """
    Reconstruct full volume from patches using averaging for overlapping regions.
    
    Args:
        patches: (B*N, C, pd, ph, pw) tensor
        locations: list of (d, h, w) positions
        volume_shape: (B, C, D, H, W)
        patch_size: tuple (pd, ph, pw)
        stride: tuple or None
    
    Returns:
        reconstructed volume: (B, C, D, H, W)
    """
    if stride is None:
        stride = patch_size
    
    B, C, D, H, W = volume_shape
    pd, ph, pw = patch_size
    
    # Initialize accumulation tensors
    device = patches.device
    reconstructed = torch.zeros(volume_shape, device=device)
    counts = torch.zeros(volume_shape, device=device)
    
    num_patches_per_volume = len(locations)
    
    for i, (d, h, w) in enumerate(locations):
        # Get patches for all volumes in batch
        for b in range(B):
            patch_idx = b * num_patches_per_volume + i
            patch = patches[patch_idx]
            
            reconstructed[b, :, d:d+pd, h:h+ph, w:w+pw] += patch
            counts[b, :, d:d+pd, h:h+ph, w:w+pw] += 1
    
    # Average overlapping regions
    reconstructed = reconstructed / counts.clamp(min=1)
    return reconstructed


def save_nifti(anomaly_map, reconstructed_volume, original_filename, save_dir, affine=None, prediction_score=None, threshold=None, true_label=None):
    """
    Save anomaly map as NIfTI file and prediction results as text file.
    
    Args:
        anomaly_map: (C, D, H, W) or (D, H, W) tensor
        original_filename: original image filename
        save_dir: directory to save the file
        affine: affine transformation matrix (optional)
        prediction_score: anomaly score (optional)
        threshold: detection threshold (optional)
        true_label: ground truth label (optional)
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Remove channel dimension if present
    if anomaly_map.dim() == 4:
        anomaly_map = anomaly_map.squeeze(0)
        reconstructed_volume = reconstructed_volume.squeeze(0)
    
    # Convert to numpy and ensure correct shape (D, H, W)
    anomaly_np = anomaly_map.cpu().numpy()
    recon_np = reconstructed_volume.cpu().numpy()

    affine = nib.load("/gpfs/work2/0/prjs1633/anomaly_detection/SHOMRI/one_mask.nii.gz").affine
    
    nifti_img = nib.Nifti1Image(anomaly_np, affine)
    recon_img =nib.Nifti1Image(recon_np, affine)
    
    # Create output filename
    base_name = os.path.splitext(os.path.splitext(original_filename)[0])[0]  # Remove .nii.gz
    output_path = os.path.join(save_dir, f"{base_name}_anomaly_map.nii.gz")
    output_path1 = os.path.join(save_dir, f"{base_name}_reconstruction.nii.gz")
    
    # Save NIfTI
    nib.save(nifti_img, output_path)
    nib.save(recon_img, output_path1)
    print(f"Saved anomaly map to: {output_path}")
    
    # Save prediction results if provided
    if prediction_score is not None:
        result_path = os.path.join(save_dir, f"{base_name}_prediction.txt")
        with open(result_path, 'w') as f:
            f.write(f"Filename: {original_filename}\n")
            f.write(f"Anomaly Score: {prediction_score:.6f}\n")
            
            if threshold is not None:
                predicted_label = "ANOMALOUS" if prediction_score > threshold else "NORMAL"
                f.write(f"Threshold: {threshold:.6f}\n")
                f.write(f"Predicted: {predicted_label}\n")
            
            if true_label is not None:
                f.write(f"True Label: {true_label}\n")
                
                if threshold is not None:
                    correct = (prediction_score > threshold) == (true_label != 'good' and true_label != 0)
                    f.write(f"Correct: {correct}\n")
        
        print(f"Saved prediction to: {result_path}")


def mdps(args):
    config = OmegaConf.load(args.config)
    print(config.data.category)
    
    is_3d = config.data.name == 'SHOMRI'
    
    # Load model with appropriate patch size
    if is_3d:
        model_input_size = config.data.get('patch_size', [32, 32, 32])[0]
    else:
        model_input_size = config.data.image_size
    
    unet = UNetModel(
        model_input_size, 
        32, 
        dropout=0.0, 
        n_heads=4,
        in_channels=config.data.imput_channel
    )
    
    checkpoint_path = os.path.join(
        os.getcwd(), 
        config.model.checkpoint_dir, 
        config.data.category, 
        str(config.model.ckpt)
    )
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    
    # Handle checkpoint format
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Remove 'module.' prefix if exists
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    unet.load_state_dict(state_dict)
    unet.to(config.model.device)
    unet.eval()

    # Load appropriate dataset
    
    if config.data.name == 'SHOMRI':
        patch_size = config.data.get('patch_size', [64, 64, 64])
        test_dataset = SHOMRI(
            root_dir=config.data.data_dir,
            patch_size=tuple(patch_size),
            patches_per_volume=1,  
            is_train=False,
            cache_rate=0.5,  
        )
    else:
        raise ValueError(f"Unknown dataset: {config.data.name}")
        
    testloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1 if is_3d else config.data.batch_size,
        shuffle=False,
        num_workers=config.model.num_workers,
        drop_last=False,
    )
    
    resnet = Resnet(config).to(config.model.device)
    resnet.eval()
    
    labels_list = []
    predictions = []
    anomaly_map_list = []
    reconstructions_list = []
    gt_list = []
    
    # For 3D: store data for later saving with threshold
    save_data_3d = []
    
    # Get patch size for 3D processing
    if is_3d:
        patch_size = tuple(config.data.get('patch_size', [64, 64, 64]))
        stride = config.data.get('inference_stride', patch_size)
        save_dir = config.data.get('save_dir', './anomaly_maps')
        print(f"Using patch size: {patch_size}, stride: {stride}")
        print(f"Saving anomaly maps to: {save_dir}")
        print(f"\nStarting evaluation on {len(testloader)} batches...")
    
    if config.model.mask_steps == 0:
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(testloader):
                if is_3d:
                    data_dict = batch_data
                    data = data_dict['image'].to(config.model.device)

                    if len(data.shape) == 4:  # (B, D, H, W) - missing channel
                        data = data.unsqueeze(1)  # -> (B, 1, D, H, W)
                    elif len(data.shape) == 5:  # (B, C, D, H, W) - correct
                        pass
                    else:
                        raise ValueError(f"Unexpected data shape: {data.shape}")

                    labels = [data_dict['label']] if isinstance(data_dict['label'], str) else data_dict['label']
                    targets = data_dict.get('mask', torch.zeros_like(data))
                    filenames = [data_dict['filename']] if isinstance(data_dict.get('filename'), str) else data_dict.get('filename', [f'volume_{batch_idx}.nii.gz'])
                    affine = data_dict.get('affine', None)
                    
                    print(f"Processing batch {batch_idx+1}/{len(testloader)}: {labels}")
                    
                    # Extract patches from full volume
                    B, C, D, H, W = data.shape
                    patches, locations = extract_patches_3d(data, patch_size, stride)
                    
                    # Process patches in mini-batches to avoid OOM
                    patch_batch_size = config.data.get('inference_patch_batch_size', 4)
                    num_patches = len(locations)

                    all_reconstructed_patches = []

                    for i in range(config.model.test_repeat):
                        for patch_start in range(0, num_patches, patch_batch_size):
                            patch_end = min(patch_start + patch_batch_size, num_patches)
                            patch_batch = patches[patch_start:patch_end]
                            
                            # Run diffusion inference on patch batch
                            test_steps = torch.Tensor([config.model.test_steps]).type(torch.int64).to(config.model.device)
                            at = compute_alpha(test_steps.long(), config)
                            
                            noisy_image = at.sqrt() * patch_batch + (1 - at).sqrt() * torch.randn_like(patch_batch)
                            seq = range(0, config.model.test_steps, config.model.skip)
                            reconstructed = sample(patch_batch, noisy_image, seq, unet, config, w=config.model.w)
                            data_reconstructed = reconstructed[-1]
                            all_reconstructed_patches.append(data_reconstructed)

                    # Concatenate all patches from all repeats
                    all_reconstructed_patches = torch.cat(all_reconstructed_patches, dim=0)

                    # Average across test repeats
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
                    
                    # Reconstruct full volume
                    reconstructed_volume = reconstruct_from_patches_3d(
                        averaged_patches, 
                        locations, 
                        data.shape, 
                        patch_size, 
                        stride
                    )
                    
                    reconstructed_volume = reconstructed_volume[:, :, :data.shape[2], :data.shape[3], :data.shape[4]]
                    reconstructions_list.append(reconstructed_volume)
                    # Compute anomaly map
                    anomaly_map = distance(reconstructed_volume, data, resnet, config) / 2
                    
                    # Store results
                    anomaly_map_list.append(anomaly_map)
                    gt_list.append(targets)
                    
                    # Process predictions and labels, and store for later saving
                    for idx, (pred, label) in enumerate(zip(anomaly_map, labels)):
                        labels_list.append(0 if label == 'good' else 1)
                        k = 500
                        pred_flat = pred.reshape(1, -1)
                        pred_soft = F.softmax(pred_flat, dim=1)
                        k_max, _ = pred_soft.topk(k, largest=True)
                        score = torch.sum(k_max).item()
                        predictions.append(score)
                        
                        # Store data for saving after threshold is computed
                        filename = filenames[idx] #if idx < len(filenames) else f'volume_{batch_idx}_{idx}.nii.gz'
                        current_affine = affine[idx].cpu().numpy() if affine is not None and idx < len(affine) else None
                        save_data_3d.append({
                            'anomaly_map': anomaly_map[idx],
                            'reconstructed_volume':reconstructions_list[idx],
                            'filename': filename,
                            'affine': current_affine,
                            'score': score,
                            'label': label
                        })
                    
                else:
                    pass
                #     data, targets, labels = batch_data
                #     anomaly_batch = []
                #     data = data.to(config.model.device)
                #     test_steps = torch.Tensor([config.model.test_steps]).type(torch.int64).to(config.model.device)
                #     at = compute_alpha(test_steps.long(), config)
                    
                #     seq = range(0, config.model.test_steps, config.model.skip)
                #     for i in range(config.model.test_repeat):
                #         noisy_image = at.sqrt() * data + (1 - at).sqrt() * torch.randn_like(data)
                #         reconstructed = sample(data, noisy_image, seq, unet, config, w=config.model.w)
                #         data_reconstructed = reconstructed[-1]
                #         anomaly_map = distance(data_reconstructed, data, resnet, config) / 2
                #         anomaly_batch.append(anomaly_map.unsqueeze(0))
                    
                #     anomaly_batch = torch.cat(anomaly_batch, dim=0)
                #     anomaly_map = torch.mean(anomaly_batch, dim=0)
                    
                #     # Apply center crop for 2D datasets
                #     transform = transforms.Compose([
                #         transforms.CenterCrop((224)),
                #     ])
                    
                #     if config.data.name == 'MVTec':
                #         anomaly_map = transform(anomaly_map)
                #         targets = transform(targets)

                # # Store results
                # anomaly_map_list.append(anomaly_map)
                # gt_list.append(targets)
                
                # # Process predictions and labels
                # for pred, label in zip(anomaly_map, labels):
                #     labels_list.append(0 if label == 'good' else 1)
                #     k = 500
                #     pred = pred.reshape(1, -1)
                #     pred = F.softmax(pred, dim=1)
                #     k_max, idx = pred.topk(k, largest=True)
                #     score = torch.sum(k_max)
                #     predictions.append(score.item())

    # Print summary before computing metrics
    print(f"\nEvaluation complete!")
    print(f"Total samples: {len(labels_list)}")
    print(f"Normal samples: {labels_list.count(0)}")
    print(f"Abnormal samples: {labels_list.count(1)}")

    if len(set(labels_list)) < 2:
        raise ValueError(f"Only found {set(labels_list)} class(es) in test set. Need both normal and abnormal samples!")

    # Compute metrics
    threshold, _, _ = metric(labels_list, predictions, anomaly_map_list, gt_list)
    print(f"Detection threshold: {threshold}")
    
    # Save 3D anomaly maps with prediction outcomes
    if is_3d and save_data_3d:
        print(f"\nSaving {len(save_data_3d)} anomaly maps with predictions...")
        for data in save_data_3d:
            save_nifti(
                data['anomaly_map'],
                data['reconstructed_volume'],
                data['filename'],
                save_dir,
                data['affine'],
                data['score'],
                threshold,
                data['label']
            )
        print(f"All anomaly maps and predictions saved to: {save_dir}")

    
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
    mdps(args)