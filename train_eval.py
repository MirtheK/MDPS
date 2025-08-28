import os
import argparse
import time
from omegaconf import OmegaConf
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision.transforms import transforms
from src.dataset import MVTec, BTAD, SHOMRI
from src.diffusion import diffusion_loss, sample, sample_mask, compute_alpha
from src.models.unet import UNetModel
from src.models.resnet import Resnet
from src.compare import distance
import torch.amp as amp
from torchmetrics.classification import BinaryAUROC, BinaryAccuracy, BinaryF1Score, BinaryPrecision, BinaryRecall, BinaryAveragePrecision
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def get_2d_slice_from_3d(images, slice_index=None):
    """
    Extracts a 2D slice from the middle of a 3D image batch.
    Images are expected in (N, C, D, H, W) format.
    """
    if len(images.shape) != 5:
        return images
        
    N, C, D, H, W = images.shape
    
    if slice_index is None:
        slice_index = D // 2
    
    if not (0 <= slice_index < D):
        slice_index = D // 2
        print(f"Warning: Slice index is out of bounds. Using middle slice {slice_index}.")
        
    slices_2d = images[:, :, slice_index, :, :].squeeze(2)
    return slices_2d


def get_anomaly_scores(data, unet, resnet, config, mask=None):
    """
    Generates anomaly scores for a batch of data.
    """
    anomaly_batch = []
    
    if mask is not None:
        # Masking-based reconstruction
        seq = range(0, config.model.test_steps, config.model.skip)
        for _ in range(0, config.model.mask_repeat):
            reconstructed = sample_mask(data, mask, seq, unet, config, w=config.model.w)
            data_reconstructed = reconstructed[-1]
            anomaly_map = distance(data_reconstructed, data, resnet, config) / 2
            anomaly_batch.append(anomaly_map.unsqueeze(0))
    else:
        # Standard reconstruction from noise
        test_steps = torch.Tensor([config.model.test_steps]).type(torch.int64).to(config.model.device)
        at = compute_alpha(test_steps.long(), config)
        seq = range(0, config.model.test_steps, config.model.skip)
        for _ in range(0, config.model.test_repeat):
            noisy_image = at.sqrt() * data + (1 - at).sqrt() * torch.randn_like(data).to(config.model.device)
            reconstructed = sample(data, noisy_image, seq, unet, config, w=config.model.w)
            data_reconstructed = reconstructed[-1]
            anomaly_map = distance(data_reconstructed, data, resnet, config) / 2
            anomaly_batch.append(anomaly_map.unsqueeze(0))
            
    anomaly_batch = torch.cat(anomaly_batch, dim=0)
    anomaly_map = torch.mean(anomaly_batch, dim=0)
    
    return anomaly_map, anomaly_batch


def evaluator(unet, resnet, testloader, config):
    """
    Evaluates the model on the test set and calculates various metrics.
    """
    unet.eval()
    resnet.eval()
    
    device = config.model.device
    
    labels_list = []
    predictions = []
    
    # Define a transform to handle MVTec's specific cropping
    transform_mvtec = transforms.Compose([
        transforms.CenterCrop((224, 224)), 
    ]) 

    with torch.no_grad():
        if config.model.mask_steps == 0:
            print("Running evaluation with no masking")
            for data, targets, labels in testloader:
                data = data.to(device)
                
                anomaly_map, _ = get_anomaly_scores(data, unet, resnet, config)
                
                if config.data.name == 'MVTec':
                    anomaly_map = transform_mvtec(anomaly_map)

                for pred, label in zip(anomaly_map, labels):
                    labels_list.append(0 if label == 'good' else 1)
                    k = config.model.k_topk
                    pred = pred.reshape(1, -1)
                    pred = F.softmax(pred, dim=1)
                    k_max, _ = pred.topk(k, largest=True)
                    score = torch.sum(k_max)
                    predictions.append(score.item())
        else:
            print("Running two-stage evaluation with masking")
            # First pass to generate masks
            mask_anomaly_map_list = []
            for data, _, _ in testloader:
                data = data.to(device)
                anomaly_map, _ = get_anomaly_scores(data, unet, resnet, config)
                mask_anomaly_map_list.append(anomaly_map)

            mask_anomaly_map_list = torch.cat(mask_anomaly_map_list, dim=0)
            
            pixel_min = torch.min(mask_anomaly_map_list)
            pixel_max = torch.max(mask_anomaly_map_list)
            thresholds = pixel_min + config.model.mask0_thresholds * (pixel_max - pixel_min)
            mask_init = torch.where(mask_anomaly_map_list > thresholds, 1, torch.zeros_like(mask_anomaly_map_list))
            
            # Second pass for final anomaly detection
            for batch_idx, (data, targets, labels) in enumerate(testloader):
                data = data.to(device)
                start_idx = batch_idx * config.data.batch_size
                end_idx = start_idx + data.shape[0]
                mask = mask_init[start_idx:end_idx]
                
                anomaly_map, _ = get_anomaly_scores(data, unet, resnet, config, mask=mask)
                
                if config.data.name == 'MVTec':
                    anomaly_map = transform_mvtec(anomaly_map)
                
                for pred, label in zip(anomaly_map, labels):
                    labels_list.append(0 if label == 'good' else 1)
                    k = config.model.k_topk
                    pred = pred.reshape(1, -1)
                    pred = F.softmax(pred, dim=1)
                    k_max, _ = pred.topk(k, largest=True)
                    score = torch.sum(k_max)
                    predictions.append(score.item())

    labels_tensor = torch.tensor(labels_list).to(device)
    predictions_tensor = torch.tensor(predictions).to(device)
    
    metrics = {
        'AUROC': BinaryAUROC().to(device),
        'F1Score': BinaryF1Score().to(device),
        'Precision': BinaryPrecision().to(device),
        'Recall': BinaryRecall().to(device),
        'AveragePrecision': BinaryAveragePrecision().to(device)
    }

    results = {}
    for name, metric_fn in metrics.items():
        results[name] = metric_fn(predictions_tensor, labels_tensor).item()

    unet.train()
    resnet.train()
    return results


def trainer(args):
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    config = OmegaConf.load(args.config)
    print(config.data.name)
    device = config.model.device

    writer = SummaryWriter(log_dir=os.path.join(config.model.checkpoint_dir, 'tensorboard_logs'))

    # UNet and Resnet models
    unet = UNetModel(config.data.image_size, 64, dropout=0.0, n_heads=4, in_channels=config.data.imput_channel)
    resnet = Resnet(config).to(device)
    print("UNet Num params: ", sum(p.numel() for p in unet.parameters()))
    unet = unet.to(device)

    # Log the model graph with a dummy input
    if len(config.data.image_size) == 3: # 3D images
        dummy_input = torch.randn(1, config.data.imput_channel, config.data.image_size[0], config.data.image_size[1], config.data.image_size[2]).to(device)
    else: # 2D images
        dummy_input = torch.randn(1, config.data.imput_channel, config.data.image_size, config.data.image_size).to(device)

    writer.add_graph(unet, dummy_input)

    optimizer = torch.optim.Adam(
        unet.parameters(), lr=config.model.learning_rate, weight_decay=config.model.weight_decay
    )

    # --- Dataset Loading ---
    if config.data.name == 'MVTec':
        train_dataset = MVTec(root=config.data.data_dir, category=config.data.category, config=config, is_train=True)
        test_dataset = MVTec(root=config.data.data_dir, category=config.data.category, config=config, is_train=False)
    elif config.data.name == 'BTAD':
        train_dataset = BTAD(root=config.data.data_dir, category=config.data.category, config=config, is_train=True)
        test_dataset = BTAD(root=config.data.data_dir, category=config.data.category, config=config, is_train=False)
    elif config.data.name == 'SHOMRI':
        train_dataset = SHOMRI(root=config.data.data_dir, config=config, is_train=True)
        test_dataset = SHOMRI(root=config.data.data_dir, config=config, is_train=False)
    else:
        raise ValueError(f"Unknown dataset name: {config.data.name}")

    trainloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.model.num_workers,
        drop_last=True,
    )
    
    testloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.model.num_workers,
        drop_last=False,
    )

    if not os.path.exists(os.path.join(config.model.checkpoint_dir, config.data.category)):
        os.makedirs(os.path.join(config.model.checkpoint_dir, config.data.category))

    scaler = amp.GradScaler(device)
    best_auroc = 0.0

    total_start_time = time.time()

    for epoch in range(config.model.epochs):
        start_time = time.time()
        unet.train()
        
        for step, batch in enumerate(trainloader):
            images = batch[0].to(device)
            t = torch.randint(1, config.model.diffusion_steps, (images.shape[0],), device=device).long()
            optimizer.zero_grad()
            
            with amp.autocast(device):
                loss = diffusion_loss(unet, images, t, config) 
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            writer.add_scalar('Loss/train_step', loss.item(), epoch * len(trainloader) + step)

        epoch_time = time.time() - start_time
        writer.add_scalar('Time/epoch', epoch_time, epoch)
        print(f"Epoch {epoch} | Loss: {loss.item():.4f} | Time: {epoch_time:.2f}s")
        
        writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)
        
        if epoch == 0:
            images_2d = get_2d_slice_from_3d(images[:8])
            if images_2d.shape[1] == 1:
                images_to_grid = images_2d.repeat(1, 3, 1, 1)
            else:
                images_to_grid = images_2d
            img_grid = vutils.make_grid(images_to_grid, normalize=True, scale_each=True)
            writer.add_image('Training Images (2D slice)', img_grid, global_step=epoch)

        if epoch % config.model.epochs_checkpoint == 0 or epoch == config.model.epochs - 1:
            metrics = evaluator(unet, resnet, testloader, config)
            
            print(f"--- Evaluation at Epoch {epoch} ---")
            for metric_name, value in metrics.items():
                writer.add_scalar(f"Evaluation/{metric_name}", value, epoch)
                print(f"{metric_name}: {value:.4f}")
            print("-----------------------------------")

            if metrics['AUROC'] > best_auroc:
                best_auroc = metrics['AUROC']
                model_save_path = os.path.join(config.model.checkpoint_dir, config.data.category, 'best_model.pth')
                print(f"New best AUROC: {best_auroc:.4f}. Saving model to {model_save_path}")
                torch.save(unet.state_dict(), model_save_path)
    
    total_training_time = time.time() - total_start_time
    writer.add_scalar('Total Training Time', total_training_time)
    
    writer.close()
    
    print(f"\nTraining finished. Total training time: {total_training_time:.2f}s")


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
    trainer(args)