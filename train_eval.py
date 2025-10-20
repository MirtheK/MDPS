import os
import argparse
from omegaconf import OmegaConf
import time
import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms import transforms
from src.dataset import SHOMRI
from src.diffusion import diffusion_loss
from src.models.unet import UNetModel
import torch.amp as amp
from torchmetrics.classification import Accuracy, AUROC, F1Score, Precision, Recall, AveragePrecision
from torchmetrics import StructuralSimilarityIndexMeasure
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils


os.environ['CUDA_VISIBLE_DEVICES'] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def evaluator(model, testloader, config, device):
    """
    Evaluates the model on the test set and calculates various metrics.
    """
    model.eval()
    
    accuracy = Accuracy(task="binary", num_classes=2).to(device)
    auroc = AUROC(task="binary", num_classes=2).to(device)
    f1 = F1Score(task="binary", num_classes=2).to(device)
    precision = Precision(task="binary", num_classes=2).to(device)
    recall = Recall(task="binary", num_classes=2).to(device)
    ap = AveragePrecision(task="binary", num_classes=2).to(device)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    
    total_loss = 0.0
    
    with torch.no_grad():
        for data, target, labels in testloader:
            images = data.to(device)
  
            # labels = labels.to(device) 
            labels = (labels.sum(dim=(1,2,3,4)) > 0).long()  

            anomaly_map, _ = get_anomaly_scores(data, unet, resnet, config)
            
            labels_list.append(0 if label == 'good' else 1)
            k = config.model.k_topk
            pred = pred.reshape(1, -1)
            pred = F.softmax(pred, dim=1)
            k_max, _ = pred.topk(k, largest=True)
            score = torch.sum(k_max)
            predictions.append(score.item())

            accuracy.update(preds, labels)
            auroc.update(scores, labels)
            f1.update(preds, labels)
            precision.update(preds, labels)
            recall.update(preds, labels)
            ap.update(scores, labels)
            ssim.update(reconstructed_images, images)

            loss = diffusion_loss(model, images, torch.randint(1, config.model.diffusion_steps, (images.shape[0],), device=device).long(), config)
            total_loss += loss.item()
            
    avg_loss = total_loss / len(testloader)
    
    metrics = {
        'loss': avg_loss,
        'accuracy': accuracy.compute(),
        'auroc': auroc.compute(),
        'f1': f1.compute(),
        'precision': precision.compute(),
        'recall': recall.compute(),
        'ap': ap.compute(),
        'ssim': ssim.compute()
    }
    
    accuracy.reset()
    auroc.reset()
    f1.reset()
    precision.reset()
    recall.reset()
    ap.reset()
    ssim.reset()
    
    model.train() 
    return metrics


def trainer(args):
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    config = OmegaConf.load(args.config)
    device = config.model.device
    print(config.data.name)

    writer = SummaryWriter(log_dir=os.path.join(config.model.checkpoint_dir, 'tensorboard_logs'))

    model = UNetModel(config.data.image_size, 64, dropout=0.0, n_heads=4 ,in_channels=config.data.imput_channel)
    print("Num params: ", sum(p.numel() for p in model.parameters()))
    model = model.to(config.model.device)
    model.train()
    model = torch.nn.DataParallel(model)


    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.model.learning_rate, weight_decay=config.model.weight_decay
    )
    if config.data.name == 'MVTec':
        train_dataset = MVTec(
            root= config.data.data_dir,
            category=config.data.category,
            config = config,
            is_train=True,
        )
    if config.data.name == 'BTAD':
        train_dataset = BTAD(
            root= config.data.data_dir,
            category=config.data.category,
            config = config,
            is_train=True,
        )

    if config.data.name == 'SHOMRI':
        train_dataset = SHOMRI(
            root= config.data.data_dir,
            config = config,
            is_train=True,
        )
        test_dataset = SHOMRI(
            root= config.data.data_dir,
            config = config,
            is_train=False,
        )

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
        shuffle=True,
        num_workers=config.model.num_workers,
        drop_last=True,
    )


    if not os.path.exists('checkpoints'):
        os.mkdir('checkpoints')
    if not os.path.exists(config.model.checkpoint_dir):
        os.mkdir(config.model.checkpoint_dir)

    scaler = amp.GradScaler('cuda')
    best_auroc = 0.0

    total_start_time = time.time()

    for epoch in range(config.model.epochs):
        start_time = time.time()
        model.train()
        
        for step, batch in enumerate(trainloader):
            images = batch[0].to(device)
            t = torch.randint(1, config.model.diffusion_steps, (images.shape[0],), device=device).long()
            optimizer.zero_grad()
            
            with amp.autocast(device):
                loss = diffusion_loss(model, images, t, config) 
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Log training loss to TensorBoard at each step
            writer.add_scalar('Loss/train_step', loss.item(), epoch * len(trainloader) + step)

        epoch_time = time.time() - start_time
        writer.add_scalar('Time/epoch', epoch_time, epoch)
        print(f"Epoch {epoch} | Loss: {loss.item():.4f} | Time: {epoch_time:.2f}s")
        
        # Log learning rate to TensorBoard
        writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)
        
        # Log a batch of training images to see how they look
        if epoch == 0:
            img_grid = vutils.make_grid(images[:8], normalize=True, scale_each=True)
            # writer.add_image('Training Images', img_grid, global_step=epoch)

        # Evaluate the model every 50 epochs or on the last epoch
        if testloader and (epoch % 50 == 0 or epoch == config.model.epochs - 1):
            metrics = evaluator(model, testloader, config, device)
            
            print(f"--- Evaluation at Epoch {epoch} ---")
            for metric_name, value in metrics.items():
                # Ensure the value is a scalar before logging
                if isinstance(value, torch.Tensor):
                    value = value.item()
                writer.add_scalar(f"Evaluation/{metric_name}", value, epoch)
                print(f"{metric_name.replace('_', ' ').title()}: {value:.4f}")
            print("-----------------------------------")

            if metrics['auroc'] > best_auroc:
                best_auroc = metrics['auroc']
                model_save_path = os.path.join(config.model.checkpoint_dir, 'best_model_by_auroc.pth')
                print(f"New best AUROC: {best_auroc:.4f}. Saving model to {model_save_path}")
                torch.save(model.state_dict(), model_save_path)
    
    total_training_time = time.time() - total_start_time
    writer.add_scalar('Total Training Time', total_training_time)
    
    # Close the SummaryWriter
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
    
    trainer(args)