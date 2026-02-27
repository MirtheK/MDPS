import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
import argparse
from omegaconf import OmegaConf
import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms import transforms
from src.dataset import SHOMRI, SHOMRIGridPatches 
from src.diffusion import diffusion_loss
from src.models.unet import UNetModel


def trainer(args):
    config = OmegaConf.load(args.config)
    
    # Get patch size from config (add this to your config file)
    patch_size = config.data.get('patch_size', (64, 64, 64))
    print(f"Training with patch size: {patch_size}")

    model = UNetModel(
        patch_size[0],  # Use patch size instead of full image size
        32, 
        dropout=0.0, 
        n_heads=4,
        in_channels=config.data.imput_channel
    )
    print("Num params: ", sum(p.numel() for p in model.parameters()))
    model = model.to(config.model.device)
    model = model.float()
    model.train()

    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config.model.learning_rate, 
        weight_decay=config.model.weight_decay
    )
    
    if config.data.name == 'SHOMRI':
        patches_per_volume = config.data.get('patches_per_volume', 4)
        
        # Random patches (recommended - more variation)
        train_dataset = SHOMRI(
            root_dir=config.data.data_dir,
            patch_size=patch_size,
            patches_per_volume=patches_per_volume,
            is_train=True,
            cache_rate=1.0,  # Cache all volumes in memory
        )
        
        # train_dataset = SHOMRIGridPatches(
        #     root_dir=config.data.data_dir,
        #     patch_size=patch_size,
        #     is_train=True,
        #     cache_rate=1.0,
        # )
    
    trainloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.model.num_workers,
        drop_last=True,
        pin_memory=True,  # Added for faster GPU transfer
    )
    
    print(f"Dataset size: {len(train_dataset)} patches")
    print(f"Batches per epoch: {len(trainloader)}")
    
    if not os.path.exists('checkpoints'):
        os.mkdir('checkpoints')
    if not os.path.exists(config.model.checkpoint_dir):
        os.mkdir(config.model.checkpoint_dir)

    scaler = torch.amp.GradScaler('cuda')

    for epoch in range(config.model.epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        for step, batch in enumerate(trainloader):
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
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            # Print progress every 10 steps
            if epoch % 1 == 0 and step == 0:
                print(f"Epoch {epoch} | Loss: {loss.item():.4f}")
        
        # Print epoch summary
        avg_loss = epoch_loss / num_batches
        print(f"Epoch {epoch} completed | Average Loss: {avg_loss:.4f}")

        # Save checkpoint - CORRECT INDENTATION (inside epoch loop, outside step loop)
        if epoch % config.model.epochs_checkpoint == 0:
            model_save_dir = os.path.join(os.getcwd(), config.model.checkpoint_dir, config.data.category)
            if not os.path.exists(model_save_dir):
                os.mkdir(model_save_dir)
            print('saving model')
            torch.save(model.state_dict(), os.path.join(model_save_dir, str(epoch)))
            torch.cuda.empty_cache()


        # # Save checkpoint
        # if epoch % config.model.epochs_checkpoint == 0:
        #     model_save_dir = os.path.join(
        #         os.getcwd(), 
        #         config.model.checkpoint_dir, 
        #         config.data.category
        #     )
        #     if not os.path.exists(model_save_dir):
        #         os.makedirs(model_save_dir, exist_ok=True)
            
        #     checkpoint_path = os.path.join(model_save_dir, f"epoch_{epoch}.pt")
        #     print(f'Saving model to {checkpoint_path}')
            
        #     torch.save({
        #         'epoch': epoch,
        #         'model_state_dict': model.state_dict(),
        #         'optimizer_state_dict': optimizer.state_dict(),
        #         'loss': avg_loss,
        #     }, checkpoint_path)
            
            torch.cuda.empty_cache()


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

