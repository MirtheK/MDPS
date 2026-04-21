import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
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
    
    patch_size = config.data.get('patch_size')
    print(f"Training with patch size: {patch_size}")

    model = UNetModel(
        patch_size[0], 
        64, 
        dropout=0.0, 
        n_heads=4,
        in_channels=config.data.imput_channel
    )
    print("Num params: ", sum(p.numel() for p in model.parameters()))
    model = model.to(config.model.device)
    model = model.float()

    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config.model.learning_rate, 
        weight_decay=config.model.weight_decay
    )

    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=config.model.device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            # New-style checkpoint
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint["epoch"] + 1
            print(f"Resumed from epoch {checkpoint['epoch']} -> starting at {start_epoch}")
        else:
            # Old-style checkpoint (weights only, like your epoch-1900 file)
            model.load_state_dict(checkpoint)
            start_epoch = args.start_epoch  # pass manually via --start_epoch 1900
            print(f"Loaded weights only. Starting from epoch {start_epoch}")

    model.train()

    patches_per_volume = config.data.get('patches_per_volume')
    
    train_dataset = SHOMRI(
        root_dir=config.data.data_dir,
        patch_size=patch_size,
        patches_per_volume=patches_per_volume,
        is_train=True,
        cache_rate=2.0, 
    )
        
    trainloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.model.num_workers,
        drop_last=True,
        pin_memory=True, 
    )
    
    print(f"Dataset size: {len(train_dataset)} patches")
    print(f"Batches per epoch: {len(trainloader)}")
    
    if not os.path.exists('checkpoints'):
        os.mkdir('checkpoints')
    if not os.path.exists(config.model.checkpoint_dir):
        os.mkdir(config.model.checkpoint_dir)

    for epoch in range(start_epoch, config.model.epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        for step, batch in enumerate(trainloader):
            # Maybe try: t = torch.randint(0, config.model.diffusion_steps, (batch[0].shape[0],), device=config.model.device).long()
            t = torch.randint(
                0, 
                config.model.diffusion_steps, 
                (batch[0].shape[0],), 
                device=config.model.device
            ).long()

            x = batch[0].to(config.model.device, non_blocking=True).float()
            
            optimizer.zero_grad()
            loss = diffusion_loss(model, x, t, config)
            loss.backward()
            optimizer.step()

            if epoch % 1 == 0 and step == 0:
                print(f"Epoch {epoch} | Loss: {loss.item()}")

            if epoch % config.model.epochs_checkpoint == 0 and step == 0:
                model_save_dir = os.path.join(os.getcwd(), config.model.checkpoint_dir, config.data.category)
                if not os.path.exists(model_save_dir):
                    os.mkdir(model_save_dir)
                print('saving model')
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                }, os.path.join(model_save_dir, f"ckpt_{epoch}.pt"))
           



def parse_args():
    parser = argparse.ArgumentParser('MDPS')    
    parser.add_argument('-cfg', '--config', help='config file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint file to resume from')
    parser.add_argument('--start_epoch', type=int, default=0,
                        help='Epoch to start from (only needed for weights-only checkpoints)')
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

# import os
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# import argparse
# from omegaconf import OmegaConf
# import numpy as np
# import torch
# import torch.nn as nn
# from torchvision.transforms import transforms
# from src.dataset import SHOMRI, SHOMRIGridPatches 
# from src.diffusion import diffusion_loss
# from src.models.unet import UNetModel


# def trainer(args):
#     config = OmegaConf.load(args.config)
    
#     patch_size = config.data.get('patch_size')
#     print(f"Training with patch size: {patch_size}")

#     model = UNetModel(
#         patch_size[0], 
#         64, 
#         dropout=0.0, 
#         n_heads=4,
#         in_channels=config.data.imput_channel
#     )
#     print("Num params: ", sum(p.numel() for p in model.parameters()))
#     model = model.to(config.model.device)
#     model = model.float()
#     model.train()

#     optimizer = torch.optim.Adam(
#         model.parameters(), 
#         lr=config.model.learning_rate, 
#         weight_decay=config.model.weight_decay
#     )
    
#     patches_per_volume = config.data.get('patches_per_volume')
    
#     train_dataset = SHOMRI(
#         root_dir=config.data.data_dir,
#         patch_size=patch_size,
#         patches_per_volume=patches_per_volume,
#         is_train=True,
#         cache_rate=1.0, 
#     )
        
    
#     trainloader = torch.utils.data.DataLoader(
#         train_dataset,
#         batch_size=config.data.batch_size,
#         shuffle=True,
#         num_workers=config.model.num_workers,
#         drop_last=True,
#         pin_memory=True, 
#     )
    
#     print(f"Dataset size: {len(train_dataset)} patches")
#     print(f"Batches per epoch: {len(trainloader)}")
    
#     if not os.path.exists('checkpoints'):
#         os.mkdir('checkpoints')
#     if not os.path.exists(config.model.checkpoint_dir):
#         os.mkdir(config.model.checkpoint_dir)

#     # scaler = torch.amp.GradScaler('cuda')

#     for epoch in range(config.model.epochs):
#         epoch_loss = 0.0
#         num_batches = 0
        
#         for step, batch in enumerate(trainloader):
#             t = torch.randint(
#                 1, 
#                 config.model.diffusion_steps, 
#                 (batch[0].shape[0],), 
#                 device=config.model.device
#             ).long()
            
#             optimizer.zero_grad()
            

#             loss = diffusion_loss(model, batch[0], t, config)
            
#             loss.backward()
#             optimizer.step()
#             if epoch % 1 == 0 and step == 0:
#                 print(f"Epoch {epoch} | Loss: {loss.item()}")
#             if epoch % config.model.epochs_checkpoint == 0 and step ==0:
#                 model_save_dir = os.path.join(os.getcwd(), config.model.checkpoint_dir, config.data.category)
#                 if not os.path.exists(model_save_dir):
#                     os.mkdir(model_save_dir)
#                 print('saving model')
#                 torch.save(model.state_dict(), os.path.join(model_save_dir, str(epoch)))
            
           
#             torch.cuda.empty_cache()


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

