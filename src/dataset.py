import os
from glob import glob
from pathlib import Path
from typing import Optional, Any, Callable, Tuple, Union
import numpy as np
import torch
import torch.utils.data
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
import nibabel as nib
import torch.nn.functional as F

class MVTec(torch.utils.data.Dataset):
    def __init__(self, root, category, config, is_train=True):
        self.image_transform = transforms.Compose(
            [
                transforms.Resize((config.data.image_size, config.data.image_size)),  
                transforms.ToTensor(), 
                transforms.Lambda(lambda t: (t * 2) - 1)
            ]
        )
        self.config = config
        self.mask_transform = transforms.Compose(
            [
                transforms.Resize((config.data.image_size, config.data.image_size)),
                transforms.ToTensor(),
            ]
        )
        if is_train:
            if category:
                self.image_files = glob(
                    os.path.join(root, category, "train", "good", "*.png")
                )
            else:
                self.image_files = glob(
                    os.path.join(root, "train", "good", "*.png")
                )
        else:
            if category:
                self.image_files = glob(os.path.join(root, category, "test", "*", "*.png"))
            else:
                self.image_files = glob(os.path.join(root, "test", "*", "*.png"))
        self.is_train = is_train

    def __getitem__(self, index):
        image_file = self.image_files[index]
        image = Image.open(image_file)
        image = self.image_transform(image)
        if(image.shape[0] == 1):
            image = image.expand(3, self.config.data.image_size, self.config.data.image_size)
        if self.is_train:
            label = 'good'
            return image, label
        else:
            if self.config.data.mask:
                if os.path.dirname(image_file).endswith("good"):
                    target = torch.zeros([1, image.shape[-2], image.shape[-1]])
                    label = 'good'
                else :
                    if self.config.data.name == 'MVTec':
                        target = Image.open(
                            image_file.replace("/test/", "/ground_truth/").replace(
                                ".png", "_mask.png"
                            )
                        )
                    else:
                        target = Image.open(
                            image_file.replace("/test/", "/ground_truth/"))
                    target = self.mask_transform(target)
                    label = 'defective'
            else:
                if os.path.dirname(image_file).endswith("good"):
                    target = torch.zeros([1, image.shape[-2], image.shape[-1]])
                    label = 'good'
                else :
                    target = torch.zeros([1, image.shape[-2], image.shape[-1]])
                    label = 'defective'
                
            return image, target, label

    def __len__(self):
        if self.is_train:
            return int(len(self.image_files)*1)
        else:
            return len(self.image_files)


class BTAD(torch.utils.data.Dataset):
    def __init__(self, root, category, config, is_train=True):
        self.image_transform = transforms.Compose(
            [
                transforms.Resize((config.data.image_size, config.data.image_size)),  
                transforms.ToTensor(), # Scales data into [0,1] 
                transforms.Lambda(lambda t: (t * 2) - 1) # Scale between [-1, 1] 
            ]
        )
        self.config = config
        self.mask_transform = transforms.Compose(
            [
                transforms.Resize((config.data.image_size, config.data.image_size)),
                transforms.ToTensor(), # Scales data into [0,1] 
            ]
        )

        if is_train:
            if category:
                self.image_files = glob(
                    os.path.join(root, category, "train", "ok", "*.png")
                )
                if len(self.image_files) == 0:
                    self.image_files = glob(os.path.join(root, category, "train", "ok", "*.bmp"))
        else:
            if category:
                self.image_files = glob(os.path.join(root, category, "test", "*", "*.png"))
                if len(self.image_files) == 0:
                    self.image_files = glob(os.path.join(root, category, "test", "*", "*.bmp"))

        self.is_train = is_train

    def __getitem__(self, index):
        image_file = self.image_files[index]
        image = Image.open(image_file)
        image = self.image_transform(image)
        if(image.shape[0] == 1):
            image = image.expand(3, self.config.data.image_size, self.config.data.image_size)
        if self.is_train:
            label = 'good'
            return image, label
        else:
            if self.config.data.mask:
                if os.path.dirname(image_file).endswith("ok"):
                    target = torch.zeros([1, image.shape[-2], image.shape[-1]])
                    label = 'good'
                else :
                    mask_path = image_file.replace("/test/", "/ground_truth/")
                    if os.path.exists(mask_path):
                        target = Image.open(mask_path)
                    else:
                        mask_path = mask_path.replace('.png', '.bmp')
                        if not os.path.exists(mask_path):
                            mask_path = mask_path.replace('.bmp', '.png')
                        target = Image.open(mask_path)
                    target = self.mask_transform(target)
                    label = 'defective'
                
            return image, target, label

    def __len__(self):
        return len(self.image_files)


class Resize3d:
    """
    Resizes a 3D tensor to a specified size (depth, height, width).
    Assumes input tensor has shape (C, D, H, W).
    """
    def __init__(self, size: Union[int, Tuple[int, int, int]]):
        if isinstance(size, int):
            self.size = (size, size, size)
        else:
            self.size = size

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # F.interpolate expects input to be (N, C, D, H, W)
        # We need to add a batch dimension for F.interpolate
        x = x.unsqueeze(0)
        x = F.interpolate(x, size=self.size, mode='trilinear', align_corners=False)
        return x.squeeze(0)

class ToTensor3d:
    """
    Converts a numpy array to a PyTorch tensor and adds a channel dimension.
    Assumes input is a 3D numpy array (D, H, W) and converts to (1, D, H, W).
    """
    def __call__(self, x: np.ndarray) -> torch.Tensor:
        x = torch.from_numpy(x).float()
        # Add a channel dimension, common for medical volumes
        x = x.unsqueeze(0)
        return x

class SHOMRI(Dataset):
    def __init__(self, root: str, config: Any, is_train: bool = True):
        
        # Use custom 3D transforms
        self.image_transform = transforms.Compose([
            ToTensor3d(),
            Resize3d(config.data.image_size),
            transforms.Lambda(lambda t: (t * 2) - 1)
        ])
        self.config = config
        self.mask_transform = transforms.Compose([
            ToTensor3d(),
            Resize3d(config.data.image_size),
        ])

        if is_train:
            self.image_files = glob(os.path.join(root, "train", "NORMAL", "*.nii.gz"))
        else:
            self.image_files = glob(os.path.join(root, "test", "*", "*.nii.gz"))
        
        self.is_train = is_train

    def __getitem__(self, index: int) -> Any:
        image_file = self.image_files[index]
        
        # Load the NIfTI file and get the data as a numpy array
        image_data = nib.load(image_file).get_fdata()
        image = self.image_transform(image_data)
        
        # The original code expands a single channel to 3 channels for 2D images.
        # This is not common for 3D medical volumes. The UNet should be configured for 1 channel.
        # If you need to replicate this, you could repeat the channel dimension.
        # e.g., if image.shape[0] == 1: image = image.repeat(3, 1, 1, 1)

        if self.is_train:
            label = 'good'
            return image, label
        else:
            if self.config.data.mask:
                if os.path.dirname(image_file).endswith("NORMAL"):
                    # Create an empty mask for 'good' images
                    target = torch.zeros([1, self.config.data.image_size, self.config.data.image_size, self.config.data.image_size])
                    label = 'good'
                else:
                    # Construct the path to the ground truth mask file
                    mask_file_path = image_file.replace("/test/", "/ground_truth/").replace(".nii.gz", "_mask.nii.gz")
                    
                    if os.path.exists(mask_file_path):
                        target_data = nib.load(mask_file_path).get_fdata()
                        target = self.mask_transform(target_data)
                    else:
                        # Fallback for images without a mask file
                        target = torch.zeros([1, self.config.data.image_size, self.config.data.image_size, self.config.data.image_size])
                    
                    label = 'defective'
            else:
                if os.path.dirname(image_file).endswith("NORMAL"):
                    target = torch.zeros([1, self.config.data.image_size, self.config.data.image_size, self.config.data.image_size])
                    label = 'good'
                else:
                    target = torch.zeros([1, self.config.data.image_size, self.config.data.image_size, self.config.data.image_size])
                    label = 'defective'
            
            return image, target, label

    def __len__(self) -> int:
        return len(self.image_files)
