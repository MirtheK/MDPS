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


# class BTAD(torch.utils.data.Dataset):
#     def __init__(self, root, category, config, is_train=True):
#         self.image_transform = transforms.Compose(
#             [
#                 transforms.Resize((config.data.image_size, config.data.image_size)),  
#                 transforms.ToTensor(), # Scales data into [0,1] 
#                 transforms.Lambda(lambda t: (t * 2) - 1) # Scale between [-1, 1] 
#             ]
#         )
#         self.config = config
#         self.mask_transform = transforms.Compose(
#             [
#                 transforms.Resize((config.data.image_size, config.data.image_size)),
#                 transforms.ToTensor(), # Scales data into [0,1] 
#             ]
#         )

#         if is_train:
#             if category:
#                 self.image_files = glob(
#                     os.path.join(root, category, "train", "ok", "*.png")
#                 )
#                 if len(self.image_files) == 0:
#                     self.image_files = glob(os.path.join(root, category, "train", "ok", "*.bmp"))
#         else:
#             if category:
#                 self.image_files = glob(os.path.join(root, category, "test", "*", "*.png"))
#                 if len(self.image_files) == 0:
#                     self.image_files = glob(os.path.join(root, category, "test", "*", "*.bmp"))

#         self.is_train = is_train

#     def __getitem__(self, index):
#         image_file = self.image_files[index]
#         image = Image.open(image_file)
#         image = self.image_transform(image)
#         if(image.shape[0] == 1):
#             image = image.expand(3, self.config.data.image_size, self.config.data.image_size)
#         if self.is_train:
#             label = 'good'
#             return image, label
#         else:
#             if self.config.data.mask:
#                 if os.path.dirname(image_file).endswith("ok"):
#                     target = torch.zeros([1, image.shape[-2], image.shape[-1]])
#                     label = 'good'
#                 else :
#                     mask_path = image_file.replace("/test/", "/ground_truth/")
#                     if os.path.exists(mask_path):
#                         target = Image.open(mask_path)
#                     else:
#                         mask_path = mask_path.replace('.png', '.bmp')
#                         if not os.path.exists(mask_path):
#                             mask_path = mask_path.replace('.bmp', '.png')
#                         target = Image.open(mask_path)
#                     target = self.mask_transform(target)
#                     label = 'defective'
                
#             return image, target, label

#     def __len__(self):
#         return len(self.image_files)


# class Resize3d:
#     """
#     Resizes a 3D tensor to a specified size (depth, height, width).
#     Assumes input tensor has shape (C, D, H, W).
#     """
#     def __init__(self, size: Union[int, Tuple[int, int, int]]):
#         if isinstance(size, int):
#             self.size = (size, size, size)
#         else:
#             self.size = size

#     def __call__(self, x: torch.Tensor) -> torch.Tensor:
#         x = x.unsqueeze(0)
#         x = F.interpolate(x, size=self.size, mode='trilinear', align_corners=False)
#         return x.squeeze(0)

# class ToTensor3d:
    # """
    # Converts a numpy array to a PyTorch tensor and adds a channel dimension.
    # Assumes input is a 3D numpy array (D, H, W) and converts to (1, D, H, W).
    # """
    # def __call__(self, x: np.ndarray) -> torch.Tensor:
    #     x = torch.from_numpy(x).float()
    #     x = x.unsqueeze(0)
    #     return x


# class SHOMRI(Dataset):
#     def __init__(self, root: str, config: Any, is_train: bool = True):
        
#         # Use custom 3D transforms
#         self.image_transform = transforms.Compose([
#             ToTensor3d(),
#             Resize3d(config.data.image_size),
#             transforms.Lambda(lambda t: (t * 2) - 1)
#         ])
#         self.config = config
#         self.mask_transform = transforms.Compose([
#             ToTensor3d(),
#             Resize3d(config.data.image_size),
#         ])

#         if is_train:
#             self.image_files = glob(os.path.join(root, "train", "NORMAL", "*.nii.gz"))
#         else:
#             self.image_files = glob(os.path.join(root, "test", "*", "*.nii.gz"))
        
#         self.is_train = is_train

#     def __getitem__(self, index: int) -> Any:
#         image_file = self.image_files[index]
        
#         image_data = nib.load(image_file).get_fdata()
#         image = self.image_transform(image_data)
        

#         if self.is_train:
#             label = 'good'
#             return image, label
#         else:
#             if self.config.data.mask:
#                 if os.path.dirname(image_file).endswith("NORMAL"):
#                     # Create an empty mask for 'good' images
#                     target = torch.zeros([1, self.config.data.image_size, self.config.data.image_size, self.config.data.image_size])
#                     label = 'good'
#                 else:
#                     # Construct the path to the ground truth mask file
#                     mask_file_path = image_file.replace("/test/", "/ground_truth/").replace(".nii.gz", "_mask.nii.gz")
                    
#                     if os.path.exists(mask_file_path):
#                         target_data = nib.load(mask_file_path).get_fdata()
#                         target = self.mask_transform(target_data)
#                     else:
#                         # Fallback for images without a mask file
#                         target = torch.zeros([1, self.config.data.image_size, self.config.data.image_size, self.config.data.image_size])
                    
#                     label = 'defective'
#             else:
#                 if os.path.dirname(image_file).endswith("NORMAL"):
#                     target = torch.zeros([1, self.config.data.image_size, self.config.data.image_size, self.config.data.image_size])
#                     label = 'good'
#                 else:
#                     target = torch.zeros([1, self.config.data.image_size, self.config.data.image_size, self.config.data.image_size])
#                     label = 'defective'
            
#             return image, target, label

#     def __len__(self) -> int:
#         return len(self.image_files)



import os
import glob
from typing import Any, Dict, List, Tuple
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

from monai.transforms import (
    Compose,
    LoadImaged,
    ScaleIntensityRanged,
    RandSpatialCropd,
    ToTensorD,
    EnsureTypeD,
    ResizeD,
    EnsureChannelFirstd
)
from monai.data import CacheDataset

def get_data_list(root_dir: str, image_key: str, mask_key: str, is_train: bool) -> List:
    """
    Generates a list of dictionaries containing file paths for images and metadata.
    This structure is required for MONAI's dictionary-based transforms.
    """
    data_list = []
    
    if is_train:
        image_files = glob.glob(os.path.join(root_dir, "train", "NORMAL", "*.nii.gz"))
        for img_path in image_files:
            data_list.append({image_key: img_path, "label": "good"})
            

    else:
        normal_images = glob.glob(os.path.join(root_dir, "test", "NORMAL", "*.nii.gz"))
        for img_path in normal_images:

            data_list.append({image_key: img_path, mask_key: "/projects/prjs1633/anomaly_detection/SHOMRI/zero_mask.nii.gz", "label": "good"})


        defective_images = glob.glob(os.path.join(root_dir, "test", "ABNORMAL", "*.nii.gz"))
        for img_path in defective_images:
            # Construct the path to the ground truth mask file
            mask_path = img_path.replace("/test/", "/ground_truth/").replace(".nii.gz", "_mask.nii.gz")
            if not os.path.exists(mask_path):
                mask_path = "/projects/prjs1633/anomaly_detection/SHOMRI/one_mask.nii.gz" # Use dummy for mask, mask is now array of ones
                
            data_list.append({image_key: img_path, mask_key: mask_path, "label": "defective"})

    return data_list


class SHOMRI(Dataset):
    """
    Wraps the MONAI CacheDataset to return a simple tuple (image, target_mask, label),
    while efficiently handling 3D file loading and patching internally.
    """
    def __init__(self, root: str, config: Any, is_train: bool = True):
        self.image_key = "image"
        self.mask_key = "mask"
        self.is_train = is_train
        self.image_size = config.data.image_size 

        data_list = get_data_list(root, self.image_key, self.mask_key, is_train)

        keys_to_load = [self.image_key]
        if not is_train:
            keys_to_load.append(self.mask_key)

        base_transforms = [
            LoadImaged(keys=self.image_key), 
            EnsureChannelFirstd(keys=self.image_key), 
            ResizeD(keys=self.image_key, spatial_size=(config.data.image_size, config.data.image_size, config.data.image_size)),
            ScaleIntensityRanged(keys=self.image_key, a_min=-1.0, a_max=1.0),
        ]

        if is_train:
            self.transforms = Compose(base_transforms)
            
        else:
            mask_transforms = [
                LoadImaged(keys=self.mask_key), 
                EnsureChannelFirstd(keys=self.mask_key),
                ResizeD(keys=self.mask_key, spatial_size=(config.data.image_size, config.data.image_size, config.data.image_size), mode="nearest")
            ]
            self.transforms = Compose(base_transforms + mask_transforms)
            
            
        self.monai_dataset = CacheDataset(
            data=data_list,
            transform=self.transforms,
            cache_rate=1.0, 
            num_workers=4
        )
        
    def __len__(self) -> int:
        return len(self.monai_dataset)

    def __getitem__(self, index: int) -> Tuple:
        data = self.monai_dataset[index]
        image = data[self.image_key] 
        label = data["label"]  
        

        if self.is_train:
            target_mask = torch.zeros_like(image) 
            return image, target_mask, label
        else:
            target_mask = (torch.ones_like(image))
            return image, target_mask, label
