import os
import glob
from typing import Any, List
import numpy as np
import torch
from torch.utils.data import Dataset
import nibabel as nib
from monai.transforms import Compose, LoadImage, EnsureChannelFirst, ToTensor

from monai.transforms import (
    Compose,
    LoadImaged,
    ScaleIntensityRanged,
    RandSpatialCropSamplesd,
    EnsureTypeD,
    ResizeD,
    EnsureChannelFirstd,
    RandCropByPosNegLabeld,
)
from monai.data import PatchDataset, CacheDataset, PersistentDataset
from monai.utils import first

def get_train_patch_sampler(image_key, patch_size, num_samples):
    """
    Returns a callable (RandSpatialCropSamplesd) that takes a dictionary 
    (the full volume) and returns a list of dictionaries (the patches).
    """
    sampler = RandSpatialCropSamplesd(
        keys=[image_key],
        roi_size=patch_size, 
        num_samples=num_samples,
        random_center=True,
        random_size=False,
    )
    return sampler


def get_data_list(root_dir: str, image_key: str, mask_key: str, is_train: bool) -> List[dict]:
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

            mask_path = img_path.replace("/test/", "/ground_truth/").replace(".nii.gz", "_mask.nii.gz")
            if not os.path.exists(mask_path):
                mask_path = "/projects/prjs1633/anomaly_detection/SHOMRI/one_mask.nii.gz"
                
            data_list.append({image_key: img_path, mask_key: mask_path, "label": "defective"})
    return data_list




# class SHOMRI(Dataset):
#     """
#     Implements a PatchDataset for training to sample random patches from NIfTI volumes,
#     following a similar mechanism to nnUNet for efficiency.
#     """
#     def __init__(self, root: str, config: Any, is_train: bool = True):
#         self.image_key = "image"
#         self.mask_key = "mask"
#         self.is_train = is_train

#         self.patch_size = tuple(config.data.get("patch_size", (32, 32, 32))) 
#         self.samples_per_image = config.data.get("num_samples_per_image", 4)
        
#         data_list = get_data_list(root, self.image_key, self.mask_key, is_train)

#         base_transforms = [
#             LoadImaged(keys=self.image_key), 
#             EnsureChannelFirstd(keys=self.image_key),
#             ScaleIntensityRanged(keys=self.image_key, a_min=-1.0, a_max=1.0, b_min=0.0, b_max=1.0, clip=True), 
#         ]
        
#         if is_train:
#             train_cache_transforms = Compose(base_transforms + [
#                 EnsureTypeD(keys=[self.image_key], data_type="tensor")
#             ])

#             inner_dataset = CacheDataset(
#                 data=data_list,
#                 transform=train_cache_transforms, 
#                 cache_rate=1.0, 
#                 num_workers=0 
#             )
            
#             patch_sampler = get_train_patch_sampler(
#                 image_key=self.image_key, 
#                 patch_size=self.patch_size, 
#                 num_samples=self.samples_per_image
#             )

#             self.monai_dataset = PatchDataset(
#                 data=inner_dataset,
#                 patch_func=lambda x: patch_sampler(x),
#                 samples_per_image=self.samples_per_image,

#             )

#         else:
#             mask_transforms = [
#                 LoadImaged(keys=self.mask_key), 
#                 EnsureChannelFirstd(keys=self.mask_key),
#             ]

#             self.transforms = Compose(base_transforms + mask_transforms + [
#                 EnsureTypeD(keys=[self.image_key, self.mask_key], data_type="tensor")
#             ])
            
#             self.monai_dataset = CacheDataset(
#                 data=data_list,
#                 transform=self.transforms, 
#                 cache_rate=1.0, 
#                 num_workers=0
#             )
            
#     def __len__(self) -> int:
#         return len(self.monai_dataset)

#     def __getitem__(self, index):
#         data_dict = self.monai_dataset[index]

#         if isinstance(data_dict_or_list, list):
#             return tuple(d[self.image_key] for d in data_dict_or_list)

#         # normal cache dataset case
#         if self.is_train:
#             return (data_dict_or_list[self.image_key],)
#         else:
#             return data_dict_or_list
#         # if self.is_train:
#         #     return (data_dict[self.image_key],)
#         # else:
#         #     return data_dict



class SHOMRI(torch.utils.data.Dataset):
    def __init__(self, root_dir, is_train=True, transform=None):
        """
        Args:
            root_dir (str): Root directory containing train/test folders.
            is_train (bool): Whether to use the training set (default is True).
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.is_train = is_train
        self.transform = transform
        self.image_files = []
        
        if self.is_train:
            normal_dir = os.path.join(root_dir, 'train', 'NORMAL')
            self.image_files = [os.path.join(normal_dir, f) for f in os.listdir(normal_dir) if f.endswith('.nii.gz')]
        else:

            normal_dir = os.path.join(root_dir, 'test', 'NORMAL')
            abnormal_dir = os.path.join(root_dir, 'test', 'ABNORMAL')
            self.image_files = [os.path.join(normal_dir, f) for f in os.listdir(normal_dir) if f.endswith('.nii.gz')]
            self.image_files += [os.path.join(abnormal_dir, f) for f in os.listdir(abnormal_dir) if f.endswith('.nii.gz')]

    def __getitem__(self, index):
        image_file = self.image_files[index]
        
        img = nib.load(image_file)
        image = img.get_fdata() 

        if image.ndim == 3:
            image = np.expand_dims(image, axis=0) 
        
        image = torch.tensor(image, dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        if self.is_train:
            label = 'good'
            return image, label
        else:
            if os.path.dirname(image_file).endswith("NORMAL"):
                label = 'good'
                target = torch.zeros_like(image)
            else:
                label = 'defective'
                target = torch.ones_like(image) 

            return image, target, label

    def __len__(self):
        return len(self.image_files)