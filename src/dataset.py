import os
import glob
from typing import Any, List
import numpy as np
import torch
from torch.utils.data import Dataset
import nibabel as nib

from monai.transforms import (
    Compose,
    Resized,
    LoadImaged,
    ScaleIntensityRanged,
    RandSpatialCropd,
    EnsureTyped,
    EnsureChannelFirstd,
)
from monai.data import CacheDataset


def get_data_list(root_dir: str, image_key: str = "image", is_train: bool = True) -> List[dict]:
    """
    Generates a list of dictionaries containing file paths for images.
    """
    data_list = []
    
    if is_train:
        normal_dir = os.path.join(root_dir, "train", "NORMAL")
        image_files = glob.glob(os.path.join(normal_dir, "*.nii.gz"))
        for img_path in image_files:
            data_list.append({image_key: img_path, "label": "good"})
    else:
        # Test set (not used during training but kept for completeness)
        normal_dir = os.path.join(root_dir, "test", "NORMAL")
        abnormal_dir = os.path.join(root_dir, "test", "ABNORMAL")
        
        normal_images = glob.glob(os.path.join(normal_dir, "*.nii.gz"))
        for img_path in normal_images:
            data_list.append({image_key: img_path, "label": "good"})
            
        abnormal_images = glob.glob(os.path.join(abnormal_dir, "*.nii.gz"))
        for img_path in abnormal_images:
            data_list.append({image_key: img_path, "label": "defective"})
    
    return data_list


class SHOMRI(Dataset):
    """
    nnUNet-style patch sampling dataset for 3D medical images.
    Samples multiple random patches per volume per epoch.
    """
    def __init__(self, root_dir: str, patch_size: tuple = (64, 64, 64), 
                 patches_per_volume: int = 4, is_train: bool = True, 
                 cache_rate: float = 1.0):
        """
        Args:
            root_dir: Root directory containing train/test folders
            patch_size: Size of patches to extract (D, H, W)
            patches_per_volume: Number of random patches per volume per epoch
            is_train: Whether to use training set
            cache_rate: Fraction of dataset to cache in memory (1.0 = cache all)
        """
        self.image_key = "image"
        self.is_train = is_train
        self.patch_size = tuple(patch_size)
        self.patches_per_volume = patches_per_volume
        
        # Get list of data files
        data_list = get_data_list(root_dir, self.image_key, is_train)
        
        if len(data_list) == 0:
            raise ValueError(f"No images found in {root_dir}")
        
        num_volumes = len(data_list)
        num_patches = num_volumes * patches_per_volume if is_train else num_volumes
        print(f"Found {num_volumes} volumes for {'training' if is_train else 'testing'}")
        print(f"Total patches per epoch: {num_patches}")
        
        # Define transforms
        if is_train:
            # Training: Load full volumes, cache them, then crop on-the-fly
            self.base_transforms = Compose([
                LoadImaged(keys=[self.image_key]),
                EnsureChannelFirstd(keys=[self.image_key], channel_dim="no_channel"),
                Resized(keys=[self.image_key], spatial_size=[128, 128, 128], mode="trilinear"),
                ScaleIntensityRanged(
                    keys=[self.image_key], 
                    a_min=-1.0, 
                    a_max=1.0, 
                    b_min=0.0, 
                    b_max=1.0, 
                    clip=True
                ),
                EnsureTyped(keys=[self.image_key], dtype=torch.float32),
            ])
            
            # Cache full volumes
            self.cached_volumes = CacheDataset(
                data=data_list,
                transform=self.base_transforms,
                cache_rate=cache_rate,
                num_workers=4,
            )
            
            # Random crop transform (applied on-the-fly, NOT cached)
            self.crop_transform = RandSpatialCropd(
                keys=[self.image_key],
                roi_size=self.patch_size,
                random_center=True,
                random_size=False,
            )
            
        else:
            # Test: Load full volumes without cropping
            self.transforms = Compose([
                LoadImaged(keys=[self.image_key]),
                EnsureChannelFirstd(keys=[self.image_key], channel_dim="no_channel"),
                Resized(keys=[self.image_key], spatial_size=[128, 128, 128], mode="trilinear"),
                ScaleIntensityRanged(
                    keys=[self.image_key],
                    a_min=-1.0,
                    a_max=1.0,
                    b_min=0.0,
                    b_max=1.0,
                    clip=True
                ),
                EnsureTyped(keys=[self.image_key], dtype=torch.float32),
            ])
            
            self.cached_volumes = CacheDataset(
                data=data_list,
                transform=self.transforms,
                cache_rate=cache_rate,
                num_workers=4,
            )
    
    def __len__(self) -> int:
        if self.is_train:
            return len(self.cached_volumes) * self.patches_per_volume
        return len(self.cached_volumes)
    
    def __getitem__(self, index):
        """
        Returns a random patch from a volume.
        
        For training: Maps index to volume, then samples random patch
        For testing: Returns full volume
        """
        if self.is_train:
            # Map linear index to volume index
            volume_idx = index % len(self.cached_volumes)
            
            # Get cached full volume
            volume_dict = self.cached_volumes[volume_idx]
            
            # Apply random crop (this happens on-the-fly, so different each time)
            patch_dict = self.crop_transform(volume_dict)
            
            # Return just the image patch as a tuple for compatibility
            return (patch_dict[self.image_key],)
        else:
            # Return full volume for testing
            data_dict = self.cached_volumes[index]
            return data_dict


class SHOMRIGridPatches(Dataset):
    """
    Alternative: Extract all non-overlapping patches from each volume.
    More systematic, ensures full coverage of each volume.
    """
    def __init__(self, root_dir: str, patch_size: tuple = (64, 64, 64),
                 is_train: bool = True, cache_rate: float = 1.0):
        """
        Args:
            root_dir: Root directory
            patch_size: Size of patches (D, H, W)
            is_train: Whether to use training set
            cache_rate: Fraction to cache
        """
        self.image_key = "image"
        self.is_train = is_train
        self.patch_size = tuple(patch_size)
        
        data_list = get_data_list(root_dir, self.image_key, is_train)
        
        if len(data_list) == 0:
            raise ValueError(f"No images found in {root_dir}")
        
        if is_train:
            # Load and cache full volumes
            self.base_transforms = Compose([
                LoadImaged(keys=[self.image_key]),
                EnsureChannelFirstd(keys=[self.image_key], channel_dim="no_channel"),
                Resized(keys=[self.image_key], spatial_size=[128, 128, 128], mode="trilinear"),
                ScaleIntensityRanged(
                    keys=[self.image_key],
                    a_min=-1.0,
                    a_max=1.0,
                    b_min=0.0,
                    b_max=1.0,
                    clip=True
                ),
                EnsureTyped(keys=[self.image_key], dtype=torch.float32),
            ])
            
            self.cached_volumes = CacheDataset(
                data=data_list,
                transform=self.base_transforms,
                cache_rate=cache_rate,
                num_workers=4,
            )
            
            # Pre-compute patch locations for 128x128x128 volumes
            # Assumes all volumes are same size
            volume_size = (128, 128, 128)
            self.patch_locations = []
            
            # Calculate non-overlapping grid
            for d in range(0, volume_size[0], patch_size[0]):
                for h in range(0, volume_size[1], patch_size[1]):
                    for w in range(0, volume_size[2], patch_size[2]):
                        if (d + patch_size[0] <= volume_size[0] and 
                            h + patch_size[1] <= volume_size[1] and 
                            w + patch_size[2] <= volume_size[2]):
                            self.patch_locations.append((d, h, w))
            
            self.patches_per_volume = len(self.patch_locations)
            total_patches = len(self.cached_volumes) * self.patches_per_volume
            
            print(f"Found {len(self.cached_volumes)} volumes")
            print(f"Patches per volume: {self.patches_per_volume}")
            print(f"Total patches: {total_patches}")
            
        else:
            self.transforms = Compose([
                LoadImaged(keys=[self.image_key]),
                EnsureChannelFirstd(keys=[self.image_key], channel_dim="no_channel"),
                Resized(keys=[self.image_key], spatial_size=[128, 128, 128], mode="trilinear"),
                ScaleIntensityRanged(
                    keys=[self.image_key],
                    a_min=-1.0,
                    a_max=1.0,
                    b_min=0.0,
                    b_max=1.0,
                    clip=True
                ),
                EnsureTyped(keys=[self.image_key], dtype=torch.float32),
            ])
            
            self.cached_volumes = CacheDataset(
                data=data_list,
                transform=self.transforms,
                cache_rate=cache_rate,
                num_workers=4,
            )
    
    def __len__(self) -> int:
        if self.is_train:
            return len(self.cached_volumes) * self.patches_per_volume
        return len(self.cached_volumes)
    
    def __getitem__(self, index):
        if self.is_train:
            # Map index to volume and patch location
            volume_idx = index // self.patches_per_volume
            patch_idx = index % self.patches_per_volume
            
            # Get cached volume
            volume_dict = self.cached_volumes[volume_idx]
            volume = volume_dict[self.image_key]
            
            # Extract specific patch
            d, h, w = self.patch_locations[patch_idx]
            patch = volume[
                :,
                d:d+self.patch_size[0],
                h:h+self.patch_size[1],
                w:w+self.patch_size[2]
            ]
            
            return (patch,)
        else:
            data_dict = self.cached_volumes[index]
            return data_dict
