import os
import glob
from typing import Any, List
import numpy as np
import torch
from torch.utils.data import Dataset
import nibabel as nib
from copy import deepcopy

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
        image_files = sorted(glob.glob(os.path.join(normal_dir, "*.nii.gz")))
        for img_path in image_files:
            # filename = os.path.basename(img_path)
            data_list.append({image_key: img_path, "label": "good", "filename": os.path.basename(img_path)})
    else:
        # Test set
        normal_dir = os.path.join(root_dir, "test", "NORMAL")
        abnormal_dir = os.path.join(root_dir, "test", "ABNORMAL/img")
        
        normal_images = sorted(glob.glob(os.path.join(normal_dir, "*.nii.gz")))
        for img_path in normal_images:
            # filename = os.path.basename(img_path)
            data_list.append({image_key: img_path, "label": "good", "filename": os.path.basename(img_path)})
            
        abnormal_images = sorted(glob.glob(os.path.join(abnormal_dir, "*.nii.gz")))
        for img_path in abnormal_images:
            # filename = os.path.basename(img_path)
            data_list.append({image_key: img_path, "label": "defective", "filename": os.path.basename(img_path)})
    
    return data_list


class SHOMRI(Dataset):
    """
    FIXED: Non-blocking patch sampling dataset for 3D medical images.
    
    KEY FIXES:
    1. Deep copy volumes before cropping to avoid blocking
    2. Persistent workers compatible
    3. Thread-safe transforms
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
                    a_min= 0, 
                    a_max= 600, 
                    b_min=0.0, 
                    b_max=1.0, 
                    clip=True
                ),
                EnsureTyped(keys=[self.image_key], dtype=torch.float32),
            ])
            
            # Cache full volumes
            # IMPORTANT: Use copy_cache=True to avoid blocking issues
            self.cached_volumes = CacheDataset(
                data=data_list,
                transform=self.base_transforms,
                cache_rate=cache_rate,
                num_workers=4,
                copy_cache=True,  # CRITICAL: Makes a copy when retrieving cached items
            )
            
            # Random crop transform (applied on-the-fly, NOT cached)
            # Each worker needs its own random state - handled by PyTorch
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
                    a_min= 0,
                    a_max= 600,
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
                copy_cache=True,  # CRITICAL: For thread safety
            )
    
    def __len__(self) -> int:
        if self.is_train:
            return len(self.cached_volumes) * self.patches_per_volume
        return len(self.cached_volumes)
    
    def __getitem__(self, index):
        """
        Returns a random patch from a volume.
        
        FIXED: Deep copies volume before cropping to avoid blocking issues
        """
        if self.is_train:
            # Map linear index to volume index
            volume_idx = index % len(self.cached_volumes)
            
            # Get cached full volume
            volume_dict = self.cached_volumes[volume_idx]
            
            # CRITICAL FIX: Deep copy to avoid in-place modification blocking
            # Without this, multiple workers accessing same cached volume causes blocking
            volume_dict_copy = {
                self.image_key: volume_dict[self.image_key].clone(),  # Clone the tensor
            }
            
            # Apply random crop (different each time due to copy)
            patch_dict = self.crop_transform(volume_dict_copy)
            
            # Return just the image patch as a tuple for compatibility
            return (patch_dict[self.image_key],)
        else:
            # Return full volume for testing
            data_dict = self.cached_volumes[index]
            return data_dict


class SHOMRIGridPatches(Dataset):
    """
    FIXED: Non-blocking grid patch extraction.
    Extract all non-overlapping patches from each volume.
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
                    a_min= 0,
                    a_max= 600,
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
                copy_cache=True,  # CRITICAL: Avoid blocking
            )
            
            # Pre-compute patch locations for 128x128x128 volumes
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
                    a_min= 0,
                    a_max= 600,
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
                copy_cache=True,  # CRITICAL: Avoid blocking
            )
    
    def __len__(self) -> int:
        if self.is_train:
            return len(self.cached_volumes) * self.patches_per_volume
        return len(self.cached_volumes)
    
    def __getitem__(self, index):
        """
        FIXED: Uses slicing (creates a view) which is thread-safe
        """
        if self.is_train:
            # Map index to volume and patch location
            volume_idx = index // self.patches_per_volume
            patch_idx = index % self.patches_per_volume
            
            # Get cached volume
            volume_dict = self.cached_volumes[volume_idx]
            volume = volume_dict[self.image_key]
            
            # Extract specific patch (slicing creates a new view, thread-safe)
            d, h, w = self.patch_locations[patch_idx]
            patch = volume[
                :,
                d:d+self.patch_size[0],
                h:h+self.patch_size[1],
                w:w+self.patch_size[2]
            ].clone()  # CRITICAL: Clone to avoid blocking
            
            return (patch,)
        else:
            data_dict = self.cached_volumes[index]
            return data_dict



class SHOMRIPreExtracted(Dataset):
    """
    BEST OPTION FOR AVOIDING BLOCKING: Pre-extract all patches during initialization.
    
    Pros:
    - Zero blocking issues
    - Fastest training (no on-the-fly operations)
    - Works with persistent_workers=True
    
    Cons:
    - Uses more memory (stores all patches)
    - Longer initialization time
    - Less data augmentation (patches are fixed)
    """
    def __init__(self, root_dir: str, patch_size: tuple = (64, 64, 64), 
                 patches_per_volume: int = 4, is_train: bool = True):
        """
        Args:
            root_dir: Root directory
            patch_size: Size of patches to extract
            patches_per_volume: Number of random patches per volume
            is_train: Whether to use training set
        """
        self.image_key = "image"
        self.is_train = is_train
        self.patch_size = tuple(patch_size)
        
        print(f"Pre-extracting patches (this may take a few minutes)...")
        
        # Get data list
        data_list = get_data_list(root_dir, self.image_key, is_train)
        
        if len(data_list) == 0:
            raise ValueError(f"No images found in {root_dir}")
        
        # Load and transform volumes
        transforms = Compose([
            LoadImaged(keys=[self.image_key]),
            EnsureChannelFirstd(keys=[self.image_key], channel_dim="no_channel"),
            Resized(keys=[self.image_key], spatial_size=[128, 128, 128], mode="trilinear"),
            ScaleIntensityRanged(
                keys=[self.image_key],
                a_min= 0,
                a_max= 600,
                b_min=0.0,
                b_max=1.0,
                clip=True
            ),
            EnsureTyped(keys=[self.image_key], dtype=torch.float32),
        ])
        
        # Pre-extract all patches
        self.patches = []
        
        for vol_data in data_list:
            volume_dict = transforms(vol_data)
            volume = volume_dict[self.image_key]
            
            if is_train:
                # Extract random patches
                C, D, H, W = volume.shape
                for _ in range(patches_per_volume):
                    # Random crop coordinates
                    d_start = np.random.randint(0, D - patch_size[0] + 1)
                    h_start = np.random.randint(0, H - patch_size[1] + 1)
                    w_start = np.random.randint(0, W - patch_size[2] + 1)
                    
                    patch = volume[
                        :,
                        d_start:d_start+patch_size[0],
                        h_start:h_start+patch_size[1],
                        w_start:w_start+patch_size[2]
                    ].clone()
                    
                    self.patches.append(patch)
            else:
                # Store full volume
                self.patches.append(volume)
        
        print(f"Pre-extracted {len(self.patches)} patches")
    
    def __len__(self) -> int:
        return len(self.patches)
    
    def __getitem__(self, index):
        if self.is_train:
            return (self.patches[index],)
        else:
            # For testing, return as dict
            return {self.image_key: self.patches[index]}