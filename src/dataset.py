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
    RandFlipd,
    RandRotate90d,
    RandAffined
)
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityRangePercentilesd,
    ScaleIntensityd,
    RandFlipd,
    RandRotate90d,
    EnsureTyped,
)
from monai.data import CacheDataset, DataLoader, Dataset, pad_list_data_collate


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
        normal_dir = os.path.join(root_dir, "test2", "NORMAL")
        abnormal_dir = os.path.join(root_dir, "test2", "ABNORMAL")
        
        normal_images = sorted(glob.glob(os.path.join(normal_dir, "*.nii.gz")))
        for img_path in normal_images:
            # filename = os.path.basename(img_path)
            data_list.append({image_key: img_path, "label": "good", "filename": os.path.basename(img_path)})
            
        abnormal_images = sorted(glob.glob(os.path.join(abnormal_dir, "*.nii.gz")))
        for img_path in abnormal_images:
            # filename = os.path.basename(img_path)
            data_list.append({image_key: img_path, "label": "defective", "filename": os.path.basename(img_path)})
    
    return data_list



def get_transforms(is_train: bool = True, image_key: str = "image"):
    """
    Returns MONAI transforms pipeline.
    Intensity is normalized to [-1, 1] as required by the diffusion model.
    """
    base_transforms = [
        LoadImaged(keys=[image_key]),
        EnsureChannelFirstd(keys=[image_key]),
        Resized(keys=[image_key], spatial_size=[128, 128, 128], mode="trilinear"),
        # First clip extreme outliers using percentiles, then scale to [-1, 1]
        ScaleIntensityRangePercentilesd(
            keys=[image_key],
            lower=0.5,
            upper=99.5,
            b_min=-1.0,
            b_max=1.0,
            clip=True,
        ),
        EnsureTyped(keys=[image_key], dtype=torch.float32),
    ]

    if is_train:
        augmentation_transforms = [
            RandFlipd(keys=[image_key], prob=0.5, spatial_axis=0),
            RandFlipd(keys=[image_key], prob=0.5, spatial_axis=1),
            RandFlipd(keys=[image_key], prob=0.5, spatial_axis=2),
            RandRotate90d(keys=[image_key], prob=0.5, spatial_axes=(0, 1)),
        ]
        return Compose(base_transforms + augmentation_transforms)

    return Compose(base_transforms)


def SHOMRI(
    root_dir: str,
    batch_size: int,
    image_key: str = "image",
    is_train: bool = True,
    num_workers: int = 4,
    cache_rate: float = 1.0,
):
    def custom_collate(batch):
        return {
            "image":    pad_list_data_collate([b["image"] for b in batch]),
            "label":    [b["label"]    for b in batch],
            "filename": [b["filename"] for b in batch],
        }
    """
    Builds a MONAI CacheDataset and DataLoader.

    Args:
        root_dir:    Root directory of the dataset.
        batch_size:  Batch size for the DataLoader.
        image_key:   Key used for images in the data dicts.
        is_train:    Whether to build the training set (with augmentation) or test set.
        num_workers: Number of DataLoader worker processes.
        cache_rate:  Fraction of data to cache in RAM (1.0 = full cache).
    """
    data_list = get_data_list(root_dir, image_key=image_key, is_train=is_train)
    transforms = get_transforms(is_train=is_train, image_key=image_key)

    ds = CacheDataset(
        data=data_list,
        transform=transforms,
        cache_rate=cache_rate,
        num_workers=num_workers,
    )

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=is_train,       # shuffle only for training
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=custom_collate,
    )

    return ds, loader



def get_data_list_with_masks(root_dir: str, image_key: str = "image", is_train: bool = True) -> List[dict]:
    """
    Generates a list of dictionaries containing file paths for images and their masks.
    Masks are loaded from root_dir/mask/<same_filename> regardless of train/test split.
    """
    data_list = []
    mask_dir = os.path.join(root_dir, "mask")

    if is_train:
        normal_dir = os.path.join(root_dir, "train", "NORMAL")
        image_files = sorted(glob.glob(os.path.join(normal_dir, "*.nii.gz")))
        for img_path in image_files:
            filename = os.path.basename(img_path)
            mask_path = os.path.join(mask_dir, filename)
            data_list.append({
                image_key: img_path,
                "mask": mask_path,
                "label": "good",
                "filename": filename,
            })
    else:
        normal_dir = os.path.join(root_dir, "test2", "NORMAL")
        abnormal_dir = os.path.join(root_dir, "test2", "ABNORMAL")

        normal_images = sorted(glob.glob(os.path.join(normal_dir, "*.nii.gz")))
        for img_path in normal_images:
            filename = os.path.basename(img_path)
            mask_path = os.path.join(mask_dir, filename)
            data_list.append({
                image_key: img_path,
                "mask": mask_path,
                "label": "good",
                "filename": filename,
            })

        abnormal_images = sorted(glob.glob(os.path.join(abnormal_dir, "*.nii.gz")))
        for img_path in abnormal_images:
            filename = os.path.basename(img_path)
            mask_path = os.path.join(mask_dir, filename)
            data_list.append({
                image_key: img_path,
                "mask": mask_path,
                "label": "defective",
                "filename": filename,
            })

    return data_list


def get_transforms_mask(is_train: bool = True, image_key: str = "image"):
    """
    Returns MONAI transforms pipeline.
    - Image: resized trilinear + intensity scaled to [-1, 1]
    - Mask:  resized nearest-neighbor, no intensity scaling
    """
    base_transforms = [
        LoadImaged(keys=[image_key, "mask"]),
        EnsureChannelFirstd(keys=[image_key, "mask"]),
        # Image: trilinear interpolation
        Resized(keys=[image_key], spatial_size=[128, 128, 128], mode="trilinear"),
        # Mask: nearest-neighbor to preserve label integers
        Resized(keys=["mask"], spatial_size=[128, 128, 128], mode="nearest"),
        ScaleIntensityRangePercentilesd(
            keys=[image_key],
            lower=0.5,
            upper=99.5,
            b_min=-1.0,
            b_max=1.0,
            clip=True,
        ),
        EnsureTyped(keys=[image_key], dtype=torch.float32),
        EnsureTyped(keys=["mask"], dtype=torch.long),
    ]

    if is_train:
        augmentation_transforms = [
            # Apply flips and rotations consistently to both image and mask
            RandFlipd(keys=[image_key, "mask"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=[image_key, "mask"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=[image_key, "mask"], prob=0.5, spatial_axis=2),
            RandRotate90d(keys=[image_key, "mask"], prob=0.5, spatial_axes=(0, 1)),
        ]
        return Compose(base_transforms + augmentation_transforms)

    return Compose(base_transforms)


def SHOMRI_with_mask(
    root_dir: str,
    batch_size: int,
    image_key: str = "image",
    is_train: bool = True,
    num_workers: int = 4,
    cache_rate: float = 1.0,
):
    def custom_collate(batch):
        return {
            "image":    pad_list_data_collate([b["image"] for b in batch]),
            "mask":     pad_list_data_collate([b["mask"]  for b in batch]),
            "label":    [b["label"]    for b in batch],
            "filename": [b["filename"] for b in batch],
        }

    """
    Builds a MONAI CacheDataset and DataLoader.

    Args:
        root_dir:    Root directory of the dataset.
        batch_size:  Batch size for the DataLoader.
        image_key:   Key used for images in the data dicts.
        is_train:    Whether to build the training set (with augmentation) or test set.
        num_workers: Number of DataLoader worker processes.
        cache_rate:  Fraction of data to cache in RAM (1.0 = full cache).
    """
    data_list = get_data_list_with_masks(root_dir, image_key=image_key, is_train=is_train)
    transforms = get_transforms_mask(is_train=is_train, image_key=image_key)

    ds = CacheDataset(
        data=data_list,
        transform=transforms,
        cache_rate=cache_rate,
        num_workers=num_workers,
    )

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=custom_collate,
    )

    return ds, loader
