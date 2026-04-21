import nibabel as nib
import numpy as np
import glob
import os

def check_dataset(root_dir):
    files = sorted(glob.glob(os.path.join(root_dir, "**", "*.nii.gz"), recursive=True))
    for f in files:
        img = nib.load(f).get_fdata()
        if img.min() == img.max():
            print(f"Constant image: {f} | value: {img.min()}")
        elif img.max() == 0:
            print(f"Empty image: {f}")
        else:
            print(f"OK: {f} | min: {img.min():.2f} max: {img.max():.2f}")

check_dataset("/projects/prjs1633/anomaly_detection/SHOMRI/train/NORMAL")