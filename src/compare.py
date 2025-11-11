import torch
import torch.nn.functional as F
from torchvision.transforms import transforms
import math 
import numpy as np 
from skimage.filters import gaussian
from typing import Any, List, Tuple


def gaussian_blur(x: torch.Tensor, sigma, kernel_size=None, mode='reflect'):
    """
    
    """
    x_np = x.detach().cpu().numpy()
    
    out = np.empty_like(x_np)

    # Loop over batch and channels
    for n in range(x_np.shape[0]):
        for c in range(x_np.shape[1]):
            out[n, c] = gaussian(
                x_np[n, c],
                sigma=sigma,
                mode=mode,
                preserve_range=True,
            )

    return torch.from_numpy(out).to(x.device, dtype=x.dtype)


def distance(input1, input2, resnet, config):
    sigma = 4
    kernel_size = 2 * int(4 * sigma + 0.5) + 1
    anomaly_map = 0
    device = config.model.device
    input1 = input1.to(device)
    input2 = input2.to(device)
    i_d = MSE(input1, input2)
    f_d = LPIPS(input1, input2, resnet, config)
    f_d = torch.Tensor(f_d).to(device)
    max_f_d = torch.max(f_d)
    max_i_d = torch.max(i_d)
    anomaly_map += f_d + config.model.eta * max_f_d/max_i_d * i_d
    
    anomaly_map = gaussian_blur(anomaly_map, sigma)  
                                
    anomaly_map = torch.sum(anomaly_map, dim=1, keepdim=True)
    return anomaly_map


def MSE(output, target):
    transform = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / (2)),
        transforms.Normalize([0.5], [0.5])
        ])
    output = transform(output)
    target = transform(target)
    distance_map = torch.mean(torch.abs(output - target), dim=1, keepdim=True)
    return distance_map


def LPIPS(output, target, resnet, config):
    resnet.eval()
    def normalize(tensor):
        return transforms.Normalize([0.5], [0.5])((tensor + 1) / 2)
    
    target_normalized = normalize(target)
    output_normalized = normalize(output)
    target_features = resnet(target_normalized)
    output_features = resnet(output_normalized)
    
    out_size = config.data.image_size
    device = config.model.device
    anomaly_map = torch.zeros(target_features[0].shape[0], 1, out_size, out_size, out_size, device=device)

    for i in range(1, len(target_features)):
        target_patches = patchify(target_features[i])
        output_patches = patchify(output_features[i])
        similarity_map = F.cosine_similarity(target_patches, output_patches)
        a_map = 1 - similarity_map 
        a_map = torch.mean(a_map, dim=0, keepdim=True) 
        a_map = a_map.unsqueeze(dim=1) 

        interpolated_a_map = F.interpolate(a_map, size=out_size, mode='trilinear', align_corners=True)
        anomaly_map += interpolated_a_map

    return anomaly_map



def patchify(features: torch.Tensor, patchsize: int = 3, stride: int = 1) -> torch.Tensor:
    """
    Extracts 3D patches from a feature tensor using a sliding window approach.
    Returns a flattened tensor of patches.
    Input shape: (N, C, D, H, W)
    Output shape: (N * N_patches, C, patchsize, patchsize, patchsize)
    """
    n, c, d, h, w = features.shape
    
    patches_d = (d - patchsize) // stride + 1
    patches_h = (h - patchsize) // stride + 1
    patches_w = (w - patchsize) // stride + 1
    

    patches = features.unfold(2, patchsize, stride) \
                      .unfold(3, patchsize, stride) \
                      .unfold(4, patchsize, stride)
    
    patches = patches.permute(0, 1, 5, 6, 7, 2, 3, 4)
    patches = patches.reshape(n, c, patchsize, patchsize, patchsize, patches_d * patches_h * patches_w)
    patches = patches.permute(0, 5, 1, 2, 3, 4)
    patches = patches.reshape(n * patches_d * patches_h * patches_w, c, patchsize, patchsize, patchsize)

    return patches  

