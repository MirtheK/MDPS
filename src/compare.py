import torch
import torch.nn.functional as F
from kornia.filters import gaussian_blur2d
from torchvision.transforms import transforms
import math 


import torch
import torch.nn.functional as F
from torchvision.transforms import transforms
import math
from typing import Any, List, Tuple

def get_gaussian_kernel1d(kernel_size: int, sigma: float) -> torch.Tensor:
    """Generates a 1D Gaussian kernel."""
    coords = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g /= g.sum()
    return g

def gaussian_blur3d(input_tensor: torch.Tensor, kernel_size: Tuple[int, int, int], sigma: Tuple[float, float, float]) -> torch.Tensor:
    """
    Applies a 3D Gaussian blur to a tensor by applying a 1D Gaussian filter
    sequentially along each spatial dimension.
    Input shape: (N, C, D, H, W).
    """
    assert len(kernel_size) == 3 and len(sigma) == 3

    # Generate 1D kernels for each dimension
    kernel_d = get_gaussian_kernel1d(kernel_size[0], sigma[0]).to(input_tensor.device)
    kernel_h = get_gaussian_kernel1d(kernel_size[1], sigma[1]).to(input_tensor.device)
    kernel_w = get_gaussian_kernel1d(kernel_size[2], sigma[2]).to(input_tensor.device)

    # Reshape kernels for 3D convolution
    kernel_d = kernel_d.view(1, 1, -1, 1, 1)
    kernel_h = kernel_h.view(1, 1, 1, -1, 1)
    kernel_w = kernel_w.view(1, 1, 1, 1, -1)

    padding_d = kernel_size[0] // 2
    padding_h = kernel_size[1] // 2
    padding_w = kernel_size[2] // 2

    # Apply convolution for each dimension
    # The input needs to be a float tensor for convolution
    output = input_tensor.float()
    
    # Blur along Depth dimension
    output = F.conv3d(output, kernel_d.repeat(output.shape[1], 1, 1, 1, 1), padding=(padding_d, 0, 0), groups=output.shape[1])
    
    # Blur along Height dimension
    output = F.conv3d(output, kernel_h.repeat(output.shape[1], 1, 1, 1, 1), padding=(0, padding_h, 0), groups=output.shape[1])
    
    # Blur along Width dimension
    output = F.conv3d(output, kernel_w.repeat(output.shape[1], 1, 1, 1, 1), padding=(0, 0, padding_w), groups=output.shape[1])

    return output

# def distance(input1: torch.Tensor, input2: torch.Tensor, resnet: torch.nn.Module, config: Any) -> torch.Tensor:
#     """
#     Combines MSE and LPIPS-like distance for 3D volumes to create an anomaly map.
#     Assumes inputs are 5D tensors (N, C, D, H, W).
#     """
#     sigma = 4
#     kernel_size = 2 * int(4 * sigma + 0.5) + 1
#     device = config.model.device
#     input1 = input1.to(device)
#     input2 = input2.to(device)

#     i_d = MSE(input1, input2)
#     f_d = LPIPS(input1, input2, resnet, config)
#     f_d = torch.Tensor(f_d).to(device)
    
#     max_f_d = torch.max(f_d)
#     max_i_d = torch.max(i_d)
    
#     anomaly_map = f_d + config.model.eta * max_f_d / max_i_d * i_d
    
#     # Use the custom 3D Gaussian blur function
#     anomaly_map = gaussian_blur3d_custom(anomaly_map, 
#                                          kernel_size=(kernel_size, kernel_size, kernel_size), 
#                                          sigma=(sigma, sigma, sigma))
    
#     # Sum over the channel dimension (dim=1)
#     anomaly_map = torch.sum(anomaly_map, dim=1, keepdim=True)
#     return anomaly_map


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
    
    anomaly_map = gaussian_blur3d(anomaly_map, 
                                  kernel_size=(kernel_size, kernel_size), 
                                  sigma=(sigma, sigma))
    anomaly_map = torch.sum(anomaly_map, dim=1, keepdim=True)
    return anomaly_map

def MSE(output, target):
    transform = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / (2)),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        transforms.Normalize([0.5], [0.5])
        ])
    output = transform(output)
    target = transform(target)
    # distance_map = torch.mean(torch.abs(output - target), dim=1).unsqueeze(1)
    distance_map = torch.mean(torch.abs(output - target), dim=1, keepdim=True)
    return distance_map

def LPIPS(output, target, resnet, config):
    resnet.eval()
    def normalize(tensor):
        # return transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])((tensor + 1) / 2)
        return transforms.Normalize([0.5], [0.5])((tensor + 1) / 2)
    
    target_normalized = normalize(target)
    output_normalized = normalize(output)
    target_features = resnet(target_normalized)
    output_features = resnet(output_normalized)
    
    out_size = config.data.image_size
    device = config.model.device
    anomaly_map = torch.zeros(target_features[0].shape[0], 1, out_size, out_size, device=device)
    for i in range(1, len(target_features)):
        target_patches = patchify(target_features[i])
        output_patches = patchify(output_features[i])

        target_patches_perm = target_patches.permute(0, 2, 3, 4, 1)
        output_patches_perm = output_patches.permute(0, 2, 3, 4, 1)
        
        similarity_map = F.cosine_similarity(target_patches_perm, output_patches_perm, dim=-1)
        a_map = 1 - similarity_map

        # similarity_map = F.cosine_similarity(target_patches, output_patches)
        # a_map = 1 - similarity_map
        # a_map = a_map.unsqueeze(dim=1)
        # interpolated_a_map = F.interpolate(a_map, size=out_size, mode='bilinear', align_corners=True)
        # anomaly_map += interpolated_a_map
        n_patches_dim = round(a_map.shape[0] ** (1/3))
        a_map = a_map.reshape(target_features[i].shape[0], n_patches_dim, n_patches_dim, n_patches_dim)
        a_map = a_map.unsqueeze(1) # Add channel dim
        
        interpolated_a_map = F.interpolate(a_map, size=(out_size, out_size, out_size), mode='trilinear', align_corners=True)
        anomaly_map += interpolated_a_map

    return anomaly_map


# def patchify(features, return_spatial_info=False):
    # patchsize = 3
    # stride = 1
    # padding = int((patchsize - 1) / 2)
    # unfolder = torch.nn.Unfold(
    #     kernel_size=patchsize, stride=stride, padding=padding, dilation=1
    # )
    # unfolded_features = unfolder(features)
    # number_of_total_patches = []
    # for s in features.shape[-2:]:
    #     n_patches = (
    #         s + 2 * padding - 1 * (patchsize - 1) - 1
    #     ) / stride + 1
    #     number_of_total_patches.append(int(n_patches))
    # unfolded_features = unfolded_features.reshape(
    #     *features.shape[:2], patchsize, patchsize, -1
    # )
    # unfolded_features = unfolded_features.permute(0, 4, 1, 2, 3)
    # max_features = torch.mean(unfolded_features, dim=(3,4))
    # features = max_features.reshape(features.shape[0], int(math.sqrt(max_features.shape[1])) , int(math.sqrt(max_features.shape[1])), max_features.shape[-1]).permute(0,3,1,2)
    # if return_spatial_info:
    #     return unfolded_features, number_of_total_patches
    # return features


def patchify(features: torch.Tensor, patchsize: int = 3, stride: int = 1) -> torch.Tensor:
    """
    Extracts 3D patches from a feature tensor using a sliding window approach.
    Returns a flattened tensor of patches.
    Input shape: (N, C, D, H, W)
    Output shape: (N * N_patches, C, patchsize, patchsize, patchsize)
    """
    n, c, d, h, w = features.shape
    
    # Calculate number of patches along each dimension
    patches_d = (d - patchsize) // stride + 1
    patches_h = (h - patchsize) // stride + 1
    patches_w = (w - patchsize) // stride + 1
    
    # Use tensor.unfold for each spatial dimension
    patches = features.unfold(2, patchsize, stride) \
                      .unfold(3, patchsize, stride) \
                      .unfold(4, patchsize, stride)
    
    # Reshape the patches tensor to have a shape (N*N_patches, C, p_d, p_h, p_w)
    patches = patches.permute(0, 1, 5, 6, 7, 2, 3, 4)
    patches = patches.reshape(n, c, patchsize, patchsize, patchsize, patches_d * patches_h * patches_w)
    patches = patches.permute(0, 5, 1, 2, 3, 4)
    patches = patches.reshape(n * patches_d * patches_h * patches_w, c, patchsize, patchsize, patchsize)

    return patches  

