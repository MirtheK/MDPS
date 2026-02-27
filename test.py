import torch
import numpy as np
from omegaconf import OmegaConf
from src.diffusion import compute_alpha

config = OmegaConf.load("/projects/prjs1633/repositories/MDPS/config/train.yaml")

print(f"{'t':>6} | {'at (signal retain)':>18} | {'SNR':>8} | verdict")
print("-" * 55)

for t in [50, 100, 150, 200, 250, 300, 400, 500]:
    device = config.model.device
    t_tensor = torch.tensor([t], dtype=torch.long)
    t_tensor = t_tensor.to(device)
    at = compute_alpha(t_tensor, config)          # signal retention factor
    signal_weight = at.sqrt().item()
    noise_weight  = (1 - at).sqrt().item()
    snr = signal_weight / (noise_weight + 1e-8)
    
    verdict = ""
    if snr > 1.5:   verdict = "too little noise (anomaly survives)"
    elif snr > 0.5: verdict = " good range"
    elif snr > 0.2: verdict = " borderline"
    else:           verdict = " near-pure noise"
    
    print(f"{t:>6} | {signal_weight:>18.4f} | {snr:>8.4f} | {verdict}")
