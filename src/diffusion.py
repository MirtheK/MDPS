import torch
import numpy as np


def diffusion_loss(model, x_0, t, config):
    device = config.model.device
    x_0 = x_0.to(device, non_blocking=True)
    if not hasattr(diffusion_loss, "_betas_tensor"):
        betas = np.linspace(
            config.model.beta_start, 
            config.model.beta_end, 
            config.model.diffusion_steps, 
            dtype=np.float32 
        )
        diffusion_loss._betas_tensor = torch.tensor(betas, dtype=torch.float32, device=device)
        diffusion_loss._alphas_cumprod = torch.cumprod(1.0 - diffusion_loss._betas_tensor, dim=0)  
    betas_tensor = diffusion_loss._betas_tensor
    alphas_cumprod = diffusion_loss._alphas_cumprod
    noise = torch.randn_like(x_0, device=device)
    alpha_t = alphas_cumprod.index_select(0, t).view(-1, 1, 1, 1, 1)
    noisy_input = alpha_t.sqrt() * x_0 + (1 - alpha_t).sqrt() * noise
    model_output = model(noisy_input, t)
    loss = (noise - model_output).pow(2).mean()
    return loss

def sample(y0, x, seq, model, config, w):
    with torch.no_grad():
        n = x.size(0)
        device = x.device
        seq_next = [-1] + list(seq[:-1])
        xs = [x]
        seq_list = list(reversed(seq))
        seq_next_list = list(reversed(seq_next))
        t_all     = torch.tensor(seq_list,      dtype=torch.long, device=device)
        t_next_all = torch.tensor(seq_next_list, dtype=torch.long, device=device)
        at_all      = compute_alpha(t_all,      config)   # (num_steps,)
        at_next_all = compute_alpha(t_next_all, config)   # (num_steps,)

        for index in range(len(seq_list)):
            i = seq_list[index]
            at      = at_all[index]   
            at_next = at_next_all[index]
            t = (torch.ones(n, device=device) * i).long()
            xt = xs[-1]
            et = model(xt, t)
            if w == 0:
                et_hat = et
            else:
                guidance = condition_score(model, xt, et, y0, t, config)
                et_hat = et + (1 - at).sqrt() * w * guidance

            x0_t = (xt - et_hat * (1 - at).sqrt()) / at.sqrt()
            c1 = ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et_hat
            xs.append(xt_next)

    return xs[-1].cpu()

def sample_mask(y0, mask, seq, model, config, w):
    with torch.no_grad():
        n = y0.size(0)
        device = y0.device
        seq_list      = list(reversed(seq))
        seq_next_list = [-1] + list(seq[:-1])
        seq_next_list = list(reversed(seq_next_list))

        t_all      = torch.tensor(seq_list,      dtype=torch.long, device=device)
        t_next_all = torch.tensor(seq_next_list, dtype=torch.long, device=device)
        at_all      = compute_alpha(t_all,      config)  # (num_steps,)
        at_next_all = compute_alpha(t_next_all, config)  # (num_steps,)

        z = torch.randn_like(y0)
        at_T = at_all[0]  # largest t = most noise = first in reversed seq
        x = at_T.sqrt() * y0 + (1 - at_T).sqrt() * z

        xs = [x]

        for index in range(len(seq_list)):
            at      = at_all[index]
            at_next = at_next_all[index]
            t = (torch.ones(n, device=device) * seq_list[index]).long()

            xt = xs[-1]
            xt_noise = at.sqrt() * y0 + (1 - at).sqrt() * z
            xt = torch.where(mask == 1, xt, xt_noise)

            et = model(xt, t)
            if w == 0:
                et_hat = et
            else:
                # Posterior guidance — only applied in anomaly region (mask==1)
                guidance = condition_score(model, xt, et, y0, t, config)
                et_hat = et + (1 - at).sqrt() * w * guidance
                # Normal regions use plain et, anomaly regions use guided et
                # et_hat = torch.where(mask == 0, et, et_guided)
            x0_t = (xt - et_hat * (1 - at).sqrt()) / at.sqrt()
            c1   = ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            c2   = ((1 - at_next) - c1 ** 2).sqrt()

            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et_hat
            xs.append(xt_next)

    return xs[-1].cpu()

def condition_score(model, xt, et, x_guidance, t, config):
    """
    Posterior guidance: gradient of log p(y|xt).
    Implements the guidance term from Eq. 16 in the paper:
    rho * sqrt(1-alpha_t) * grad_xt ||y - x0_prior||^2
    The rho * sqrt(1-alpha_t) scaling is applied by the caller.
    """
    with torch.enable_grad():
        x_in = xt.detach().requires_grad_(True)
        et = model(x_in, t)
        at = compute_alpha(t.long(),config)
        x_0_hat = (x_in - et * (1 - at).sqrt()) / at.sqrt()
        difference_x = x_0_hat-x_guidance
        norm_x = torch.linalg.norm(difference_x)
        test_grad = torch.autograd.grad(outputs=norm_x, inputs=x_in)[0]
    return test_grad
    # at = compute_alpha(t.long(), config)
    # x0_prior = (xt - et * (1 - at).sqrt()) / at.sqrt()
    # guidance = y0 - x0_prior
    # guidance = guidance / (guidance.norm() + 1e-8)
    # return guidance

def compute_alpha(t, config):
    betas = np.linspace(config.model.beta_start, config.model.beta_end, config.model.diffusion_steps, dtype=np.float32)
    betas = torch.tensor(betas).type(torch.float)
    beta = torch.cat([torch.zeros(1).to(betas.device), betas], dim=0)
    beta = beta.to(config.model.device)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1, 1)
    return a