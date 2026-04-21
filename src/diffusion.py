import torch
import numpy as np


def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = np.linspace(0, timesteps, steps)
    alphas_cumprod = np.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod /= alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, 0, 0.999)


def diffusion_loss(model, x_0, t, config):
    device = config.model.device
    if (not hasattr(diffusion_loss, "_betas_tensor") or
        diffusion_loss._betas_tensor.device != device):

        betas = cosine_beta_schedule(config.model.diffusion_steps)

        # betas = np.linspace(
        #     config.model.beta_start,
        #     config.model.beta_end,
        #     config.model.diffusion_steps,
        #     dtype=np.float32
        # )

        diffusion_loss._betas_tensor = torch.tensor(betas, device=device, dtype=torch.float32)
        diffusion_loss._alphas_cumprod = torch.cumprod(
            1.0 - diffusion_loss._betas_tensor, dim=0
        )
    betas_tensor = diffusion_loss._betas_tensor
    alphas_cumprod = diffusion_loss._alphas_cumprod
    noise = torch.randn_like(x_0, device=device)
    alpha_t = alphas_cumprod.index_select(0, t)
    while alpha_t.dim() < x_0.dim():
        alpha_t = alpha_t.unsqueeze(-1)
    noisy_input = alpha_t.sqrt() * x_0 + (1 - alpha_t).sqrt() * noise
    model_output = model(noisy_input, t)
    loss = (noise - model_output).pow(2).mean()
    return loss


# Module-level cache for alpha values — computed once, reused everywhere
_alpha_cache = {}

def compute_alpha(t, config):
    key = (config.model.beta_start, config.model.beta_end, config.model.diffusion_steps, config.model.device)
    if key not in _alpha_cache:
        betas = np.linspace(config.model.beta_start, config.model.beta_end,
                            config.model.diffusion_steps, dtype=np.float32)
        betas = torch.tensor(betas, dtype=torch.float32)
        beta = torch.cat([torch.zeros(1), betas], dim=0).to(config.model.device)
        _alpha_cache[key] = (1 - beta).cumprod(dim=0)
    alphas = _alpha_cache[key]
    return alphas.index_select(0, t + 1).view(-1, 1, 1, 1, 1)


def sample(y0, x, seq, model, config, w):
    with torch.no_grad():
        n = x.size(0)
        device = x.device
        seq_list      = list(reversed(seq))
        seq_next_list = [-1] + list(seq[:-1])
        seq_next_list = list(reversed(seq_next_list))

        t_all      = torch.tensor(seq_list,      dtype=torch.long, device=device)
        t_next_all = torch.tensor(seq_next_list, dtype=torch.long, device=device)
        at_all      = compute_alpha(t_all,      config)
        at_next_all = compute_alpha(t_next_all, config)

        xt = x
        for index in range(len(seq_list)):
            i       = seq_list[index]
            at      = at_all[index]
            at_next = at_next_all[index]
            t = (torch.ones(n, device=device) * i).long()

            et = model(xt, t)
            if w == 0:
                et_hat = et
            else:
                guidance = condition_score(model, xt, et, y0, t, at, config)
                et_hat = et + (1 - at).sqrt() * w * guidance

            x0_t   = (xt - et_hat * (1 - at).sqrt()) / at.sqrt()
            c1     = ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            c2     = ((1 - at_next) - c1 ** 2).sqrt()
            xt     = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et_hat
    return xt.cpu()


def sample_mask(y0, mask, seq, model, config, w):
    with torch.no_grad():
        n = y0.size(0)
        device = y0.device
        seq_list      = list(reversed(seq))
        seq_next_list = [-1] + list(seq[:-1])
        seq_next_list = list(reversed(seq_next_list))

        t_all      = torch.tensor(seq_list,      dtype=torch.long, device=device)
        t_next_all = torch.tensor(seq_next_list, dtype=torch.long, device=device)
        at_all      = compute_alpha(t_all,      config)
        at_next_all = compute_alpha(t_next_all, config)

        z    = torch.randn_like(y0)
        at_T = at_all[0]
        xt   = at_T.sqrt() * y0 + (1 - at_T).sqrt() * z

        for index in range(len(seq_list)):
            at      = at_all[index]
            at_next = at_next_all[index]
            t = (torch.ones(n, device=device) * seq_list[index]).long()

            xt_noise = at.sqrt() * y0 + (1 - at).sqrt() * z
            xt = torch.where(mask == 1, xt, xt_noise)

            et = model(xt, t)
            if w == 0:
                et_hat = et
            else:
                guidance = condition_score(model, xt, et, y0, t, at, config)
                et_hat = et + (1 - at).sqrt() * w * guidance

            x0_t = (xt - et_hat * (1 - at).sqrt()) / at.sqrt()
            c1   = ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            c2   = ((1 - at_next) - c1 ** 2).sqrt()
            xt   = at_next.sqrt() * x0_t + c1 * torch.randn_like(xt) + c2 * et_hat

    return xt.cpu()

def condition_score(model, xt, et, x_guidance, t, at, config):
    with torch.enable_grad():
        x_in = xt.detach().requires_grad_(True)
        et_recomputed = model(x_in, t)  # need model here
        x_0_hat = (x_in - et_recomputed * (1 - at).sqrt()) / at.sqrt()
        norm_x = torch.linalg.norm(x_0_hat - x_guidance)
        test_grad = torch.autograd.grad(outputs=norm_x, inputs=x_in)[0]
    return test_grad
# def condition_score(xt, et, x_guidance, t, at, config):
#     """
#     Posterior guidance: gradient of log p(y|xt).
#     at is passed in directly — already computed by the caller, no recomputation needed.
#     """
#     with torch.enable_grad():
#         x_in = xt.detach().requires_grad_(True)
#         x_0_hat = (x_in - et.detach() * (1 - at).sqrt()) / at.sqrt()
#         norm_x = torch.linalg.norm(x_0_hat - x_guidance)
#         test_grad = torch.autograd.grad(outputs=norm_x, inputs=x_in)[0]
#     return test_grad