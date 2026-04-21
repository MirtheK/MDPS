# import libraries

import numpy as np
import torch
import torch.nn.functional as F

import argparse
import os
import pickle
import matplotlib.pyplot as plt
import nibabel as nib
from scipy.ndimage import gaussian_filter

from monai.utils import set_determinism
from torch.amp import GradScaler, autocast
from monai.data import DataLoader
from generative.inferers import DiffusionInferer
from generative.networks.schedulers import DDPMScheduler
from generative.networks.nets import DiffusionModelUNet
from src.dataset import SHOMRI, SHOMRI_with_mask

def train_partial_diffusion(args):
    # configuration
    task_name = args.task_name
    train_data_path = args.train_data_path
    n_epochs = args.epoch
    batch_size = args.batch_size
    lr = args.learning_rate
    K = args.K
    set_determinism(99)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # training dataset
    train_ds, train_loader = SHOMRI_with_mask(
        root_dir=train_data_path,
        batch_size=batch_size,
        is_train=True,
    )

    # build model
    model = DiffusionModelUNet(
        spatial_dims=3,
        in_channels=2,
        out_channels=1,
        num_res_blocks=(2, 2, 2),
        num_channels=(32, 64, 64),
        attention_levels=(False, False, False),
        num_head_channels=64
    )
    model.to(device)

    # scheduler, optimizer and inferer
    # here we still define a diffusion process of 1000 timesteps
    scheduler = DDPMScheduler(num_train_timesteps=1000, schedule='linear_beta')
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    inferer = DiffusionInferer(scheduler)

    # training loop
    scaler = GradScaler()
    loss_list = []
    
    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0
        print(f"Epoch {epoch + 1} / {n_epochs}", flush=True)

        for i, batch in enumerate(train_loader):  
            image = batch["image"].to(device)
            mask = batch["mask"].to(device)

            optimizer.zero_grad(set_to_none=True)

            with autocast('cuda', enabled=True):
                noise = torch.randn_like(image).to(device)
                timesteps = torch.randint(0, K, (image.shape[0],), device=image.device).long()
                noisy_image = scheduler.add_noise(original_samples=image, noise=noise, timesteps=timesteps)

                model_input = torch.cat([noisy_image, mask], dim=1)

                noise_pred = model(model_input, timesteps=timesteps, context=None)
                loss = F.mse_loss(noise_pred.float(), noise.float())

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()

        loss_list.append(epoch_loss / (i + 1))


    # save the model
    model_save_path = os.path.join('/projects/prjs1633/checkpoints/SHOMRI/', task_name)
    os.mkdir(model_save_path)
    torch.save(model.state_dict(), os.path.join(model_save_path, task_name + '.pth'))

    # plot the loss curve
    plt.figure(figsize=(10, 5))
    plt.title("Learning Curves", fontsize=20)
    plt.plot(np.linspace(1, n_epochs, n_epochs), loss_list, color="C0", linewidth=2.0, label="Train")
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    plt.xlabel("Epochs", fontsize=16)
    plt.ylabel("Loss", fontsize=16)
    plt.legend(prop={"size": 14})
    plt.savefig(os.path.join(model_save_path, 'loss_curve_' + task_name + '.png'))

    # plot the log loss curve
    plt.figure(figsize=(10, 5))
    plt.title("Log10 Learning Curves", fontsize=20)
    plt.plot(np.linspace(1, n_epochs, n_epochs), np.log10(loss_list), color="C0", linewidth=2.0,
             label="Train")
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    plt.xlabel("Epochs", fontsize=16)
    plt.ylabel("Loss", fontsize=16)
    plt.legend(prop={"size": 14})
    plt.savefig(os.path.join(model_save_path, 'log_loss_curve_' + task_name + '.png'))


def inference_partial_diffusion(args):
    # configuration
    test_data_path = args.test_data_path
    model_path = args.model_path
    result_save_path = args.result_save_path
    K_inf = args.K_inf
    set_determinism(99)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # test dataset
    test_ds, test_loader = SHOMRI(
        root_dir=test_data_path,
        batch_size=1,          
        is_train=False,
         )

    # build model
    model = DiffusionModelUNet(
        spatial_dims=3,
        in_channels=2,
        out_channels=1,
        num_res_blocks=(2, 2, 2),
        num_channels=(32, 64, 64),
        attention_levels=(False, False, False),
        num_head_channels=64
    )
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.to(device)
    model.eval()

    scheduler = DDPMScheduler(num_train_timesteps=1000, schedule='linear_beta')

    # partial diffusion timesteps
    timesteps = torch.linspace(K_inf, 0, K_inf + 1, dtype=torch.int32)

    for i, batch in enumerate(test_loader):
        image_test = batch["image"].to(device)
        filename   = batch["filename"][0]   
        label      = batch["label"][0]
        true_label = ("NORMAL" if label == "good" else "ABNORMAL").upper()

        noise = torch.randn_like(image_test).to(image_test)

        # start point of partial diffusion x_{K_inf}
        x_K = scheduler.add_noise(original_samples=image_test, noise=noise, timesteps=timesteps[0])

        # initialise an x_t
        x_t = x_K
        with autocast('cuda', enabled=True):
            with torch.no_grad():
                for t in timesteps:
                    # Get model prediction
                    model_input = torch.cat([x_t, image_test], dim=1)
                    model_output = model(model_input, timesteps=torch.Tensor((t,)).to(image_test.device), context=None)
                    # compute the previous sample: x_t -> x_{t-1}
                    x_t, _ = scheduler.step(model_output, t, x_t)
                    print(f'Timestep {t.item()} is Done', flush=True)

        # postprocess x0
        x0 = x_t[0].cpu().detach().numpy()
        x0 = x0.squeeze(0)

        # save
        os.makedirs(result_save_path, exist_ok=True)

        base_name = filename.replace(".nii.gz", "")
        affine = nib.load(os.path.join(test_data_path, "test", true_label, filename)).affine
        input_img = image_test[0, 0].cpu().detach().numpy()


        # save reconstruction
        nib.save(nib.Nifti1Image(x0, affine=affine),
                os.path.join(result_save_path, f"{base_name}_recon.nii.gz")
                )

        # save anomaly map: MSE then Gaussian Blur
        # anomaly_map = np.abs(x0 - input_img)
        anomaly_map = (x0 - input_img) ** 2
        anomaly_map = gaussian_filter(anomaly_map, sigma=4)

        nib.save(nib.Nifti1Image(anomaly_map, affine=affine),
                os.path.join(result_save_path, f"{base_name}_anomaly_map.nii.gz")
                )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Partial diffusion")
    parser.add_argument('--mode', type=str, default='train', help='train, test')

    # for training mode
    # paths
    parser.add_argument('--task_name', type=str, default='pd_project', help='Task name')
    parser.add_argument('--train_data_path', type=str, default=None, help='Path of training data')
    # training hyperparameters
    parser.add_argument('--epoch', type=int, default=100, help='Total number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.00001, help='Learning rate')
    # partial diffusion
    parser.add_argument('--K', type=int, default=100, help='Partial diffusion max timestep in training')

    # for test mode
    # paths
    parser.add_argument('--test_data_path', type=str, default=None, help='Path of test data')
    parser.add_argument('--model_path', type=str, default=None, help='Path of trained model')
    parser.add_argument('--result_save_path', type=str, default=None, help='Path of saving results')
    # partial diffusion
    parser.add_argument('--K_inf', type=int, default=50, help='Partial diffusion max timestep in testing')

    # parse
    args = parser.parse_args()
    for key, value in vars(args).items():
        print(f"{key}: {value}")

    if args.mode == 'train':
        print(f'In {args.mode} mode', flush=True)
        train_partial_diffusion(args)
    elif args.mode == 'test':
        print(f'In {args.mode} mode', flush=True)
        inference_partial_diffusion(args)
    else:
        raise ValueError(f"Mode {args.mode} is not supported")
