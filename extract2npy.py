# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Extract features from a pre-trained VAE and save them to npy files.
example usage:
    torchrun extract2npy.py --data-dir /mnt/disks/vae/.cache/autoencoders/data/ILSVRC2012_train/data --features-dir data/features --checkpoint-path /mnt/disks/sci/ldm/logs/2024-09-29T04-01-24_autoencoder_kl_32x32x4/checkpoints/epoch=000002.ckpt
"""
import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
from collections import OrderedDict
from PIL import Image
import argparse
import logging
import os
import tqdm

from ldm.util import instantiate_from_config
from omegaconf import OmegaConf, DictConfig

def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


def main(args):
    """
    Trains a new DiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    dist.init_process_group("nccl")
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    local_batch_size = args.global_batch_size // dist.get_world_size()
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}, local_batch_size={local_batch_size}. ")

    # Create model:
    print("Creating model...")
    f = 8
    assert args.image_size % f == 0, "Image size must be divisible by 8 (for the VAE encoder)."

    model_config = OmegaConf.load('configs/autoencoder/autoencoder_kl_32x32x4.yaml')['model']
    sd = torch.load(args.checkpoint_path, map_location="cpu")['state_dict']

    vae = instantiate_from_config(model_config)
    vae.load_state_dict(sd,strict=False)
    vae.to(device)
    vae.eval()

    # Setup data:
    print("Setting up datatraining...")
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    dataset = ImageFolder(args.data_dir, transform=transform)
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=False,
        seed=args.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size = local_batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    # Setup a feature folder:
    if rank == 0:
        os.makedirs(args.features_dir, exist_ok=True)
        os.makedirs(os.path.join(args.features_dir, 'features'), exist_ok=True)
        os.makedirs(os.path.join(args.features_dir, 'labels'), exist_ok=True)

    print("Start training...")
    feature_id = 0
    if dist.get_rank() == 0:  
        pbar = tqdm.tqdm(loader)
    else:
        pbar = loader
    for x, y in pbar:
        x = x.to(device)
        with torch.no_grad():
            # Map input images to latent space + normalize latents:
            x = vae.encode(x).sample()
            
        x = x.detach().cpu().numpy()    # (1, 4, 32, 32)
        y = y.detach().cpu().numpy()    # (1,)
        for i in range(x.shape[0]):
            np.save(
                f'{args.features_dir}/features/{feature_id}.npy',
                x[i:i+1])
            np.save(
                f'{args.features_dir}/labels/{feature_id}.npy',
                y[i:i+1])

            feature_id += 1

if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--checkpoint-path", type=str, default='/mnt/disks/sci/ldm/logs/2024-09-29T04-01-24_autoencoder_kl_32x32x4/checkpoints/epoch=000001.ckpt')
    parser.add_argument("--features-dir", type=str, default="data/features/epoch1")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    main(args)