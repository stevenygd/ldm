# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
!!Warning!!: This script still has bugs.
Extract features from a pre-trained VAE and save them to tfrecords.
example usage:
    torchrun extract2tfrecord.py --data-dir /mnt/disks/vae/.cache/autoencoders/data/ILSVRC2012_train/data --checkpoint-path /mnt/disks/sci/ldm/logs/2024-09-29T04-01-24_autoencoder_kl_32x32x4/checkpoints/epoch=000002.ckpt --record-file-dir /mnt/disks/vae/epoch2/imagenet256_tfdata_sharded/
    torchrun extract2tfrecord.py --data-dir /mnt/disks/sci/data/imagenet_train/ --checkpoint-path /mnt/disks/sci/ldm/logs/2024-09-29T04-01-24_autoencoder_kl_32x32x4/checkpoints/epoch=000001.ckpt --record-file-dir /mnt/disks/sci/ldm/epoch1/imagenet256_tfdata_sharded/
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
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os
import os.path as osp
import shutil
import tqdm

from ldm.util import instantiate_from_config
from omegaconf import OmegaConf, DictConfig
from npy2tfrecord import _bytes_feature, _float_feature, _int64_feature, make_tf_example, benchmark_reading_tf_dataset

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
    print("\nCreating model...")
    f = 8
    assert args.image_size % f == 0, "Image size must be divisible by 8 (for the VAE encoder)."

    model_config = OmegaConf.load('configs/autoencoder/autoencoder_kl_32x32x4.yaml')['model']
    sd = torch.load(args.checkpoint_path, map_location="cpu")['state_dict']

    vae = instantiate_from_config(model_config)
    vae.load_state_dict(sd,strict=False)
    vae.to(device)
    vae.eval()

    # Setup data:
    print("\nSetting up datatraining...")
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
        drop_last=False
    )
    
    # Make record file directory
    shutil.rmtree(osp.dirname(args.record_file_dir), ignore_errors=True)
    os.makedirs(osp.dirname(args.record_file_dir))
    record_file_format = osp.join(args.record_file_dir, "%0.5d-of-%0.5d.tfrecords") 

    n_batches = len(loader)
    batch_per_shard = max( int(np.ceil(args.min_exs_per_shard / local_batch_size)), 
                        int(np.ceil(n_batches // args.n_shards)))
    num_shards = int(np.ceil(n_batches / batch_per_shard))
    assert n_batches - num_shards * batch_per_shard < batch_per_shard
    print("#batches=%d, #shards=%d, #batch/shard=%d" % (n_batches, num_shards, batch_per_shard))

    # Extract features:
    print("\nStart Extraction...")

    loader_iter = iter(loader)
    for shard_id in tqdm(range(num_shards)):
        record_file_sharded = record_file_format % (shard_id, num_shards)
        with tf.io.TFRecordWriter(record_file_sharded) as writer:
            for _ in range(batch_per_shard):
                try:
                    x, y = next(loader_iter)
                    x = x.to(device)
                    with torch.no_grad():
                        # Map input images to latent space + normalize latents:
                        x = vae.encode(x).sample()
                    x = x.detach().cpu().numpy()    # (B, 4, 32, 32)
                    y = y.detach().cpu().numpy()    # (B,)
                    
                    for feature, label in zip(x, y):
                        d, w, h = feature.shape
                        tf_example = make_tf_example(feature, label, d, w, h)
                        writer.write(tf_example.SerializeToString())

                except StopIteration:
                    break
                    

    # Benchmark reading the tf dataset:
    benchmark_reading_tf_dataset(osp.join(args.record_file_dir, "*.tfrecords"))

if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default='/mnt/disks/vae/.cache/autoencoders/data/ILSVRC2012_train/data')
    parser.add_argument("--checkpoint-path", type=str, default='/mnt/disks/sci/ldm/logs/2024-09-29T04-01-24_autoencoder_kl_32x32x4/checkpoints/epoch=000001.ckpt')
    parser.add_argument("--record-file-dir", type=str, default="/mnt/disks/vae/epoch1/imagenet256_tfdata_sharded/")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--n-shards", type=int, default=1000)
    parser.add_argument("--min-exs-per-shard", type=int, default=256)
    args = parser.parse_args()
    main(args)