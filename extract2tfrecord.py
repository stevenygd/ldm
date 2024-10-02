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
# import torch.distributed as dist
# from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
# from torch.utils.data.distributed import DistributedSampler
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
import tensorflow as tf

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
    local_batch_size = args.global_batch_size 

    # Create model:
    print("\nCreating model...")
    f = 8
    assert args.image_size % f == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    model_config = OmegaConf.load('configs/autoencoder/autoencoder_kl_32x32x4.yaml')['model']
    sd = torch.load(args.checkpoint_path, map_location="cpu")['state_dict']
    vae = instantiate_from_config(model_config)
    vae.load_state_dict(sd,strict=False)
    # vae.to(device)
    vae.cuda()
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
    loader = DataLoader(
        dataset,
        batch_size = local_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )

    # Create sharding
    record_file = args.tfrecord_pattern
    record_file_dir = osp.dirname(record_file % (args.image_size, 0, 0))
    shutil.rmtree(record_file_dir, ignore_errors=True)
    os.makedirs(record_file_dir)
    n_shards = args.max_num_shards
    min_exs_per_shard = args.min_data_per_shard

    # Sharding
    n_batches = len(loader)
    exs_per_shard = max(int(np.ceil(min_exs_per_shard / args.global_batch_size)),
                        int(np.ceil(n_batches // n_shards)))
    num_shards = int(np.ceil(n_batches / exs_per_shard))
    assert n_batches - num_shards * exs_per_shard < exs_per_shard
    print("#examples=%d, #shards=%d, #ex/shard=%d total=%d len(ds)=%s"
          % (n_batches, num_shards, exs_per_shard, 
             num_shards * exs_per_shard, len(loader)))
    breakpoint()

    print("Start training...")
    loader_iter = iter(loader)
    for shard_id in tqdm.tqdm(range(num_shards)):
        record_file_sharded = record_file % (args.image_size, shard_id, num_shards)
        print(record_file_sharded)
        with tf.io.TFRecordWriter(record_file_sharded) as writer:
            for _ in tqdm.tqdm(range(exs_per_shard), leave=False):
                try:
                    x, y = next(loader_iter)
                    x = x.cuda()
                    with torch.no_grad():
                        # Map input images to latent space + normalize latents:
                        z = vae.encode(x).sample()
                    z = z.detach().cpu().numpy().reshape(-1, *z.shape[-3:])    
                    y = np.array(y).reshape(z.shape[0])
                    for i in range(z.shape[0]):
                        features = z[i].reshape(*z.shape[-3:])
                        label = int(y[i])
                        tf_example = make_tf_example(features, label)
                        writer.write(tf_example.SerializeToString())
                except StopIteration:
                    break
                
    # Benchmark reading the tf dataset:
    benchmark_reading_tf_dataset(osp.join(args.record_file_dir, "*.tfrecords"))

if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default='/mnt/disks/vae/.cache/autoencoders/data/ILSVRC2012_train/data')
    parser.add_argument("--checkpoint-path", type=str, default='/mnt/disks/sci/ldm/logs/2024-09-29T04-01-24_autoencoder_kl_32x32x4/checkpoints/epoch=000002.ckpt')
    parser.add_argument("--record-file-dir", type=str, default="/mnt/disks/vae/epoch1/imagenet256_tfdata_sharded/")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--tfrecord-pattern", type=str,
        default="data/imagenet%d_flax_tfdata_sharded/%0.5d-of-%0.5d.tfrecords")
    parser.add_argument("--max-num-shards", type=int, default=1_000)
    parser.add_argument("--min-data-per-shard", type=int, default=1_000)
    args = parser.parse_args()
    main(args)