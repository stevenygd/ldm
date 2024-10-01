# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT using PyTorch DDP.
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
    latent_size = args.image_size // f

    model_config = OmegaConf.load('configs/autoencoder/autoencoder_kl_32x32x4.yaml')['model']
    sd = torch.load(args.checkpoint_dir, map_location="cpu")['state_dict']

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
    dataset = ImageFolder(args.data_path, transform=transform)
    
    ## Shuffle dataset(shuffle=False is not available in this setting):
    # import random
    # random.shuffle(dataset.samples)

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

    # Make record file directory: 
    shutil.rmtree(osp.dirname(args.record_file_path), ignore_errors=True)
    os.makedirs(osp.dirname(args.record_file_path))
    record_file_format = osp.join(args.record_file_path, "%0.5d-of-%0.5d.tfrecords")

    # Calculate number of shards and examples per shard:
    num_images = len(dataset)
    exs_per_shard = max(args.min_exs_per_shard, int(np.ceil(num_images // args.n_shards)))
    num_shards = int(np.ceil(num_images // exs_per_shard))
    assert num_images - num_shards * exs_per_shard < exs_per_shard
    print("#examples=%d, #shards=%d, #ex/shard=%d" % (num_images, num_shards, exs_per_shard))
    
    # Define functions for creating tfrecords and benchmarking reading tfrecords (from feature2tfrecord.py):
    import tensorflow as tf
    import tensorflow_datasets as tfds
    def _bytes_feature(value):
        """Returns a bytes_list from a string / byte."""
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _float_feature(value):
        """Returns a float_list from a float / double."""
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))

    def _int64_feature(value):
        """Returns an int64_list from a bool / enum / int / uint."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def make_tf_example(features, label, d, w, h):
        feature = {
            'y': _int64_feature(label),
            "x": _bytes_feature(tf.io.serialize_tensor(features))
        }
        return tf.train.Example(features=tf.train.Features(feature=feature))

    def benchmark_reading_tf_dataset(record_file):
        """
            [record_file] is a pattern of <dir_name>/%0.5d-of-%0.5d.tfrecords
        """
        files = tf.io.matching_files(record_file)
        files = tf.random.shuffle(files)
        shards = tf.data.Dataset.from_tensor_slices(files)
        raw_ds = shards.interleave(tf.data.TFRecordDataset)
        raw_ds = raw_ds.shuffle(buffer_size=10000)

        # Create a dictionary describing the features.
        def _parse_fn_(example_proto):
            feature_description = {
                'y': tf.io.FixedLenFeature([], tf.int64),
                'x': tf.io.FixedLenFeature([], tf.string), 
            }
            parsed_ex = tf.io.parse_single_example(example_proto, feature_description)
            return {
            "x": tf.io.parse_tensor(parsed_ex["x"], out_type=tf.float32),
            "y": parsed_ex["y"], 
            }

        ds = raw_ds.map(_parse_fn_, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds = ds.batch(256).prefetch(buffer_size=tf.data.AUTOTUNE)
        for raw_record in ds.take(10):
            print(raw_record["x"].shape, raw_record["x"].dtype)
            print(raw_record["y"].shape, raw_record["y"].dtype)
        tfds.benchmark(ds, batch_size=256)

    # Extract features & Create tfrecords:
    stop = False
    print("Start training...")
    train_steps = 0
    if dist.get_rank() == 0:  
        pbar = tqdm.tqdm(loader)
    else:
        pbar = loader
    for x, y in pbar:
        
        x = x.to(device)
        y = y.to(device)
        with torch.no_grad():
            # Map input images to latent space + normalize latents:
            x = vae.encode(x).sample().mul_(0.18215)
            
        x = x.detach().cpu().numpy()    # (B, 4, 32, 32)
        y = y.detach().cpu().numpy()    # (B,)
        for i in range(x.shape[0]):
            shard_id = train_steps // exs_per_shard
            # if shard_id >= num_shards:
            #     stop=True
            #     print("Shard finished. Breaking..")
            #     break
            record_file_sharded = record_file_format % (shard_id, num_shards)
            with tf.io.TFRecordWriter(record_file_sharded) as writer:
                features = x[i]
                d, w, h = features.shape
                label = y[i]
                tf_example = make_tf_example(features, label, d, w, h)
                writer.write(tf_example.SerializeToString())

            train_steps += 1
            if train_steps % exs_per_shard == 0:
                print(f"Shard {shard_id}({exs_per_shard*(shard_id)}-{exs_per_shard*(shard_id+1)-1}) finished.")
        # if stop:
        #     break

    # Benchmark reading the tf dataset:
    benchmark_reading_tf_dataset(osp.join(args.record_file_path, "*.tfrecords"))

if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default='/mnt/disks/vae/.cache/autoencoders/data/ILSVRC2012_train/data')
    parser.add_argument("--checkpoint-dir", type=str, default='/mnt/disks/sci/ldm/logs/2024-09-29T04-01-24_autoencoder_kl_32x32x4/checkpoints/epoch=000002.ckpt')
    # parser.add_argument("--features-path", type=str, default="features")
    # parser.add_argument("--results-dir", type=str, default="results")
    # parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    # parser.add_argument("--num-classes", type=int, default=1000)
    # parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    # parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
    parser.add_argument("--num-workers", type=int, default=4)
    # parser.add_argument("--log-every", type=int, default=100)
    # parser.add_argument("--ckpt-every", type=int, default=50_000)
    parser.add_argument("--n-shards", type=int, default=1000)
    parser.add_argument("--min-exs-per-shard", type=int, default=256)
    parser.add_argument("--record-file-path", type=str, default="/mnt/disks/vae/epoch2/imagenet256_tfdata_sharded/")
    args = parser.parse_args()
    main(args)