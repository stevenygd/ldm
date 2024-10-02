'''
Decode extracted features from tfrecords and reconstruct images using a trained VAE model.
Usage:
    python tfrecord2recon.py --checkpoint-dir ... --features-dir ... --recon-npz-dir ... 
'''

import jax
import os.path as osp
import tensorflow as tf
# tf.config.experimental.set_visible_devices([], "GPU")
# tf.config.experimental.set_visible_devices([], "TPU")

import argparse
from omegaconf import OmegaConf
import numpy as np
import tensorflow_datasets as tfds
from tqdm import tqdm
from ldm.util import instantiate_from_config
from scripts.sample_diffusion import custom_to_np
import torch

#################################################################################
#From datasets/imagenet_feature_tfds.py

def create_tf_dataset(record_file, global_batch_size, num_processes=1):
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
    x = tf.io.parse_tensor(parsed_ex["x"], out_type=tf.float32)
    y = parsed_ex["y"]
    return x, y

  ds = raw_ds.map(_parse_fn_, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  ds = ds.shuffle(buffer_size=1_000).repeat()
  local_batch_size = global_batch_size // num_processes 
  ds = ds.batch(local_batch_size)
  ds = ds.batch(num_processes)
  ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
  return ds

def make_data_loader(args, **kwargs):
    record_file = osp.join(args.feature_path, "*.tfrecords")
    if "rng" in kwargs:
      seed = int(
        jax.random.randint(
          kwargs["rng"], shape=(1,), minval=0, maxval=100_000)[0]
      )
    else:
      seed = 0
    tf.random.set_seed(seed)
    global_batch_size = (
        kwargs["local_batch_size"] if "local_batch_size" in kwargs 
        else args.global_batch_size)
    num_processes = (
        kwargs["num_processes"] if "num_processes" in kwargs 
        else jax.local_device_count())
    ds = create_tf_dataset(
      record_file, global_batch_size, num_processes=num_processes)
    return ds, iter(ds)

#################################################################################

def main(args):
    
    N = args.num_samples
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Data Loader
    dataloader_args = OmegaConf.create({
        "feature_path": args.features_dir,
        "image_size": args.image_size,
        "global_batch_size": args.global_batch_size,
    })
    rng = jax.random.PRNGKey(args.global_seed)
    _, dsiter = make_data_loader(dataloader_args, rng=rng) 

    # Create model
    model_config = OmegaConf.load('configs/autoencoder/autoencoder_kl_32x32x4.yaml')['model']
    sd = torch.load(args.checkpoint_dir, map_location="cpu")['state_dict']
    vae = instantiate_from_config(model_config)
    vae.load_state_dict(sd,strict=False)
    vae.to(device)
    vae.eval()

    # Reconstruct
    recons = []
    with tqdm(total=N, desc="Reconstructing") as pbar:
        for x, _ in dsiter:
            z = torch.from_numpy(x[0].numpy()).to(device)
            rec = vae.decode(z)
            recons.append(custom_to_np(rec).numpy())

            pbar.update(len(z))
            if len(recons)*len(z) >= N:
                break
        
    recons = np.concatenate(recons, axis=0)
    recons = recons[:N]
    print('Reconstructions done: {}'.format(recons.shape))

    
    npz_file_path = osp.join(args.recon_npz_dir, f'imagenet{args.image_size}_uniformsamp_recon_{N//1000}k.npz')
    np.savez(npz_file_path, arr_0=recons)
    print(f'Reconstructions saved to {npz_file_path}')

  
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=str, default='/mnt/disks/sci/ldm/logs/2024-09-29T04-01-24_autoencoder_kl_32x32x4/checkpoints/epoch=000002.ckpt')
    parser.add_argument("--features-dir", type=str, default="/mnt/disks/sci/ldm/epoch2/imagenet256_tfdata_sharded/")
    parser.add_argument("--recon-npz-dir", type=str, default="/mnt/disks/sci/ldm/epoch2")
    parser.add_argument("--num-samples", type=int, default=10000)
    parser.add_argument("--skip-samples", type=int, default=50000)
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--global-batch-size", type=int, default=64)
    parser.add_argument("--global-seed", type=int, default=0)
    args = parser.parse_args()
    main(args)