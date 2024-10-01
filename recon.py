import argparse, os, sys, datetime, glob, importlib, csv
import numpy as np
import time
from tqdm import tqdm
import torch
import torchvision
import pytorch_lightning as pl

from packaging import version
from omegaconf import OmegaConf
from torch.utils.data import random_split, DataLoader, Dataset, Subset
from functools import partial
from PIL import Image

from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, LearningRateMonitor
from pytorch_lightning.utilities.distributed import rank_zero_only
from pytorch_lightning.utilities import rank_zero_info

from ldm.data.base import Txt2ImgIterableBaseDataset
from ldm.util import instantiate_from_config

from scripts.sample_diffusion import custom_to_pil, custom_to_np

def custom_to_torch(img, device='cuda'):
    x = np.array(img)[None, ...]
    print('before re-normalize',x.min(), x.max(), x.shape)
    x = (torch.tensor(x.transpose(0,3,1,2))/255).float()
    x = (2*x-1)
    return x.to(device)

if __name__ == "__main__":

    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    sys.path.append(os.getcwd())

    f = 8
    # checkpoint_dir = f'models/first_stage_models/kl-f{f}/model.ckpt'
    checkpoint_dir = '/mnt/disks/sci/ldm/logs/2024-09-29T04-01-24_autoencoder_kl_32x32x4/checkpoints/epoch=000000.ckpt'
    model_config = OmegaConf.load('configs/autoencoder/autoencoder_kl_32x32x4.yaml')['model']

    N = 10000
    npz_dir = f'/mnt/disks/sci/data/imagenet256_trainset_balanced_{N//1000}k.npz'
    recon_npz_dir = f'/mnt/disks/vae/imagenet256_recon_{N//1000}k.npz'

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # model
    sd = torch.load(checkpoint_dir, map_location="cpu")['state_dict']
    model = instantiate_from_config(model_config)
    model.load_state_dict(sd,strict=False)
    model.to(device)
    model.eval()

    ## Single Image Reconstruction
    # img = Image.open('img_vqgan.png')
    # x = custom_to_torch(img, device)
    # x_r, _ = model(x)
    # img_r = custom_to_pil(x_r[0])

    # img_r.save(f'img_recon_f{f}.png')
    # print(f'Reconstructed image saved to img_f{f}.png')

    ## 
    npz = np.load(npz_dir)['arr_0']
    npz = (npz/255).astype(np.float32)
    npz = (2*npz-1)
    xs = torch.tensor(npz).permute(0,3,1,2)
    del npz
    print('Data loaded: {}'.format(xs.shape))

    recons = []

    batch_size = 16
    n_batches = len(xs)//batch_size
    for i in tqdm(range(n_batches)):
        start = i*batch_size
        end = min((i+1)*batch_size, len(xs))

        x = xs[start:end].to(device)
        import pdb; pdb.set_trace()
        x_r, _ = model(x)
        x_r = custom_to_np(x_r).numpy()
        
        recons.append(x_r)

    # for x in tqdm(xs):
    #     x = x[None,...].to(device)
    #     x_r, _ = model(x)
    #     x_r = custom_to_np(x_r).numpy()
    #     recons.append(x_r)
    
    recons = np.concatenate(recons, axis=0)
    print('Reconstructions done: {}'.format(recons.shape))

    np.savez(recon_npz_dir, arr_0=recons)
    print(f'Reconstructions saved to {recon_npz_dir}')

