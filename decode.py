import argparse, os, sys, datetime, glob
import os.path as osp
import numpy as np
from tqdm import tqdm
import torch
from PIL import Image
from torch.utils.data import TensorDataset, DataLoader
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from scripts.sample_diffusion import custom_to_np

BUCKET_DIR = "/mnt/disks/sci/checkpoints/"
LOCAL_DIR = "/mnt/sdb/ldm/checkpoints/"

def get_inout_dir(expr_name, ckpt_step, args):

    model_string_name = OmegaConf.load(osp.join(BUCKET_DIR, expr_name, "config.yaml")).model.name.replace("/", "-")
    file_name = f"{model_string_name}-"\
                  f"size-{args.image_size}-" \
                  f"cfg-{args.inference.cfg_scale}-" \
                  f"seed-{args.global_seed}-" \
                  f"step-{args.inference.num_sampling_steps}-" \
                  f"nsmp-{args.inference.num_fid_samples}"
    if args.inference.get("mode", "ddpm") == "ddim":
        file_name = f"{file_name}-ddim"
    elif args.inference.get("mode", "ddpm") == "rectflow":
        file_name = f"{file_name}-rectflow"
    
    in_npy_path = osp.join(
        BUCKET_DIR, expr_name, "features", str(ckpt_step), f"{file_name}.npy")
    if not osp.isfile(in_npy_path):
        return False
    
    out_npz_folder = osp.join(BUCKET_DIR, expr_name, "samples", str(ckpt_step))
    if not os.path.exists(out_npz_folder):
        os.makedirs(out_npz_folder, exist_ok=True)
    out_npz_path = osp.join(out_npz_folder, f"{file_name}.npz")

    out_img_folder = osp.join(LOCAL_DIR, expr_name, "samples", str(ckpt_step), file_name)
    if not os.path.exists(out_img_folder):
        os.makedirs(out_img_folder, exist_ok=True)
    
    return (in_npy_path, out_npz_path, out_img_folder)

def main(args):

    expr_name = args.expr_name
    expr_config = OmegaConf.create({
        "inference": {
            "cfg_scale": args.cfg_scale,
            "num_fid_samples": args.num_samples,
            "mode": args.sampling_mode,
            "num_sampling_steps": args.num_sampling_steps,
        },
        "image_size": args.image_size,
        "global_seed": args.global_seed
    })

    if int(args.resume_step) < 0:
        ckpt_steps = [ckpt for ckpt in os.listdir(osp.join(BUCKET_DIR, expr_name, 'checkpoints')) if ckpt.isdigit()]
        ckpt_steps = sorted(ckpt_steps, key=lambda x: -int(x))
        print(f"This run will iterate over {len(ckpt_steps)} checkpoints: {ckpt_steps}")
    else:
        ckpt_steps = [str(args.resume_step)]
        print(f"This run will decode for single checkpoint: {str(args.resume_step)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    for ckpt_step in ckpt_steps:
        print(f"\n\n---- Processing {expr_name} at {ckpt_step} ----")
         
        dirs = get_inout_dir(expr_name, ckpt_step, expr_config)
        if not dirs:
            print(f"---- Features not found for {expr_name} at {ckpt_step} ----")
            continue
        in_npy_path, out_npz_path, out_img_folder = dirs
        if osp.isfile(out_npz_path):
            saved_npz = np.load(out_npz_path)['arr_0']
            if saved_npz.shape == (args.num_samples, args.image_size, args.image_size, 3):
                print(f"---- Already reconstructed: {out_npz_path} ----")
            else:
                print(f"{out_npz_path} exists but shape is {saved_npz.shape}, deleting...")
                os.remove(out_npz_path)

        # load latent features
        features = np.load(in_npy_path)
        assert features.ndim==4 and features.shape[0] == expr_config.inference.num_fid_samples, \
            f"Wrong features shape: {features.shape}"
        
        # load model
        try: 
            model.eval()
        except NameError:
            model_checkpoint_dir = '/mnt/disks/sci/ldm/logs/2024-09-29T04-01-24_autoencoder_kl_32x32x4/checkpoints/epoch=000002.ckpt'
            model_config = OmegaConf.load('configs/autoencoder/autoencoder_kl_32x32x4.yaml')['model']
            
            sd = torch.load(model_checkpoint_dir, map_location="cpu")['state_dict']
            model = instantiate_from_config(model_config)
            model.load_state_dict(sd,strict=False)
            model.to(device)
            model.eval()
        
        # data loader
        dataset = TensorDataset(torch.tensor(features))
        batch_size = args.batch_size
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)

        # reconstruct
        recons = []
        i = 0
        for (z,) in tqdm(dataloader):
            z = z.to(device)
            x_r = model.decode(z) # (batch_size, 3, 256, 256), (-1, 1) torch.float32
            x_r = custom_to_np(x_r).numpy() # (batch_size, 256, 256, 3) (0, 255) np.uint8
            recons.append(x_r)
            for img in x_r:
                Image.fromarray(img).save(osp.join(out_img_folder, f"pid0-{i:06d}.png"))
                i += 1
        recons = np.concatenate(recons, axis=0)
        print('Reconstructions done: {}'.format(recons.shape))

        np.savez(out_npz_path, arr_0=recons)
        print(f'---- Reconstructions saved to {out_npz_path} ----')
    
    return

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--expr-name", type=str, required=True)
    parser.add_argument("--resume-step", type=str, default="-1")
    parser.add_argument("--batch-size", type=int, default=256)

    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--num-samples", type=int, default=10000)
    parser.add_argument("--sampling-mode", type=str, default="rectflow")
    parser.add_argument("--num-sampling-steps", type=int, default=50)
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    args = parser.parse_args()

    main(args)
    

