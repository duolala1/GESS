import argparse, os, sys, glob
import torch
import torch.nn as nn
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange, repeat
from torchvision.utils import make_grid
from utils import *
import time
from multiprocessing import cpu_count
from torchvision import transforms
from ldm.util import instantiate_from_config, parallel_data_prefetch
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.modules.encoders.modules import FrozenClipImageEmbedder, FrozenCLIPTextEmbedder
import cv2
from torchvision.transforms import InterpolationMode
from ldm.models.diffusion.resizer import Resizer
BICUBIC = InterpolationMode.BICUBIC
# torch.cuda.set_device(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs"
    )

    parser.add_argument(
        "--skip_grid",
        action='store_true',
        help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
    )

    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )

    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )

    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=1,
        help="sample this often",
    )

    parser.add_argument(
        "--H",
        type=int,
        default=768,        # 768
        help="image height, in pixel space",
    )

    parser.add_argument(
        "--W",
        type=int,
        default=768,
        help="image width, in pixel space",
    )

    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="how many samples to produce for each given prompt. A.k.a batch size",
    )

    parser.add_argument(
        "--n_rows",
        type=int,
        default=0,
        help="rows in the grid (default: n_samples)",
    )

    parser.add_argument(
        "--scale",
        type=float,
        default=5.0,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/retrieval-augmented-diffusion/768x768.yaml",
        help="path to config which constructs model",
    )

    parser.add_argument(
        "--ckpt",
        type=str,
        default="./all_data/model.ckpt",
        help="path to checkpoint of model",
    )

    parser.add_argument(
        "--clip_type",
        type=str,
        default="ViT-L/14",
        # default="ViT-B/32",
        help="which CLIP model to use for retrieval and NN encoding",
    )
    parser.add_argument(
        "--sub",
        type=str,
        default="3",
    )
    parser.add_argument(
        "--test_num",
        type=int,
        default=50,
        
    )

    parser.add_argument(
        "--use_adaptive",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--use_CS",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--all_data_dir",
        type=str,
        default="./all_data/"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data/"
    )
    parser.add_argument(
        "--filename",
        type=str,
        default="sub3",
    )
    parser.add_argument(
        "--cyclegan_low",
        type=int,
        default=1
    )
    
    opt = parser.parse_args()

    transform = transforms.Compose([
                transforms.Resize((opt.W, opt.H), interpolation=BICUBIC),       
                transforms.ToTensor(),
            ])
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}", device)

    model = model.to(device)

    if opt.plms:
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    batch_size = opt.n_samples
    n_rows = opt.n_rows if opt.n_rows > 0 else batch_size

    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))
    grid_count = len(os.listdir(outpath)) - 1

    print(f"sampling scale for cfg is {opt.scale:.2f}")


    # load pic, semfeats
    pi_c, feats, rep_feats, low_feats = np.load(opt.data_dir +"preprocessed_feats_s"+opt.sub + ".npz").values()
    raw_imgs = np.load("./data/test_imgs.npz")["raw_imgs"]
    
    raw_imgs = process_imgs(raw_imgs, transform)
    low_feats = process_imgs(low_feats, transform=transform768)
    pi_s = torch.from_numpy(1 - pi_c).to(device)
    feats = torch.from_numpy(feats).to(device)
    rep_feats = torch.from_numpy(rep_feats).to(device)

    use_lowlevel = True
    test_num = int(opt.test_num)
    use_CS = bool(opt.use_CS) 
    use_adaptive = bool(opt.use_adaptive)   
    
    print("test_num", test_num, "use_CS", use_CS, "use_adaptive", use_adaptive)

    torch.manual_seed(0)

    with torch.no_grad():
            with model.ema_scope():
        
                for n in trange(opt.n_iter, desc="Sampling"):
                    all_samples = list()
                    gen_samples = list()
                    for i in range(0, test_num):
                        print(i)
                        c = feats[i].unsqueeze(0).unsqueeze(0).float().to(device)
                        rep_c = rep_feats[i].unsqueeze(0).unsqueeze(0).float().to(device)

                        c = torch.cat((c, rep_c), dim=1).to(device)

                        uc = None
                        if opt.scale != 1.0:
                            uc = torch.zeros_like(c)
                        
                        ref_z = None
                        if use_lowlevel:
                            post = model.first_stage_model.encode(low_feats[i].unsqueeze(0).to(device))
                            ref_z = model.get_first_stage_encoding(post).detach()

                        shape = [16, opt.H // 16, opt.W // 16]  # note: currently hardcoded for f16 model

                        
                        if use_CS:  ref_img = ref_z
                        else: ref_img =  None
                        if use_adaptive: ref_img_w = pi_s[i]
                        else: ref_img_w = None

                        samples_ddim, _ = sampler.sample(ref_img=ref_img, ref_img_w=ref_img_w, S=opt.ddim_steps,
                                                        conditioning=c,
                                                        batch_size=c.shape[0],
                                                        shape=shape,
                                                        verbose=False,
                                                        unconditional_guidance_scale=opt.scale,
                                                        unconditional_conditioning=uc,
                                                        eta=opt.ddim_eta,
                                                        )

                        x_samples_ddim = model.decode_first_stage(samples_ddim)
                        x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                        
                        all_samples.append(raw_imgs[i].unsqueeze(0).to(device))
                        all_samples.append(x_samples_ddim)
                        gen_samples.append(x_samples_ddim)

                    if not opt.skip_grid:
                        # additionally, save as grid
                        grid = torch.stack(all_samples, 0)
                        gen_samples = torch.stack(gen_samples, 0).squeeze()
                        
                        grid = rearrange(grid, 'n b c h w -> (n b) c h w')
                        grid = make_grid(grid, nrow=8, padding=10)

                        # to image
                        grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                        gen_samples = 255. * rearrange(gen_samples, 'b c h w -> b h w c').cpu().numpy()
                        print("gen_samples", gen_samples.shape)
                        if opt.filename is None:
                            Image.fromarray(grid.astype(np.uint8)).save(os.path.join(outpath, f'grid-{grid_count:04}.jpg'))
                        else:
                            Image.fromarray(grid.astype(np.uint8)).save(os.path.join(outpath,  opt.filename+f'.jpg'))
                        grid_count += 1

    print(f"Your samples are ready and waiting for you here: \n{outpath} \nEnjoy.")
