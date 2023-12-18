import torch
import os
import numpy as np
import diffusers
import random
import pickle

from PIL import Image
from tqdm.auto import tqdm
from diffusers.pipelines.stable_diffusion import StableDiffusionPipeline
from diffusers.pipelines.controlnet import StableDiffusionControlNetPipeline
from diffusers import DDIMScheduler, ControlNetModel, AutoencoderKL, StableDiffusionXLControlNetPipeline, UniPCMultistepScheduler, DPMSolverMultistepScheduler

import transformers
from transformers import CLIPTextModel, CLIPTokenizer
import torch.nn.functional as F
from torchvision import transforms
import copy
from _config import *
from lcm_pipeline import LatentConsistencyModelPipeline
from lcm_scheduler import LCMScheduler
from diffusers import DiffusionPipeline

def loadControlNet(config):
    print(f'Version of diffusers: {diffusers.__version__}')

    # token = ## Put your access token here ##
    device= config.device
    cn_version = 'lllyasviel/control_v11p_sd15_openpose' 
    # cn_version = 'thibaud/controlnet-sd21-openpose-diffusers'
    """
    'lllyasviel/control_v11p_sd15_openpose'
    'thibaud/controlnet-sd21-openpose-diffusers'
    'thibaud/controlnet-openpose-sdxl-1.0'
    'TencentARC/t2i-adapter-openpose-sdxl-1.0'
    """
    ######## ControlNet
    if cn_version == 'lllyasviel/control_v11p_sd15_openpose':
        controlnet = ControlNetModel.from_pretrained(cn_version, torch_dtype=torch.float16)
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16
        ).to(device)
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    elif cn_version == 'thibaud/controlnet-sd21-openpose-diffusers':
        controlnet = ControlNetModel.from_pretrained(cn_version, torch_dtype=torch.float16)
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-1", controlnet=controlnet, torch_dtype=torch.float16
        ).to(device)
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    elif cn_version == 'thibaud/controlnet-openpose-sdxl-1.0':
        controlnet = ControlNetModel.from_pretrained(cn_version, torch_dtype=torch.float16)
        pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", controlnet=controlnet, torch_dtype=torch.float16
        ).to(device)
        pipe.enable_model_cpu_offload()
    elif cn_version == 'TencentARC/t2i-adapter-openpose-sdxl-1.0':
        adapter = T2IAdapter.from_pretrained(
                "TencentARC/t2i-adapter-openpose-sdxl-1.0", torch_dtype=torch.float16
                ).to("cuda")

        model_id = 'stabilityai/stable-diffusion-xl-base-1.0'
        euler_a = EulerAncestralDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
        vae=AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
        pipe = StableDiffusionXLAdapterPipeline.from_pretrained(
            model_id, vae=vae, adapter=adapter, scheduler=euler_a, torch_dtype=torch.float16, variant="fp16", 
        ).to("cuda")
        pipe.enable_xformers_memory_efficient_attention()
    else:
        pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5").to(device)
        
    pipe.safety_checker = lambda images, clip_input: (images, False)


    # orig_vae = copy.deepcopy(pipe.vae)


    timesteps = pipe.scheduler.timesteps
    sp_sz = pipe.unet.sample_size
    bzs = 1

    for n, _module in pipe.unet.up_blocks.named_modules():
        if _module.__class__.__name__ == "Attention":
            # print('up')
            mod_forward_orig = _module.__class__.__call__

    return timesteps, sp_sz, bzs, mod_forward_orig, pipe

    

def loadLCMNet(config):
    print(f'Version of diffusers: {diffusers.__version__}')

    # token = ## Put your access token here ##
    device= config.device
    cn_version = "SimianLuo/LCM_Dreamshaper_v7"
        

    scheduler = LCMScheduler.from_pretrained("SimianLuo/LCM_Dreamshaper_v7", subfolder="scheduler")

    pipe = LatentConsistencyModelPipeline.from_pretrained("SimianLuo/LCM_Dreamshaper_v7", scheduler=scheduler)
    
    
    # pipe = DiffusionPipeline.from_pretrained("SimianLuo/LCM_Dreamshaper_v7")

    # # To save GPU memory, torch.float16 can be used, but it may compromise image quality.

    # pipe.to(torch_device="cuda", torch_dtype=torch.float32)
    pipe.to("cuda", dtype=torch.float16)
    num_inference_steps = 4


    timesteps = pipe.scheduler.timesteps
    sp_sz = pipe.unet.sample_size
    bzs = 1

    for n, _module in pipe.unet.up_blocks.named_modules():
        if _module.__class__.__name__ == "Attention":
            # print('up')
            mod_forward_orig = _module.__class__.__call__

    return timesteps, sp_sz, bzs, mod_forward_orig, pipe


