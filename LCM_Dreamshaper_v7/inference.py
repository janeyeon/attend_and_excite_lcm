from lcm_pipeline import LatentConsistencyModelPipeline
from lcm_scheduler import LCMScheduler

from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import CLIPTokenizer, CLIPTextModel, CLIPImageProcessor

import os
import torch
from tqdm import tqdm
from safetensors.torch import load_file

# Input Prompt:
prompt = "a young happy man and an old sad woman"

# Save Path:
save_path = "./lcm_images"
os.makedirs(save_path, exist_ok=True)


# Origin SD Model ID
model_id = "digiplay/DreamShaper_7"


# Initalize Diffusers Model:
vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet", device_map=None, low_cpu_mem_usage=False, local_files_only=False)
safety_checker = StableDiffusionSafetyChecker.from_pretrained(model_id, subfolder="safety_checker")
feature_extractor = CLIPImageProcessor.from_pretrained(model_id, subfolder="feature_extractor")


# Initalize Scheduler:
scheduler = LCMScheduler(beta_start=0.00085, beta_end=0.0120, beta_schedule="scaled_linear", prediction_type="epsilon")


# Replace the unet with LCM:
lcm_unet_ckpt = "./LCM_Dreamshaper_v7_4k.safetensors"
ckpt = load_file(lcm_unet_ckpt)
m, u = unet.load_state_dict(ckpt, strict=False)
if len(m) > 0:
    print("missing keys:")
    print(m)
if len(u) > 0:
    print("unexpected keys:")
    print(u)


# LCM Pipeline:
pipe = LatentConsistencyModelPipeline(vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet, scheduler=scheduler, safety_checker=safety_checker, feature_extractor=feature_extractor)
pipe = pipe.to("cuda")


# Output Images:
images = pipe(prompt=prompt, num_images_per_prompt=4, num_inference_steps=4, guidance_scale=2.5, lcm_origin_steps=50).images

# Save Images:
for i in tqdm(range(len(images))):
    output_path = os.path.join(save_path, "{}.png".format(i))
    image = images[i]
    image.save(output_path) 






