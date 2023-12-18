from lcm_pipeline import LatentConsistencyModelPipeline
from lcm_scheduler import LCMScheduler
import matplotlib.pyplot as plt
import torch
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
import torch
import random
import numpy as np
seed = 30
deterministic = True

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
if deterministic:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

g = torch.Generator('cuda').manual_seed(seed)

scheduler = LCMScheduler.from_pretrained("SimianLuo/LCM_Dreamshaper_v7", subfolder="scheduler")
safety_checker = StableDiffusionSafetyChecker.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="safety_checker")
pipe = LatentConsistencyModelPipeline.from_pretrained("SimianLuo/LCM_Dreamshaper_v7", scheduler=scheduler, safety_checker=safety_checker, generator = g)
#custom_pipeline="latent_consistency_txt2img", custom_revision="main", revision="fb9c5d", generator = g)

pipe.to("cuda", dtype=torch.float32)

prompt = "a young happy man and an old sad woman"
images = pipe(prompt=prompt, guidance_scale=2.5, num_inference_steps=2, lcm_origin_steps=50, output_type="pil").images

plt.imsave('sample.png',np.array(images[0]))