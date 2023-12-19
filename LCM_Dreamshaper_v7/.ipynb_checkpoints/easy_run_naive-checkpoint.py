from lcm_pipeline_naive import LatentConsistencyModelPipeline
from lcm_scheduler import LCMScheduler
import matplotlib.pyplot as plt
import torch
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
import torch
import random
import numpy as np
import pandas as pd
import time
from tqdm import tqdm
from pathlib import Path

#custom_pipeline="latent_consistency_txt2img", custom_revision="main", revision="fb9c5d", generator = g)

dataset_path = './datasets/phrases3.csv'
dataset = pd.read_csv(dataset_path)
dataset_name = dataset_path.split("/")[-1].split('.')[0]
ts = time.time()
count = 0
time_total = 0
scheduler = LCMScheduler.from_pretrained("SimianLuo/LCM_Dreamshaper_v7", subfolder="scheduler")
safety_checker = StableDiffusionSafetyChecker.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="safety_checker")
for seed in [42,54]:
    g = torch.Generator('cuda').manual_seed(seed)
    stable = LatentConsistencyModelPipeline.from_pretrained("SimianLuo/LCM_Dreamshaper_v7", scheduler=scheduler, safety_checker=safety_checker, generator = g).to('cuda')
    for i in tqdm(range(len(dataset)), desc="Prompt idx"):
        output_path = Path('./outputs')
        dataset_prompt_output_path = output_path / dataset_name / f"{i:003}"
        dataset_prompt_output_path.mkdir(exist_ok=True, parents=True)
        prompt = dataset.iloc[i].prompt
        token_indices = dataset.iloc[i].item_indices
          
        print(f"Seed: {seed}")
        img_path = dataset_prompt_output_path / f'LCM_{seed}.png'
        if img_path.exists():
            continue
        ts = time.time()
        # image = run_on_prompt(prompt=config.prompt,
        #                       model=stable,
        #                       controller=controller,
        #                       token_indices=token_indices,
        #                       seed=g,
        #                       config=config)
        image = stable(prompt=prompt, guidance_scale=2.5, num_inference_steps=4, lcm_origin_steps=50, output_type="pil").images
        image = image[0]
        te = time.time()
        image.save(img_path)
        time_total += (te-ts)
        count += 1
        # except:
            #     print('FAILED:',i, config.prompt)

te = time.time()
print(f"*** Total time spent: {time_total:.4f} ***")
print(f"*** For one image: {time_total/count:.4f}")
with open(f"{output_path}/Time_LCM_Pharses3.txt", 'w') as f:
    f.write(f"{time_total/count:.4f}") 
#pipe.to("cuda", dtype=torch.float32)

# prompt = "a young happy man and an old sad woman"
# images = pipe(prompt=prompt, guidance_scale=2.5, num_inference_steps=2, lcm_origin_steps=50, output_type="pil").images

# plt.imsave('sample.png',np.array(images[0]))