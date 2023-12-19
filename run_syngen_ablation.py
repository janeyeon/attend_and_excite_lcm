import pprint
from typing import List
import time
from tqdm import tqdm
import pyrallis
import torch
from PIL import Image
from config import RunConfig
# from pipeline_attend_and_excite import AttendAndExcitePipeline
from pipeline_attend_and_excite_syngen_ablation import AttendAndExciteSynGenPipeline
import sys
sys.path.append(' ')
from utils.ptp_utils import AttentionStore
from utils import ptp_utils
from utils import vis_utils
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
import warnings
from diffusers import LCMScheduler
import numpy as np
import pandas as pd
import random

warnings.filterwarnings("ignore", category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

def load_model(config: RunConfig):
    def disabled_safety_checker(images, clip_input):
        if len(images.shape)==4:
            num_images = images.shape[0]
            return images, [False]*num_images
        else:
            return images, False
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    # safety_checker = StableDiffusionSafetyChecker.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="safety_checker")
    stable = AttendAndExciteSynGenPipeline.from_pretrained("SimianLuo/LCM_Dreamshaper_v7", dtype=torch.float32).to(device)
    tokenizer = stable.tokenizer
    stable.scheduler = LCMScheduler.from_config(stable.scheduler.config)
    stable.scheduler.set_timesteps(num_inference_steps=config.n_inference_steps,original_inference_steps=50,device=device)
    stable.safety_checker = disabled_safety_checker

    return stable


def get_indices_to_alter(stable, prompt: str) -> List[int]:
    token_idx_to_word = {idx: stable.tokenizer.decode(t)
                         for idx, t in enumerate(stable.tokenizer(prompt)['input_ids'])
                         if 0 < idx < len(stable.tokenizer(prompt)['input_ids']) - 1}
    pprint.pprint(token_idx_to_word)
    # token_indices = input("Please enter the a comma-separated list indices of the tokens you wish to "
    #                       "alter (e.g., 2,5): ")
    # token_indices = [int(i) for i in token_indices.split(",")]
    token_indices = [3, 7]
    print(f"Altering tokens: {[token_idx_to_word[i] for i in token_indices]}")

    return token_indices


def run_on_prompt(prompt: str,
                  model: AttendAndExciteSynGenPipeline,
                  controller: AttentionStore,
                  token_indices: List[int],
                  seed: torch.Generator,
                  config: RunConfig) -> Image.Image:
    if controller is not None:
        ptp_utils.register_attention_control(model, controller)
    prompt = [prompt]
    outputs = model(prompt=prompt,
                    attention_store=controller,
                    indices_to_alter=token_indices, # jubin change
                    attention_res=config.attention_res,
                    guidance_scale=config.guidance_scale,
                    generator=seed,
                    num_inference_steps=config.n_inference_steps,
                    max_iter_to_alter=config.max_iter_to_alter,
                    run_standard_sd=config.run_standard_sd,
                    thresholds=config.thresholds,
                    scale_factor=config.scale_factor,
                    scale_range=config.scale_range,
                    smooth_attentions=config.smooth_attentions,
                    sigma=config.sigma,
                    kernel_size=config.kernel_size,
                    sd_2_1=config.sd_2_1, 
                    tokenizer=model.tokenizer,
                    config=config)
    image = outputs.images[0]
    return image


@pyrallis.wrap()
def main(config: RunConfig):
    METHOD = config.method.split("_")[-1] # jubin change
    stable = load_model(config)
    images = []

    count = 0 # jubin change
    failed_idx = [] # jubin change
    time_total = 0 # jubin change

    dataset = pd.read_csv(config.dataset_path) # jubin change
    dataset_name = config.dataset_path.split("/")[-1].split('.')[0] # jubin change
    ts = time.time()
    for i in tqdm(range(len(dataset)), desc="Prompt idx"):
        i = config.idx if config.idx != -1 else i
        config.prompt = dataset.iloc[i].prompt # jubin change
        token_indices = dataset.iloc[i].item_indices # jubin change
        token_indices = eval(token_indices) if isinstance(token_indices, str) else token_indices # jubin change
        for j in range(2):
            seed = random.randint(0, 10000000)
            dataset_prompt_output_path = config.output_path / dataset_name / f"{i:003}" # jubin change
            dataset_prompt_output_path.mkdir(exist_ok=True, parents=True) # jubin change
            img_path = dataset_prompt_output_path / f'ablation_{METHOD}_{seed}.png' # jubin change
            if img_path.exists(): # jubin change
                continue          # jubin change
            print(f"Seed: {seed}")

            ts = time.time()
            g = torch.Generator('cuda').manual_seed(seed)
            controller = AttentionStore()
            image = run_on_prompt(prompt=config.prompt,
                                  model=stable,
                                  controller=controller,
                                  token_indices=token_indices,
                                  seed=g,
                                  config=config)
            te = time.time() # jubin change
            time_total += (te-ts) # jubin change
            count += 1 # jubin change 
            image.save(img_path)

            # except: # jubin change
            #     print('FAILED:',i, config.prompt) # jubin change
            #     failed_idx.append(f"{i}_{seed}") # jubin change
        if config.debug or (config.idx != -1): 
            break
            
    te = time.time() # jubin change
    print(f"*** Total time spent: {time_total:.4f} ***") # jubin change
    print(f"*** For one image: {time_total/count:.4f}") # jubin change
    with open(f"{config.output_path}/Time_{METHOD}_{dataset_name}.txt", 'w') as f: # jubin change
        f.write(f"{time_total/count:.4f}") # jubin change
        
    print("Failed prompt idx & seed") # jubin change
    print(failed_idx) # jubin change
    
if __name__ == '__main__':
    main()