import pprint
from typing import List
import pandas as pd
from tqdm import tqdm
import pyrallis
import time
import torch
from PIL import Image
import matplotlib.pyplot as plt
from config import RunConfig
# from pipeline_attend_and_excite import AttendAndExcitePipeline
from pipeline_attend_and_excite_syngen import AttendAndExciteSynGenPipeline
import sys
sys.path.append(' ')
from utils.ptp_utils import AttentionStore
from utils import ptp_utils
from utils import vis_utils
from LCM_Dreamshaper_v7.lcm_pipeline import LatentConsistencyModelPipeline
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
import warnings
from diffusers import LCMScheduler
import os

warnings.filterwarnings("ignore", category=UserWarning)
METHOD = 'LCM_CFG'

def load_model(config: RunConfig):
    safety_checker = StableDiffusionSafetyChecker.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="safety_checker")
    scheduler = LCMScheduler.from_pretrained("SimianLuo/LCM_Dreamshaper_v7", subfolder="scheduler")

    stable = LatentConsistencyModelPipeline.from_pretrained("SimianLuo/LCM_Dreamshaper_v7", scheduler=scheduler, safety_checker=safety_checker).to('cuda')
    stable.scheduler.set_timesteps(num_inference_steps=config.n_inference_steps,                                   original_inference_steps=50)
    tokenizer = stable.tokenizer

    return stable


def get_indices_to_alter(stable, prompt: str) -> List[int]:
    # token_idx_to_word = {idx: stable.tokenizer.decode(t)
    #                      for idx, t in enumerate(stable.tokenizer(prompt)['input_ids'])
    #                      if 0 < idx < len(stable.tokenizer(prompt)['input_ids']) - 1}
    # pprint.pprint(token_idx_to_word)
    # token_indices = input("Please enter the a comma-separated list indices of the tokens you wish to "
    #                       "alter (e.g., 2,5): ")
    # token_indices = [int(i) for i in token_indices.split(",")]
    token_indices = [3, 7]
    # print(f"Altering tokens: {[token_idx_to_word[i] for i in token_indices]}")

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
                    indices_to_alter=[],
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
                    tokenizer=model.tokenizer)
    image = outputs.images[0]
    return image


@pyrallis.wrap()
def main(config: RunConfig):
    #torch.autograd.set_detect_anomaly(True)
    safety_checker = StableDiffusionSafetyChecker.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="safety_checker")
    scheduler = LCMScheduler.from_pretrained("SimianLuo/LCM_Dreamshaper_v7", subfolder="scheduler")
    stable = LatentConsistencyModelPipeline.from_pretrained("SimianLuo/LCM_Dreamshaper_v7", scheduler=scheduler, safety_checker=safety_checker).to('cuda')
    stable.scheduler.set_timesteps(num_inference_steps=config.n_inference_steps,                                   original_inference_steps=50)
    tokenizer = stable.tokenizer
    count = 0
    time_total = 0
    if config.dataset_path != '':
        dataset = pd.read_csv(config.dataset_path)
        dataset_name = config.dataset_path.split("/")[-1].split('.')[0]
        ts = time.time()
        for i in tqdm(range(len(dataset)), desc="Prompt idx"):
            dataset_prompt_output_path = config.output_path / dataset_name / f"{i:003}"
            dataset_prompt_output_path.mkdir(exist_ok=True, parents=True)

                
            config.prompt = dataset.iloc[i].prompt
            token_indices = dataset.iloc[i].item_indices
            for seed in config.seeds:
                print(f"Seed: {seed}")
                img_path = dataset_prompt_output_path / f'{METHOD}_{config.model}_{seed}.png'
                if img_path.exists():
                    continue
                ts = time.time()
                g = torch.Generator('cuda:0').manual_seed(seed)
                image = stable(prompt=config.prompt, guidance_scale=2.5, num_inference_steps=4, lcm_origin_steps=50, output_type="pil").images
                image = np.array(image[0])
#                     image = run_on_prompt(prompt=config.prompt,
#                                           model=stable,
#                                           controller=controller,
#                                           token_indices=token_indices,
#                                           seed=g,
#                                           config=config)
                te = time.time()
                image.save(img_path)
                time_total += (te-ts)
                count += 1
                #except:
                #    print('FAILED:',i, config.prompt)
                
        te = time.time()
        print(f"*** Total time spent: {time_total:.4f} ***")
        print(f"*** For one image: {time_total/count:.4f}")
        with open(f"{config.output_path}/Time_{METHOD}_{config.model}_{dataset_name}.txt", 'w') as f:
            f.write(f"{time_total/count:.4f}") 
            
    

        # save a grid of results across all seeds
        joined_image = vis_utils.get_image_grid(images)
        joined_image.save(config.output_path / f'{config.prompt}.png')


if __name__ == '__main__':
    main()