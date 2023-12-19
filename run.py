import pprint
from typing import List
import time

import pyrallis
import torch
from PIL import Image

import pandas as pd
from config import RunConfig
from pipeline_attend_and_excite import AttendAndExcitePipeline
from utils import ptp_utils, vis_utils
from utils.ptp_utils import AttentionStore

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

METHOD = "A&E"

def load_model(config: RunConfig):
    def disabled_safety_checker(images, clip_input):
        if len(images.shape)==4:
            num_images = images.shape[0]
            return images, [False]*num_images
        else:
            return images, False
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    if config.sd_2_1:
        stable_diffusion_version = "stabilityai/stable-diffusion-2-1-base"
    else:
        stable_diffusion_version = "runwayml/stable-diffusion-v1-5"
    stable = AttendAndExcitePipeline.from_pretrained(stable_diffusion_version).to(device)
    stable.safety_checker = disabled_safety_checker

    return stable


def get_indices_to_alter(stable, prompt: str) -> List[int]:
    token_idx_to_word = {idx: stable.tokenizer.decode(t)
                         for idx, t in enumerate(stable.tokenizer(prompt)['input_ids'])
                         if 0 < idx < len(stable.tokenizer(prompt)['input_ids']) - 1}
    pprint.pprint(token_idx_to_word)
    token_indices = input("Please enter the a comma-separated list indices of the tokens you wish to "
                          "alter (e.g., 2,5): ")
    token_indices = [int(i) for i in token_indices.split(",")]
    print(f"Altering tokens: {[token_idx_to_word[i] for i in token_indices]}")
    return token_indices


def run_on_prompt(prompt: List[str],
                  model: AttendAndExcitePipeline,
                  controller: AttentionStore,
                  token_indices: List[int],
                  seed: torch.Generator,
                  config: RunConfig) -> Image.Image:
    if controller is not None:
        ptp_utils.register_attention_control(model, controller)
        
    outputs = model(prompt=prompt,
                    attention_store=controller,
                    indices_to_alter=token_indices,
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
                    sd_2_1=config.sd_2_1)
    image = outputs.images[0]
    return image


@pyrallis.wrap()
def main(config: RunConfig):
    global METHOD
    METHOD = 'Baseline' if config.run_standard_sd == True else METHOD
    stable = load_model(config)
    torch.autograd.set_detect_anomaly(True)
    stable = load_model(config)
    time_total = 0
    count = 0
    if config.dataset_path != '':
        dataset = pd.read_csv(config.dataset_path)
        dataset_name = config.dataset_path.split("/")[-1].split('.')[0]
        for i in range(len(dataset)):
            config.prompt = dataset.iloc[i].prompt
            token_indices = dataset.iloc[i].item_indices
            token_indices = eval(token_indices) if isinstance(token_indices, str) else token_indices
            
            for seed in config.seeds:
                dataset_prompt_output_path = config.output_path / dataset_name / f"{i:003}"
                dataset_prompt_output_path.mkdir(exist_ok=True, parents=True)
                img_path = dataset_prompt_output_path / f'{METHOD}_{config.model}_{seed}.png'
                if img_path.exists():
                    continue
                else:
                    count += 1
                    
                print(f"Seed: {seed}")
                g = torch.Generator('cuda').manual_seed(seed)
                controller = AttentionStore()
                ts = time.time()
                image = run_on_prompt(prompt=config.prompt,
                                      model=stable,
                                      controller=controller,
                                      token_indices=token_indices,
                                      seed=g,
                                      config=config)
                te = time.time()
                time_total += (te-ts)
                image.save(img_path)
        print(f"*** Total time spent: {te-ts:.4f} ***")
        print(f"*** For one image: {(te-ts)/((i+1)*len(config.seeds)):.4f}")
        with open(f"{config.output_path}/Time_{METHOD}_{config.model}_{dataset_name}_cnt{counts}.txt", 'w') as f:
            f.write(f"{(time_total / counts):.4f}") 
    else:
        token_indices = get_indices_to_alter(stable, config.prompt) if config.token_indices is None else config.token_indices

        images = []
        for seed in config.seeds:
            print(f"Seed: {seed}")
            g = torch.Generator('cuda').manual_seed(seed)
            controller = AttentionStore()
            image = run_on_prompt(prompt=config.prompt,
                                  model=stable,
                                  controller=controller,
                                  token_indices=token_indices,
                                  seed=g,
                                  config=config)
            prompt_output_path = config.output_path / config.prompt
            prompt_output_path.mkdir(exist_ok=True, parents=True)
            image.save(prompt_output_path / f'{seed}.png')
            images.append(image)

        # save a grid of results across all seeds
        joined_image = vis_utils.get_image_grid(images)
        joined_image.save(config.output_path / f'{config.prompt}.png')


if __name__ == '__main__':
    main()