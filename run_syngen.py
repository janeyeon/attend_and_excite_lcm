import pprint
from typing import List

import pyrallis
import torch
from PIL import Image
from config import RunConfig
# from pipeline_attend_and_excite import AttendAndExcitePipeline
from pipeline_attend_and_excite_syngen import AttendAndExciteSynGenPipeline
import sys
sys.path.append(' ')
from utils.ptp_utils import AttentionStore
from utils import ptp_utils
from utils import vis_utils
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
import warnings
from diffusers import LCMScheduler

warnings.filterwarnings("ignore", category=UserWarning)


def load_model(config: RunConfig):
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    safety_checker = StableDiffusionSafetyChecker.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="safety_checker")
    stable = AttendAndExciteSynGenPipeline.from_pretrained("SimianLuo/LCM_Dreamshaper_v7",safety_checker=safety_checker, dtype=torch.float32).to(device)
    tokenizer = stable.tokenizer
    stable.scheduler = LCMScheduler.from_config(stable.scheduler.config)
    stable.scheduler.set_timesteps(num_inference_steps=config.n_inference_steps,                                   original_inference_steps=50,device=device)
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
    torch.autograd.set_detect_anomaly(True)
    stable = load_model(config)
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