import pprint
from typing import List

import torch
from PIL import Image
# from config_controlnet_boxdiff import RunConfig

import numpy as np

from controlnet_aux import OpenposeDetector
from controlnet_aux.open_pose import draw_poses

import os
import json
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import PIL.Image
# from controlnet_aux.open_pose import PoseResult, BodyResult, Keypoint
# from controlnet_aux.util import HWC3
import cv2
import math


from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List
from drawer import draw_rectangle, DashedImageDraw

import matplotlib.pyplot as plt

from _config import *
from _utils import *
from _group_dict import *
from _load_model import *
from _generate_map import *
from _preprocess_patch import *
from gd import *
# from temp_list import *

from diffusers.image_processor import VaeImageProcessor
from diffusers.loaders import FromSingleFileMixin, LoraLoaderMixin, TextualInversionLoaderMixin
from diffusers.models import AutoencoderKL, ControlNetModel, UNet2DConditionModel
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import (
    is_accelerate_available,
    is_accelerate_version,
    is_compiled_module,
    logging,
    randn_tensor,
    replace_example_docstring,
)
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
# from .multicontrolnet import MultiControlNetModel
import copy
from diffusers.models.attention_processor import (
    AttnProcessor2_0,
    LoRAAttnProcessor2_0,
    LoRAXFormersAttnProcessor,
    XFormersAttnProcessor,
)
import argparse

def getGroupIds_auto(group_dictionary):
    group_ids = list(group_dictionary.keys())
    group_ids.remove('shape')
    group_ids.remove('global_caption')
    return group_ids

def get_canvas(H, W):
    global_canvas_H = int(max(64*math.ceil(H/64), 512))
    global_canvas_W = int(max(64*math.ceil(W/64), 512))

    return global_canvas_H, global_canvas_W

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--key', type=str)
    parser.add_argument('--seed', type=int)
    # name_list = {'

    args = parser.parse_args()
    config = RunConfig()


    image_idx = args.key

    try: 
        seed = args.seed
    except:
        print(f"seed is {config.seed}")
    else:
        print(f"seed is {seed}")
        config.seed = seed
        config.generator=torch.Generator().manual_seed(seed)
        

    print(f"\n\n\nREAL IDX: {image_idx}\n\n\n")

    os.makedirs("./hicon_outputs", exist_ok=True)
    
    # input_temp = 
    # output_temp = args.output_temp



    # with open("./dataset/group_dictionary_dic.json", "r") as st_json:
    # #     group_dicts = json.load(st_json)
    # # with open("./dataset/data_299.json", "r") as st_json:
    #     group_dicts = json.load(st_json)
    # group_dictionary = group_dicts[image_idx]

    if not os.path.exists(config.output_path / image_idx ):
        # If it doesn't exist, create it
        os.makedirs(config.output_path / image_idx )


    input_temp = \
"""
Image size 640 480;
Task: Estimation of person and object bounding boxes of an image;
Number of group boxes: 3;
Number of people and objects of Group0: P 3; O 0;
Number of people and objects of Group1: P 1; O 1;
Number of people and objects of Group2: P 4; O 0;
Description;
Global : an Star Wars Characters on the Alien planet, background is sparkling Milky way and lots of stars.;
Group0 : the Star Wars Characters on the Alien planet are in the left side.;
Group1 : the Darth Vader with a light saber in the middle and front side.;
Group2 : the Star Wars Characters on the Alien planet in the right side.;
Group0 bounding box; [ xmin 1 ymin 154 xmax 168 ymax 273 ];
Group1 bounding box; [ xmin 263 ymin 105 xmax 413 ymax 413 ];
Group2 bounding box; [ xmin 430 ymin 154 xmax 640 ymax 344 ];
Group0;
P0: a stormtrooper attacking people;
P1: a stormtrooper attacking people;
P2: a stormtrooper, handsome, holding a lightsabor, highly detailed;
Group1;
P0: A Darth Vader, handsome, holding a lightsabor, highly detailed;
O0: a lightsaber;
Group2;
P0: a stormtrooper attacking people;
P1: a stormtrooper attacking peoples;
P2: a stormtrooper attacking people;
P3: a stormtrooper attacking people;
"""

    #output_temp_3
    output_temp = \
"""\
Image size 640 480;
Task: Add keypoints of person;
Number of people and objects of Group0: P 3; O 0;
P0: [ person a 45 173 b 32 185 c 15 189 d 3 215 e 15 224 f 49 181 g 52 206 h 45 218 i 20 224 j 20 248 k 20 268 l 41 222 m 45 248 n 48 269 o 42 171 p 47 171 q 34 171 r 47 171 ]; [ xmin 1 ymin 154 xmax 61 ymax 269 ];
P1: [ person a 75 179 b 69 194 c 58 194 d 50 216 e 58 232 f 79 194 g 84 218 h 92 235 i 63 235 j 63 261 k 60 277 l 77 234 m 82 261 n 85 278 o 72 176 p 77 176 q 66 177 r 78 177 ]; [ xmin 42 ymin 160 xmax 103 ymax 273 ];
P2: [ person a 126 170 b 143 183 c 155 183 d 157 206 e 134 211 f 131 183 g 121 206 h 115 216 i 152 222 j 155 252 k 155 275 l 136 223 m 129 252 n 125 275 o 126 168 p 127 168 q 144 170 r 132 170 ]; [ xmin 102 ymin 154 xmax 168 ymax 273 ];
Number of people and objects of Group1: P 1; O 1;
P0: [ person a 323 141 b 338 183 c 307 185 d 276 221 e 265 192 f 369 182 g 388 231 h 377 267 i 321 284 j 291 332 k 295 391 l 362 284 m 381 337 n 395 400 o 321 136 p 333 133 q 321 138 r 346 133 ]; [ xmin 263 ymin 105 xmax 413 ymax 413 ];
O0: [ xmin 307 ymin 160 xmax 375 ymax 277 ];
Number of people and objects of Group0: P 4; O 0;
P0: [ person a 450 164 b 450 176 c 436 176 d 432 187 e 432 198 f 465 176 g 467 188 h 465 199 i 440 205 j 440 231 k 439 261 l 457 205 m 459 231 n 460 261 o 448 163 p 453 163 q 446 163 r 457 163 ]; [ xmin 430 ymin 154 xmax 466 ymax 308 ];
P1: [ person a 497 166 b 505 175 c 493 175 d 491 186 e 489 197 f 518 175 g 522 188 h 522 199 i 499 201 j 499 221 k 499 238 l 514 200 m 514 221 n 514 238 o 497 164 p 502 164 q 497 164 r 507 164 ]; [ xmin 488 ymin 158 xmax 518 ymax 300 ];
P2: [ person a 549 169 b 549 181 c 540 181 d 535 193 e 533 198 f 558 181 g 561 193 h 564 199 i 542 207 j 542 227 k 542 245 l 556 207 m 556 227 n 556 245 o 547 168 p 551 168 q 544 169 r 554 169 ]; [ xmin 528 ymin 161 xmax 561 ymax 304 ];
P3: [ person a 631 178 b 620 190 c 605 190 d 601 204 e 601 215 f 635 190 g 638 205 h 638 218 i 605 221 j 605 245 k 605 270 l 622 221 m 622 245 n 622 270 o 628 177 p 632 177 q 622 178 r 634 178 ]; [ xmin 582 ymin 165 xmax 640 ymax 344 ];
"""
    group_dictionary = makeGroupdict_custom(input_temp, output_temp)

    timesteps, sp_sz, bsz, mod_forward_orig, pipe = loadControlNet(config)


    ### DenseDiff Global Param
    reg_part = 0.7
    sreg = .3
    creg = 1.
    COUNT = 0
    DENSEDIFF_ON = True
    NUM_INFERENCE_STEPS = config.num_inference_steps

    ### padding
    use_padded_latents = True


    ### Hierachical Refinement
    
    def upcast_vae(pipe):
        dtype = pipe.vae.dtype
        pipe.vae.to(dtype=torch.float32)
        use_torch_2_0_or_xformers = isinstance(
            pipe.vae.decoder.mid_block.attentions[0].processor,
            (
                AttnProcessor2_0,
                XFormersAttnProcessor,
                LoRAXFormersAttnProcessor,
                LoRAAttnProcessor2_0,
            ),
        )
        # if xformers or torch_2_0 is used attention block does not need
        # to be in float32 which can save lots of memory
        if use_torch_2_0_or_xformers:
            pipe.vae.post_quant_conv.to(dtype)
            pipe.vae.decoder.conv_in.to(dtype)
            pipe.vae.decoder.mid_block.to(dtype)
        return pipe

    def get_timesteps(pipe, num_inference_steps, strength, device, reverse_ratio):
        #reverse_ratio: 1: original image, 0: noise image
        # get the original timestep using init_timestep
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = pipe.scheduler.timesteps[t_start * pipe.scheduler.order :]

        # t_mix = int(1000 * (1 - reverse_ratio))
        # mix_steps = int(t_mix/1000 * 50)

        # pipe.scheduler.set_timesteps(num_inference_steps, device=device)
        # timesteps = pipe.scheduler.timesteps
        # timesteps_list = list(range(t_mix, 0, -20))
        # timesteps = torch.tensor(timesteps_list, dtype=torch.int, device='cuda')
        return timesteps


    def double_image(input_image, enlarge_ratio, probability=0.9):
        input_image = np.array(input_image)
        # input_image = input_image.transpose((1, 0, 2))
        distance=1
        # Get the dimensions of the input image
        height, width, channels = input_image.shape

        # Calculate the new dimensions
        new_height = height * enlarge_ratio
        new_width = width * enlarge_ratio

        # Create an empty expanded image
        expanded_image = cv2.resize(input_image, (new_width, new_height), cv2.INTER_LANCZOS4)

        # Loop through each pixel in the expanded image
        for y in range(new_height):
            for x in range(new_width):
                if random.random() > probability:
                    # Find a random pixel within the specified distance in the original image
                    source_x = min(max(0, int(x / enlarge_ratio) + random.randint(-distance, distance)), width - distance - 1)
                    source_y = min(max(0, int(y / enlarge_ratio) + random.randint(-distance, distance)), height - distance - 1)
                    # Copy the pixel from the original image to the expanded image
                    expanded_image[y, x] = input_image[source_y, source_x]
                else:
                    pass
        
        expanded_image = np.array(expanded_image) / 255.
        expanded_image = torch.tensor(expanded_image, device='cuda', dtype=torch.float32).unsqueeze(0).permute(0, 3, 1, 2)
        return expanded_image




    def prepare_latents(pipe, image, timestep, batch_size, num_images_per_prompt, dtype, device, generator=None):
        """
        if image 가 latent이면 -> 그대로 냅둠
        image가 image 면 -> pipe.vae.encoder 통과

        이후 scheduler에 맞게 noise를  추가하여 init_latents생성후 return
        """
        if not isinstance(image, (torch.Tensor, PIL.Image.Image, list)):
            raise ValueError(
                f"`image` has to be of type `torch.Tensor`, `PIL.Image.Image` or list but is {type(image)}"
            )

        image = image.to(device=device, dtype=dtype)

        batch_size = batch_size * num_images_per_prompt

        if image.shape[1] == 4:
            init_latents = image

        else:
            if isinstance(generator, list) and len(generator) != batch_size:
                raise ValueError(
                    f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                    f" size of {batch_size}. Make sure the batch size matches the length of the generators."
                )

            elif isinstance(generator, list):
                init_latents = [
                    pipe.vae.encode(image[i : i + 1]).latent_dist.sample(generator[i]) for i in range(batch_size)
                ]
                init_latents = torch.cat(init_latents, dim=0)
            else:
                init_latents = pipe.vae.encode(image).latent_dist.sample(generator)

            init_latents = pipe.vae.config.scaling_factor * init_latents

        if batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] == 0:
            # expand init_latents for batch_size
            deprecation_message = (
                f"You have passed {batch_size} text prompts (`prompt`), but only {init_latents.shape[0]} initial"
                " images (`image`). Initial images are now duplicating to match the number of text prompts. Note"
                " that this behavior is deprecated and will be removed in a version 1.0.0. Please make sure to update"
                " your script to pass as many initial images as text prompts to suppress this warning."
            )
            deprecate("len(prompt) != len(image)", "1.0.0", deprecation_message, standard_warn=False)
            additional_image_per_prompt = batch_size // init_latents.shape[0]
            init_latents = torch.cat([init_latents] * additional_image_per_prompt, dim=0)
        elif batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] != 0:
            raise ValueError(
                f"Cannot duplicate `image` of batch size {init_latents.shape[0]} to {batch_size} text prompts."
            )
        else:
            init_latents = torch.cat([init_latents], dim=0)

        shape = init_latents.shape
        noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)

        # get latents
        init_latents = pipe.scheduler.add_noise(init_latents, noise, timestep)
        latents = init_latents

        return latents


    ##################################################################################################
    def mod_forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, config=None):
        residual = hidden_states
        if self.spatial_norm is not None:
                hidden_states = attn.spatial_norm(hidden_states, temb)
        hidden_states_orig = hidden_states
        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        
        
        batch_size, sequence_length, _ = hidden_states.shape
        
        attention_mask = self.prepare_attention_mask(attention_mask, sequence_length, batch_size=batch_size)
        
        query = self.to_q(hidden_states)
        query = self.head_to_batch_dim(query)
        
        context_states = text_cond if encoder_hidden_states is not None else hidden_states
        key = self.to_k(context_states)
        value = self.to_v(context_states)
        key = self.head_to_batch_dim(key)
        value = self.head_to_batch_dim(value)
        
        #################################################
        
        if DENSEDIFF_ON and (COUNT < NUM_INFERENCE_STEPS*reg_part):
            # print('DENSE')
            dtype = query.dtype
            if self.upcast_attention:
                query = query.float()
                key = key.float()
                
                
            sim = torch.baddbmm(torch.empty(query.shape[0], query.shape[1], key.shape[1], 
                                            dtype=query.dtype, device=query.device),
                                query, key.transpose(-1, -2), beta=0, alpha=self.scale)
            
            treg = torch.pow(timesteps[COUNT]/1000, 5)

            ## reg at self-attn
            if encoder_hidden_states is None:
                min_value = sim[int(sim.size(0)/2):].min(-1)[0].unsqueeze(-1)
                max_value = sim[int(sim.size(0)/2):].max(-1)[0].unsqueeze(-1)  
                mask = sreg_maps[sim.size(1)].repeat(self.heads,1,1)
                size_reg = reg_sizes[sim.size(1)].repeat(self.heads,1,1)
                
                sim[int(sim.size(0)/2):] += (mask>0)*size_reg*sreg*treg*(max_value-sim[int(sim.size(0)/2):])
                sim[int(sim.size(0)/2):] -= ~(mask>0)*size_reg*sreg*treg*(sim[int(sim.size(0)/2):]-min_value)
                
            ## reg at cross-attn
            else:
                min_value = sim[int(sim.size(0)/2):].min(-1)[0].unsqueeze(-1)
                max_value = sim[int(sim.size(0)/2):].max(-1)[0].unsqueeze(-1)  
                mask = creg_maps[sim.size(1)].repeat(self.heads,1,1)
                size_reg = reg_sizes[sim.size(1)].repeat(self.heads,1,1)
                
                sim[int(sim.size(0)/2):] += (mask>0)*size_reg*creg*treg*(max_value-sim[int(sim.size(0)/2):])
                sim[int(sim.size(0)/2):] -= ~(mask>0)*size_reg*creg*treg*(sim[int(sim.size(0)/2):]-min_value)

            attention_probs = sim.softmax(dim=-1)
            attention_probs = attention_probs.to(dtype)
        

        else:
            # print('No')
            attention_probs = self.get_attention_scores(query, key, attention_mask)
            
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = self.batch_to_head_dim(hidden_states)
        
        # linear proj
        hidden_states = self.to_out[0](hidden_states)
        # dropout
        hidden_states = self.to_out[1](hidden_states)
        # print(residual.shape, hidden_states.shape)
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        if self.residual_connection:
            hidden_states = hidden_states + residual
            

        hidden_states = hidden_states / self.rescale_output_factor

        return hidden_states


    def attention2Mod(pipe):
        for n, _module in pipe.unet.up_blocks.named_modules():
            if _module.__class__.__name__ == "Attention":
                _module.__class__.__call__ = mod_forward


    def attention2Orig(pipe, mod_orig):
        for n, _module in pipe.unet.up_blocks.named_modules():
            if _module.__class__.__name__ == "Attention":
                _module.__class__.__call__ = mod_orig



    ############################################################################################################
    """
    view
    pose_image_for_views

    full_prompt_for_views

    text_cond_for_views
    sreg_maps_for_views
    reg_sizes_for_views
    creg_maps_for_views

    global_prompt
    global_H
    global_W
    global_canvas_H
    global_canvas_W
    output_path

    """

    # strength = config.strength
    # print(f'strength: {strength}')

    # if config.gen_mode == 'global_focus':
    #     # mix_mask_for_views = None
    #     img2img_input_path = None
    # elif config.gen_mode == 'group_focus':
    #     # mix_mask_for_views = group_mix_mask_for_views
    #     # mix_mask = inst_obj_mix_mask#group_mix_mask
    #     img2img_input_path = config.output_path / f'results_1.png'
        
    # elif config.gen_mode == 'inst_obj_focus':
    #     # mix_mask_for_views = inst_obj_mix_mask_for_views
    #     # mix_mask = inst_obj_mix_mask
    #     img2img_input_path = config.output_path / f'results_2.png'

    attention2Mod(pipe)

    for hierar_step in range(3):
        if hierar_step == 0:
            continue
            # initial generation 
            config.gen_mode = 'global_focus'
            config.zoom_ratio = 1
            config.use_img2img = False
            config.enlarge_ratio = 2
            config.strength = 0.
            config.mix_ratio = 0.5
            config.reverse_ratio = 1.
            config.img2img_input_path = None
        elif hierar_step == 1:
            continue
            config.gen_mode = 'group_focus'
            config.zoom_ratio = 2
            config.use_img2img = True
            config.enlarge_ratio = 2
            config.strength = 0.7
            config.mix_ratio = 0.5
            config.reverse_ratio = 0.5
            config.img2img_input_path = config.output_path / image_idx  / f'results_global_focus_{image_idx}_{config.seed}.png' # Later
        else:
            config.gen_mode = 'inst_obj_focus'
            config.zoom_ratio = 4
            config.use_img2img = True
            config.enlarge_ratio = 2
            config.strength = 0.5
            config.mix_ratio = 0.5
            config.reverse_ratio = 0.5
            config.img2img_input_path = config.output_path /image_idx / f'results_group_focus_{image_idx}_{config.seed}.png' # Later

        print(f"!!!!!!!!!!!!!!!!!{config.gen_mode, config.zoom_ratio}!!!!!!!!!!!!!!!!!")

        # initialize 
        reg_part = 0.7
        sreg = .3
        creg = 1.
        COUNT = 0
        DENSEDIFF_ON = True
        NUM_INFERENCE_STEPS = config.num_inference_steps

        global_prompt = group_dictionary['global_caption']
        global_W, global_H = group_dictionary['shape']
        global_H *= config.zoom_ratio
        global_W *= config.zoom_ratio
        global_canvas_H, global_canvas_W = get_canvas(global_H, global_W)

        group_ids = getGroupIds_auto(group_dictionary)

        pose_map,\
        group_L_maps,\
        inst_obj_L_maps,\
        inst_obj_boxes, \
        inst_obj_L_maps_small,\
        group_maps,\
        inst_maps,\
        obj_maps,\
        group_prompt_dic,\
        inst_obj_prompt_dic,\
        poses, \
        pose_masks, \
        nose2neck_lengths,\
        group_boxes,\
        inst_boxes,\
        obj_boxes, \
        inst_obj_maps,\
        pose_box_map, \
        group_mix_mask, \
        inst_obj_mix_mask = generate_map( 
                    global_canvas_H=global_canvas_H, 
                    global_canvas_W=global_canvas_W, 
                    config=config, 
                    global_prompt=global_prompt,
                    group_ids=group_ids,
                    group_dictionary=group_dictionary,
                    image_idx=image_idx
                    )


        views = get_views(global_canvas_H, global_canvas_W, window_size=config.window_size, stride=config.stride, circular_padding=False)

        attention2Mod(pipe)

        with torch.no_grad():
            ### 0. Loading
            controlnet = pipe.controlnet._orig_mod if is_compiled_module(pipe.controlnet) else pipe.controlnet
            batch_size = 1
            device = pipe._execution_device
            do_classifier_free_guidance = config.guidance_scale > 1.0 #True
            
            control_guidance_start, control_guidance_end = [config.control_guidance_start], [config.control_guidance_end]
            
            ### 1. Encode input prompt
            text_encoder_lora_scale = (
                config.cross_attention_kwargs.get("scale", None) if config.cross_attention_kwargs is not None else None
            )
            global_prompt_embeds = pipe._encode_prompt(
                global_prompt,
                device,
                config.num_images_per_prompt,
                do_classifier_free_guidance,
                negative_prompt=config.negative_prompt,
                prompt_embeds=config.prompt_embeds,
                negative_prompt_embeds=config.negative_prompt_embeds,
                lora_scale=text_encoder_lora_scale,
            ) # Not used


            # 6. Prepare latent variables

            
            ### 2. Prepare timesteps ### 3. Prepare latent variables
            pipe.scheduler.set_timesteps(config.num_inference_steps, device=device)
            num_channels_latents = pipe.unet.config.in_channels

            attention2Orig(pipe, mod_forward_orig)

            # normal에서는 false
            timesteps = pipe.scheduler.timesteps

            latents = pipe.prepare_latents(
                batch_size,
                num_channels_latents,
                global_canvas_H,
                global_canvas_W,
                global_prompt_embeds.dtype,
                device,
                config.generator,
                None,
            ) # -> 주석 읽어보기



            if config.use_img2img:
                COUNT = int(NUM_INFERENCE_STEPS / config.reverse_ratio)
                t_mix = int(1000 * (1 - config.reverse_ratio))
                mix_steps = int(t_mix/1000 * 50)
                timesteps_list = list(range(t_mix+1 - int(1000/NUM_INFERENCE_STEPS), 0, -int(1000/NUM_INFERENCE_STEPS)))
                timesteps = torch.tensor(timesteps_list, dtype=torch.int, device='cuda')
                low_res_image = Image.open(config.img2img_input_path)
                resize_image = double_image(low_res_image, config.enlarge_ratio)

                is_odd_h = (global_H % 2)
                is_odd_w = (global_W % 2)

                img2img_input = torch.ones((1,3,global_canvas_H, global_canvas_W), dtype=torch.half, device=config.device)
                img2img_input[:,:,:global_H-is_odd_h,:global_W-is_odd_w] = resize_image
                
                prior_image_latent = pipe.vae.encode(img2img_input*2 - 1).latent_dist.sample() * 0.18215
                latents = pipe.scheduler.add_noise(prior_image_latent, latents, timesteps=torch.tensor([t_mix], dtype=torch.int, device='cuda'))


                print(f"latents_timestep:{t_mix}")

            print(f"timesteps:{timesteps}")

            
            ### 3-1. Prepare black latent variables

            pad_image = torch.ones((1, 3, global_canvas_H, global_canvas_W))*255 # global_canvas_H, global_canvas_W 사이즈의 흰색 이미지를 생성 
            pad_image = pipe.image_processor.preprocess(pad_image)
            pad_image = pad_image.to(next(iter(pipe.vae.post_quant_conv.parameters())).dtype).to(device)
            pad_latents = pipe.vae.encode(pad_image).latent_dist.mean
            pad_latents = (pipe.vae.config.scaling_factor * pad_latents).to(latents.dtype)

            attention2Mod(pipe)

            
            # 4. Define panorama grid and initialize views for synthesis.
            

            count = torch.zeros_like(latents)
            value = torch.zeros_like(latents)

                
            # 5. Prepare extra step kwargs. todo: Logic should ideally just be moved out of the pipeline
            extra_step_kwargs = pipe.prepare_extra_step_kwargs(config.generator, config.eta)
            
            # 5.1 Create tensor stating which controlnets to keep
            controlnet_keep = []
            for i in range(len(timesteps)):
                keeps = [
                    1.0 - float(i / len(timesteps) < s or (i + 1) / len(timesteps) > e)
                    for s, e in zip(control_guidance_start, control_guidance_end)
                ]
                controlnet_keep.append(keeps[0] if isinstance(controlnet, ControlNetModel) else keeps)
                    
            
            # 6. Denoising loop
            num_warmup_steps = len(timesteps) - config.num_inference_steps * pipe.scheduler.order
            with pipe.progress_bar(total=len(timesteps)) as progress_bar:
                for i, t in enumerate(timesteps):
                    count.zero_()
                    value.zero_()

                    #! When used mix_mask
                    # if config.use_img2img and i < len(timesteps)-1:
                    #     # print('hi')
                    #     # latents = latents_img
                        
                    #     if latent_timesteps[0] == timesteps[i]:
                    #         print(latent_timesteps[0], timesteps[i])
                    #         if config.gen_mode == 'group_focus':
                    #             mix_mask = group_mix_mask.to(latents.dtype).to(latents.device).unsqueeze(0).unsqueeze(0)
                    #         elif config.gen_mode == 'inst_obj_focus':
                    #             mix_mask = inst_obj_mix_mask.to(latents.dtype).to(latents.device).unsqueeze(0).unsqueeze(0)
                    #         latents = mix_mask * latents + (1-mix_mask) * latents_img
                    
                    
                    
                    # DENSEDIFF_ON = True
                    # NUM_GROUPS = len(view_batch)
                    NUM_GROUPS = len(views)

                    attention2Mod(pipe)

                    # for j, batch_view in enumerate(view_batch):
                    for view_index, batch_view in enumerate(views):
                        #!!!!! Instance/Object description manipulation
                        # inst_obj_prompt_dic[2] = 'Donald Trumph wearing a suit'

                        full_prompt_for_view_multi,\
                        prompts_for_view_multi,\
                        text_cond_for_view_multi,\
                        sreg_maps_for_view_multi,\
                        reg_sizes_for_view_multi,\
                        creg_maps_for_view_multi,\
                        excluded_prompts,\
                        pose_image_for_view = preprocess_patch(pipe, config,pose_map,
                                            group_L_maps,
                                            inst_obj_L_maps,
                                            inst_obj_L_maps_small,
                                            group_prompt_dic,
                                            inst_obj_prompt_dic,
                                            global_canvas_H,
                                            global_canvas_W,
                                            global_prompt,
                                            bsz,
                                            views, 
                                            view_index)
                        # 
                        
                        # print(f"i : {i}, view_index:{view_index}")
                        # print(f"pose_image_for_view :{pose_image_for_view}")
                        # print(f"sreg_maps_for_view_multi :{sreg_maps_for_view_multi}")
                        # print(f"reg_sizes_for_view_multi :{reg_sizes_for_view_multi}")
                        # print(f"creg_maps_for_view_multi :{creg_maps_for_view_multi}")



                        # patch 사이즈로 자른 pose image = image를 prepare_image에 넣어놓음
                        control_image_for_views = pipe.prepare_image( ################ H and W
                                        image=pose_image_for_view,
                                        width=config.window_size*8,
                                        height=config.window_size*8,
                                        batch_size=batch_size * config.num_images_per_prompt,
                                        num_images_per_prompt=config.num_images_per_prompt,
                                        device=device,
                                        dtype=controlnet.dtype,
                                        do_classifier_free_guidance=do_classifier_free_guidance,
                                        guess_mode=config.guess_mode,
                                    ) 
                        # 각 view의 각 prompt를 따로 encode해서 token으로 바꾼것들의 list를 저장해놓음
                        prompt_embeds_for_views_multi = []
                        # for prompt in full_prompt_for_views[view_index]:
                        for prompt in full_prompt_for_view_multi:
                            prompt_embeds_for_views_multi.append(pipe._encode_prompt(
                                prompt,
                                device,
                                config.num_images_per_prompt,
                                do_classifier_free_guidance,
                                negative_prompt=config.negative_prompt,
                                prompt_embeds=config.prompt_embeds,
                                negative_prompt_embeds=config.negative_prompt_embeds,
                                lora_scale=text_encoder_lora_scale,
                            ))                    

                    

                        #! View_batch = 1 always!!!!!
                        vb_size = 1
                        view_batch = views[view_index]
                        views_scheduler_status = [copy.deepcopy(pipe.scheduler.__dict__)] * len(view_batch)
                        # control_image_for_views_batch = control_image_for_views[j]
                        control_image_for_views_batch = control_image_for_views[0]
                        # if config.gen_mode in ['group_focus', 'inst_obj_focus']:
                        #     mix_mask_for_views_batch = mix_mask_for_views[view_index]
                        
                        full_prompt_for_views_batch = full_prompt_for_view_multi
                        
                        # text_cond_for_views_batch = text_cond_for_view_multi
                        # sreg_maps_for_views_batch = sreg_maps_for_view_multi
                        # reg_sizes_for_views_batch = reg_sizes_for_view_multi
                        # creg_maps_for_views_batch = creg_maps_for_view_multi

                        
                                    
                    
                        # get the latents corresponding to the current view coordinates

                        h_start, h_end, w_start, w_end = batch_view
                    


                        latents_for_view = torch.cat(
                            [
                                latents[:, :, h_start:h_end, w_start:w_end]
                            ]
                        )

                        
                        control_image = control_image_for_views_batch.to(latents.dtype).to(latents.device)
                        
                        text_cond_total = text_cond_for_view_multi
                        sreg_maps_total = sreg_maps_for_view_multi
                        reg_sizes_total = reg_sizes_for_view_multi
                        creg_maps_total = creg_maps_for_view_multi
                        # prompt_embeds_total = prompt_embeds_for_views
                        prompt_embeds_total = prompt_embeds_for_views_multi
                        views_scheduler_status_total = [views_scheduler_status[0]] * len(text_cond_total)



                        
                        for k in range(len(text_cond_total)):
                            text_cond = text_cond_total[k].to(latents.dtype).to(latents.device)
                            sreg_maps = sreg_maps_total[k]
                            reg_sizes = reg_sizes_total[k]
                            creg_maps = creg_maps_total[k]
                            prompt_embeds = prompt_embeds_total[k].to(latents.dtype).to(latents.device)

                        
                            # rematch block's scheduler status
                            pipe.scheduler.__dict__.update(views_scheduler_status_total[k])

                            
                            # expand the latents if we are doing classifier free guidance
                            latent_model_input = (
                                latents_for_view.repeat_interleave(2, dim=0)
                                if do_classifier_free_guidance
                                else latents_for_view
                            )
                        
                            latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)
                
                            # repeat prompt_embeds for batch
                            prompt_embeds_input = torch.cat([prompt_embeds] * vb_size) # TODO: view batch 에 따라 변경해야함
            
                            
                            ############# START: Controlnet, DenseDiff
                            # controlnet(s) inference
                            # 일단 guess_mode = False상태임, do_classifier_free_guidance 는 True
                            if config.guess_mode and do_classifier_free_guidance:
                                # Infer ControlNet only for the conditional batch.
                                control_model_input = latents_for_view # todo: guess mode 뭐고?
                                control_model_input = pipe.scheduler.scale_model_input(control_model_input, t)
                                controlnet_prompt_embeds = prompt_embeds_input.chunk(2)[1]
                            else:
                                control_model_input = latent_model_input
                                controlnet_prompt_embeds = prompt_embeds_input
                
                            if isinstance(controlnet_keep[i], list):
                                cond_scale = [c * s for c, s in zip(config.controlnet_conditioning_scale, controlnet_keep[i])]
                            else:
                                controlnet_cond_scale = config.controlnet_conditioning_scale
                                if isinstance(controlnet_cond_scale, list):
                                    controlnet_cond_scale = controlnet_cond_scale[0]
                                cond_scale = controlnet_cond_scale * controlnet_keep[i]
                            
                            
                            down_block_res_samples, mid_block_res_sample = pipe.controlnet(
                                control_model_input,
                                t,
                                encoder_hidden_states=controlnet_prompt_embeds,
                                controlnet_cond=control_image,
                                conditioning_scale=cond_scale,
                                guess_mode=config.guess_mode,
                                return_dict=False,
                            )
                
                            if config.guess_mode and do_classifier_free_guidance:
                                # Infered ControlNet only for the conditional batch.
                                # To apply the output of ControlNet to both the unconditional and conditional batches,
                                # add 0 to the unconditional batch to keep it unchanged.
                                down_block_res_samples = [torch.cat([torch.zeros_like(d), d]) for d in down_block_res_samples]
                                mid_block_res_sample = torch.cat([torch.zeros_like(mid_block_res_sample), mid_block_res_sample])
                            # predict the noise residual
                            noise_pred = pipe.unet(
                                latent_model_input,
                                t,
                                encoder_hidden_states=prompt_embeds_input,
                                cross_attention_kwargs=config.cross_attention_kwargs,
                                down_block_additional_residuals=down_block_res_samples,
                                mid_block_additional_residual=mid_block_res_sample,
                                return_dict=False,
                            )[0]
                
                            # perform guidance
                            if do_classifier_free_guidance:
                                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2) # noise_pred 반으로 두조각 내기 
                                noise_pred = noise_pred_uncond + config.guidance_scale * (noise_pred_text - noise_pred_uncond)
                
                            ############# END: Controlnet, DenseDiff
                            
                        
                            # compute the previous noisy sample x_t -> x_t-1
                            latents_denoised_batch = pipe.scheduler.step(
                                noise_pred, t, latents_for_view, **extra_step_kwargs
                            ).prev_sample

                            
                            # save views scheduler status after sample
                            views_scheduler_status[0] = copy.deepcopy(pipe.scheduler.__dict__)

                            
                            value[:, :, h_start:h_end, w_start:w_end] += latents_denoised_batch
                            count[:, :, h_start:h_end, w_start:w_end] += 1
                        torch.cuda.empty_cache()


                        
                    # take the MultiDiffusion step. Eq. 5 in MultiDiffusion paper: https://arxiv.org/abs/2302.08113
                    latents = torch.where(count > 0, value / count, value)
                    COUNT += 1
                    

                    
                    if use_padded_latents:               
                        soft_patting_ratio = 0.5
                        border_offset_H =  int( (global_canvas_H - global_H)/8*soft_patting_ratio)
                        border_offset_W =  int( (global_canvas_W - global_W)/8*soft_patting_ratio)
                        shape = pad_latents.shape
                        noise = randn_tensor(shape, generator=config.generator, device=device, dtype=pad_latents.dtype)
                        if i != len(timesteps) - 1:
                            pad_latents_noisy = pipe.scheduler.add_noise(pad_latents, noise, timesteps[i+1]).to(latents.dtype)
                        else:
                            pad_latents_noisy = pad_latents
                        
                        latents[:, :, int(global_H/8)+border_offset_H:, :] = pad_latents_noisy[:, :, int(global_H/8)+border_offset_H:, :]
                        latents[:, :, :, int(global_W/8)+border_offset_W:] = pad_latents_noisy[:, :, :, int(global_W/8)+border_offset_W:]
                    
            
                    # call the callback, if provided
                    if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % pipe.scheduler.order == 0):
                        progress_bar.update()
                        if config.callback is not None and i % config.callback_steps == 0:
                            config.callback(i, t, latents)
                            

                attention2Mod(pipe)
                if not config.output_type == "latent":
                    latents = latents.to(next(iter(pipe.vae.post_quant_conv.parameters())).dtype)
                    image = pipe.vae.decode(latents / pipe.vae.config.scaling_factor, return_dict=False)[0]
                    image, has_nsfw_concept = pipe.run_safety_checker(image, device, latents.dtype)
                    has_nsfw_concept = None
                else:
                    image = latents
                    has_nsfw_concept = None
            
                if has_nsfw_concept is None:
                    do_denormalize = [True] * image.shape[0]
                else:
                    do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]
                image = image[:,:,:global_H,:global_W]
                image = pipe.image_processor.postprocess(image, output_type=config.output_type, do_denormalize=do_denormalize)
            
        

        print(f'* Global prompt: {global_prompt}')
        for excluded_prompt in excluded_prompts:
            print(f"[Warning]'{excluded_prompt}' is excluded because it exceeds the token limits.")
        print(f'- image idx: {config.image_idx}')
        print(f'- seed: {config.seed}')
        print(f'- reg_part: {reg_part}')
        print(f'- size: {image[0].size}')

        print()
        print(f'* Groups')
        for i, (group_x1, group_y1, group_x2, group_y2) in enumerate(group_boxes):
            key = int(group_maps[i,:,:].max().item())
            print(f'green: {group_prompt_dic[key]}')

        print()
        print(f'* Instance and Objects')
        for i, (x1, y1, x2, y2) in enumerate(inst_obj_boxes):
            key = int(inst_obj_maps[i,:,:,1].max().item())
            print(f'{config.color[i]}: {inst_obj_prompt_dic[key]}')

        

        image[0].save(config.output_path /image_idx/ f'results_{config.gen_mode}_{image_idx}_{config.seed}.png')
        image[0].save(f"./hicon_outputs/{image_idx}_{config.seed}.png")



        combined_pil = Image.fromarray(np.concatenate([np.asarray(pose_box_map)[:global_H,:global_W,:]]+[np.asarray(image[0])], 1)) # 사이즈 맞춰야함
        combined_pil.save(config.output_path / image_idx/ f'results_combined_{config.gen_mode}_{image_idx}_{config.seed}.png')
        # combined_pil.save(f"{config.image_idx}.png")
        # display(combined_pil)
        # break
        
        # torch.cuda.empty_cache()