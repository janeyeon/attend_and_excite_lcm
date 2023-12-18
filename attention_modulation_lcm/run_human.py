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
from dataset.total_list import *

from diffusers.image_processor import VaeImageProcessor
from diffusers.loaders import FromSingleFileMixin, LoraLoaderMixin, TextualInversionLoaderMixin
from diffusers.models import AutoencoderKL, ControlNetModel, UNet2DConditionModel
from diffusers.schedulers import KarrasDiffusionSchedulers
# from diffusers.utils import (
#     is_accelerate_available,
#     is_accelerate_version,
#     is_compiled_module,
#     logging,
#     randn_tensor,
#     replace_example_docstring,
# )
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
# from .multicontrolnet import MultiControlNetModel

from diffusers import DiffusionPipeline
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
  



    # with open("./dataset/dict_real.json", "r") as st_json:
    # with open("./dataset/dict_gt.json", "r") as st_json:
    #     print(f"open dict!!!")
    # #     group_dicts = json.load(st_json)
    # # with open("./dataset/data_299.json", "r") as st_json:
    #     group_dicts = json.load(st_json)
    # group_dictionary = group_dicts[image_idx]
    
    task_name = 'crowd_caption'

    config.output_path = config.output_path / task_name / image_idx

    
    ############################################################################################

    # config.output_path = config.output_path /  image_idx


    input_temp, output_temp = return_temp(image_idx)
    group_dictionary = makeGroupdict_custom(input_temp, output_temp)


    if not os.path.exists(config.output_path):
        # If it doesn't exist, create it
        os.makedirs(config.output_path)


    timesteps, sp_sz, bsz, mod_forward_orig, pipe = loadLCMNet(config)


    ### DenseDiff Global Param
    # reg_part = 0.7
    reg_part = 0.7
    sreg = .3
    # reg_part = 0
    # sreg = 1.
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


        return timesteps





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
        # noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        noise = torch.randn(shape, generator=generator, device=device, dtype=dtype)
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

        if encoder_hidden_states is not None:
            key  =   key[key.size(0)//2:,  ...]
            value  =  value[value.size(0)//2:,  ...]
    
        #################################################
        
        if DENSEDIFF_ON and (COUNT < NUM_INFERENCE_STEPS*reg_part):
            # print('DENSE')
            dtype = query.dtype
            if self.upcast_attention:
                query = query.float()
                key = key.float()
            
                # query  =  torch.concat([query, query], dim=0)
            sim = torch.baddbmm(torch.empty(query.shape[0], query.shape[1], key.shape[1], 
                                            dtype=query.dtype, device=query.device),
                                query, key.transpose(-1, -2), beta=0, alpha=self.scale)
            

            
            # treg = torch.pow(timesteps[COUNT]/1000, 5)
            treg = torch.pow(timesteps[COUNT]/1000, 15)


            ## reg at self-attn
            if encoder_hidden_states is None:
                min_value = sim[int(sim.size(0)/2):].min(-1)[0].unsqueeze(-1)
                max_value = sim[int(sim.size(0)/2):].max(-1)[0].unsqueeze(-1) 
                
                mask = sreg_maps[sim.size(1)].repeat(int(self.heads/2),1,1)
                size_reg = reg_sizes[sim.size(1)].repeat(int(self.heads/2),1,1)
                # print(f" self  size_reg: {size_reg.shape}, sreg: {sreg}, treg: {treg} max_value: {max_value.shape}, sim[int(sim.size(0)/2):]: {sim[int(sim.size(0)/2):].shape} ") 

                sim[int(sim.size(0)/2):] += (mask>0)*size_reg*sreg*treg*(max_value-sim[int(sim.size(0)/2):])
                sim[int(sim.size(0)/2):] -= ~(mask>0)*size_reg*sreg*treg*(sim[int(sim.size(0)/2):]-min_value)
                
            ## reg at cross-attn
            #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            else: 

                min_value = sim[int(sim.size(0)/2):].min(-1)[0].unsqueeze(-1)
                max_value = sim[int(sim.size(0)/2):].max(-1)[0].unsqueeze(-1) 
                
                mask = creg_maps[sim.size(1)].repeat(int(self.heads/2),1,1)
                size_reg = reg_sizes[sim.size(1)].repeat(int(self.heads/2),1,1)
                # print(f" cross size_reg: {size_reg.shape}, sreg: {sreg}, treg: {treg} max_value: {max_value.shape}, sim[int(sim.size(0)/2):]: {sim[int(sim.size(0)/2):].shape} ") 

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


        # img_sim = hidden_states
        # print(f"attention_probs: {hidden_states.shape}")
        # img = img_sim * 255 / hidden_states.min()
        # img = torch.mean(img, dim=0).detach().cpu().numpy() *255
        # im  = Image.fromarray(img.astype(np.uint8))
        # im.save(f"attn_image.png")
 

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


    # initial generation 
    config.gen_mode = 'global_focus'
    config.zoom_ratio = 1
    config.use_img2img = False
    config.enlarge_ratio = 2
    config.strength = 0.
    config.mix_ratio = 0.5
    config.reverse_ratio = 1.
    config.img2img_input_path = None
    

    print(f"!!!!!!!!!!!!!!!!!{config.gen_mode, config.zoom_ratio}!!!!!!!!!!!!!!!!!")

    # initialize 
    reg_part = 1
    sreg = .3
    # reg_part = 0
    # sreg = 1.
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


    views = get_views(config.default_H, config.default_W, window_size=config.window_size, stride=config.stride, circular_padding=False)

    attention2Mod(pipe)

    with torch.no_grad():
        ### 0. Loading
        # controlnet = pipe.controlnet._orig_mod if is_compiled_module(pipe.controlnet) else pipe.controlnet
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
            prompt_embeds=config.prompt_embeds,
        ) # Not used


        # 6. Prepare latent variables

        
        ### 2. Prepare timesteps ### 3. Prepare latent variables
        pipe.scheduler.set_timesteps(config.num_inference_steps, lcm_origin_steps= config.lcm_origin_steps)
        num_channels_latents = pipe.unet.config.in_channels

        attention2Orig(pipe, mod_forward_orig)

        # normal에서는 false
        timesteps = pipe.scheduler.timesteps

        latents = pipe.prepare_latents(
            batch_size* config.num_images_per_prompt,
            num_channels_latents,
            config.default_H,
            config.default_W,
            global_prompt_embeds.dtype,
            device,
            config.generator,
            None,
        ) # -> 주석 읽어보기

        print(f"timesteps:{timesteps}")

        
        ### 3-1. Prepare black latent variables

        pad_image = torch.ones((1, 3, config.default_H, config.default_W))*255 # global_canvas_H, global_canvas_W 사이즈의 흰색 이미지를 생성 
        pad_image = pipe.image_processor.preprocess(pad_image)
        pad_image = pad_image.to(next(iter(pipe.vae.post_quant_conv.parameters())).dtype).to(device)
        pad_latents = pipe.vae.encode(pad_image).latent_dist.mean
        pad_latents = (pipe.vae.config.scaling_factor * pad_latents).to(latents.dtype)

        attention2Mod(pipe)

        
        # 4. Define panorama grid and initialize views for synthesis.
        

        count = torch.zeros_like(latents)
        value = torch.zeros_like(latents)

            
        # 5. Prepare extra step kwargs. todo: Logic should ideally just be moved out of the pipeline
        # extra_step_kwargs = pipe.prepare_extra_step_kwargs(config.generator, config.eta)
        
    
        # 6. Denoising loop
        num_warmup_steps = len(timesteps) - config.num_inference_steps * pipe.scheduler.order
        with pipe.progress_bar(total=config.num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                count.zero_()
                value.zero_()

                
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
                                        config.default_H,
                                        config.default_W,
                                        global_prompt,
                                        bsz,
                                        views, 
                                        view_index)
                    
                    # 각 view의 각 prompt를 따로 encode해서 token으로 바꾼것들의 list를 저장해놓음
                    prompt_embeds_for_views_multi = []
                    # for prompt in full_prompt_for_views[view_index]:
                    for prompt in full_prompt_for_view_multi:
                        prompt_embeds_for_views_multi.append(pipe._encode_prompt(
                            prompt,
                            device,
                            config.num_images_per_prompt,
                            prompt_embeds=config.prompt_embeds,
                        ))                    

                

                    #! View_batch = 1 always!!!!!
                    vb_size = 1
                    view_batch = views[view_index]
                    views_scheduler_status = [copy.deepcopy(pipe.scheduler.__dict__)] * len(view_batch)
                    
                    full_prompt_for_views_batch = full_prompt_for_view_multi
                    
                    # get the latents corresponding to the current view coordinates

                    h_start, h_end, w_start, w_end = batch_view
                


                    latents_for_view = torch.cat(
                        [
                            latents[:, :, h_start:h_end, w_start:w_end]
                        ]
                    )

                                    
                    text_cond_total = text_cond_for_view_multi
                    sreg_maps_total = sreg_maps_for_view_multi
                    reg_sizes_total = reg_sizes_for_view_multi
                    creg_maps_total = creg_maps_for_view_multi
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
    
            
                        bs  =  batch_size * config.num_images_per_prompt
                        w = torch.tensor(config.guidance_scale).repeat(bs)
                        w_embedding = pipe.get_w_embedding(w, embedding_dim=256).to(device=device, dtype=latents.dtype)
                        ts = torch.full((bs,), t, device=device, dtype=torch.long)

                        
                        noise_pred = pipe.unet(
                            latents,
                            ts,
                            timestep_cond=w_embedding,
                            encoder_hidden_states=prompt_embeds_input,
                            cross_attention_kwargs=config.cross_attention_kwargs, 
                            return_dict=False)[0]
            
                        # # perform guidance
                        # if do_classifier_free_guidance:
                        #     noise_pred_uncond, noise_pred_text = noise_pred.chunk(2) # noise_pred 반으로 두조각 내기 
                        #     noise_pred = noise_pred_uncond + config.guidance_scale * (noise_pred_text - noise_pred_uncond)
            
                        ############# END: Controlnet, DenseDiff
                        
                    
                        # compute the previous noisy sample x_t -> x_t-1
                        # latents_denoised_batch = pipe.scheduler.step(
                        #     noise_pred, i,  t, latents_for_view, return_dict=False
                        # ).prev_sample

                        latents_denoised_batch, denoised = pipe.scheduler.step(
                            noise_pred, i,  t, latents_for_view, return_dict=False,  generator=config.generator
                        )

                        
                        # save views scheduler status after sample
                        views_scheduler_status[0] = copy.deepcopy(pipe.scheduler.__dict__)

                        progress_bar.update()
                        value[:, :, h_start:h_end, w_start:w_end] += latents_denoised_batch
                        count[:, :, h_start:h_end, w_start:w_end] += 1
                    


                    
                # take the MultiDiffusion step. Eq. 5 in MultiDiffusion paper: https://arxiv.org/abs/2302.08113
                latents = torch.where(count > 0, value / count, value)
                COUNT += 1
            

            attention2Mod(pipe)
            # attention2Orig(pipe)
            if not config.output_type == "latent":
                # latents = latents.to(next(iter(pipe.vae.post_quant_conv.parameters())).dtype)
                image = pipe.vae.decode(denoised / pipe.vae.config.scaling_factor, return_dict=False)[0]
                image, has_nsfw_concept = pipe.run_safety_checker(image, device, latents.dtype)
                has_nsfw_concept = None
            else:
                image = denoised
                has_nsfw_concept = None
        
            if has_nsfw_concept is None:
                do_denormalize = [True] * image.shape[0]
            else:
                do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]
            # image = image[:,:,:global_H,:global_W]
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

    

    image[0].save(config.output_path/ f'results_{config.gen_mode}_{image_idx}_{config.seed}.png')
    image[0].save(f"./hicon_outputs/{task_name}_{image_idx}_{config.seed}.png")






