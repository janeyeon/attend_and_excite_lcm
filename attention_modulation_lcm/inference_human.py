import os
import random
import pickle
import argparse
import pdb
import datetime
import hashlib

import numpy as np
from PIL import Image
from tqdm.auto import tqdm

import torch
import torch.nn.functional as F
from torchvision import transforms
import diffusers
from diffusers.pipelines.stable_diffusion import StableDiffusionPipeline
from diffusers.pipelines import DiffusionPipeline
from diffusers import DDIMScheduler, LCMScheduler
import transformers
from transformers import CLIPTextModel, CLIPTokenizer


from _config import *
# from _utils import *
from _group_dict import *
# from _load_model import *
from _generate_map_human import *
from _generate_map import *
from _preprocess_patch import *
from gd import *
from attention import *
from dataset.total_list import *

"""
Example usage

- Basic usage
python inference_densediff.py --model SD --batch_size 1 -s 50 -idx 1 (-s = num inference steps)
python inference_densediff.py --model LCM --batch_size 1 -s 4 -idx 1

- Change dense diffusion parameters
python inference_densediff.py --model LCM --batch_size 1 -s 16 --reg_part 0.5 --creg 1.5 --sreg 0.5 --pow_time 3 --idx 1

- Generate images for multiple indices
python inference_densediff.py --model LCM --batch_size 1 -s 16 --idx 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19

- Generate without dense diffusion (add -w argument)
python inference_densediff.py --model LCM --batch_size 1 -s 16 -w -idx 1 
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='LCM', choices=['LCM', 'SD'])
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--idx', type=int, default=[1], nargs="*",
                        help='dense diffusion dataset image mask & caption index')
    parser.add_argument('-s', '--num_inference_steps', type=int, default=50)
    parser.add_argument('--reg_part', type=float, default=0.8)
    parser.add_argument('--sreg', type=float, default=.4)
    parser.add_argument('--creg', type=float, default=1)
    parser.add_argument('--pow_time', type=float, default=3)
    parser.add_argument('-w', '--wo_modulation', action=argparse.BooleanOptionalAction, default=False,
                        help='when True, run inference without dense diffusion attention manipulation')
    parser.add_argument('--debug', type=str)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--key', type=str)
    parser.add_argument('--attention', type=bool, default=True)
    args = parser.parse_args()
    
    
    ## Set hyperparameters
    device= "cuda"
    num_inference_steps = args.num_inference_steps 
    reg_part = args.reg_part if not args.wo_modulation else 0
    sreg = args.sreg
    creg = args.creg
    image_idx = args.key
    config = RunConfig()


    #get human centric dataset 
    input_temp, output_temp = return_temp(image_idx)
    group_dictionary = makeGroupdict_custom(input_temp, output_temp)
    config.output_path = config.output_path /  image_idx
    os.makedirs(config.output_path, exist_ok=True)


    
    ## Load Model
    if args.model == 'LCM':
        pipe = DiffusionPipeline.from_pretrained("SimianLuo/LCM_Dreamshaper_v7")
        pipe.to(device=device, dtype=torch.float16)
        num_inference_steps = num_inference_steps
        lcm_origin_steps = 50
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
        pipe.scheduler.set_timesteps(num_inference_steps=num_inference_steps,
                                     original_inference_steps=lcm_origin_steps,
                                     device=device)
    else:
        pipe = diffusers.StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            safety_checker=None,
            variant="fp16",
            cache_dir='./models/diffusers/'
        ).to(device)
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe.scheduler.set_timesteps(num_inference_steps)

        
    ## Set attn modulation variables
    num_attn_layers = 32
    timesteps = pipe.scheduler.timesteps
    sp_sz = pipe.unet.sample_size
    num_attn_layers = 32
    bsz = args.batch_size
    idx = args.idx

    # timesteps, sp_sz, bsz, mod_forward_orig, pipe = loadLCMNet(config)
    mod_counts = []

    print("=== Experiment Settings ===")
    print("- Model:", args.model, "/ N inference steps:", num_inference_steps, "/ Batch size:", bsz)
    print("- Regulation part:", reg_part, "/ Self attention regulation:", sreg, "/ Cross attention regulation:", creg, "/ Time regulation:", args.pow_time)
    # print("Chosen timesteps:", timesteps)

    
    ## attention modulation function
    def mod_forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None):
        global COUNT, treg, sret, creg, sreg_maps, creg_maps, reg_sizes, text_cond, step_store, attn_stores
        STEP = COUNT // 32

        if COUNT % 32 == 0 and STEP > 0:
            attn_stores.append(step_store)
            step_store = {"down_cross": [], "mid_cross": [], "up_cross": [],
                          "down_self": [],  "mid_self": [],  "up_self": []}

        residual = hidden_states 

        if self.spatial_norm is not None:
            hidden_states = self.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape)
        attention_mask = self.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if self.group_norm is not None:
            hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = self.to_q(hidden_states)

        sa_ = True if encoder_hidden_states is None else False
        encoder_hidden_states = text_cond if encoder_hidden_states is not None else hidden_states
        if self.norm_cross:
            encoder_hidden_states = self.norm_encoder_hidden_states(encoder_hidden_states)

        key = self.to_k(encoder_hidden_states)
        value = self.to_v(encoder_hidden_states)

        query = self.head_to_batch_dim(query)
        key = self.head_to_batch_dim(key)
        value = self.head_to_batch_dim(value)

        if sa_ == False and args.model == 'LCM':
            key =  key[key.size(0)//2:,  ...]
            value = value[value.size(0)//2:,  ...]

        # modulate attention with dense diffusion
        if (COUNT/num_attn_layers < num_inference_steps*reg_part):
            mod_counts.append(COUNT)
            dtype = query.dtype
            if self.upcast_attention:
                query = query.float()
                key = key.float()

            sim = torch.baddbmm(torch.empty(query.shape[0], query.shape[1], key.shape[1], 
                                            dtype=query.dtype, device=query.device),
                                query, key.transpose(-1, -2), beta=0, alpha=self.scale)
            treg = torch.pow(timesteps[COUNT//num_attn_layers]/1000, args.pow_time)
            reg_map = sreg_maps if sa_ else creg_maps
            w_reg = sreg if sa_ else creg

            # manipulate attention
            batch_idx = int(sim.size(0)/2) if args.model != 'LCM' else 0 # why do we have to apply below operations for latter half of sim???
            min_value = sim[batch_idx:].min(-1)[0].unsqueeze(-1)
            max_value = sim[batch_idx:].max(-1)[0].unsqueeze(-1)  
            mask = reg_map[sim.size(1)].repeat(self.heads,1,1)
            size_reg = reg_sizes[sim.size(1)].repeat(self.heads,1,1)

            sim[batch_idx:] += (mask>0)*size_reg*w_reg*treg*(max_value-sim[batch_idx:])
            sim[batch_idx:] -= ~(mask>0)*size_reg*w_reg*treg*(sim[batch_idx:]-min_value)

            attention_probs = sim.softmax(dim=-1)
            attention_probs = attention_probs.to(dtype)
        else: # get original attention
            attention_probs = self.get_attention_scores(query, key, attention_mask)

        COUNT += 1
        if attention_probs.shape[1] <= 32 ** 2: # save attention in each place(up, down, mid) when attention shape is small
            step_store[f"{self.place_in_unet.lower()}_{'self' if sa_ else 'cross'}"].append(attention_probs)

        #################################################        
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = self.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = self.to_out[0](hidden_states)
        # dropout
        hidden_states = self.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if self.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / self.rescale_output_factor

        return hidden_states

    
    ## change call function of attn layers in Unet 
    for _module in pipe.unet.modules():
        n = _module.__class__.__name__
        if 'CrossAttn' in n:
            for place in ['Up', 'Down', 'Mid']:
                if place in n:
                    curr_place = place

        if n == "Attention":
            _module.__class__.__call__ = mod_forward
            _module.place_in_unet = curr_place

    
    ## Main function which generates modulated image
    def generate_index_img(idx):
        global COUNT, treg, sret, creg, sreg_maps, creg_maps, reg_sizes, text_cond, step_store, attn_stores
        
        global_prompt = group_dictionary['global_caption']
        global_W, global_H = group_dictionary['shape']
        global_canvas_H, global_canvas_W = get_canvas(global_H, global_W)
        group_ids = getGroupIds_auto(group_dictionary)

        prompt_wanted = generate_map_human( 
                    global_canvas_H=global_canvas_H, 
                    global_canvas_W=global_canvas_W, 
                    config=config, 
                    global_prompt=global_prompt,
                    group_ids=group_ids,
                    group_dictionary=group_dictionary,
                    image_idx=image_idx
                    )
        
        prompts_idx[idx] = prompt_wanted[0]
        prompts = prompt_wanted
        layout_num = 3
        ## prepare text condition embeddings
        ############
        text_input = pipe.tokenizer(prompts, padding="max_length", return_length=True, return_overflowing_tokens=False, 
                                    max_length=pipe.tokenizer.model_max_length, truncation=True, return_tensors="pt")
        print(f"text_input: {text_input['input_ids'].shape}")
        cond_embeddings = pipe.text_encoder(text_input.input_ids.to(device))[0]

        uncond_input = pipe.tokenizer([""]*bsz, padding="max_length", max_length=pipe.tokenizer.model_max_length,
                                      truncation=True, return_tensors="pt")
        uncond_embeddings = pipe.text_encoder(uncond_input.input_ids.to(device))[0]


        ## set layout image masks
        ############
        layout_img_ = np.asarray(Image.open(str(config.output_path) + "/save_pose.png").resize([sp_sz*8,sp_sz*8]))[..., :3]
        unique, counts = np.unique(np.reshape(layout_img_,(-1,layout_num)), axis=0, return_counts=True)
        sorted_idx = np.argsort(-counts)
        layouts_ = []

        for i in range(len(prompts)-1):
            if (unique[sorted_idx[i]] == np.zeros(layout_num)).all() or (unique[sorted_idx[i]] == np.ones(layout_num) * 255).all():
                layouts_ = [((layout_img_ == unique[sorted_idx[i]]).sum(-1)==layout_num).astype(np.uint8)] + layouts_
            else:
                layouts_.append(((layout_img_ == unique[sorted_idx[i]]).sum(-1)==layout_num).astype(np.uint8))

            save_img = Image.fromarray(layouts_[-1]*255)
            save_img.save(str(config.output_path) + f"/save_pose_{i}.png")

        layouts = [torch.FloatTensor(l).unsqueeze(0).unsqueeze(0).cuda() for l in layouts_]
        print(f" layouts_: {np.asarray(layouts_).shape}")
        layouts = F.interpolate(torch.cat(layouts),(sp_sz,sp_sz),mode='nearest')
        ############
        print('\n'.join(prompts))
        Image.fromarray(np.concatenate([255*_.squeeze().cpu().numpy() for _ in layouts], 1).astype(np.uint8))

        ###########################
        ###### prep for sreg ###### 
        ###########################
        sreg_maps = {}
        reg_sizes = {}
        for r in range(4):
            res = int(sp_sz/np.power(2,r))
            layouts_s = F.interpolate(layouts,(res, res),mode='nearest')
            layouts_s = (layouts_s.view(layouts_s.size(0),1,-1)*layouts_s.view(layouts_s.size(0),-1,1)).sum(0).unsqueeze(0).repeat(bsz,1,1)
            reg_sizes[np.power(res, 2)] = 1-1.*layouts_s.sum(-1, keepdim=True)/(np.power(res, 2))
            sreg_maps[np.power(res, 2)] = layouts_s


        ###########################
        ###### prep for creg ######
        ###########################
        pww_maps = torch.zeros(1, 77, sp_sz, sp_sz).to(device)
        for i in range(1,len(prompts)):
            wlen = text_input['length'][i] - 2
            widx = text_input['input_ids'][i][1:1+wlen]
            for j in range(77):
                diff = wlen - len(text_input['input_ids'][0][j:j+wlen])
                if diff != 0: 
                    widx = widx[:wlen-diff]
                # print(f"input: {text_input['input_ids'][0][j:j+wlen]}, widx: {widx}, diff: {diff}")
                if (text_input['input_ids'][0][j:j+wlen] == widx).sum() == wlen:
                    pww_maps[:,j:j+wlen,:,:] = layouts[i-1:i]
                    cond_embeddings[0][j:j+wlen] = cond_embeddings[i][1:1+wlen]
                    print(prompts[i], i, '-th segment is handled.')
                    break

        creg_maps = {}
        for r in range(4):
            res = int(sp_sz/np.power(2,r))
            layout_c = F.interpolate(pww_maps,(res,res),mode='nearest').view(1,77,-1).permute(0,2,1).repeat(bsz,1,1)
            creg_maps[np.power(res, 2)] = layout_c


        ###########################    
        #### prep for text_emb ####
        ###########################
        text_cond = torch.cat([uncond_embeddings, cond_embeddings[:1].repeat(bsz,1,1)])

        ## generate images
        COUNT = 0
        attn_stores = []
        step_store = {"down_cross": [], "mid_cross": [], "up_cross": [],
                      "down_self": [],  "mid_self": [],  "up_self": []}

        with torch.no_grad():
            latents = torch.randn(bsz,4,sp_sz,sp_sz, generator=torch.Generator().manual_seed(args.seed)).to(device) 
            if args.model == 'LCM':
                with torch.autocast('cuda'):
                    image = pipe(prompts[:1]*bsz, latents=latents,
                                 num_inference_steps=num_inference_steps,
                                 lcm_origin_steps=lcm_origin_steps,
                                 guidance_scale=8.0).images
            else:
                image = pipe(prompts[:1]*bsz, latents=latents).images

        imgs = [ Image.fromarray(np.asarray(image[i])) for i in range(len(image)) ]
        if imgs[0].size[0] > 512:
            imgs = [ x.resize((512,512)) for x in imgs ]
        
        imgs_idx[idx] = imgs
        attentions_idx[idx] = attn_stores
        
        ## save images
        
        # save_path = f'./outputs/{idx:02}/'
        save_path = config.output_path 
        os.makedirs(save_path, exist_ok=True)

        for i, img in enumerate(imgs):
            img_name = f'{args.model}_{args.num_inference_steps}_reg-ratio{reg_part:.1f}_sreg{sreg}_creg{creg}_pow_time{args.pow_time}_{args.wo_modulation*"_woModulation"}_{hash_key}.png'
            if img.size[0] > 512:
                img = img.resize((512,512)) # in order to compare LCM with SD
            img.save(str(save_path)+'/'+img_name)
        
        # return attn_stores

   
    ## Generate images for given indices  
    imgs_idx = dict()
    prompts_idx = dict()
    attentions_idx = dict()
    time_hash = datetime.datetime.now().time()
    hash_key = hashlib.sha1(str(time_hash).encode()).hexdigest()[:6]
    for i in args.idx:
        print(f"=== Generate image for index {i} ===")
        generate_index_img(i)
        if args.attention: 
            cas, sas = get_attention_timesteps(pipe, attentions_idx[i], prompts_idx[i], 24, ['down','up'], 0, 2)
            save_images_into_one(cas, config, f'attention_{args.num_inference_steps}_reg-ratio{reg_part:.1f}_sreg{sreg}_creg{creg}_pow_time{args.pow_time}_{args.wo_modulation*"_woModulation"}_{hash_key}.png')
    # pdb.set_trace()