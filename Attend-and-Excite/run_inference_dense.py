from typing import List, Dict, Optional
import torch

import sys 
sys.path.append(".")
sys.path.append("..")

from pipeline_attend_and_excite import AttendAndExcitePipeline
from config import RunConfig
from run import run_on_prompt, get_indices_to_alter
from utils import vis_utils
from utils import ptp_utils
from utils.ptp_utils import AttentionStore
from config import RunConfig


import transformers
from transformers import CLIPTextModel, CLIPTokenizer
import torch.nn.functional as F
from torchvision import transforms
from tqdm.auto import tqdm
from PIL import Image
import argparse
import pickle
import numpy as np
import datetime
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from diffusers import DDIMScheduler, LCMScheduler
import hashlib
import gc
if __name__ == '__main__':
    command = "--model LCM --batch_size 0 -s 10 --reg_part 0.3 --idx 5 ".split()

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='LCM', choices=['LCM', 'SD'])
    parser.add_argument('--batch_size', type=int, default=0)
    parser.add_argument('--idx', type=int, default=[1], nargs="*",
                        help='dense diffusion dataset image mask & caption index')
    parser.add_argument('-s', '--num_inference_steps', type=int, default=50)
    parser.add_argument('--reg_part', type=float, default=.3)
    parser.add_argument('--sreg', type=float, default=.3)
    parser.add_argument('--creg', type=float, default=1)
    parser.add_argument('--pow_time', type=float, default=5)
    parser.add_argument('-w', '--wo_modulation', action='store_true', default=False,
                        help='when True, run inference without dense diffusion attention manipulation')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--debug', type=str)
    args = parser.parse_args(command)

    NUM_DIFFUSION_STEPS = 4
    GUIDANCE_SCALE = 7.5
    MAX_NUM_WORDS = 77
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    safety_checker = StableDiffusionSafetyChecker.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="safety_checker")
    pipe = AttendAndExcitePipeline.from_pretrained("SimianLuo/LCM_Dreamshaper_v7",safety_checker=safety_checker, dtype=torch.float32).to(device)
    tokenizer = pipe.tokenizer
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    pipe.scheduler.set_timesteps(num_inference_steps=4,
                                     original_inference_steps=50, device = device)
    reg_part = args.reg_part if not args.wo_modulation else 0
    num_inference_steps = args.num_inference_steps 

    sreg = args.sreg
    creg = args.creg
    #attn modulation variables
    num_attn_layers = 32
    timesteps = pipe.scheduler.timesteps
    sp_sz = pipe.unet.sample_size
    bsz = 1
    idx = [1]
    mod_counts = []

#     def get_attention_maps_per_token(self):
#         global COUNT, treg, sret, creg, sreg_maps, creg_maps, reg_sizes, text_cond, step_store, attn_stores

#         from_where=("up", "down", "mid")
#         out = []
#         # res : 사이즈 다를 수도 
#         res = 24
#         # num_pixels = res ** 2
#         num_pixels = 144
#         # step_store[f"{location}_cross"] : 12 x 12 x 77

#         for location in from_where:
#             for item in step_store[f"{location}_cross"]:
#                 if item.shape[1] != num_pixels:
#                     out.append(item.reshape(-1, res, res, item.shape[-1]))
#                     # out.append(cross_maps)
#         out = torch.cat(out, dim=0)
#         out = out.sum(0) / out.shape[0]
#         attention_maps = out

#         attention_maps_list = self._get_attention_maps_list(
#             attention_maps=attention_maps
#         )

#         return attention_maps_list

#     def _aggregate_max_ftn(self, attention_store,
#                         indices_to_alter,
#                         attention_res: int = 16,
#                         smooth_attentions: bool = False,
#                         sigma: float = 0.5,
#                         kernel_size: int = 3,
#                         normalize_eot: bool = False):
#         global COUNT, treg, sret, creg, sreg_maps, creg_maps, reg_sizes, text_cond, step_store, attn_stores

        
#         from_where=("up", "down", "mid")
#         out = []
#         # res : 사이즈 다를 수도 
#         res = 24
#         # num_pixels = res ** 2
#         num_pixels = 144
#         # step_store[f"{location}_cross"] : 12 x 12 x 77

#         for location in from_where:
#             for item in step_store[f"{location}_cross"]:
#                 print(f"item.shape: {item.shape} in {location}")
#                 if item.shape[1] != num_pixels:
#                     out.append(item.reshape(-1, res, res, item.shape[-1]))
#                     # out.append(cross_maps)
#         out = torch.cat(out, dim=0)
#         out = out.sum(0) / out.shape[0]
#         attention_maps = out


#         max_attention_per_index = self._compute_max_attention_per_index(
#             attention_maps=attention_maps,
#             indices_to_alter=indices_to_alter,
#             smooth_attentions=smooth_attentions,
#             sigma=sigma,
#             kernel_size=kernel_size,
#             normalize_eot=normalize_eot)

#         print(f"max_attention_per_index: {max_attention_per_index}")


#         return max_attention_per_index



    def mod_forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None):
        global COUNT, treg, sret, creg, sreg_maps, creg_maps, reg_sizes, text_cond
        # , step_store, attn_stores
        # attn_stores = []
        STEP = COUNT // 32
        # if COUNT % 32 == 0 and STEP > 0:
        #     attn_stores.append(step_store)
        #     step_store = {"down_cross": [], "mid_cross": [], "up_cross": [],
        #                   "down_self": [],  "mid_self": [],  "up_self": []}

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
        torch.cuda.empty_cache()

        query = self.head_to_batch_dim(query)
        key = self.head_to_batch_dim(key)
        value = self.head_to_batch_dim(value)
        if sa_ == False and args.model == 'LCM':
            key =  key[key.size(0)//2:,  ...]
            value = value[value.size(0)//2:,  ...]

        # modulate attention with dense diffusion
        if (- < num_inference_steps*reg_part):
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

        del sim
        del size_reg
        del mask
        torch.cuda.empty_cache()

        COUNT += 1
        # if attention_probs.shape[1] <= 32 ** 2: # save attention in each place(up, down, mid) when attention shape is small
        #     step_store[f"{self.place_in_unet.lower()}_{'self' if sa_ else 'cross'}"].append(attention_probs)

        #################################################        
        hidden_states = torch.bmm(attention_probs, value)
        del attention_probs
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
    # pipe.__class__._aggregate_and_get_attention_maps_per_token = get_attention_maps_per_token
    # pipe.__class__._aggregate_and_get_max_attention_per_token = _aggregate_max_ftn
    
    for _module in pipe.unet.modules():
        n = _module.__class__.__name__
        # if 'CrossAttn' in n:
        #     for place in ['Up', 'Down', 'Mid']:
        #         if place in n:
        #             curr_place = place

        if n == "Attention":
            _module.__class__.__call__ = mod_forward
            # _module.place_in_unet = curr_place

    with open('/home/jeeit17/s1/2_GENAI_CONS/attention_modulation_lcm/dataset/valset.pkl', 'rb') as f:
        dataset = pickle.load(f)
        layout_img_root = '/home/jeeit17/s1/2_GENAI_CONS/attention_modulation_lcm/dataset/valset_layout/'

#     def run_and_display(prompts: List[str],
#                         latents : None,
#                         controller: AttentionStore,
#                         indices_to_alter: List[int],
#                         generator: torch.Generator,
#                         run_standard_sd: bool = False,
#                         scale_factor: int = 20,
#                         thresholds: Dict[int, float] = {0:0.05, 10: 0.5, 20: 0.8},
#                         max_iter_to_alter: int = 25,
#                         display_output: bool = False,
#                         sd_2_1: bool = False):
#         config = RunConfig(prompt=prompts[0],
#                            run_standard_sd=run_standard_sd,
#                            scale_factor=scale_factor,
#                            thresholds=thresholds,
#                            max_iter_to_alter=max_iter_to_alter,
#                            sd_2_1=sd_2_1)

#         image = run_on_prompt(model=pipe,
#                               latents = latents,
#                               prompt=prompts,
#                               controller=controller,
#                               token_indices=indices_to_alter,
#                               seed=generator,
#                               config=config)
#         torch.cuda.empty_cache()
#         return image

    def generate_index_img(idx):
        global COUNT, treg, sret, creg, sreg_maps, creg_maps, reg_sizes, text_cond
        # , step_store, attn_stores

        layout_img_path = layout_img_root+str(idx)+'.png'
        prompts = [dataset[idx]['textual_condition']] + dataset[idx]['segment_descriptions']
        prompts_idx[idx] = prompts[0]
        ## prepare text condition embeddings
        ############
        text_input = pipe.tokenizer(prompts, padding="max_length", return_length=True, return_overflowing_tokens=False, 
                                    max_length=pipe.tokenizer.model_max_length, truncation=True, return_tensors="pt")
        cond_embeddings = pipe.text_encoder(text_input.input_ids.to(device))[0]

        uncond_input = pipe.tokenizer([""]*bsz, padding="max_length", max_length=pipe.tokenizer.model_max_length,
                                      truncation=True, return_tensors="pt")
        uncond_embeddings = pipe.text_encoder(uncond_input.input_ids.to(device))[0]

        for i in range(1,len(prompts)):
            wlen = text_input['length'][i] - 2
            widx = text_input['input_ids'][i][1:1+wlen]
            for j in range(77):
                if (text_input['input_ids'][0][j:j+wlen] == widx).sum() == wlen:
                    break

        ## set layout image masks
        ############
        layout_img_ = np.asarray(Image.open(layout_img_path).resize([sp_sz*8,sp_sz*8]))[:,:,:3]
        unique, counts = np.unique(np.reshape(layout_img_,(-1,3)), axis=0, return_counts=True)
        sorted_idx = np.argsort(-counts)

        layouts_ = []

        for i in range(len(prompts)-1):
            if (unique[sorted_idx[i]] == [0, 0, 0]).all() or (unique[sorted_idx[i]] == [255, 255, 255]).all():
                layouts_ = [((layout_img_ == unique[sorted_idx[i]]).sum(-1)==3).astype(np.uint8)] + layouts_
            else:
                layouts_.append(((layout_img_ == unique[sorted_idx[i]]).sum(-1)==3).astype(np.uint8))

        layouts = [torch.FloatTensor(l).unsqueeze(0).unsqueeze(0).cuda() for l in layouts_]
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
        # attn_stores = []
        # step_store = {"down_cross": [], "mid_cross": [], "up_cross": [],
        #               "down_self": [],  "mid_self": [],  "up_self": []}
        generator = torch.Generator().manual_seed(args.seed)
        latents = torch.randn(bsz,4,sp_sz,sp_sz, generator=generator).to(device) 
        token_indices = get_indices_to_alter(pipe, prompts[0])
        prompts = [prompts[0]]
        controller = AttentionStore()
        config = RunConfig(prompt=prompts,
                       run_standard_sd=False,
                       scale_factor=20,
                       thresholds={0:0.05, 10: 0.5, 20: 0.8},
                       max_iter_to_alter=25,
                       sd_2_1=False)
        g = torch.Generator('cuda').manual_seed(40)
        image = run_on_prompt(prompt=prompts,
                              model=pipe,
                              latents = latents,
                              controller=controller,
                              token_indices=token_indices,
                              seed=g,
                              config=config)


        imgs = [Image.fromarray(np.asarray(image[i])) for i in range(len(image))]
        if imgs[0].size[0] > 512:
            imgs = [ x.resize((512,512)) for x in imgs ]


        save_path = './'
        os.makedirs(save_path, exist_ok=True)

        for i, img in enumerate(imgs):
            img_n = f'{args.model}_' + img_name
            if img.size[0] > 512:
                img = img.resize((512,512)) # in order to compare LCM with SD
            img.save(str(save_path)+'/'+img_n)
            
          ## Generate images for given indices  
   
    imgs_idx = dict()
    prompts_idx = dict()
    attentions_idx = dict()
    time_hash = datetime.datetime.now().time()
    hash_key = hashlib.sha1(str(time_hash).encode()).hexdigest()[:6]
    for i in args.idx:
        print(f"=== Generate image for index {i} ===")
        img_name= f'{hash_key}_seed{args.seed}_{args.num_inference_steps}_reg-ratio{reg_part:.1f}.png'
        generate_index_img(i)
        if args.attention: 
            cas, sas = get_attention_timesteps(pipe, attentions_idx[i], prompts_idx[i], 24, ['down','up'], 0, 2)
            print(f"saved at {img_name}")
            save_images_into_one(cas, config, 'attention_' + img_name)
            torch.cuda.empty_cache()
    # pdb.set_trace()