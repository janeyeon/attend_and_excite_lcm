from _utils import *
import torch
import torch.nn.functional as F
from IPython.display import display
import matplotlib.pyplot as plt
import time
def preprocess_patch(pipe,
                    config,
                    pose_map,
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
                    view_index):
    ########################## Get Views and Preprocess the Maps ##############################
    """
    1. Now we have
    - pose_map (512, 640, 3)
    - group_L_maps (1, 64, 80), inst_obj_L_maps(6, 64, 80, 2)
    - global_prompt, group_prompt_dic, inst_obj_prompt_dic
    - global_canvas_H, global_canvas_W, global_H, global_W

    2. We need to prepare for each patch
    - pose image for ControlNet
    >  pose_map
    - full text prompts for SD & sreg_maps, creg_maps, reg_sizes, text_cond for DenseDiff
    > group_L_maps, inst_obj_L_maps, global_prompt, group_prompt_dic, inst_obj_prompt_dic
    """

    with torch.no_grad():
        # View 좌표들은 latent 기준임
        
        pose_image_for_views = []

        #! 안씀
        inst_obj_small_L_maps = inst_obj_L_maps_small
        group_mix_mask = group_L_maps.sum(0).bool().to(torch.uint8)
        inst_obj_mix_mask = inst_obj_small_L_maps[:,:,:,0].sum(0).bool().to(torch.uint8)
        # display(Image.fromarray(np.concatenate([255*group_mix_mask.squeeze().cpu().numpy()], 1).astype(np.uint8)))#.resize((config.window_size,config.window_size)))
        # display(Image.fromarray(np.concatenate([255*inst_obj_mix_mask.squeeze().cpu().numpy()], 1).astype(np.uint8)))#.resize((config.window_size,config.window_size)))

        full_prompt_for_views = []
        prompts_for_views = []
        text_cond_for_views = []
        
        sreg_maps_for_views = []
        reg_sizes_for_views = []
        creg_maps_for_views = []
        
        exceed_cnt = 0
        # for j, (h_start, h_end, w_start, w_end) in enumerate(views):
        (h_start, h_end, w_start, w_end) = views[view_index]
        ## pose image
        pose_image_for_view = Image.fromarray(pose_map[h_start*8:h_end*8, w_start*8:w_end*8,:])
        # pose_image_for_views.append(pose_image_for_view)
        # display(Image.fromarray(np.concatenate([np.array(pose_image_for_view)], 1).astype(np.uint8)))#.resize((config.window_size,config.window_size)))

        ## obatin keys which contribute on the patch
        group_L_map_for_view = group_L_maps[:, h_start:h_end, w_start:w_end]
        view_H, view_W = group_L_map_for_view.shape[-2:]
        inst_obj_L_map_for_view = inst_obj_L_maps[:,h_start:h_end, w_start:w_end,:] # 1번째 channel에는 0 or group key / 2번째 channel에는 0 or instance key


        ######
        
        group_group_keys_total = group_L_map_for_view.amax(dim=(1,2)).long()
        group_group_ids_total = torch.arange(len(group_group_keys_total))[group_group_keys_total>0].long() #group이 아닌 background는 제외됨
        group_group_keys_total = group_group_keys_total[group_group_keys_total>0] # 해당 view에 (key가) 존재하는 group의 key
        
        inst_obj_group_keys_total = inst_obj_L_map_for_view[:,:,:,0].amax(dim=(1,2)).long() # non group group key TODO
        inst_obj_group_ids_total = torch.arange(len(inst_obj_group_keys_total))[inst_obj_group_keys_total>0].long()
        inst_obj_group_keys_total= inst_obj_group_keys_total[inst_obj_group_keys_total>0] # 해당 view에 (key가) 존재하는 instance/object들의 group key

        inst_obj_inst_obj_keys_total = inst_obj_L_map_for_view[:,:,:,1].amax(dim=(1,2)).long()
        
        inst_obj_inst_obj_ids_total = torch.arange(len(inst_obj_inst_obj_keys_total))[inst_obj_inst_obj_keys_total>0].long()
        inst_obj_inst_obj_keys_total = inst_obj_inst_obj_keys_total[inst_obj_inst_obj_keys_total>0] # 해당 view에 (key가) 존재하는 instance/object들의 instance/object key
        
        ## text prompt, text cond
        group_prompts_total = [group_prompt_dic[k.item()] for k in group_group_keys_total] #group_group_keys_total과 매칭되는 prompt들 -> group 갯수만큼 존재
        inst_obj_prompts_total = [inst_obj_prompt_dic[k.item()] for k in inst_obj_inst_obj_keys_total] # inst_obj_inst_obj_keys_total과 매칭되는 prompt -> instance / object 갯수만큼 존재
        
        full_prompt_for_view_multi = []
        prompts_for_view_multi = []
        text_cond_for_view_multi = []
        
        sreg_maps_for_view_multi = []
        reg_sizes_for_view_multi = []
        creg_maps_for_view_multi = []

        layouts_multi = []

        # Consider token limit
        # Todo: consider layout cum 
        k1_now = 0
        k2_now = 0

        excluded_prompts = []

        condition = True
        
        while condition:
            if len(inst_obj_prompts_total) == 0 :
                condition =  not (k1_now == len(group_prompts_total))
            else:
                condition = not ((k1_now == len(group_prompts_total)) and (k2_now == len(inst_obj_prompts_total)))
            ############################################   here
            # Exclude if num_token exceeds the limits
            #! Global style
            full_prompt_for_view = '((masterpiece, realistic)), sharp focus, high resolution, award–winning photo.' + global_prompt
            # full_prompt_for_view = '8K UHD real photo. ' + global_prompt
            # full_prompt_for_view = 'the Simpsons style. ' + global_prompt
            # full_prompt_for_view = 'Monet style painting.' + global_prompt
            # full_prompt_for_view = 'Picasso style painting.' + global_prompt
            # full_prompt_for_view = '4K, Mysterious, Dark, An realistic photo of harry potter series.' + global_prompt
            # full_prompt_for_view = 'vivid colors, 4k, fantasy, An beautiful fairy tale maincharacters.' + global_prompt
            excluded_prompts = []
            # k1 = len(group_prompts)-1
            # k2 = len(inst_obj_prompts)-1

            # Exception handle for: 해당 view에 instance나 object 없는 경우 방지
            k2 = k2_now
            k1 = k1_now
            # instance first!!!!!!
            for k2 in range(k2_now, len(inst_obj_prompts_total)):
                prompt = inst_obj_prompts_total[k2]
                # print(f"k2: {k2} | prompt: {prompt}")
                prev_full_prompt_for_view = full_prompt_for_view
                full_prompt_for_view += prompt + '. '
                token_count = len(pipe.tokenizer(full_prompt_for_view)['input_ids'])
                if token_count > 77:
                    full_prompt_for_view = prev_full_prompt_for_view
                    # print(f"From Group '{prompt}', will be included in next patch.")
                    # for excluded_prompt in inst_obj_prompts[k2:]:
                    #     print(f"[Warning]' Instance/Object {excluded_prompt}' is excluded because it exceeds the token limits.")
                    #     if excluded_prompt not in excluded_prompts:
                    #         excluded_prompts.append(excluded_prompt)
                        # exceed_cnt += 1
                    # k2 -= 1 # k 번째까지만 가능, Group 나누기
                    
                    break 
            else:
                k2+=1

            for k1 in range(k1_now, len(group_prompts_total)):
                prompt = group_prompts_total[k1]
                # print(f"k1: {k1} | prompt: {prompt}")
                prev_full_prompt_for_view = full_prompt_for_view
                #!##############################################  231107 Remove obj
                # full_prompt_for_view += prompt + '. '
                token_count = len(pipe.tokenizer(full_prompt_for_view)['input_ids']) - 2
                if token_count > 77-2:
                    full_prompt_for_view = prev_full_prompt_for_view
                    # print(f"From Group '{prompt}', will be included in next patch.")
                    # for excluded_prompt in group_prompts[k1:]:
                    #     print(f"[Warning] Group '{excluded_prompt}' is excluded because it exceeds the token limits.")
                    #     if excluded_prompt not in excluded_prompts:
                    #         excluded_prompts.append(excluded_prompt)
                    #     exceed_cnt += 1
                    # k1 -= 1 # k 번째까지만 가능, Group 나누기
                    break
                
            else:
                k1+=1
            
            
            
            group_prompts = group_prompts_total[k1_now:k1]
            inst_obj_prompts = inst_obj_prompts_total[k2_now:k2]
    
            group_group_ids = group_group_ids_total[k1_now:k1]
            group_group_keys = group_group_keys_total[k1_now:k1]
            inst_obj_group_ids = inst_obj_group_ids_total[k2_now:k2]
            inst_obj_group_keys = inst_obj_group_keys_total[k2_now:k2]
            inst_obj_inst_obj_ids = inst_obj_inst_obj_ids_total[k2_now:k2]
            inst_obj_inst_obj_keys = inst_obj_inst_obj_keys_total[k2_now:k2]

            
            k1_prev = k1_now 
            k2_prev = k2_now 
            
            k1_now = k1
            k2_now = k2
            if not len(inst_obj_prompts_total) == 0:
                if (not config.use_multi_turn_patch) or (k1_now == len(group_prompts_total) and k2_now == len(inst_obj_prompts_total)):
                    pass
                else:
                    full_prompt_for_view = full_prompt_for_view.replace(global_prompt, '')
            full_prompt_for_view_multi.append(full_prompt_for_view)
            ## layouts
            
            
            group_layouts = group_L_map_for_view.bool().to(torch.uint8)
            group_layouts = group_layouts[group_group_ids]
            inst_obj_layouts = inst_obj_L_map_for_view[:,:,:,0].bool().to(torch.uint8)
            inst_obj_layouts = inst_obj_layouts[inst_obj_inst_obj_ids]
            # print(f"inst_obj_layouts: {inst_obj_layouts.shape}")
            #!##############################################  231107 Remove obj
            # layouts_orig = torch.cat([inst_obj_layouts, group_layouts],dim=0)
            layouts_orig = inst_obj_layouts
            
            # me = mutually exclusive (라고 짐작 - 하연)
            inst_obj_layouts_me = []
            excluded_inst_obj_layout_ids = []
            if k1_prev==0 and k2_prev==0:
                layout_cum = torch.zeros((config.window_size, config.window_size)).bool().to(torch.uint8)
            # layout_cum = torch.zeros((config.window_size, config.window_size)).bool().to(torch.uint8)
            for k in range(len(inst_obj_layouts)):
                # layout_cum = layouts_orig[:k2].sum(0).bool().to(torch.uint8)
                inst_obj_layout_me = (inst_obj_layouts[k]*(1-layout_cum)).unsqueeze(0)
                layout_cum = (layout_cum + inst_obj_layouts[k]).bool().to(torch.uint8)
                if inst_obj_layout_me.sum().item() == 0:
                    excluded_inst_obj_layout_ids.append(k)
                else:
                    inst_obj_layouts_me.append(inst_obj_layout_me)
            group_layouts_me = []
            excluded_group_layout_ids = []
            #!##############################################  231107 Remove obj
            # for k in range(len(group_layouts)):
            #     # layout_cum = layouts_orig[:len(inst_obj_layouts)+k1].sum(0).bool().to(torch.uint8)
            #     group_layout_me = (group_layouts[k]*(1-layout_cum)).unsqueeze(0)
            #     layout_cum = (layout_cum + group_layouts[k]).bool().to(torch.uint8)
            #     if group_layout_me.sum().item() == 0:
            #         excluded_group_layout_ids.append(k)
            #     else:
            #         group_layouts_me.append(group_layout_me)
    
            if len(group_layouts_me) > 0 and len(inst_obj_layouts_me) > 0:
                layouts_me = torch.cat(inst_obj_layouts_me + group_layouts_me,dim=0)
            elif len(group_layouts_me) == 0 and len(inst_obj_layouts_me) > 0:
                layouts_me = torch.cat(inst_obj_layouts_me,dim=0)
            elif len(group_layouts_me) > 0 and len(inst_obj_layouts_me) == 0:
                layouts_me = torch.cat(group_layouts_me,dim=0)
            else:
                layouts_me = None

            # layouts_me = torch.cat(inst_obj_layouts_me,dim=0)
    
            #!!!!!!!!!!!!!!!!!!!!!
            if config.use_mutually_exclusive_layout:
                layouts = layouts_me
                
                group_prompts_me = []
                group_group_ids_me = []
                group_group_keys_me = []
                # for k in range(len(group_prompts)):
                #     if k not in excluded_group_layout_ids:
                #         group_prompts_me.append(group_prompts[k])  
                #         group_group_ids_me.append(group_group_ids[k])  
                #         group_group_keys_me.append(group_group_keys[k])  
    
                inst_obj_prompts_me = []
                inst_obj_group_ids_me = []
                inst_obj_group_keys_me = []
                inst_obj_inst_obj_ids_me = []
                inst_obj_inst_obj_keys_me = []

                for k in range(len(inst_obj_prompts)):
                    if k not in excluded_inst_obj_layout_ids:
                        inst_obj_prompts_me.append(inst_obj_prompts[k])
                        try:
                            inst_obj_group_ids_me.append(inst_obj_group_ids[k]) 

                            inst_obj_group_keys_me.append(inst_obj_group_keys[k])
                        except:
                            print(f"inst_obj_prompts:{inst_obj_prompts}, K :{k} k2:{k2}, k2_prev:{k2_prev} inst_obj_group_keys : {inst_obj_group_keys}, inst_obj_prompts_total:{inst_obj_prompts_total}, inst_obj_group_keys_total: {inst_obj_group_keys_total}")
                          
                        inst_obj_inst_obj_ids_me.append(inst_obj_inst_obj_ids[k])  
                        inst_obj_inst_obj_keys_me.append(inst_obj_inst_obj_keys[k])  
    
                group_prompts = group_prompts_me
                group_group_ids = group_group_ids_me
                group_group_keys = group_group_keys_me
    
                inst_obj_prompts = inst_obj_prompts_me
                inst_obj_group_ids = inst_obj_group_ids_me
                inst_obj_group_keys = inst_obj_group_keys_me
                inst_obj_inst_obj_ids = inst_obj_inst_obj_ids_me
                inst_obj_inst_obj_keys = inst_obj_inst_obj_keys_me
            else:
                layouts = torch.cat([inst_obj_layouts, group_layouts],dim=0)
    
            if layouts == None:
                if (config.use_multi_turn_patch) and (k1_prev != 0 or k2_prev != 0):
                    continue
                #!!!!!!!!!!!!!!!!!!!!!
                else:
                    layout_bg = torch.ones_like(group_L_map_for_view.bool().to(torch.uint8)[0])
                    layouts = layout_bg[None,None,:,:]
            else:
                #!!!!!!!!!!!!!!!!!!!!!
                if (not config.use_multi_turn_patch) or (k1_now == len(group_prompts_total) and k2_now == len(inst_obj_prompts_total)):
                    layouts_multi.append(layouts)
                    layout_bg = 1-layout_cum.unsqueeze(0)
                    layouts = torch.cat([layout_bg, layouts],dim=0)[:,None,:,:]
                else:
                    layouts_multi.append(layouts)
                    layouts = layouts[:,None,:,:]
            # i  = 0
            # for layout in layouts:
            #     image = Image.fromarray(np.concatenate([255*layout.squeeze().cpu().numpy()], 1).astype(np.uint8)).resize((config.window_size,config.window_size))
            #     plt.xlabel(f"layout: {i}")
            #     plt.imshow(image)
            #     plt.show()
            #     i += 1
            # time.sleep(1000000000000000000000000000000000000000)
                # display(Image.fromarray(np.concatenate([255*layout.squeeze().cpu().numpy()], 1).astype(np.uint8)).resize((config.window_size,config.window_size)))
            # display(Image.fromarray(np.concatenate([255*_.squeeze().cpu().numpy() for _ in layouts], 1).astype(np.uint8)))
            #!!!!!!!!!!!!!!!!!!!!!
            if (not config.use_multi_turn_patch) or (k1_now == len(group_prompts_total) and k2_now == len(inst_obj_prompts_total)):
                # prompts_for_view = [full_prompt_for_view] + [global_prompt] + inst_obj_prompts + group_prompts
                #!##############################################  231107 Remove obj
                prompts_for_view = [full_prompt_for_view] + [global_prompt] + inst_obj_prompts 
                # prompts_for_view = [full_prompt_for_view] + [global_prompt] + inst_obj_prompts + [global_prompt]
                # prompts_for_text_input = [full_prompt_for_view] 
            else:
                #!##############################################  231107 Remove obj
                # prompts_for_view = [full_prompt_for_view]  + inst_obj_prompts + group_prompts
                prompts_for_view = [full_prompt_for_view]  + inst_obj_prompts 

                # prompts_for_text_input = [full_prompt_for_view]
            prompts_for_view_multi.append(prompts_for_view)
            # for p in prompts_for_view:
            #     print(p)
            
            
            # Text embedding
            
            if config.use_consistent_text_embedding:
                consistent_group_prompts = [global_prompt + '. ' + group_prompt for group_prompt in group_prompts] 
                consistent_inst_obj_prompts =  [global_prompt + '. ' + group_prompt_dic[group_key.item()] +'. '+ inst_obj_prompt 
                                                for inst_obj_prompt, group_key in zip(inst_obj_prompts, inst_obj_group_keys)] 
                
                consistent_prompts_for_view = [full_prompt_for_view] + [global_prompt]  + consistent_inst_obj_prompts + consistent_group_prompts
                consistent_text_input = pipe.tokenizer(consistent_prompts_for_view, padding="max_length", return_length=True, return_overflowing_tokens=False, 
                                        max_length=pipe.tokenizer.model_max_length, truncation=True, return_tensors="pt")
                consistent_cond_embeddings = pipe.text_encoder(consistent_text_input.input_ids.to(config.device))[0]
                
                text_input = pipe.tokenizer(prompts_for_view, padding="max_length", return_length=True, return_overflowing_tokens=False, 
                                        max_length=pipe.tokenizer.model_max_length, truncation=True, return_tensors="pt")
                cond_embeddings = pipe.text_encoder(text_input.input_ids.to(config.device))[0]


                # text_input = pipe.tokenizer(prompts_for_text_input, padding="max_length", return_length=True, return_overflowing_tokens=False, 
                #                         max_length=pipe.tokenizer.model_max_length, truncation=True, return_tensors="pt")
                # cond_embeddings = pipe.text_encoder(text_input.input_ids.to(config.device))[0]
                
                for k in range(2,len(prompts_for_view)):
                    wlen = text_input['length'][k] - 2
                    consistent_wlen = consistent_text_input['length'][k] - 2
                    cond_embeddings[k][1:1+wlen] = consistent_cond_embeddings[k][consistent_wlen-wlen+1:consistent_wlen+1]
            #!!!!!!!!!!!!!!!!!!!!!
            else:
                text_input = pipe.tokenizer(prompts_for_view, padding="max_length", return_length=True, return_overflowing_tokens=False, 
                                        max_length=pipe.tokenizer.model_max_length, truncation=True, return_tensors="pt")
                cond_embeddings = pipe.text_encoder(text_input.input_ids.to(config.device))[0]
                # text_input = pipe.tokenizer(prompts_for_text_input, padding="max_length", return_length=True, return_overflowing_tokens=False, 
                #                         max_length=pipe.tokenizer.model_max_length, truncation=True, return_tensors="pt")
                # cond_embeddings = pipe.text_encoder(text_input.input_ids.to(config.device))[0]
                
    
            
            uncond_input = pipe.tokenizer([""]*bsz, padding="max_length", max_length=pipe.tokenizer.model_max_length,
                                        truncation=True, return_tensors="pt")
            uncond_embeddings = pipe.text_encoder(uncond_input.input_ids.to(config.device))[0]
            
    
            ## sreg_maps, reg_sizes, creg_maps from layouts
            # sreg_maps, reg_sizes
            sreg_maps = {}
            reg_sizes = {}
            print(f"layouts: {layouts.shape}")
            for r in range(4):
                res1, res2 = int(view_H/np.power(2,r)), int(view_W/np.power(2,r))
                layouts_s = F.interpolate(layouts,(res1, res2),mode='nearest').to(config.device)
                layouts_s = (layouts_s.view(layouts_s.size(0),1,-1)*layouts_s.view(layouts_s.size(0),-1,1)).sum(0).unsqueeze(0).repeat(1,1,1)
                reg_sizes[res1*res2] = 1-1.*layouts_s.sum(-1, keepdim=True)/(res1*res2)
                sreg_maps[res1*res2] = layouts_s
            

            
            
            sreg_maps_for_view_multi.append(sreg_maps)
            reg_sizes_for_view_multi.append(reg_sizes)

            
            
            # creg_maps
            # creg_maps
            pww_maps = torch.zeros(1, 77, view_H, view_W).to(config.device)
            
            # print(f"total: {text_input['input_ids'][0]}")
            # print(f"prompts_for_view: {prompts_for_view}")
            for k in range(1,len(prompts_for_view)): # global, instance, group
            # k = 2 # total, global, instance1, instance2,  group
                if k ==1 : #global
                    wlen = text_input['length'][k] - 3 # token 길이
                    widx = text_input['input_ids'][k][1:wlen+1] # 단어가 뭔지 나타내는 key들, 앞뒤토큰 버림 = wlen만큼의 길이
                else: 
                    wlen = text_input['length'][k] - 2 # token 길이
                    widx = text_input['input_ids'][k][1:wlen+1] # 단어가 뭔지 나타내는 key들, 앞뒤토큰 버림 = wlen만큼의 길이
                # print(f"wlen: {wlen}, widx: {widx}")
                # tokenizer에서는 token간의 attention이 없음
                # HI and HI I'm hayeon 이 두개가 동일함
                # text_input['input_ids'][0] <- total text를 의미
                # text_input['input_ids'][i] <- 각 word들의 text를 의미
                for l in range(77):
                    # full text에서 wlen만큼 sliding을 하다가 -> widx와 matching이 되는게 생기면 -> sum 했을때 11111 -> 그 갯수가 wlen-1이 됨
                    # if (text_input['input_ids'][0][l:l+wlen-1] == widx[:-1]).sum() == wlen-1:
                    if (text_input['input_ids'][0][l:l+wlen] == widx).sum() == wlen:
                        # if k < len(prompts_for_view)-1:
                        pww_maps[0,l:l+wlen,:,:] = layouts[k-1]# full prompt에서 그 위치에 해당하는 layout을 넣어줘라
                        cond_embeddings[0][l:l+wlen] = cond_embeddings[k][1:1+wlen] # 그 단어의 강조, 원래의 text embedding넣어주기  (더 살도록)
                        break
            
            
            creg_maps = {}
            for r in range(4):
                res1, res2 = int(view_H/np.power(2,r)), int(view_W/np.power(2,r))
                layout_c = F.interpolate(pww_maps,(res1,res2),mode='nearest').view(1,77,-1).permute(0,2,1).repeat(1,1,1)
                # layout_c = F.interpolate(pww_maps,(res1,res2),mode='bilinear').view(1,77,-1).permute(0,2,1).repeat(1,1,1)
                creg_maps[res1*res2] = layout_c
            creg_maps_for_view_multi.append(creg_maps)
            
        
            
            ## refocused text cond
            text_cond_for_view = torch.cat([uncond_embeddings, cond_embeddings[:1].repeat(bsz,1,1)])
            text_cond_for_view_multi.append(text_cond_for_view)
            
            if not config.use_multi_turn_patch:
                break
                
            
        # print(f'exceed_cnt: {exceed_cnt}')
        # print(f'len(views): {len(views)}')
        # print(f"full_prompt_for_view_multi: {full_prompt_for_view_multi}")

        return full_prompt_for_view_multi,\
                prompts_for_view_multi,\
                text_cond_for_view_multi,\
                sreg_maps_for_view_multi,\
                reg_sizes_for_view_multi,\
                creg_maps_for_view_multi,\
                excluded_prompts,\
                pose_image_for_view