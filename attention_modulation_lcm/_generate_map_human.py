import torch
import numpy as np
from PIL import Image
from controlnet_aux.open_pose import PoseResult, BodyResult, Keypoint
from controlnet_aux.util import HWC3
from drawer import draw_rectangle, DashedImageDraw
import torch.nn as nn
import torch.nn.functional as F
from _utils import *
import random

def generate_map_human(
        global_canvas_H, 
        global_canvas_W, 
        config, 
        global_prompt,
        group_ids,
        group_dictionary,
        image_idx
    ):
    
    group_maps = []
    inst_maps = []
    obj_maps= []

    group_prompt_dic = {0:''}
    inst_obj_prompt_dic = {0:''}
    global_prompt = global_prompt.replace('.',' ')
    prompt_wanted = [global_prompt]
    whole_prompt = global_prompt

    poses = []
    pose_masks = []
    nose2neck_lengths = []
    length_order = []

    group_boxes = []
    inst_boxes = []
    obj_boxes= []

    prev_inst_points = 0
    prev_obj_points = 0

    # h_factor  = config.default_H / global_canvas_H
    # w_factor  = config.default_W / global_canvas_W
    h_factor  = 1
    w_factor  = 1


    with torch.no_grad():
        for i, group_idx in enumerate(group_ids):
            print(f'group_idx: {group_idx}')
            chosen_group = group_dictionary[str(group_idx)]
            if group_idx == 'non_group_person':
                group_prompt_dic[i+1] = ''
                group_map = torch.zeros(1,config.default_H, config.default_W)
                group_maps.append(group_map)
                group_boxes.append((0,0,0,0))
                
            else:
                group_y1 = int(chosen_group['group_bbox']['y']*config.zoom_ratio*h_factor)
                group_x1 = int(chosen_group['group_bbox']['x']*config.zoom_ratio*w_factor)
                
                group_H = int(chosen_group['group_bbox']['height']*config.zoom_ratio*h_factor)
                group_W = int(chosen_group['group_bbox']['width']*config.zoom_ratio*w_factor)
        
                group_y2 = group_y1 + group_H
                group_x2 = group_x1 + group_W

                group_map = torch.zeros(1,config.default_H, config.default_W)
                group_map[0,group_y1:group_y2, group_x1:group_x2] = i+1 # group key
                group_maps.append(group_map)
        
                group_prompt = chosen_group['group_caption'][0]#.replace('people', '').replace('person', '').replace('man', '').replace('woman', '').replace('men', '').replace('women', '').replace('boy', '').replace('girl', '').replace('player', '')
                group_prompt = group_prompt.split('.')[0]#+'.'
                
                #! Change here!!!!!!!!!!!!!!
                group_prompt_dic[i+1] = group_prompt
                # group_prompt_dic[i+1] = group_dictionary['global_caption']
                group_boxes.append((group_x1,group_y1,group_x2,group_y2))
                
                
                
        
            ## Human keylayout
            inst_ids = chosen_group['instance'].keys()
            print(f'instance_ids: {inst_ids}')
            
            for j, inst_idx in enumerate(inst_ids):
                # prompt
                if 'caption' not in chosen_group['instance'][inst_idx].keys():
                    continue
                else:
                    inst_prompt = chosen_group['instance'][inst_idx]['caption']


                inst_prompt = inst_prompt.replace('.','')
                article = "" if len(inst_prompt) > 1 else "a "

                inst_obj_prompt_dic[j+1+prev_inst_points] = article + inst_prompt #!!!!!!!!!!!!!!!!!!!!!!!!
                prompt_wanted.append(article + inst_prompt)
                whole_prompt += ' ' + article + inst_prompt

                if group_idx == 'non_group_person':
                    print('non_group!!!!!!!!')
                    print(inst_prompt.replace('.',''))
                    print('non_group!!!!!!!!')
                    
                # map
                x1 = int(chosen_group['instance'][inst_idx]['x']*config.zoom_ratio*w_factor)
                y1 = int(chosen_group['instance'][inst_idx]['y']*config.zoom_ratio*h_factor)
                h = int(chosen_group['instance'][inst_idx]['height']*config.zoom_ratio*h_factor)
                w = int(chosen_group['instance'][inst_idx]['width']*config.zoom_ratio*w_factor)
                x2 = x1+w
                y2 = y1+h

                inst_map = torch.zeros(1, config.default_H, config.default_W, 2)
                inst_map[0,y1:y2, x1:x2,0] = i+1 # group key
                inst_map[0,y1:y2, x1:x2,1] = j+prev_inst_points+1 # instance key
                inst_maps.append(inst_map)
                inst_boxes.append((x1,y1,x2,y2))

                # pose
                key_data = chosen_group['instance'][inst_idx]['keypoint']
                pose = PoseResult(body=BodyResult(
                                    keypoints=[
                                        # Keypoint(x=key_data[key_idx * 3]*config.zoom_ratio/config.default_W,
                                        #         y=key_data[key_idx * 3+1]*config.zoom_ratio/config.default_H,
                                        #         score=key_data[key_idx * 3 + 2],
                                        #         id=key_idx
                                        #         ) for key_idx in range(18)],
                                        Keypoint(x=key_data[key_idx * 3]*config.zoom_ratio*w_factor/config.default_W,
                                                y=key_data[key_idx * 3+1]*config.zoom_ratio*w_factor/config.default_W,
                                                score=key_data[key_idx * 3 + 2],
                                                id=key_idx
                                                ) for key_idx in range(18)],
                                    total_score=0,
                                    total_parts=0
                                    )
                                ,left_hand=None, right_hand=None, face=None)
                poses.append(pose)
                
                # pose mask
                pose_mask = np.zeros((config.default_H, config.default_W, 3), dtype=np.int32)
                pose_mask, nose2neck_length = draw_bodyline(pose_mask, pose.body.keypoints, stickwidth_alpha=1., return_nose2neck_length=True)
                pose_mask = pose_mask.sum(-1)
                pose_mask[pose_mask>0]=j+1
                pose_mask = torch.from_numpy(pose_mask).unsqueeze(0)
                pose_masks.append(pose_mask)
                nose2neck_lengths.append(nose2neck_length)

            prev_inst_points += len(inst_ids) #!!!!!!!!!!!!!!!!!!!!!!!!

            # ordering 
            # ### Human ordering according to nose2neck lengths
            #! Why prompt is not oredered?

            length_order = np.argsort(nose2neck_lengths)[::-1]
            nose2neck_lengths = [nose2neck_lengths[i] for i in length_order]
            inst_maps = [inst_maps[i] for i in length_order]
            inst_boxes = [inst_boxes[i] for i in length_order]
            poses = [poses[i] for i in length_order]
            pose_masks = [pose_masks[i] for i in length_order]
                
            #!##############################################  231107 Remove obj
            ## Object layout
            # -------------------------------------------------------------------------------------
            obj_ids = chosen_group['obj'].keys()
            print(f'obj_ids: {obj_ids}')
            
            for j, obj_idx in enumerate(obj_ids):
                # prompt
                if 'caption' not in chosen_group['obj'][obj_idx].keys():
                    continue
                else:
                    obj_prompt = chosen_group['obj'][obj_idx]['caption']
                article = "" if len(obj_prompt) > 1 else "a "
                inst_obj_prompt_dic[j+1001+prev_obj_points] = article +  obj_prompt #!!!!!!!!!!!!!!!!!!!!!!!!
                if group_idx == 'non_group_person':
                    print('obj_prompt!!!!!!!!')
                    print(obj_prompt.replace('.',''))
                    print('obj_prompt!!!!!!!!')
                # map
                x1 = int(chosen_group['obj'][obj_idx]['x']*config.zoom_ratio*w_factor)
                y1 = int(chosen_group['obj'][obj_idx]['y']*config.zoom_ratio*h_factor)
                h = int(chosen_group['obj'][obj_idx]['height']*config.zoom_ratio*h_factor)
                w = int(chosen_group['obj'][obj_idx]['width']*config.zoom_ratio*w_factor)
                x2 = x1+w
                y2 = y1+h

                obj_map = torch.zeros(1,config.default_H, config.default_W, 2)
                obj_map[0,y1:y2, x1:x2,0] = i+1
                obj_map[0,y1:y2, x1:x2,1] = j+prev_obj_points+1001
                obj_maps.append(obj_map)
                obj_boxes.append((x1,y1,x2,y2))
            prev_obj_points += len(obj_ids) #!!!!!!!!!!!!!!!!!!!!!!!!
        
        #-------------------------------------------------------------------------------------
        ### for instance 
        inst_maps = torch.cat(inst_maps, dim=0)
        inst_sizes = inst_maps[:,:,:,0].clone().bool().to(torch.uint8).sum(-1).sum(-1)


        ### Total pose map
        pose_map = np.zeros((config.default_H, config.default_W, 3), dtype=np.uint8)
        for pose in poses:
            pose_map = draw_bodypose(pose_map, pose.body.keypoints, stickwidth=2)    
        pose_map = HWC3(pose_map)
        pose_map_pil = Image.fromarray(pose_map)
        pose_masks = torch.cat(pose_masks, dim=0)
        #!!!!!!!!!!!!!!!!!!!!!!!
        H, W = pose_masks.shape[1], pose_masks.shape[2]
        save_img = np.zeros((H, W, 3), dtype=np.uint8)
        for idx, pose_mask in enumerate(pose_masks):
            save_img[pose_mask>0] = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
        
        save_img = save_img.astype(np.uint8)
        save_pose = Image.fromarray(save_img*255)
        save_pose.save(config.output_path / f'save_pose.png')

        ### Group ordering according to the size
        group_maps = torch.cat(group_maps,dim=0).unsqueeze(1) 
        group_sizes = group_maps[:,0,:,:].clone().bool().to(torch.uint8).sum(-1).sum(-1)
        group_sizes_ids = group_sizes.argsort().view(-1)
        
        group_maps =  torch.cat([group_maps[i].unsqueeze(0) for i in group_sizes_ids], dim=0)
        group_boxes =  [group_boxes[i] for i in group_sizes_ids]

        pose_masks_binary = pose_masks.clone().bool().to(torch.uint8).unsqueeze(3)
        pose_masked_inst_maps = pose_masks_binary * inst_maps


        group_maps_group_keys = group_maps.amax(dim=(2,3)).unsqueeze(2).unsqueeze(3).long()
        group_L_maps = F.interpolate(group_maps,(config.default_H//8, config.default_W//8), mode='nearest').squeeze(1).long()

        for j, k in enumerate(group_maps_group_keys):
            map = group_L_maps[j].clone() 
            map[map>0] = k
            group_L_maps[j] = map

        group_mix_mask = group_L_maps.sum(0).bool().to(torch.uint8)



        #-------------------------------------------------------------------------------------
        ### for object 
        try:
            small_obj_ids, big_obj_ids = [], []
            obj_maps = torch.cat(obj_maps, dim=0)
                

        except:
            obj_maps = torch.zeros(1,config.default_H, config.default_W, 2)
            obj_sizes = torch.zeros([1])
            # inst_obj_maps = torch.zeros(1,config.default_H, config.default_W, 2)
            # inst_obj_boxes = []
            # inst_obj_maps_small = torch.zeros(1,config.default_H, config.default_W, 2)
            # inst_obj_maps_group_keys = torch.zeros([1]).long()
            # inst_obj_maps_instance_keys = torch.zeros([1]).long()
            # inst_obj_L_maps = F.interpolate(inst_obj_maps.permute(0,3,1,2),(config.default_H//8, config.default_W//8), mode='nearest').permute(0,2,3,1).long()
            # inst_obj_L_maps_small = F.interpolate(inst_obj_maps.permute(0,3,1,2),(config.default_H//8, config.default_W//8), mode='nearest').permute(0,2,3,1).long()
            # inst_obj_small_L_maps = inst_obj_L_maps_small
            # inst_obj_mix_mask = inst_obj_small_L_maps[:,:,:,0].sum(0).bool().to(torch.uint8)
            # TODO
        # else:
        obj_sizes = obj_maps[:,:,:,0].clone().bool().to(torch.uint8).sum(-1).sum(-1)
        obj_size_ids = obj_sizes.argsort().view(-1)
        for obj_idx in obj_size_ids:
            if (obj_sizes[obj_idx].item()<inst_sizes[0].item()):
                small_obj_ids.append(obj_idx.item())
            else:
                big_obj_ids.append(obj_idx.item())

        ### Instance object maps
        if config.use_object_size_ordering:
            inst_obj_maps = torch.cat([obj_maps[small_obj_ids].view(-1,config.default_H, config.default_W,2),
                                    pose_masked_inst_maps, ########### or inst_maps
                                    obj_maps[big_obj_ids].view(-1,config.default_H, config.default_W,2)], dim=0)
        
            inst_obj_boxes = np.array(obj_boxes)[small_obj_ids].tolist() + inst_boxes + np.array(obj_boxes)[big_obj_ids].tolist()
        else:
            inst_obj_maps = torch.cat([pose_masked_inst_maps, obj_maps], dim=0)
            inst_obj_boxes = inst_boxes + obj_boxes

        inst_obj_maps_small = torch.cat([obj_maps[small_obj_ids].view(-1,config.default_H, config.default_W,2),
                                    pose_masked_inst_maps], dim=0)
        
        inst_obj_maps_group_keys = inst_obj_maps[:,:,:,0].amax(dim=(1,2)).unsqueeze(1).unsqueeze(2).long()
        inst_obj_maps_instance_keys = inst_obj_maps[:,:,:,1].amax(dim=(1,2)).unsqueeze(1).unsqueeze(2).long()
        #!!!!!!!!!!!!!!!!!!!!!
        inst_obj_L_maps  = F.interpolate(inst_obj_maps.permute(0,3,1,2),(config.default_H//8, config.default_W//8), mode='nearest').permute(0,2,3,1).long()
        inst_obj_L_maps_small =  F.interpolate(inst_obj_maps_small.permute(0,3,1,2),(config.default_H//8, config.default_W//8), mode='nearest').permute(0,2,3,1).long()

        for j, k in enumerate(inst_obj_maps_group_keys):
            map = inst_obj_L_maps[j,:,:,0].clone() 
            map[map>0] = k
            inst_obj_L_maps[j,:,:,0] = map
        for j, k in enumerate(inst_obj_maps_instance_keys):
            map = inst_obj_L_maps[j,:,:,1].clone() 
            map[map>0] = k
            inst_obj_L_maps[j,:,:,1] = map
        inst_obj_small_L_maps = inst_obj_L_maps_small
        inst_obj_mix_mask = inst_obj_small_L_maps[:,:,:,0].sum(0).bool().to(torch.uint8)

        
        ### Visualization 
        ### TODO
        # Obj Inst는 Group idx
        # Group box 색깔
        
        # print(f'* Global: {global_prompt}')
        box_map = Image.fromarray(np.zeros((config.default_H, config.default_W, 3), dtype=np.uint8) + 0)
        # print(f'box_map.size = {box_map.size}')
        draw = DashedImageDraw(box_map)
        print()
        print(f'* Group')
        for i, (group_x1, group_y1, group_x2, group_y2) in enumerate(group_boxes):
            # print(groups_map.shape)
            draw.dashed_rectangle([(group_x1, group_y1), (group_x2, group_y2)], dash=(5, 5), outline='green', width=5)
            
            key = int(group_maps[i].max().item())
            print(f'green: {group_prompt_dic[key]}')
        
        print(f'* Instance and Objects')
        print(f"inst_obj_prompt_dic: {inst_obj_prompt_dic}")
        for i, (x1, y1, x2, y2) in enumerate(inst_obj_boxes):
            draw.dashed_rectangle([(x1, y1), (x2, y2)], dash=(5, 5), outline=config.color[i], width=5)
            key = int(inst_obj_maps[i,:,:,1].max().item())
            #! key and inst_obj_prompt_dic match fail
            print(f'{config.color[i]}: {inst_obj_prompt_dic[key]}')
        pose_map_pil.save(config.output_path / f'Pose_{image_idx}.png')        
        box_map.save(config.output_path  / f'Boxes_{image_idx}.png')
        pose_box_map = Image.fromarray(np.array(box_map) + pose_map)
        # display(pose_box_map)
        pose_box_map.save(config.output_path / f'PoseBoxes_{image_idx}.png')
        whole_prompt += '.'
        prompt_wanted.insert(0, whole_prompt)

        


        return prompt_wanted