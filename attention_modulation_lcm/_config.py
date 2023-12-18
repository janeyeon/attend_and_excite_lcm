from dataclasses import dataclass, field
from pathlib import Path
from typing import List
import torch


# COUNT = 0
# DENSEDIFF_ON = 0
# NUM_GROUPS = 0
# NUM_INFERENCE_STEPS = 0
# NUM_ATTENTIONS = 0
# REG_PART = 0.5

@dataclass
class RunConfig:
    output_path: Path = Path("./outputs")
    gen_from_json: bool = True
    json_path: str = "./dataset/crowd_caption_grit_keypoint_obj_caption_openpose2.json"
    # image_idx: int = 327
    # image_idx: int = 34
    image_idx: int = 338 #837 #853

    window_size: int = 64
    stride: int = 8
    view_batch_size = 1
    
    
    zero_pad_latents: bool = True
    use_keypoint_focused_layout: bool = True
    use_randomly_distributed_layout: bool = False
    use_mutually_exclusive_layout: bool = True
    use_object_size_ordering: bool = False
    use_nongroup: bool = False
    
    use_centering_group_box: bool = True
    use_consistent_text_embedding: bool = False

    use_multi_turn_patch: bool = False

    device: str = 'cuda'
    
    
    # Guiding text prompt
    prompt: str = None
    # Which token indices to alter with attend-and-excite
    token_indices: List[int] = None
    bbox: List[list] = field(default_factory=lambda: [[], []])
    color1 =  ['blue', 'red', 'purple', 'orange', 'yellow','brown', 'cyan', 'magenta', 'greenyellow', 'ivory', 'gold', 'navy', 'ivory', 'gold', 'navy','honeydew', 'rosybrown', 'tomato', 'ghostwhite', 'indigo', 'tan', 'hotpink']#field(default_factory=lambda:)
    color = color1 * 10
    control_body_hand_face = [1, 0, 0]


    ### Hierachical Refine
    
    gen_mode = 'normal' # CHOICE(['global_focus', 'group_focus', 'inst_obj_focus', 'normal'])
    
    if gen_mode == 'normal':
        zoom_ratio = 1.5
        use_img2img = False
        enlarge_ratio = 2
        strength = 0.
        mix_ratio = 1
        reverse_ratio = 1.
        mix_mask_for_views = None
        img2img_input_path = None
        
    elif gen_mode == 'global_focus':
        zoom_ratio = 0.5
        use_img2img = False
        enlarge_ratio = 2
        strength = 0.
        mix_ratio = 0.5
        reverse_ratio = 1.
        img2img_input_path = None
        
    elif gen_mode == 'group_focus':
        zoom_ratio = 1.0
        use_img2img = True
        enlarge_ratio = 2
        strength = 0.7
        mix_ratio = 0.5
        reverse_ratio = 0.5
        img2img_input_path = output_path / f'results_group_focus.png' # Later

        #TODO 
        # mix_mask_for_views = group_mix_mask_for_views
        # mix_mask = inst_obj_mix_mask #group_mix_mask
        # img2img_input_path = config.output_path / f'results_global_focus.png'
            
    elif gen_mode == 'inst_obj_focus':
        zoom_ratio = 2.
        use_img2img = True
        enlarge_ratio = 2
        strength = 0.5
        mix_ratio = 0.5
        reverse_ratio = 0.5
        
        img2img_input_path = output_path / f'results_inst_obj_focus.png' # Later

        #TODO
        # mix_mask_for_views = inst_obj_mix_mask_for_views
        # mix_mask = inst_obj_mix_mask
        # img2img_input_path = config.output_path / f'results_group_focus.png'

    seed = 923845
    default_H = 768
    default_W = 768
    
    # Config for diffusion model
    height = None
    width = None
    num_inference_steps = 50
    # num_inference_steps = 4
    guidance_scale = 7.5
    negative_prompt = "anime, cartoon, graphic, text, painting, crayon, graphite, abstract, glitch, deformed, mutated, ugly, disfigured"
    num_images_per_prompt= 1
    eta = 0.0
    generator=torch.Generator().manual_seed(seed)
    latents = None
    prompt_embeds = None
    negative_prompt_embeds = None
    output_type = "pil"
    return_dict = True
    callback = None
    callback_steps = 1
    cross_attention_kwargs = None
    controlnet_conditioning_scale = 1.0
    guess_mode = False
    control_guidance_start = 0.0
    control_guidance_end = 1.0
    ### Multidiffusion
    view_batch_size = 1
    circular_padding = False

    lcm_origin_steps=50

    ### padding
    use_padded_latents = True


    ### reverse_ratio
    reverse_ratio = 0.5

    def __post_init__(self):
        self.output_path.mkdir(exist_ok=True, parents=True)