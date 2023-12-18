## Dense  diffusion in Latent  consistency model with human centric description  

###  Dataset
in `dataset/dict_gt.json`,  crowd-caption  datset
```
│
└─── key number
    └───  "shape" :  image  size
    └───  "global caption" 
    └───  "0" : group key number  
          └─── "group_bbox" :  group's total bounding  box
          └─── "group_caption"
          └─── "instance"
                └─── "0"
                      └─── "x"
                      └─── "y"
                      └─── "width"
                      └─── "height"
                      └─── "caption"
                      └─── "keypoint" : human keypoint information
                └─── "1"
                ....
```

###  Installation 
```
conda create -n  -f  environment.yml  
conda activate  lcm
python run_lcm.py --key $key number$ --seed $seed number$
```

### Result
`outputs/crowd_caption_gt_select/$key number$/XXXX.png`

### To-do
- [x] Apply  crowd caption dataset  and extract human segmentation with  keypoint
- [x] Apply latent  consistency model
- [x] Apply dense diff code  into  the lcm version
- [] Extract cross  attention  map 
- [] Attention  grounding  success!!!