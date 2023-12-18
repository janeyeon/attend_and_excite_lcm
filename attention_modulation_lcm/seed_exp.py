import os
import random
import argparse



if __name__=="__main__":

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--key', type=str)
    # args = parser.parse_args()
    # image_name = args.key

    # image_list = ["ancmach", "castle", "christ", "futurecity", "gallib", "machbat", "robot", "sea", "ship", "snowforest", "spcity", "starwars", "zelda", "ironman", "queen", "french"]
    image_list = [ "ironman", "queen", "french"]

    for _ in range(100):
        for image_name in image_list:
            # select_name = random.randrange(0, len(image_list))
            seed_num = random.randint(0, 1000000)
            # image_name = image_list[select_name]

            os.system(f"python run_exp.py --key {image_name} --seed {seed_num} --h_num 2")