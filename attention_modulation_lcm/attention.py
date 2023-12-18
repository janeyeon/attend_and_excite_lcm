import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import cv2
from typing import Optional, Union, Tuple, List, Callable, Dict
from IPython.display import display
from tqdm.notebook import tqdm
import diffusers
import matplotlib.pyplot as plt
import inspect

def text_under_image(image: np.ndarray, text: str, text_color: Tuple[int, int, int] = (0, 0, 0)):
    h, w, c = image.shape
    offset = int(h * .2)
    img = np.ones((h + offset, w, c), dtype=np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    img[:h] = image
    textsize = cv2.getTextSize(text, font, 1, 2)[0]
    text_x, text_y = (w - textsize[0]) // 2, h + offset - textsize[1] // 2
    cv2.putText(img, text, (text_x, text_y ), font, 1, text_color, 2)
    return img


def view_images(images, num_rows=1, offset_ratio=0.02):
    if type(images) is list:
        num_empty = len(images) % num_rows
    elif images.ndim == 4:
        num_empty = images.shape[0] % num_rows
    else:
        images = [images]
        num_empty = 0

    empty_images = np.ones(images[0].shape, dtype=np.uint8) * 255
    images = [image.astype(np.uint8) for image in images] + [empty_images] * num_empty
    num_items = len(images)

    h, w, c = images[0].shape
    offset = int(h * offset_ratio)
    num_cols = num_items // num_rows
    image_ = np.ones((h * num_rows + offset * (num_rows - 1),
                      w * num_cols + offset * (num_cols - 1), 3), dtype=np.uint8) * 255
    for i in range(num_rows):
        for j in range(num_cols):
            image_[i * (h + offset): i * (h + offset) + h:, j * (w + offset): j * (w + offset) + w] = images[
                i * num_cols + j]

    pil_img = Image.fromarray(image_)
    # display(pil_img)
    return pil_img

def aggregate_attention(attention_store, prompts, res:int, from_where: List[str], is_cross: bool, select: int):
    out = []
    num_pixels = res ** 2
    for location in from_where:
        for item in attention_store[f"{location}_{'cross' if is_cross else 'self'}"]:
            if item.shape[1] == num_pixels:
                cross_maps = item.reshape(1, -1, res, res, item.shape[-1])[0]
                out.append(cross_maps) 
    out = torch.cat(out, dim=0)
    out = out.sum(0) / out.shape[0]
    
    return out.cpu()

def is_common_words(text):
    cws = ['a', 'an', 'the', 'in', 'for', 'of', '.', ',',
          '<|startoftext|>', '<|endoftext|>']
    return (text in cws)

def show_cross_attention(pipe, attention_store, prompts, res:int, from_where: List[str], select: int = 0, num_rows=1):
    tokens = pipe.tokenizer.encode(prompts)
    decoder = pipe.tokenizer.decode
    attention_maps = aggregate_attention(attention_store, prompts, res, from_where, True, select)
    images = []
    for i in range(len(tokens)):
        text = decoder(int(tokens[i]))
        if is_common_words(text):
            continue
        image = attention_maps[:, :, i]
        image = 255 * image / image.max()
        image = image.unsqueeze(-1).expand(*image.shape, 3)
        image = image.numpy().astype(np.uint8)
        image = np.array(Image.fromarray(image).resize((256, 256)))
        image = text_under_image(image, decoder(int(tokens[i])))
        images.append(image)
        
    return view_images(np.stack(images, axis=0), num_rows)


def show_self_attention_comp(attention_store, prompts, res:int, from_where: List[str], select: int = 0, num_rows=1, max_com=10):
    attention_maps = aggregate_attention(attention_store, prompts, res, from_where, False, select)
    attention_maps = attention_maps.numpy().reshape((res ** 2, res ** 2)).astype(float)
    u, s, vh = np.linalg.svd(attention_maps - np.mean(attention_maps, axis=1, keepdims=True))
    images = []
    for i in range(max_com):
        image = vh[i].reshape(res, res)
        image = image - image.min()
        image = 255 * image / image.max()
        image = np.repeat(np.expand_dims(image, axis=2), 3, axis=2).astype(np.uint8)
        image = Image.fromarray(image).resize((256, 256))
        image = np.array(image)
        images.append(image)

    return view_images(np.concatenate(images, axis=1), 1)


def get_attention_timesteps(pipe, attention_store_timestep, prompts, res, from_where, select, num_rows):
    cross_attns = []
    self_attns = []
    for attention_store in attention_store_timestep:
        cross_attns.append(show_cross_attention(pipe, attention_store, prompts, res, from_where, select, num_rows))
        self_attns.append(show_self_attention_comp(attention_store, prompts, res, from_where, select, num_rows))     
    
    return cross_attns, self_attns



def save_images_into_one(cas, config, name):
    resized_imgs = []
    width = 2000
    for img in cas: 
        resized_imgs.append(img.resize((width, int(img.height * (width / img.width)))))
    total_height = sum([img.height for img in resized_imgs])
    stacked_image = Image.new("RGB", (width, total_height))
    y_offset = 0
    for reimg in resized_imgs:
        stacked_image.paste(reimg, (0, y_offset))
        y_offset += reimg.height
    stacked_image.save(str(config.output_path) + f"/{name}")



