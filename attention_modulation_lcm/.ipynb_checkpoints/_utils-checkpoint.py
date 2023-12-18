from typing import List
import math
from PIL import Image
import numpy as np
from controlnet_aux.open_pose import PoseResult, BodyResult, Keypoint
import cv2


########################## Functions to handle JSON ##############################

def get_image_grid(images: List[Image.Image]) -> Image:
    num_images = len(images)
    cols = int(math.ceil(math.sqrt(num_images)))
    rows = int(math.ceil(num_images / cols))
    width, height = images[0].size
    grid_image = Image.new('RGB', (cols * width, rows * height))
    for i, img in enumerate(images):
        x = i % cols
        y = i // cols
        grid_image.paste(img, (x * width, y * height))
    return grid_image

def filter_contained_lists_with_indices(lists):
    result = []
    excluded_indices = []

    for i, list1 in enumerate(lists):
        is_contained = False
        for j, list2 in enumerate(lists):
            if i != j and all(elem in list2 for elem in list1):
                is_contained = True
                excluded_indices.append(i)
                break
        if not is_contained:
            result.append(list1)

    return result, excluded_indices

def non_overlapping_elements(list1, list2):
    result = []

    for item in list1:
        if item not in list2:
            result.append(item)

    for item in list2:
        if item not in list1:
            result.append(item)

    return result

def gather_elements_without_duplication(lists):
    all_elements = set()

    for lst in lists:
        all_elements.update(lst)

    result = list(all_elements)
    return result



def draw_bodypose(canvas: np.ndarray, keypoints: List[Keypoint], stickwidth=2) -> np.ndarray:
    """
    Draw keypoints and limbs representing body pose on a given canvas.

    Args:
        canvas (np.ndarray): A 3D numpy array representing the canvas (image) on which to draw the body pose.
        keypoints (List[Keypoint]): A list of Keypoint objects representing the body keypoints to be drawn.

    Returns:
        np.ndarray: A 3D numpy array representing the modified canvas with the drawn body pose.

    Note:
        The function expects the x and y coordinates of the keypoints to be normalized between 0 and 1.
    """
    H, W, C = canvas.shape
    # stickwidth = 4

    limbSeq = [
        [2, 3], [2, 6], [3, 4], [4, 5], 
        [6, 7], [7, 8], [2, 9], [9, 10], 
        [10, 11], [2, 12], [12, 13], [13, 14], 
        [2, 1], [1, 15], [15, 17], [1, 16], 
        [16, 18],
    ]

    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
    for (k1_index, k2_index), color in zip(limbSeq, colors):
        keypoint1 = keypoints[k1_index - 1]
        keypoint2 = keypoints[k2_index - 1]

        if keypoint1 is None or keypoint2 is None:
            continue

        Y = np.array([keypoint1.x, keypoint2.x])  * float(W)
        X = np.array([keypoint1.y, keypoint2.y]) * float(H)
        mX = np.mean(X)
        mY = np.mean(Y)
        length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
        polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
        cv2.fillConvexPoly(canvas, polygon, [int(float(c) * 0.6) for c in color])

    for keypoint, color in zip(keypoints, colors):
        if keypoint is None:
            continue

        x, y = keypoint.x, keypoint.y
        x = int(x * W)
        y = int(y * H)
        cv2.circle(canvas, (int(x), int(y)), 4, color, thickness=-1)

    return canvas

    


def draw_bodyline(canvas: np.ndarray, keypoints: List[Keypoint], stickwidth_alpha=1., return_nose2neck_length=False) -> np.ndarray:
    """
    Draw keypoints and limbs representing body pose on a given canvas.

    Args:
        canvas (np.ndarray): A 3D numpy array representing the canvas (image) on which to draw the body pose.
        keypoints (List[Keypoint]): A list of Keypoint objects representing the body keypoints to be drawn.

    Returns:
        np.ndarray: A 3D numpy array representing the modified canvas with the drawn body pose.

    Note:
        The function expects the x and y coordinates of the keypoints to be normalized between 0 and 1.
    """
    H, W, C = canvas.shape

    limbSeq = [
        [2, 3], [2, 6], [3, 4], [4, 5], 
        [6, 7], [7, 8], [2, 9], [9, 10], 
        [10, 11], [2, 12], [12, 13], [13, 14], 
        [2, 1], [1, 15], [15, 17], [1, 16], 
        [16, 18],
    ]

    keypoint1 = keypoints[0]
    keypoint2 = keypoints[1]
    if keypoint1 is None:
        try:
            keypoint1 = Keypoint(x=keypoints[16].x + keypoints[17].x,y=keypoints[16].y + keypoints[17].y)
        except:
            try:
                keypoint1 = Keypoint(x=keypoints[16].x,y=keypoints[16].y)
            except:
                keypoint1 = Keypoint(x=keypoints[17].x,y=keypoints[17].y)
    Y = np.array([keypoint1.x, keypoint2.x])  * float(W)
    X = np.array([keypoint1.y, keypoint2.y]) * float(H)
    nose2neck_length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
    stickwidth = int(nose2neck_length*stickwidth_alpha)

    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
    for (k1_index, k2_index), color in zip(limbSeq, colors):
        keypoint1 = keypoints[k1_index - 1]
        keypoint2 = keypoints[k2_index - 1]

        if keypoint1 is None or keypoint2 is None:
            continue

        Y = np.array([keypoint1.x, keypoint2.x])  * float(W)
        X = np.array([keypoint1.y, keypoint2.y]) * float(H)
        mX = np.mean(X)
        mY = np.mean(Y)
        length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
        pts = np.array([ [Y[0],X[0]],[Y[1],X[1]] ],np.int32),
        canvas = cv2.polylines(canvas, pts, isClosed=False, color=color, thickness=stickwidth )


    for keypoint, color in zip(keypoints, colors):
        if keypoint is None:
            continue

        x, y = keypoint.x, keypoint.y
        x = int(x * W)
        y = int(y * H)
        cv2.circle(canvas, (int(x), int(y)), 4, color, thickness=-1)
    if not return_nose2neck_length:
        return canvas
    else:
        return canvas, nose2neck_length
    

def get_views(panorama_height, panorama_width, window_size=64, stride=8, circular_padding=False):
    # Here, we define the mappings F_i (see Eq. 7 in the MultiDiffusion paper https://arxiv.org/abs/2302.08113)
    # if panorama's height/width < window_size, num_blocks of height/width should return 1
    panorama_height /= 8
    panorama_width /= 8
    num_blocks_height = (panorama_height - window_size) // stride + 1 if panorama_height > window_size else 1
    if circular_padding:
        num_blocks_width = panorama_width // stride if panorama_width > window_size else 1
    else:
        num_blocks_width = (panorama_width - window_size) // stride + 1 if panorama_width > window_size else 1
    total_num_blocks = int(num_blocks_height * num_blocks_width)
    views = []
    for i in range(total_num_blocks):
        h_start = int((i // num_blocks_width) * stride)
        h_end = h_start + window_size
        w_start = int((i % num_blocks_width) * stride)
        w_end = w_start + window_size
        views.append((h_start, h_end, w_start, w_end))
    return views