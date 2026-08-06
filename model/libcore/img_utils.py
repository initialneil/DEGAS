# Image processing utils.
# Contributer(s): Neil Z. Shao
# All rights reserved 2023.
import os
from pathlib import Path
import numpy as np
import cv2
import copy

# img: hwc or hw
def sub_image(img, x0:int, y0:int, x1:int, y1:int):
    if x0 >= 0 and y0 >= 0 and x1 <= img.shape[1] and y1 <= img.shape[0]:
        return img[y0:y1, x0:x1]

    if len(img.shape) == 2:
        patch = np.zeros((y1-y0, x1-x0), dtype=img.dtype)
    else:
        patch = np.zeros((y1-y0, x1-x0, img.shape[-1]), dtype=img.dtype)

    sx0 = max(x0, 0)
    sy0 = max(y0, 0)
    dx0 = sx0 - x0
    dy0 = sy0 - y0
    sx1 = min(x1, img.shape[1])
    sy1 = min(y1, img.shape[0])
    w = sx1 - sx0
    h = sy1 - sy0
    if w > 0 and h > 0:
        patch[dy0:dy0+h, dx0:dx0+w] = img[sy0:sy0+h, sx0:sx0+w]
    return patch

def readFloat2FromPng(fn, scale=10000.0, shift=32768.0):
    value_map = cv2.imread(fn, cv2.IMREAD_UNCHANGED)
    value_map = (value_map.astype(float) - shift) / scale
    h = int(value_map.shape[0])
    w = int(value_map.shape[1] / 2)
    wF = w * 2

    map2f = np.stack([value_map[0:h, 0:w], value_map[0:h, w:wF]], axis=2)
    return map2f

def writeFloat2ToPng(fn, value_map, scale=10000.0, shift=32768.0):
    value_map = (value_map * scale + shift).astype(np.uint16)
    h = int(value_map.shape[0])
    w = int(value_map.shape[1])

    map2f = np.concatenate([value_map[:, :, 0], value_map[:, :, 1]], axis=1)
    cv2.imwrite(fn, map2f)

def readFloat4FromPng(fn):
    value_map = cv2.imread(fn, cv2.IMREAD_UNCHANGED)
    value_map = (value_map.astype(float) - 32768.0) / 10000.0
    h = int(value_map.shape[0] / 2)
    w = int(value_map.shape[1] / 2)
    hF = h * 2
    wF = w * 2

    map4f = np.stack([value_map[0:h, 0:w], value_map[h:hF, 0:w], value_map[0:h, w:wF], value_map[h:hF, w:wF]], axis=2)
    return map4f

def writeNMapToUchar3(nmap_fn, nmap):
    nmap_clr = (nmap * 127 + 128).astype(np.uint8)
    cv2.imwrite(nmap_fn, nmap_clr)
    
def detachToNumpy(img):
    if len(img.shape) == 4:
        img = img.squeeze(0)
    return (img.detach().cpu() * 255.0).numpy().astype(np.uint8)

# normalize to
def colorizeWeightsMap(weights, colormap=cv2.COLORMAP_JET, 
                       min_val=None, max_val=None, 
                       to_rgb=False):
    if min_val is None:
        min_val = weights.reshape(-1, weights.shape[-1]).min(axis=0)
    if max_val is None:
        max_val = weights.reshape(-1, weights.shape[-1]).max(axis=0)

    vals = (weights - min_val) / (max_val - min_val)
    vals = (vals.clip(0, 1) * 255).astype(np.uint8)
    canvas = cv2.applyColorMap(vals, colormap=colormap)
    if to_rgb:
        return canvas[..., [2, 1, 0]]
    else:
        return canvas

# display
def cvshow(img, max_width=1920, title='image'):
    # tensor or ndarray
    if not isinstance(img, np.ndarray):
        img = img.detach().float().cpu().squeeze().numpy()

    if img.dtype == float:
        if len(img.shape) == 2:
            img = colorizeWeightsMap(img)
        else:
            img = (img * 255.0).clip(0, 255).astype(np.uint8)
    

    if max_width > 0 and img.shape[1] > max_width:
        scale = float(max_width) / img.shape[1]
        img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

    cv2.imshow(title, img)
    cv2.waitKey(100)

# tensor to image
def tensor_to_cvimage(tensor, rgb2bgr=False, to_uint8=True):
    if len(tensor.shape) == 4:
        tensor = tensor[0]
    if len(tensor.shape) == 3:
        if tensor.shape[0] == 3 or tensor.shape[0] == 4:
            tensor = tensor.permute([1, 2, 0])

    if rgb2bgr:
        if tensor.shape[-1] == 3:
            tensor = tensor[..., [2, 1, 0]]
        else:
            tensor = tensor[..., [2, 1, 0, 3]]
    
    if to_uint8:
        img = (tensor.clamp(0, 1) * 255).detach().cpu().numpy().astype(np.uint8)
    else:
        img = (tensor.clamp(0, 1)).detach().cpu().numpy()
    return img

def write_tensor_image(fn, tensor, rgb2bgr=False):
    img = tensor_to_cvimage(tensor, rgb2bgr=rgb2bgr)
    if fn is not None:
        os.makedirs(Path(fn).parent, exist_ok=True)
        cv2.imwrite(fn, img)
    return img

# draw points
def draw_pixel_points(img, pixels, radius=3, color=None, thickness=0, 
                      fontFace=0, fontScale=1.0,
                      text_start_number=None):
    canvas = copy.deepcopy(img)
    for i in range(pixels.shape[0]):
        if color is not None:
            clr = color
        else:
            clr = (np.random.rand(3) * 256).astype(int).tolist()
        cv2.circle(canvas, pixels[i].astype(int), radius, clr, thickness=thickness)
        if text_start_number is not None:
            cv2.putText(canvas, str(i + text_start_number), (pixels[i] + 10).astype(int), 
                        fontFace, fontScale, clr)
    return canvas
    
# draw pairs
def draw_pixel_pairs(img, pxls0, pxls1, pxls0_color=[0, 0, 255], pxls1_color=[255, 0, 0], 
                     line_color=[255, 255, 255], thickness=1):
    canvas = copy.deepcopy(img)
    for i in range(pxls0.shape[0]):
        cv2.circle(canvas, pxls0[i].astype(int), 1, pxls0_color, thickness=thickness)
        cv2.circle(canvas, pxls1[i].astype(int), 1, pxls1_color, thickness=thickness)
        cv2.line(canvas, pxls0[i].astype(int), pxls1[i].astype(int), line_color, thickness=thickness)
    return canvas
    
# extend maps
def extend_maps(image, mask):
    y, x = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]), indexing='ij')
    full_nbr_ys = np.stack([y - 1, y + 1, y, y], axis=0)
    full_nbr_xs = np.stack([x, x, x - 1, x + 1], axis=0)
    full_nbr_ys = np.minimum(np.maximum(full_nbr_ys, 0), image.shape[0] - 1)
    full_nbr_xs = np.minimum(np.maximum(full_nbr_xs, 0), image.shape[1] - 1)

    mask = copy.deepcopy(mask)
    image = copy.deepcopy(image)
    num_iterations = 10
    for it in range(num_iterations):
        check_mask = ~mask
        check_ys = y[check_mask]
        check_xs = x[check_mask]
        nbr_ys = full_nbr_ys[:, check_mask]
        nbr_xs = full_nbr_xs[:, check_mask]

        update_count = np.zeros(check_ys.shape)
        update_values = np.zeros(image[check_ys, check_xs].shape)
        for k in range(nbr_ys.shape[0]):
            nbr_mask = mask[nbr_ys[k], nbr_xs[k]]
            nbr_values = image[nbr_ys[k], nbr_xs[k]]
            update_count[nbr_mask] += 1
            update_values[nbr_mask] += nbr_values[nbr_mask]
            
        updated = update_count > 1
        image[check_ys[updated], check_xs[updated]] = update_values[updated] / update_count[updated, None]
        mask[check_ys[updated], check_xs[updated]] = True

    return image, mask

##################################################
import torch
import torchvision.transforms as tft

# arrange image to row x col grids
def arrange_to_grids(N):
    cols = int(np.ceil(np.sqrt(N)))
    rows = int(np.ceil(N / cols))
    return rows, cols

# [chw] image
def crop_and_resize_to(image, grid_w, grid_h):
    height, width = image.shape[1:]
    
    # crop
    crop_w = int(min(height * grid_w / grid_h, width))
    crop_h = int(min(width * grid_h / grid_w, height))
    l = (width - crop_w) // 2
    t = (height - crop_h) // 2
    image = image[:, t:t+crop_h, l:l+crop_w]

    # resize
    image = tft.Resize((grid_h, grid_w))(image)
    return image

# [chw] image
def concat_to_layout(images, rows, cols):
    N = len(images)
    k = 0
    row_images = []
    for i in range(rows):
        if k >= N:
            break

        if k + cols <= N:
            img = torch.concat(images[k:k+cols], dim=-1)
            row_images.append(img)
        else:
            t = cols + k - N
            img = torch.concat([*images[k:], 
                                torch.zeros_like(images[k].repeat(1, 1, t))
                                ], dim=-1)
            row_images.append(img)
        k += cols

    full_image = torch.concat(row_images, dim=1)
    return full_image

# merge multiple images to one
# arrange multiple [chw] images to one
# using the resolution of one image
def gridview_images(images, rows=None, cols=None, fit_to_one=True):
    if len(images) < 1:
        return None
    elif len(images) == 1:
        return images[0]
    
    if rows is None and cols is None:
        rows, cols = arrange_to_grids(len(images))
    elif rows is not None:
        cols = (int)(np.ceil(len(images) / rows))
    else:
        rows = (int)(np.ceil(len(images) / cols))

    if fit_to_one:
        full_height, full_width = images[0].shape[1:]
        grid_w = full_width // cols
        grid_h = full_height // rows
    else:
        grid_h, grid_w = images[0].shape[1:]

    images_list = []
    for image in images:
        images_list.append(crop_and_resize_to(image, grid_w, grid_h))
    show_image = concat_to_layout(images_list, rows, cols)
    return show_image

