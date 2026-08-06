#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torchvision.transforms as tft
import traceback
import socket
import json
import numpy as np
from scene.cameras import MiniCam

host = "127.0.0.1"
port = 6009

conn = None
addr = None

listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

def init(wish_host, wish_port):
    global host, port, listener
    host = wish_host
    port = wish_port
    listener.bind((host, port))
    listener.listen()
    listener.settimeout(0)

def try_connect():
    global conn, addr, listener
    try:
        conn, addr = listener.accept()
        print(f"\nConnected by {addr}")
        conn.settimeout(None)
    except Exception as inst:
        pass
            
def read():
    global conn
    messageLength = conn.recv(4)
    messageLength = int.from_bytes(messageLength, 'little')
    message = conn.recv(messageLength)
    return json.loads(message.decode("utf-8"))

def send(message_bytes, verify):
    global conn
    if message_bytes != None:
        conn.sendall(message_bytes)
    conn.sendall(len(verify).to_bytes(4, 'little'))
    conn.sendall(bytes(verify, 'ascii'))

def receive():
    message = read()

    width = message["resolution_x"]
    height = message["resolution_y"]

    if width != 0 and height != 0:
        try:
            do_training = bool(message["train"])
            fovy = message["fov_y"]
            fovx = message["fov_x"]
            znear = message["z_near"]
            zfar = message["z_far"]
            do_shs_python = bool(message["shs_python"])
            do_rot_scale_python = bool(message["rot_scale_python"])
            keep_alive = bool(message["keep_alive"])
            scaling_modifier = message["scaling_modifier"]
            world_view_transform = torch.reshape(torch.tensor(message["view_matrix"]), (4, 4)).cuda()
            world_view_transform[:,1] = -world_view_transform[:,1]
            world_view_transform[:,2] = -world_view_transform[:,2]
            full_proj_transform = torch.reshape(torch.tensor(message["view_projection_matrix"]), (4, 4)).cuda()
            full_proj_transform[:,1] = -full_proj_transform[:,1]
            custom_cam = MiniCam(width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform)
        except Exception as e:
            print("")
            traceback.print_exc()
            raise e
        return custom_cam, do_training, do_shs_python, do_rot_scale_python, keep_alive, scaling_modifier
    else:
        return None, None, None, None, None, None

########## routine ##########
def send_image_to_network(image, verify):
    global conn
    if conn == None:
        try_connect()
    if conn != None:
        try:
            custom_cam, do_training, do_shs_python, _, _, scaling_modifer = receive()

            net_image = torch.zeros((3, custom_cam.image_height, custom_cam.image_width))
            if image.shape[1] > net_image.shape[1] or image.shape[2] > net_image.shape[2]:
                step = max(image.shape[1] / net_image.shape[1], image.shape[2] / net_image.shape[2])
                step = max(int(np.ceil(step)), 1)
                image = image[:3, ::step, ::step]
            top = (net_image.shape[1] - image.shape[1]) // 2
            left = (net_image.shape[2] - image.shape[2]) // 2
            net_image[:, top:top+image.shape[1], left:left+image.shape[2]] = image

            net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
            send(net_image_bytes, verify)
        except Exception as e: 
            conn = None

def render_to_network(model, pipe, verify, gt_image=None, view_cam=None, view_scaling=None,
                      return_image=False, return_camera=False):
    do_training = True
    net_image = None
    global conn

    if conn == None:
        try_connect()
    if conn != None:
        try:
            custom_cam, do_training, do_shs_python, _, _, scaling_modifer = receive()
            if view_cam is not None:
                custom_cam = view_cam
            if view_scaling is not None:
                scaling_modifer = view_scaling

            with torch.no_grad():
                net_image = model.render_to_camera(custom_cam, pipe, background='white',
                                                   scaling_modifer=scaling_modifer)["render"]
                
            custom_cam.view_scaling = scaling_modifer

            if gt_image is not None:
                step = int(max(max(gt_image.shape[1] / 200, gt_image.shape[2] / 200), 1))
                img = gt_image[:, ::step, ::step]
                net_image[:, :img.shape[1], :img.shape[2]] = img

            net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
            send(net_image_bytes, verify)

        except Exception as e: 
            conn = None
    
    if return_image and return_camera:
        return do_training, net_image, custom_cam
    elif return_image:
        return do_training, net_image
    elif return_camera:
        return do_training, custom_cam
    return do_training

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
            img = torch.concat([*images[k:], images[k].repeat(1, 1, t)], dim=-1)
            row_images.append(img)
        k += cols

    full_image = torch.concat(row_images, dim=1)
    return full_image

# render multiple images to grids
def render_multiple_to_network(models, pipe, verify, view_cam=None, view_scaling=None,
                               return_image=False, return_camera=False):
    if len(models) <= 0:
        if return_image:
            return None, None
        else:
            return None
    if len(models) == 1:
        return render_to_network(models[0], pipe, verify, view_cam=view_cam, view_scaling=view_scaling,
                                 return_image=return_image, return_camera=return_camera)

    do_training = True
    net_image = None
    global conn

    rows, cols = arrange_to_grids(len(models))

    if conn == None:
        try_connect()
    if conn != None:
        try:
            custom_cam, do_training, do_shs_python, _, _, scaling_modifer = receive()
            if view_cam is not None:
                custom_cam = view_cam

            if view_scaling is not None:
                scaling_modifer = view_scaling

            width, height = custom_cam.image_width, custom_cam.image_height
            grid_w = width // cols
            grid_h = height // rows

            net_images = []
            with torch.no_grad():
                for model in models:
                    image = model.render_to_camera(custom_cam, pipe, background='white',
                                                   scaling_modifer=scaling_modifer)["render"]
                    image = crop_and_resize_to(image, grid_w, grid_h)
                    net_images.append(image)

            custom_cam.view_scaling = scaling_modifer

            net_image = concat_to_layout(net_images, rows, cols)
            net_image = tft.Resize((height, width))(net_image)

            net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
            send(net_image_bytes, verify)

        except Exception as e: 
            conn = None
    
    if return_image and return_camera:
        return do_training, net_image, custom_cam
    elif return_image:
        return do_training, net_image
    elif return_camera:
        return do_training, custom_cam
    return do_training
