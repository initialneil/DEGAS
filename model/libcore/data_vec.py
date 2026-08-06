# DataVec for data vector.
# Contributer(s): Neil Z. Shao
# All rights reserved 2022.
from .camera import *
from .ply_utils import saveCamerasToPly
from .img_utils import writeFloat2ToPng
import json
import cv2
import os
from copy import deepcopy
from tqdm import tqdm

def _load_image_file(img_fn):
    img = cv2.imread(img_fn, cv2.IMREAD_UNCHANGED)
    return img

def _to_frame(img, float_quant=10000.0, float_quant_shift=0.0):
    # depth/weight
    if (img.dtype == np.dtype('uint16')):
        img = (img.astype(float)-float_quant_shift) / float_quant
    return img

class DataVec:
    def __init__(self, float_quantization=10000.0, float_quant_shift=0.0) -> None:
        self.frames = []
        self.cams = []
        self.cam_models = []
        self.image_formats = []
        self.sns = []
        self.frm_idx = -1
        self.images_path = []
        self.float_quantization = float_quantization
        self.float_quant_shift = float_quant_shift

    def __repr__(self) -> str:
        return '[DataVec] size = %d' % (self.size)
    
    def __len__(self):
        return self.size

    @property
    def size(self):
        return len(self.cams)

    def loadFromFile(self, fn):
        rigs = []
        try:
            with open(fn, 'r') as f:
                value = json.load(f)
                for i in range(0, len(value['rigs'])):
                    rig = Rig()
                    rig.loadFromJson(value['rigs'][i])
                    rigs.append(rig)
            
            self.sns = []
            self.cams = []
            self.image_formats = []
            for i in range(len(rigs)):
                self.sns.append(rigs[i].info)
                self.cams.append(rigs[i].cams[0])
                self.image_formats.append(rigs[i].image_format)

            if 'float_quantization' in value:
                self.float_quantization = value['float_quantization']
            if 'float_quant_shift' in value:
                self.float_quant_shift = value['float_quant_shift']
            
            return True
        except:
            return False

    def saveToFile(self, fn):
        value = {}
        value['float_quantization'] = self.float_quantization
        value['float_quant_shift'] = self.float_quant_shift
        value['rigs'] = []
        for i in range(self.size):
            rig = Rig()
            rig.cams.append(self.cams[i])
            rig.info = self.sns[i] if i < len(self.sns) else ''
            rig.image_format = self.image_formats[i] if i < len(self.image_formats) else ''
            value['rigs'].append(rig.saveToJson())

        with open(fn, 'w', encoding='utf-8') as f:
            json.dump(value, f, ensure_ascii=False, indent=4)

    def copyInfo(self, other):
        self.cams = other.cams
        self.frm_idx = other.frm_idx
        self.cam_models = other.cam_models
        self.image_formats = other.image_formats
        self.sns = other.sns
        self.images_path = other.images_path
        self.float_quantization = other.float_quantization
        self.float_quant_shift = other.float_quant_shift

    def copyData(self, other):
        self.copyInfo(other)
        self.frames = deepcopy(other.frames)
        self.cams = deepcopy(other.cams)

    def clone(self):
        clone_vec = DataVec()
        clone_vec.copyData(self)
        return clone_vec

    def toColorSet(self):
        sub_idxs = [i for i, value in enumerate(self.image_formats) if value == 'RGB']
        sub_vec = DataVec()
        sub_vec.frm_idx = self.frm_idx
        sub_vec.cams = [self.cams[i] for i in sub_idxs]
        sub_vec.frames = [self.frames[i] for i in sub_idxs]
        sub_vec.cam_models = [self.cam_models[i] for i in sub_idxs] if len(self.cam_models) > 0 else []
        sub_vec.image_formats = [self.image_formats[i] for i in sub_idxs]
        sub_vec.sns = [self.sns[i] for i in sub_idxs]
        return sub_vec, sub_idxs

    def toSubSet(self, sub_idxs):
        sub_vec = DataVec()
        sub_vec.frm_idx = sub_vec.frm_idx

        if len(self.cams) == self.size:
            sub_vec.cams = [self.cams[i] for i in sub_idxs]
        if len(self.frames) == self.size:
            sub_vec.frames = [self.frames[i] for i in sub_idxs]
        if len(self.images_path) == self.size:
            sub_vec.images_path = [self.images_path[i] for i in sub_idxs]
        if len(self.cam_models) == self.size:
            sub_vec.cam_models = [self.cam_models[i] for i in sub_idxs] if len(self.cam_models) > 0 else []
        if len(self.image_formats) == self.size:
            sub_vec.image_formats = [self.image_formats[i] for i in sub_idxs]
        if len(self.sns) == self.size:
            sub_vec.sns = [self.sns[i] for i in sub_idxs]
            
        return sub_vec
    
    def load_images(self, silent=False):
        self.frames = []
        for i in tqdm(range(0, self.size), desc='[loadDataVecFromFolder]', disable=silent):
            img_fn = self.images_path[i]
            img = _to_frame(_load_image_file(img_fn), float_quant=self.float_quantization, float_quant_shift=self.float_quant_shift)
            self.frames.append(img)

    def load_images_parallel(self, max_workers):
        self.frames = []
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executrer:
            for res in executrer.map(_load_image_file, self.images_path):
                res = _to_frame(res, float_quant=self.float_quantization, float_quant_shift=self.float_quant_shift)
                self.frames.append(res)

    def resize_to_base_resolution(self, base_resolution):
        data_vec = DataVec()
        data_vec.copyData(self)

        for i in range(data_vec.size):
            cam = data_vec.cams[i]
            image = data_vec.frames[i]
            
            cam.scaleIntrinsicsBaseWidth(base_resolution)
            image = cv2.resize(image, cam.sz, interpolation=cv2.INTER_CUBIC)

            data_vec.cams[i] = cam
            data_vec.frames[i] = image

        return data_vec
    
    # rotate the whole frameset along x-axis 180 degrees
    # all cameras originally in y-negative, rotates to y-position
    # to align with many graphics applications
    def rotate_X180_(self):
        R = cv2.Rodrigues(np.array([np.pi, 0, 0]))[0]
        for i in range(self.size):
            cam = self.cams[i]
            cam.c = R @ cam.c
            cam.R = cam.R @ R.T

    def rotate_to_Y_positive_(self):
        if self.size == 0:
            return
        
        if self.cams[0].c[1] < 0:
            self.rotate_X180_()


def loadDataVecFromFolder(dir, with_frames=True, max_workers=4):
    data_vec = DataVec()

    # load cameras
    data_vec.loadFromFile(dir + "/cameras.json")

    data_vec.frames = []
    data_vec.images_path = []

    for i in range(0, data_vec.size):
        img_fn = dir + '/%02d.png' % i
        data_vec.images_path.append(img_fn)

    # load images
    if with_frames:
        if max_workers <= 0:
            data_vec.load_images()
        else:
            data_vec.load_images_parallel(max_workers)
        
    return data_vec

def saveDataVecToFolder(dir, data_vec, with_frames=True, silent=False):
    os.makedirs(dir, exist_ok=True)
    if not os.path.exists(dir):
        print('[saveDataVecToFolder][ERROR] make dir failed: %s' % dir)
        return

    # save cameras
    data_vec.saveToFile(dir + "/cameras.json")
    float_quantization = data_vec.float_quantization
    float_quant_shift = data_vec.float_quant_shift

    # save images
    if with_frames:
        print('[saveDataVecToFolder] saving ', end='')
        for i in tqdm(range(0, len(data_vec.frames)), desc='[saveDataVecToFolder]', disable=silent):
            img = data_vec.frames[i]
            img_fn = dir + '/%02d.png' % i

            # depth
            if (img.dtype == np.float32 or img.dtype == np.float64):
                if img.shape[-1]==2:
                    writeFloat2ToPng(img_fn, img, float_quantization, float_quant_shift)
                else:
                    img = (img * float_quantization + float_quant_shift).astype(np.uint16)
                    cv2.imwrite(img_fn, img)
            else:
                cv2.imwrite(img_fn, img)

        print('[done]')

    saveCamerasToPly(dir + '/cams.ply', data_vec.cams)
