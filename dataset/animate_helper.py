import os
import json
import numpy as np
from pathlib import Path
import random
import copy
from model.bone_deformer import smplx_utils

class MotionHelper:
    def __init__(self, fn=None, tpose_fn=None) -> None:
        if fn is not None:
            self.load_motion(fn, tpose_fn=tpose_fn)

    def load_motion(self, fn, tpose_fn=None):
        if fn.endswith('.json'):
            self.load_motion_json(fn, tpose_fn=tpose_fn)
        elif fn.endswith('.pt'):
            self.load_motion_pt(fn, tpose_fn=tpose_fn)
        else:
            raise NotImplementedError
        
    def load_motion_json(self, fn, tpose_fn=None):
        with open(fn, 'r') as fp:
            cc = json.load(fp)

        smplx_from = cc['smplx_from']
        if not os.path.isabs(smplx_from):
            smplx_from = str(Path(fn).parent / smplx_from)
        smplx_params = smplx_utils.load_and_detach(smplx_from)
        
        self.groups = cc['groups']
        for key in self.groups:
            group = self.groups[key]
            group['smplx_params'] = smplx_params
            group['knn_features'] = smplx_params['body_pose']

        self.setup_motion()

    def load_motion_pt(self, fn, tpose_fn=None):
        if tpose_fn is not None:
            tpose_params = smplx_utils.load_and_detach(tpose_fn)
        else:
            tpose_params = None

        smplx_params = smplx_utils.load_and_detach(fn)
        self.groups = {
            'idle':
            {
                'smplx_params': smplx_params,
                'tpose_params': tpose_params,
                'knn_features': smplx_params['body_pose'],
                'clips': [
                    [0, smplx_params['body_pose'].shape[0] - 1],
                ]
            }
        }

        self.setup_motion()

    def setup_motion(self):
        self.states = [key for key in self.groups]
        print(f'[MotionHelper] state(s): {self.states}')

        # playing state
        self.curr_state = self.states[0]
        self.next_state = self.curr_state
        self.clip_idx = 0

        group = self.groups[self.curr_state]
        clip = group['clips'][self.clip_idx]
        self.play_idx = clip[0]

        # blending
        self.blending_params = None
        self.blending_idx = 0
        self.blending_N = 10
        z = np.linspace(-4, 4, self.blending_N)
        def sigmoid(z):
            return 1/(1 + np.exp(-z))
        w = sigmoid(z)
        self.blending_w = w

        self.blending_alpha = 0
        self.blending_thresh = 1e-3

    def set_next_state(self, next_state):
        self.next_state = next_state

    def get_next_frame(self):
        group = self.groups[self.curr_state]
        clip = group['clips'][self.clip_idx]
        if self.play_idx <= clip[1]:
            params = smplx_utils.get_smplx_params_by_idx(group['smplx_params'], self.play_idx)
            tpose_params = group.get('tpose_params', None)
            self.play_idx = self.play_idx + 1
            self.curr_params = params
            return self.blend_motion(params, tpose_params=tpose_params)
        
        else:
            self.curr_state = self.next_state
            if self.curr_state not in self.groups:
                self.curr_state = 'idle'
                
            group = self.groups[self.curr_state]
            self.clip_idx = random.randint(0, len(group['clips']) - 1)
            clip = group['clips'][self.clip_idx]
            self.play_idx = clip[0]
            print(f'[MotionHelper] next play: {self.curr_state} #{self.clip_idx} [{clip[0]}, {clip[1]}]')

            # set blending
            self.blending_params = self.curr_params
            self.blending_idx = 0

            return self.get_next_frame()
        
    def blend_motion(self, params, ema=0.9, tpose_params=None):
        if self.blending_params is None or self.blending_idx >= self.blending_N:
            # print(f'{params["global_orient"]}')
            return {
                'smplx_params': params,
                'tpose_params': tpose_params,
            }
        
        w = self.blending_w[self.blending_idx]
        params = copy.deepcopy(params)
        for key in params:
            if key.endswith('pose') or key in ['global_orient', 'transl']:
                params[key] = (1.0 - w) * self.blending_params[key] + w * params[key]

        # print(f'{self.blending_params["global_orient"]} * {1.0 - w} + {params["global_orient"]} * {w}')


        self.blending_idx = self.blending_idx + 1
        return {
            'smplx_params': params,
            'tpose_params': tpose_params,
        }




