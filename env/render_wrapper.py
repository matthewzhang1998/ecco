#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 12:03:11 2018

@author: matthewszhang
"""
import re
import pickle
import os.path as osp

import env.env_register
from util import logger

RENDER_EPISODE = 1

class render_wrapper(object):
    def __init__(self, env_name, *args, **kwargs):
        remove_render = re.compile(r'__render$')
        
        self.env_name = remove_render.sub('', env_name)
        self.env, _ = env.env_register.make_env(self.env_name, *args, **kwargs)
        self.episode_number = 0
        
        # Getting path from logger
        self.path = logger._get_path()
        self.obs_buffer = []
        self.always_render = False
        self.render_name = ''

    def step(self, action, *args, **kwargs):
        if self.always_render or self.episode_number % RENDER_EPISODE == 0:
            self.obs_buffer.append({
                    'start_state':self.env._old_ob.tolist(),
                    'action':action.tolist()
                    })
        return_tup = self.env.step(action, *args, **kwargs)
        
        if return_tup[2]:
            self.dump_render()
        return return_tup

    def reset(self, *args, **kwargs):
        self.episode_number += 1
        if self.obs_buffer:
            self.dump_render()
        
        return self.env.reset(*args, **kwargs)

    def fdynamics(self, *args, **kwargs):
        return self.env.fdynamics(*args, **kwargs)

    def reward(self, *args, **kwargs):
        return self.env.reward(*args, **kwargs)
    
    def reset_soft(self, *args, **kwargs):
        self.episode_number += 1
        if self.obs_buffer and (self.render or 
            self.episode_number % RENDER_EPISODE == 0):
            self.dump_render()
        
        return self.env.reset_soft(*args, **kwargs)
    
    def dump_render(self):
        if (self.episode_number % RENDER_EPISODE) == 0:
            file_name = osp.join(
                self.path, 'ep_{}_{}.p'.format(
                    self.episode_number, self.render_name
                )
            )
            with open(file_name, 'wb') as pickle_file:
                pickle.dump(
                    self.obs_buffer, pickle_file,
                    protocol=pickle.HIGHEST_PROTOCOL
                )
        self.obs_buffer = []

    def reward_derivative(self, *args, **kwargs):
        return self.env.reward_derivative(*args, **kwargs)
    
    