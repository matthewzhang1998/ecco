#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 12 22:16:51 2018

@author: matthewszhang
"""

import init_path
import math
from env import base_env_wrapper as bew
from env import env_register
import numpy as np
from util import logger

class env(bew.base_env):
    MC = ['gym_sokoban']
    
    def __init__(self, env_name, rand_seed, maximum_length, misc_info):
        super(env, self).__init__(
            env_name, rand_seed, maximum_length, misc_info
        )
        self._base_path = init_path.get_abs_base_dir()
        self._env.env.penalty_for_step = 0.
        
    def step(self, action):    
        action = int(action) # get int from action     
        
        reward_last = self._env.env.reward_last
        self._env.env.reward_last = 0
        
        self._env.step(action)
        ob = self._one_hot(self._env.env.room_state)
        
        reward = self._env.env.reward_last - reward_last
        reward /= 10
    
        # flatten observation
        ob = np.reshape(ob, [-1])

        self._current_step += 1
        if self._current_step >= self._maximum_length:
            done = True
        else:
            done = False # will raise warnings -> set logger flag to ignore
        self._old_ob = np.array(ob)
        
        return ob, reward, done, {}
    
    def reset(self):
        self._env.reset()
        ob = self._one_hot(self._env.env.room_state)
        ob = np.reshape(ob, [-1])
        
        self._current_step = 0
        self._old_ob = ob
        return ob, 0, False, {}
        
    def _build_env(self):
        import gym
        import gym_sokoban
        self._current_version = gym.__version__
        _env_name = {
            'gym_sokoban':'Sokoban-v2',
            'gym_sokoban_small':'Sokoban-small-v1',
            'gym_sokoban_large':'Sokoban-large-v2',
            'gym_sokoban_huge':'Sokoban-huge-v0',
            'gym_sokoban_tiny_world': 'TinyWorld-Sokoban-v2',
            'gym_sokoban_small_tiny_world': 'TinyWorld-Sokoban-small-v1',
        }
        
        # make the environments
        self._env = gym.make(_env_name[self._env_name])
        self._env_info = env_register.get_env_info(self._env_name)
    
    def _one_hot(self, ob):
        one_hot_ob = \
            (np.arange(ob.max()) == ob[...,None]-1).astype(int)
                
        return one_hot_ob
    
    def get_supervised_goal():
        return None
     
