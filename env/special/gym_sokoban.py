#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 12 22:16:51 2018

@author: matthewszhang
"""

import init_path
import time
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
        self.n_boxes = 3
        
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
        
        _room_dim_shape = self._env.env.dim_room
        geo_state_array = np.reshape(
            ob,
            [*_room_dim_shape] + [-1]
        )
        
        ground_truth_state = np.argmax(geo_state_array, axis=-1)

        self._current_step += 1
        if self._current_step >= self._maximum_length:
            done = True
        else:
            done = False # will raise warnings -> set logger flag to ignore
        self._old_ob = np.array(ob)
               
        return ob, reward, done, {}
    
    def reset(self):
        self._env.reset()
        
        self._keep_n_boxes(self.n_boxes)
        
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
            'gym_sokoban_small_tiny_world_easy': 'TinyWorld-Sokoban-small-v1',
        }
        
        if 'easy' in self._env_name:
            self.n_boxes = 1
        
        # make the environments
        self._env = gym.make(_env_name[self._env_name])
        self._env_info = env_register.get_env_info(self._env_name)
    
    def _one_hot(self, ob):
        one_hot_ob = (np.arange(ob.max()+1) == ob[...,None]).astype(int)
        
        return one_hot_ob
    
    def get_supervised_goal():
        return None
    
    def _keep_n_boxes(self, num_boxes):
        targets = np.where(self._env.env.room_fixed == 2)
        boxes = np.where(
            (self._env.env.room_state == 3)|(self._env.env.room_state == 4)
        )
        
        for i in range(num_boxes, len(targets[0])):
            self._env.env.room_fixed[targets[0][i], targets[1][i]] = 1
            if self._env.env.room_state[targets[0][i], targets[1][i]] == 2:
                self._env.env.room_state[targets[0][i], targets[1][i]] = 1
            self._env.env.room_state[boxes[0][i], boxes[1][i]] = 1
     
    def fdynamics(self, data_dict):
        action = float(data_dict['action'])
        
        _room_dim_shape = self._env.env.dim_room
        geo_state_array = np.reshape(
            data_dict['start_state'],
            [*_room_dim_shape] + [-1]
        )
        
        ground_truth_state = np.argmax(geo_state_array, axis=-1)
        
        self._env.env.room_state = ground_truth_state
        
        # no act, creates bugs
