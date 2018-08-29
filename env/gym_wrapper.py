#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 18:28:48 2018

@author: matthewszhang
"""
import init_path
from gym import spaces

class gym_wrapper(object):
    def __init__(self, _, obs_size, act_size, act_distribution, num_envs=1):
        self.observation_space = spaces.Box(0, 1, (obs_size,))
        
        if act_distribution == 'discrete':
            self.action_space = spaces.Discrete(act_size)
            
        elif act_distribution == 'continuous':
            self.action_space = spaces.Box(0, 1, (act_size,))
            
        self.num_envs = num_envs
