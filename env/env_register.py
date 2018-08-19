#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 12 19:53:57 2018

@author: matthewszhang
"""

import importlib
import re
from env.render_wrapper import render_wrapper


_ENV_INFO = {
    'gym_sokoban': {
        'path': 'env.special.gym_sokoban',
        'ob_size': 76800, 'action_size': 8,
        'action_distribution': 'discrete'
    },
    'gym_sokoban_tiny_world': {
        'path': 'env.special.gym_sokoban',
        'ob_size':300, 'action_size':8,
        'action_distribution': 'discrete'
    },
    'gym_sokoban_small_tiny_world': {
        'path': 'env.special.gym_sokoban',
        'ob_size':147, 'action_size':8,
        'action_distribution': 'discrete'
    }
}
    
def io_information(task_name):
    render_flag = re.compile(r'__render$')
    task_name = render_flag.sub('', task_name)
    
    return _ENV_INFO[task_name]['ob_size'], \
        _ENV_INFO[task_name]['action_size'], \
        _ENV_INFO[task_name]['action_distribution']


def get_env_info(task_name):
    render_flag = re.compile(r'__render$')
    task_name = render_flag.sub('', task_name)
    
    return _ENV_INFO[task_name]


def make_env(task_name, rand_seed, maximum_length, misc_info={}):
    render_flag = re.compile(r'__render$')
    if render_flag.search(task_name):
        return \
            render_wrapper(task_name, rand_seed, maximum_length, misc_info), \
            get_env_info(task_name)
    
    env_file = importlib.import_module(_ENV_INFO[task_name]['path'])
    return env_file.env(task_name, rand_seed, maximum_length, misc_info), \
         _ENV_INFO[task_name]
