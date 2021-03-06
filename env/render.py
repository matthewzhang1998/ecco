#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 14:55:24 2018

@author: matthewszhang
"""
import pickle
import time
import numpy as np

import init_path
from env.env_register import make_env

class rendering(object):
    def __init__(self, env_name):
        self.env, _ = make_env(env_name, 0, 1234)
        self.env.reset()
        
    def render(self, transition):
        self.env.fdynamics(transition)
        self.env._env.render()
        time.sleep(1/2)
        
def get_rendering_config():
    import argparse
    parser = argparse.ArgumentParser(description='Get rendering from states')
        
    parser.add_argument("--task", type=str, default='gym_cheetah',
                        help='the mujoco environment to test')
    parser.add_argument("--render_file", type=str, default='ep-0',
                        help='pickle outputs to render')
    return parser
    
def main():
    parser = get_rendering_config()
    args = parser.parse_args()
    
    render_env = rendering(args.task)
    with open(args.render_file, 'rb') as pickle_load:
        transition_data = pickle.load(pickle_load)
    
    for transition in transition_data:
        render_transition = {
                    'start_state':np.asarray(transition['start_state']),
                    'action':np.asarray(transition['action']),
                    }

        print(transition['reward'])
        render_env.render(render_transition)

if __name__ == '__main__':
    main()