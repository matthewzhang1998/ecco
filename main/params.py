#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 14:55:18 2018

@author: matthewszhang
"""
import os
import os.path as osp
from itertools import product

PATH = os.getcwd()

def env_args():
    params = {}
    
    # Fixed parameters
    params["--task"] = ["gym_sokoban_small_tiny_world"]
    params["--batch_size"] = [5000, 10000, 2500]
    params["--episode_length"] = [50]
    params["--seed"] = [1]
    params["--gamma_max"] = [1e-1]
    params["--replay_buffer_size"] = [20000]
    params["--goals_dim_min"] = [64, 128]
    params["--decoupled_managers"] = [0,1]
    params["--use_replay_buffer"] = [1]
    params["--actor_entropy_coefficient"] = [1e-2]
    params["--clip_actor"] = [5e-2]
    
    # Tuned Parameters
    return params

def product_dict(**kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in product(*vals):
        yield dict(zip(keys, instance))

N_JOBS = 12
if __name__ == "__main__":
    params = env_args()
    grid_search_params = list(product_dict(**params))
    length = len(grid_search_params)
    print(length)
    assert (length % N_JOBS == 0)
    print(length // N_JOBS)
    iterator = 0
    size = 0
    
    for instance in grid_search_params:
        file_name = "run-{}.sh".format(iterator)
        print(file_name)
        output_dir = osp.join(PATH, file_name)
        args = " "
        for key in instance:
            args += key + " " + str(instance[key]) + " "
            
        args += "--output_dir log/{}".format(iterator)
        
        with open(output_dir, "a") as f:
            f.write("python ecco_main.py" + args + ';\n')
        
        size += 1
        if size == length//N_JOBS:
            iterator += 1
            size = 0