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
    params["--task"] = ["gym_sokoban_small_tiny_world_easy__render"]
    params["--batch_size"] = [5000]
    params["--num_cache"] = [3]
    params["--episode_length"] = [100]
    params["--seed"] = [1]
    params["--dqn_batch_size"] = [128,64,32]
    params["--load_environments"] = ["env/cache/environments/sokoban"]
    params["--dqn_update_epochs"] = [1]
    params["--dqn_lr"] = [1e-4, 1e-5]
    params["--train_dqn_steps"] = [50000, 100000]
    params["--dqn_train_freq"] = [1]
    params["--seed"] = [0]
    
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
    
    file_name = "run_all.sh".format(iterator)
    output_dir = osp.join(PATH, file_name)
    
    for instance in grid_search_params:
        args = " "
        for key in instance:
            args += key + " " + str(instance[key]) + " "
            
        args += "--num_minibatches {} ".format(
            instance['--batch_size']//instance['--episode_length']
        )
        
        args += "--replay_batch_size {} ".format(instance['--batch_size'])
        args += "--output_dir log/{}".format(iterator)
        
        with open(output_dir, "a") as f:
            f.write("python dqn_transfer_st_main.py" + args + '&\n')
        
        size += 1
        if size == length//N_JOBS:
            iterator += 1
            size = 0