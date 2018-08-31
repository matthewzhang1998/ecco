#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 12:41:57 2018

@author: matthewszhang
"""

# -----------------------------------------------------------------------------
#   @author:
#       Matthew Zhang
#   @brief:
#       Several basic classical control environments that
#       1. Provide ground-truth reward function.
#       2. Has reward as a function of the observation.
#       3. has episodes with fixed length.
# -----------------------------------------------------------------------------
import init_path
import math
from env import base_env_wrapper as bew
from env import env_register
import numpy as np
from util import logger

import os, sys

def blockPrint():
    sys.stdout = open(os.devnull, 'w')
    
def enablePrint():
    sys.stdout = sys.__stdout__

class env(bew.base_env):
    # acrobot has applied sin/cos obs
    CARTPOLE = ['gym_cartpole', 'gym_cartpole_continuous']

    

    def __init__(self, env_name, rand_seed, maximum_length, misc_info):
        super(env, self).__init__(
            env_name, rand_seed, maximum_length, misc_info
        )
        self._base_path = init_path.get_abs_base_dir()
        
    def step(self, action):
        blockPrint()
        ob, reward, _, info = self._env.step(action)
        enablePrint()

        # get the end signal
        self._current_step += 1
        if self._current_step >= self._maximum_length:
            done = True
        else:
            done = False # will raise warnings -> set logger flag to ignore
        self._old_ob = np.array(ob)
        return ob, reward, done, info
    
    def reset(self):
        self._current_step = 0
        self._old_ob = self._env.reset()
        return np.array(self._old_ob), 0.0, False, {}
    
    def _build_env(self):
        import gym
        self._current_version = gym.__version__
        _env_name = {
            'gym_cartpole': 'CartPole-v1',
            'gym_cartpole_continuous': 'CartPole-v1'
            }

        # make the environments
        self._env = gym.make(_env_name[self._env_name])
        self._env_info = env_register.get_env_info(self._env_name)

    def _set_groundtruth_api(self):
        """ @brief:
                In this function, we could provide the ground-truth dynamics
                and rewards APIs for the agent to call.
                For the new environments, if we don't set their ground-truth
                apis, then we cannot test the algorithm using ground-truth
                dynamics or reward
        """
        self._set_reward_api()
        self._set_dynamics_api()
    
    def _set_dynamics_api(self):
        '''
        def fdynamics(self, data_dict):
            raise NotImplementedError

        def fdynamics_grad_state(self, data_dict):
            raise NotImplementedError

        def fdynamics_grad_action(self, data_dict):
            raise NotImplementedError

        self.fdynamics = fdynamics
        self.fdynamics_grad_state = fdynamics_grad_state
        self.fdynamics_grad_action = fdynamics_grad_action
        '''

        def fdynamics(data_dict):
            x, x_dot, theta, theta_dot = data_dict['start_state']
            action = data_dict['action']
            
            if self._env.env.steps_beyond_done == 0:
                self._env.env.steps_beyond_done = None
                
            if x < -self._env.env.x_threshold:
                action = 1
            elif x > self._env.env.x_threshold:
                action = 0

            self._env.env.state = np.array([x, x_dot, theta, theta_dot])
            
            return self._env.step(action)[0]

        self.fdynamics = fdynamics