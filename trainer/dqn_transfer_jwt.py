#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 10:32:26 2018

@author: matthewszhang
"""
import numpy as np
import tensorflow as tf
import init_path
import time
import copy
import os.path as osp

from env import env_register
from util import parallel_util
from util import misc_utils
from util import whitening_util
from util import logger
from util import replay_buffer

from collections import OrderedDict, defaultdict

from main.base_main import make_sampler, make_trainer, log_results

import os.path as osp

from .base_trainer import base_trainer

# single-threaded trainer/worker

class trainer(object):
    def __init__(self, models, args, scope='trainer'):
        self.args = args
        self._name_scope = scope
        self._network_type = models

        # the base agent
        self._base_path = init_path.get_abs_base_dir()

        # used to save the checkpoint files
        self.timesteps_so_far = 0
        self._npr = np.random.RandomState(args.seed)
        self.env = None
        self._current_env_idx = 0
        self._is_done = 0
        self._reset_flag = 0
        self.data_dict = {}
        self._environments_cache = []
    
    def run(self):
        self._set_io_size()
        self._build_models()
        self._init_whitening_stats()
        
        timer_dict = OrderedDict()
        timer_dict['Program Start'] = time.time()
        rolling_stats = defaultdict(list)
        training_return = {}
        
        while True:
            if (self.timesteps_so_far % self.args.print_frequency) == 0:
                timer_dict['** Program Total Time **'] = time.time()
                training_return['stats'] = {}
                for key in rolling_stats:
                    training_return['stats'][key] = np.mean(
                        rolling_stats[key][-self.args.print_frequency:]
                    )
                  
                if 'mean_rewards' in training_return['stats']:
                    training_return['stats']['mean_rewards'] *= \
                       self.args.episode_length
                    
                training_return['iteration'] = \
                    self.timesteps_so_far//self.args.print_frequency
                training_return['totalsteps'] = self.timesteps_so_far
                    
                log_results(training_return, timer_dict)
    
            data_dict = self._play()
            if self.timesteps_so_far >= self.args.dqn_training_start:
                stats, _ = self._train(data_dict)
            
                for key in stats:
                    rolling_stats[key].append(stats[key])
                # log and print the results
            
            if self.timesteps_so_far >= self.args.train_dqn_steps:
                self._saver.save(
                    self._session,
                    osp.join(logger._get_path(),
                        "pretrained_model_{}".format(self._name_scope)
                    )
                )
                break
            else:
                self.timesteps_so_far += 1
                
        return self._get_weights(), self._environments_cache
        
    def _build_models(self):
        self._build_session()
        
        self._network = {key: self._network_type[key](
                self.args, self._session, self._name_scope,
                self._observation_size, self._action_size,
                self._action_distribution
            ) for key in self._network_type
        }
        
        for key in self._network:
            self._network[key].build_model()
            
        self._network['base']._not_actor_network = False     
        
        self._session.run(tf.global_variables_initializer())
        self._saver = tf.train.Saver()
     
    def _build_session(self):
        # TODO: the tensorflow configuration
        config = tf.ConfigProto(device_count={'GPU': 0})  # only cpu version
        self._session = tf.Session(config=config)
        
    def _set_io_size(self):
        self._observation_size, self._action_size, \
            self._action_distribution = \
            env_register.io_information(self.args.task)
     
        
    def _build_env(self):
        if self.args.cache_environments:
            while len(self._environments_cache) < self.args.num_cache:
                _env, self._env_info = env_register.make_env(
                        self.args.dqn_task, self._npr.randint(0, 9999),
                        self.args.episode_length,
                        {'allow_monitor': self.args.monitor \
                         and self._worker_id == 0}
                    )
                _env.reset()
                
                self._environments_cache.append(copy.deepcopy(_env))
                 
        else:
            _env, self._env_info = env_register.make_env(
                        self.args.dqn_task, self._npr.randint(0, 9999),
                        self.args.episode_length,
                        {'allow_monitor': self.args.monitor
                         \
                         and self._worker_id == 0}
                    )
            _env.reset()   
            self.env = _env
                
    def _play(self):
        # runs a single step with one environment
        
        if self.env is None:
            self._build_env()
            if self.args.cache_environments:
                self.env = copy.deepcopy(
                    self._environments_cache[self._current_env_idx]
                )
                obs, reward, self._is_done, _ = self.env.reset_soft()
            else:
                obs, reward, self._is_done, _ = self.env.reset()
            self._last_obs = obs

        control_infos = {'reset': self._reset_flag}
          
        feed_dict = {'start_state': [self._last_obs]}
        
        act_dict = self._network['base'].act(feed_dict, control_infos)
        action = act_dict['actions']
        
        obs, reward, done, _ = self.env.step(action)
        data_dict = {'start_state': np.array([self._last_obs]),
                     'rewards': [reward],
                     'actions': [action],
                     'end_state': [obs],
                     'dones': np.array([done])}

        self._reset_flag = 0
        
        if done:
            if self.args.cache_environments:
                self._current_env_idx += 1
                if self._current_env_idx >= self.args.num_cache:
                    self._current_env_idx = 0
                self.env = copy.deepcopy(
                    self._environments_cache[self._current_env_idx]
                )
                obs, _, _, _ = self.env.reset_soft()
            else:
                obs, _, _, _ = self.env.reset()
            self._reset_flag = 1
                
        self._last_obs = obs
        
        return data_dict
    
    def _train(self, data_dict):
        return self._network['base'].train(data_dict, None, None)
        
    def _get_weights(self):
        weights = {
            key: self._network[key].get_weights() for key in self._network
        }
        return weights
    
    def _init_whitening_stats(self):
        self._whitening_stats = \
            whitening_util.init_whitening_stats(['state', 'diff_state'])
