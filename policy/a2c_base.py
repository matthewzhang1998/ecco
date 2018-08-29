#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 10:45:51 2018

@author: matthewszhang
"""
import numpy as np
import tensorflow as tf

from policy.baseline.a2c_wrapper import init_wrapper, \
     act_wrapper, train_wrapper, load_wrapper, save_wrapper
     
from .baseline_base import baseline_model
     
from env.env_register import reverse_gym_wrapper
from util import tf_utils
from util.baseline_utils import get_a2c_network_kwargs_from_namespace


class model(baseline_model):
    def __init__(self, parse_args, session, name_scope, *args, **kwargs):
        self.args = parse_args
        
        self._session = session
        self._name_scope = name_scope
        self._dummy_environment = reverse_gym_wrapper(
            self.args.task, 
            num_envs=self.args.batch_size//self.args.episode_length
        )
        
        self.current_episode_timestep = 0
        _network_kwargs = get_a2c_network_kwargs_from_namespace(self.args)
        
        self._gamma = self.args.a2c_gamma
        
        self.required_keys = \
            ['start_state', 'actions', 'rewards', 'end_state',
             'value', 'hidden_states']
        with tf.variable_scope(self._name_scope, reuse=tf.AUTO_REUSE):
            self.baseline_a2c_dict = init_wrapper(
                self._dummy_environment,
                self.args.a2c_network_type,
                number_steps = self.args.batch_size,
                entropy_coefficient = self.args.a2c_entropy_coefficient,
                vf_coefficient = self.args.a2c_vf_coefficient,
                gradient_clipping = self.args.a2c_gradient_max,
                learning_rate = self.args.a2c_lr,
                alpha = self.args.a2c_rms_decay,
                epsilon = self.args.a2c_rms_epsilon,
                total_timesteps = \
                    self.args.batch_size * self.args.a2c_iterations,
                learning_rate_schedule = self.args.a2c_lr_schedule,
                **_network_kwargs
            )
            
            self._set_var_list()
                 
    def train(self, data_dict, replay_buffer, train_net):
        self.current_episode_timestep = 0
        masks = np.zeros_like(data_dict['rewards'])
        
        self._generate_advantages(data_dict)
        self._discard_final_states(data_dict)
        
        if 'lstm' not in self.args.a2c_network_type:
            data_dict['hidden_states'] = None
        
        stats_dictionary = train_wrapper(
            self.baseline_a2c_dict['model'], data_dict, masks
        )
        
        stats_dictionary['avg_reward'] = np.mean(data_dict['rewards'])
        
        return stats_dictionary, data_dict
        
    def act(self, data_dict, control_info):
        masks = np.zeros((len(data_dict['start_state']),))
        
        return act_wrapper(
            self.baseline_a2c_dict['model'], data_dict, masks
        )
        
    def save_model(self, save_path=None):
        save_wrapper(self.baseline_a2c_dict['model'], save_path)
        
    def load_model(self, load_path=None):
        load_wrapper(self.baseline_a2c_dict['model'], load_path)
        
    def _generate_advantages(self, data_dict):
        # fix indices
        data_dict['advantage'] = \
            np.zeros(data_dict['value'].shape)
            
        _batch_size = len(data_dict['start_state'])
        
        for episode in range(int(_batch_size/self.args.episode_length)):
            episode_start = episode * self.args.episode_length
            episode_end = episode_start + self.args.episode_length
            for i_step in reversed(range(episode_start, episode_end)):
                if i_step < episode_end - 1:
                    delta = data_dict['rewards'][i_step] \
                        + self._gamma * \
                        data_dict['value'][i_step + 1] \
                        - data_dict['value'][i_step]
                    data_dict['advantage'][i_step] = \
                        delta + self._gamma * \
                        data_dict['advantage'][i_step+1]
                        
                else:
                    delta = data_dict['rewards'][i_step] \
                        - data_dict['value'][i_step]
                    data_dict['advantage'][i_step] = delta