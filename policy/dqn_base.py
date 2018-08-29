#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 21 14:15:34 2018

@author: matthewszhang

This file is essentially a wrapper of DQN from baselines that allows it to be
distributed and merged with the ECCO algorithm
"""
import numpy as np
import tensorflow as tf

from policy.dqn.baseline_wrapper import init_wrapper, \
     act_wrapper, train_wrapper, load_wrapper, save_wrapper
     
from env.env_register import reverse_gym_wrapper
from util import tf_utils
from util.baseline_utils import get_dqn_network_kwargs_from_namespace

class model(object):
    def __init__(self, parse_args, session, name_scope, *args, **kwargs):
        self.args = parse_args
        
        self._session = session
        self._name_scope = name_scope
        
        self._dummy_environment = reverse_gym_wrapper(self.args.task)
        self.timesteps_so_far = 0
        
        _network_kwargs = get_dqn_network_kwargs_from_namespace(self.args)
        
        self.required_keys = ['start_state', 'actions', 'rewards', 'end_state']
        with tf.variable_scope(self._name_scope):
            self.baseline_dqn_dict = init_wrapper(
                self._dummy_environment,
                self.args.dqn_network_type,
                lr = self.args.dqn_lr,
                gamma = self.args.dqn_gamma,
                param_noise = self.args.use_dqn_param_noise,
                buffer_size = self.args.dqn_buffer_size,
                prioritized_replay_alpha = self.args.dqn_prioritized_alpha,
                prioritized_replay = self.args.use_dqn_prioritized_replay,
                prioritized_replay_beta_iters = self.args.dqn_beta_iters,
                prioritized_replay_beta = self.args.dqn_prioritized_beta,
                exploration_fraction = self.args.dqn_epsilon,
                exploration_final_eps = self.args.dqn_min_epsilon,
                total_timesteps = \
                    self.args.train_dqn_iterations * self.args.batch_size,
                **_network_kwargs
            )
            self._set_var_list()
    
    def train(self, data_dict, replay_buffer, training_info={}):
        dones = np.tile(
            np.array([0] * (self.args.episode_length - 1) + [1]),
            [len(data_dict['start_state']) // self.args.episode_length]
        )
        
        data_dict = self._discard_final_states(data_dict)
        
        self.baseline_dqn_dict['replay_buffer'].add(
            np.array(data_dict['start_state']),
            np.array(data_dict['actions']),
            np.array(data_dict['rewards']),
            np.array(data_dict['end_state']),
            np.array(dones, dtype=np.float32)
        )
        
        stats = {'mean_rewards': np.mean(data_dict['rewards'])}
        
        _target_update = self.timesteps_so_far % \
            self.args.dqn_update_target_steps
        
        for _ in range(self.args.dqn_update_epochs):
            train_wrapper(
                self.baseline_dqn_dict['replay_buffer'],
                self.baseline_dqn_dict['beta_schedule'],
                self.args.use_dqn_prioritized_replay,
                self.args.dqn_prioritized_replay_eps,
                self.timesteps_so_far,
                self.baseline_dqn_dict['update_target_function'],
                self.args.dqn_batch_size,
                self.baseline_dqn_dict['train_function'],
                target_update = _target_update
            )
        
        return stats, data_dict
        
    def act(self, data_dict, control_infos={}):
        actions = []
        for i in range(len(data_dict['start_state'])):
            _dqn_feed_obs = np.array(data_dict['start_state'][i])
                
            _reset = (_dqn_feed_obs.shape[0] == 1) # first observation
            
            self.timesteps_so_far += 1
            
            actions.append(
                act_wrapper(
                    self._dummy_environment,
                    self.baseline_dqn_dict['act_function'],
                    self.baseline_dqn_dict['exploration_scheme'],
                    _dqn_feed_obs,
                    self.baseline_dqn_dict['replay_buffer'],
                    self.timesteps_so_far,
                    reset = _reset,
                    param_noise = self.args.use_dqn_param_noise
                )
            )
        
        return {'actions': np.array(actions)}
    
    def get_weights(self):
        return self._get_network_weights()
    
    def set_weights(self, weights):
        self._set_network_weights(weights)

    def _set_var_list(self):
        # collect the tf variable and the trainable tf variable
        self._trainable_var_list = [var for var in tf.trainable_variables()
                                    if self._name_scope in var.name]

        self._all_var_list = [var for var in tf.global_variables()
                              if self._name_scope in var.name]

        # the weights that actually matter
        self._network_var_list = \
            self._trainable_var_list # + self._whitening_variable

        self._set_network_weights = tf_utils.set_network_weights(
            self._session, self._network_var_list, self._name_scope
        )

        self._get_network_weights = tf_utils.get_network_weights(
            self._session, self._network_var_list, self._name_scope
        )

    def load_checkpoint(self, ckpt_path):
        load_wrapper(ckpt_path)

    def save_checkpoint(self, ckpt_path):
        save_wrapper(ckpt_path)
        
    
    # null methods covered by __init__
    def build_model(self):
        return

    def get_input_placeholder(self):
        return {}
    
    def _discard_final_states(self, data_dict):
        # remove terminated states from dictionary
        for key in data_dict:
            _temp_item = np.array(data_dict[key])
            
            if _temp_item.ndim != 0:
                # if the shape divides by the episode length + 1
                if (_temp_item.shape[0] \
                    % (self.args.episode_length + 1)) == 0:
                    
                    _temp_data = np.reshape(_temp_item,
                        [-1, self.args.episode_length + 1,
                        *_temp_item.shape[1:]])
                        
                    if key in ['motivations']:
                        _temp_data = _temp_data[:,1:]
                    else:
                        _temp_data = _temp_data[:,:-1]
                    data_dict[key] = np.reshape(_temp_data, 
                        [-1, *_temp_item.shape[1:]]
                    )
                    
        return data_dict