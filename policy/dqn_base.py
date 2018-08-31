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

from policy.baseline.dqn_wrapper import init_wrapper, \
     act_wrapper, train_wrapper, load_wrapper, save_wrapper
     
from env.env_register import reverse_gym_wrapper
from util import tf_utils
from util.baseline_utils import get_dqn_network_kwargs_from_namespace
from .baseline_base import baseline_model

class model(baseline_model):
    def __init__(self, parse_args, session, name_scope, *args, **kwargs):
        self.args = parse_args
        
        self._session = session
        self._name_scope = name_scope
        self._not_actor_network = True
        
        self._dummy_environment = reverse_gym_wrapper(self.args.task)
        self.timesteps_so_far = 0
        
        _network_kwargs = get_dqn_network_kwargs_from_namespace(self.args)
        
        total_timesteps = self.args.train_dqn_iterations * \
            self.args.dqn_batch_size + self.args.train_transfer_iterations * \
            self.args.batch_size
        
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
                exploration_fraction = self.args.dqn_epsilon * \
                     self.args.train_dqn_iterations * \
                     self.args.dqn_batch_size / \
                     (total_timesteps),
                exploration_final_eps = self.args.dqn_min_epsilon,
                grad_norm_clipping = self.args.dqn_gradient_max,
                total_timesteps = total_timesteps,
                **_network_kwargs
            )
            self._set_var_list()
    
    def train(self, data_dict, replay_buffer, training_info={}):
        dones = np.tile(
            np.array([0] * (self.args.episode_length - 1) + [1]),
            [len(data_dict['start_state']) // self.args.episode_length]
        )
        
        data_dict = self._discard_final_states(data_dict)
        
        for i in range(len(data_dict['start_state'])):
            self.baseline_dqn_dict['replay_buffer'].add(
                np.array(data_dict['start_state'][i]),
                np.array(data_dict['actions'][i]),
                np.array(data_dict['rewards'][i]),
                np.array(data_dict['end_state'][i]),
                np.array(dones[i], dtype=np.float32)
            )
        
        stats = {'mean_rewards': np.mean(data_dict['rewards'])}
        
        # this is necessary because the train and act networks are
        # different class instances
        if self._not_actor_network:
            self.timesteps_so_far += len(data_dict['start_state'])
        
        _target_update = (self.timesteps_so_far % \
            self.args.dqn_update_target_steps) == 0
        
        td_errors = []
                          
        for _ in range(self.args.dqn_update_epochs):
            td_errors.append(
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
            )
                
        stats['td_errors'] = np.mean(np.array(td_errors)**2)
        stats['epsilon'] = \
            self.baseline_dqn_dict['exploration_scheme'].value(
                self.timesteps_so_far
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
