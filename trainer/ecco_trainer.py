#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 14:38:07 2018

@author: matthewszhang
"""
import init_path
from .base_trainer import base_trainer

class trainer(base_trainer):

    def __init__(self, args, network_type, task_queue, result_queue,
                 name_scope='trainer'):
        # the base agent
        super(trainer, self).__init__(
            args=args, network_type=network_type,
            task_queue=task_queue, result_queue=result_queue,
            name_scope=name_scope
        )
        
        self._base_path = init_path.get_abs_base_dir()

    def _update_parameters(self, rollout_data, training_info):
        # get the observation list
        self._update_whitening_stats(rollout_data)
        training_data = self._preprocess_data(rollout_data)
        training_stats = {'avg_reward': training_data['avg_reward']}

        if 'train_net' in training_info:
            train_net = training_info['train_net']
            
        else:
            train_net = None

        # train the policy
        stats_dictionary, data_dictionary = \
            self._network.train(
                training_data,  self._replay_buffer,
                train_net
            )
            
        training_stats.update(stats_dictionary)
        
        self._replay_buffer.add_data(data_dictionary)
        
        return training_stats