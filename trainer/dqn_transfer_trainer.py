#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 10:32:26 2018

@author: matthewszhang
"""
import numpy as np
import tensorflow as tf
from util import parallel_util
from util import misc_utils
from util import logger
from util import replay_buffer

import os.path as osp

from .base_trainer import base_trainer

class trainer(base_trainer):        
    def run(self):
        self._set_io_size()
        self._build_models()
        self._init_replay_buffer()
        self._init_whitening_stats()

        # load the model if needed
        if self.args.ckpt_name is not None:
            self._restore_all()

        # the main training process
        while True:
            next_task = self._task_queue.get()

            if next_task[0] is None or \
                next_task[0] == parallel_util.END_SIGNAL:
                # kill the learner
                self._task_queue.task_done()
                break

            elif next_task[0] == parallel_util.TRAINER_SET_WEIGHTS:
                self._set_weights(next_task[1])

            elif next_task[0] == parallel_util.START_SIGNAL:
                # get network weights
                self._task_queue.task_done()
                self._result_queue.put(self._get_weights())

            elif next_task[0] == parallel_util.RESET_SIGNAL:
                self._task_queue.task_done()
                self._init_whitening_stats()
                self._timesteps_so_far = 0
                self._iteration = 0
                
            elif next_task[0] == parallel_util.SAVE_SIGNAL:
                _save_root = next_task[1]['net']
                _log_path = logger._get_path()
                
                _save_extension = _save_root + \
                    "_{}_{}.ckpt".format(
                        self._name_scope, self._timesteps_so_far
                    )
                    
                _save_dir = osp.join(_log_path, _save_extension)
                self._saver.save(self._session, _save_dir)

            else:
                # training
                assert next_task[0] == parallel_util.TRAIN_SIGNAL
                stats = self._update_parameters(
                    next_task[1]['data'],
                    next_task[1]['training_info']
                )
                self._task_queue.task_done()

                self._iteration += 1
                return_data = {
                    'network_weights': self._get_weights(),
                    'stats': stats,
                    'totalsteps': self._timesteps_so_far,
                    'iteration': self._iteration
                }
                self._result_queue.put(return_data)

    def _set_weights(self, network_weights):
        for key in network_weights:
            self._network[key].set_weights(network_weights[key])
        
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
        self._session.run(tf.global_variables_initializer())
        self._saver = tf.train.Saver()
            
    def _init_replay_buffer(self):
        if self._action_distribution is 'discrete':
            _action_size = None # single discrete action only
        else:
            _action_size = self._action_size
        
        # dummy buffers except for final
        self._replay_buffer = {key: replay_buffer.replay_buffer(
                False, 0, 0, 0, 1, 0, 0, False
            ) for key in self._network
        }
        
        self._replay_buffer['final'] = replay_buffer.build_replay_buffer(
            self.args, self._observation_size, _action_size,
            save_reward=True
        )
        
    def _get_weights(self):
        weights = {
            key: self._network[key].get_weights() for key in self._network
        }

        return weights
        
    def _update_parameters(self, rollout_data, training_info):
        # get the observation list
        self._update_whitening_stats(rollout_data)
        train_model = training_info['train_model']
        training_data = self._preprocess_data(rollout_data, train_model)
        training_stats = {'avg_reward': training_data['avg_reward']}

        if 'train_net' in training_info:
            train_net = training_info['train_net']
            
        else:
            train_net = None
            

        # train the policy
        stats_dictionary, data_dictionary = \
            self._network[train_model].train(
                training_data,  self._replay_buffer[train_model],
                train_net
            )
            
        training_stats.update(stats_dictionary)
        
        self._replay_buffer[train_model].add_data(data_dictionary)
        
        return training_stats
    
    def _preprocess_data(self, rollout_data, model):
        """ @brief:
                Process the data, collect the element of
                ['start_state', 'end_state', 'action', 'reward', 'return',
                 'ob', 'action_dist_mu', 'action_dist_logstd']
        """
        # get the observations
        training_data = {}

        # get the returns (might be needed to train policy)
        for i_episode in rollout_data:
            i_episode["returns"] = \
                misc_utils.get_return(i_episode["rewards"],
                                      self.args.gamma_max)

        for key in self._network[model].required_keys:
            training_data[key] = np.concatenate(
                [i_episode[key][:] for i_episode in rollout_data]
            )

        # record the length
        training_data['episode_length'] = \
            [len(i_episode['rewards']) for i_episode in rollout_data]

        # get the episodic reward
        for i_episode in rollout_data:
            i_episode['episodic_reward'] = sum(i_episode['rewards'])
        avg_reward = np.mean([i_episode['episodic_reward']
                              for i_episode in rollout_data])
        logger.info('Mean reward: {}'.format(avg_reward))

        training_data['whitening_stats'] = self._whitening_stats
        training_data['avg_reward'] = avg_reward
        training_data['rollout_data'] = rollout_data

        # update timesteps so far
        self._timesteps_so_far += len(training_data['actions'])
        return training_data
