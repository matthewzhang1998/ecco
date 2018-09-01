#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 12:35:51 2018

@author: matthewszhang
"""
'''
CREDIT TO TINGWU WANG FOR THIS CODE
'''


import numpy as np
import multiprocessing
import init_path
import os.path as osp
import tensorflow as tf
from util import parallel_util
from util import whitening_util
from util import replay_buffer
from util import logger
from util import misc_utils
from env import env_register


class base_trainer(multiprocessing.Process):

    def __init__(self, args, network_type, task_queue, result_queue,
                 name_scope='trainer'):
        multiprocessing.Process.__init__(self)
        self.args = args
        self._name_scope = name_scope

        # the base agent
        self._base_path = init_path.get_abs_base_dir()

        # used to save the checkpoint files
        self._iteration = 0
        self._best_reward = -np.inf
        self._timesteps_so_far = 0
        self._npr = np.random.RandomState(args.seed)
        self._task_queue = task_queue
        self._result_queue = result_queue
        self._network_type = network_type

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
            
            print(next_task)
            
            if next_task[0] is None or next_task[0] == parallel_util.END_SIGNAL:
                # kill the learner
                self._task_queue.task_done()
                break

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
                    next_task[1]['data'], next_task[1]['training_info']
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

    def get_experiment_name(self):
        return self.args.task + '_' + self.args.exp_id

    def _build_session(self):
        # TODO: the tensorflow configuration
        config = tf.ConfigProto(device_count={'GPU': 0})  # only cpu version
        self._session = tf.Session(config=config)

    def _build_models(self):
        self._build_session()
        
        self._network = self._network_type(
                        self.args, self._session, self._name_scope,
                        self._observation_size, self._action_size,
                        self._action_distribution
                    )
        self._network.build_model()
        self._session.run(tf.global_variables_initializer())
        self._saver = tf.train.Saver()

    def _set_io_size(self):
        self._observation_size, self._action_size, \
            self._action_distribution = \
            env_register.io_information(self.args.task)
            
    def _init_replay_buffer(self):
        if self._action_distribution is 'discrete':
            _action_size = None # single discrete action only
        else:
            _action_size = self._action_size
        
        self._replay_buffer = replay_buffer.build_replay_buffer(
            self.args, self._observation_size, _action_size,
            save_reward=True
        )

    def set_weights(self, network_weights):
        for key in network_weights:
            self._network[key].set_weights(network_weights[key])

    def _init_whitening_stats(self):
        self._whitening_stats = \
            whitening_util.init_whitening_stats(['state', 'diff_state'])

    def _update_whitening_stats(self, rollout_data,
                                key_list=['state', 'diff_state']):
        # collect the info
        for key in key_list:
            whitening_util.update_whitening_stats(
                self._whitening_stats, rollout_data, key
            )

    def _preprocess_data(self, rollout_data):
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

        for key in self._network.required_keys:
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

    def _restore_all(self):
        # TODO
        pass

    def _save_all(self):
        # TODO
        pass

    def _get_weights(self):
        weights = self._network.get_weights()

        return weights

    def _update_parameters(self, rollout_data):
        raise NotImplementedError