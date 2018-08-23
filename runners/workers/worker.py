#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 12 15:27:54 2018

@author: matthewszhang
"""
'''
CREDIT TO TINGWU WANG FOR THIS CODE
'''

import numpy as np
import tensorflow as tf
import multiprocessing
import time
import copy
import init_path
from util import logger
from util import parallel_util
from env import env_register
from env.env_utils import play_episode_with_env

class worker(multiprocessing.Process):

    def __init__(self, args, observation_size, action_size,
                 action_distribution,
                 network_type, task_queue, result_queue, worker_id,
                 name_scope='worker'):

        # the multiprocessing initialization
        multiprocessing.Process.__init__(self)
        self.args = args
        self._name_scope = name_scope
        self._worker_id = worker_id
        self._network_type = network_type
        self._npr = np.random.RandomState(args.seed + self._worker_id)

        self._observation_size = observation_size
        self._action_size = action_size
        self._action_distribution = action_distribution
        self._task_queue = task_queue
        self._result_queue = result_queue
        
        self._num_envs_required = 1
        self._env_start_index = 0
        self._envs = []
        self._environments_cache = []

        logger.info('Worker {} online'.format(self._worker_id))
        self._base_dir = init_path.get_base_dir()
        self._build_env()
        self.control_info = \
            {'use_default_goal':True, 'use_default_states':True,
             'use_cached_environments':self.args.cache_environments}

    def run(self):
        self._build_model()

        while True:
            next_task = self._task_queue.get(block=True)
            
            if next_task[0] == parallel_util.WORKER_RUNNING:
                
                self._num_envs_required = int(next_task[1])
                    
                # collect rollouts
                traj_episode = self._play()
                self._task_queue.task_done()
                for episode in traj_episode:
                    self._result_queue.put(episode)
                    

            elif next_task[0] == parallel_util.AGENT_SET_WEIGHTS:
                # set parameters of the actor policy
                self._set_weights(next_task[1])
                time.sleep(0.001)  # yield the process
                self._task_queue.task_done()

            elif next_task[0] == parallel_util.END_ROLLOUT_SIGNAL or \
                    next_task[0] == parallel_util.END_SIGNAL:
                # kill all the thread
                #logger.info("kill message for worker {}".format(self._actor_id))
                logger.info("kill message for worker")
                self._task_queue.task_done()
                break
            else:
                logger.error('Invalid task type {}'.format(next_task[0]))
        return

    def _build_model(self):
        # by defualt each work has one set of networks, but potentially they
        # could have more
        self._build_session()

        name_scope = self._name_scope
        self._network = self._network_type(
            self.args, self._session, name_scope,
            self._observation_size, self._action_size,
            self._action_distribution
        )

        with tf.variable_scope(name_scope):
            self._network.build_model()
        self._session.run(tf.global_variables_initializer())

    def _build_session(self):
        # TODO: the tensorflow configuration
        config = tf.ConfigProto(device_count={'GPU': 0})  # only cpu version
        self._session = tf.Session(config=config)

    def _build_env(self):
        
        if self.args.cache_environments:
            while len(self._environments_cache) < self.args.num_cache:
                _env, self._env_info = env_register.make_env(
                        self.args.task, self._npr.randint(0, 9999),
                        self.args.episode_length,
                        {'allow_monitor': self.args.monitor \
                         and self._worker_id == 0}
                    )
                _env.reset()
                
                self._environments_cache.append(copy.deepcopy(_env))
                 
        else:
            while len(self._envs) < self._num_envs_required:
                _env, self._env_info = env_register.make_env(
                        self.args.task, self._npr.randint(0, 9999),
                        self.args.episode_length,
                        {'allow_monitor': self.args.monitor \
                         and self._worker_id == 0}
                    )
                _env.reset()   
                self._envs.append(_env)
    
    def _play(self):
        self._build_env()
        
        if self.args.cache_environments:
            self._envs = []
            start = self._env_start_index
            end = self._env_start_index + self._num_envs_required
            while end > len(self._environments_cache):
                end = end - (len(self._environments_cache) - start)
                self._envs.extend(
                    copy.deepcopy(self._environments_cache[start:])
                )
                start = 0
                
            self._env_start_index = end
            self._envs.extend(
                copy.deepcopy(self._environments_cache[start:end])
            )
        
        traj_episode = play_episode_with_env(
            self._envs, self._act,
            self.control_info
        )
        return traj_episode

    def _act(self, data_dict,
             control_info={'use_random_action': False,
                           'use_default_goal':True,
                           'use_default_states':True,
                           'get_dummy_goal':False}):

        # call the policy network
        return self._network.act(data_dict, control_info)
   
    def _set_weights(self, network_weights):
        self._network.set_weights(network_weights)
