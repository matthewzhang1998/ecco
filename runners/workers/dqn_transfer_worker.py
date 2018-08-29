#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 11:43:34 2018

@author: matthewszhang
"""
import time
import copy
import tensorflow as tf

from runners.workers import base_worker
from env.env_utils import play_episode_with_env
from util import parallel_util, logger

class worker(base_worker.worker):
    def run(self):
        self._build_model()

        while True:
            next_task = self._task_queue.get(block=True)
            
            if next_task[0] == parallel_util.WORKER_RUNNING:
                
                self._num_envs_required = int(next_task[1]['num_envs'])
                print(self._num_envs_required)
                _rollout_model = next_task[1]['rollout_model']
                    
                # collect rollouts
                traj_episode = self._play(_rollout_model)
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
        self._network = {
            key: self._network_type[key](
                self.args, self._session, name_scope,
                self._observation_size, self._action_size,
                self._action_distribution
            ) for key in self._network_type
        }

        for key in self._network:
            self._network[key].build_model()
        self._session.run(tf.global_variables_initializer())

    
    def _play(self, _rollout_model):
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
            
        self.control_info = \
            {'use_default_goal':True, 'use_default_states':True,
             'use_cached_environments':self.args.cache_environments}
        self.control_info['rollout_model'] = _rollout_model
        
        traj_episode = play_episode_with_env(
            self._envs[:self._num_envs_required], self._act,
            self.control_info
        )
        
        return traj_episode

    def _act(self, data_dict,
             control_info={'use_random_action': False,
                           'use_default_goal':True,
                           'use_default_states':True,
                           'get_dummy_goal':False,
                           'rollout_model':'final'}):

        # call the policy network
        _rollout_model = control_info['rollout_model']
        return self._network[_rollout_model].act(data_dict, control_info)
    
    def _set_weights(self, network_weights):
        for key in network_weights:
            self._network[key].set_weights(network_weights[key])