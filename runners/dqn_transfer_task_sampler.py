#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 10:34:24 2018

@author: matthewszhang
"""

from . import task_sampler
from util import parallel_util, logger

class sampler(task_sampler.sampler):
    def _rollout_with_workers(self, rollout_info):
        _rollout_model = rollout_info['rollout_model']
        
        self._current_iteration += 1
        rollout_data = []
        
        timesteps_needed = self.args.dqn_batch_size \
            if _rollout_model == 'base' else self.args.batch_size
        num_timesteps_received = 0
        
        if _rollout_model == 'transfer':
            _rollout_model = 'base'
        
        while True:
            num_estimated_episode = int(
                timesteps_needed/self.args.episode_length
            )
            
            num_envs_per_worker = \
                num_estimated_episode / self.args.num_workers
                
            worker_infos = {'num_envs': num_envs_per_worker,
                           'rollout_model': _rollout_model}
            
            for _ in range(self.args.num_workers):
                self._task_queue.put((parallel_util.WORKER_RUNNING,
                                      worker_infos))
                
            self._task_queue.join()

            # collect the data
            for _ in range(num_estimated_episode):
                traj_episode = self._result_queue.get()
                rollout_data.append(traj_episode)
                num_timesteps_received += len(traj_episode['rewards'])
                
            logger.info('{} timesteps from {} episodes collected'.format(
                num_timesteps_received, len(rollout_data))
            )

            if num_timesteps_received >= timesteps_needed:
                break
            
        return {'data': rollout_data}