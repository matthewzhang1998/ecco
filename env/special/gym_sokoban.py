#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 12 22:16:51 2018

@author: matthewszhang
"""

import init_path
import time
from env import base_env_wrapper as bew
from env import env_register
import numpy as np
from util import logger

class env(bew.base_env):
    MC = ['gym_sokoban']
    
    def __init__(self, env_name, rand_seed, maximum_length, misc_info):
        super(env, self).__init__(
            env_name, rand_seed, maximum_length, misc_info
        )
        self._base_path = init_path.get_abs_base_dir()
        self._env.env.penalty_for_step = 0.
        self.n_boxes = 3
        
        if 'easy' in self._env_name:
            self.n_boxes = 1
            
        self._last_reward = 0
        self._episode_reward = 0

    def step(self, action):
        action = int(action) # get int from action

        self._env.step(action)
        time.sleep(1/20)
        ob = self._one_hot(self._env.env.room_state)

        # flatten observation
        ob = np.reshape(ob, [-1])

        self._current_step += 1

        data_dict = {
            'start_state': self._old_ob,
            'action': action,
            'end_state': ob
        }

        reward = self._reward(data_dict)

        if self._current_step >= self._maximum_length:
            done = True
        else:
            done = False # will raise warnings -> set logger flag to ignore
        self._old_ob = np.array(ob)
        self._episode_reward += reward

        next_info = self.get_info()
               
        return ob, reward, done, {}
    
    def reset(self):
        self._env.reset()

        self._episode_reward = 0
        self._env.env.reward_last = 0
        
        self._last_reward = 0
        
        self._keep_n_boxes(self.n_boxes)
        
        ob = self._one_hot(self._env.env.room_state)
        ob = np.reshape(ob, [-1])
        
        self._current_step = 0
        self._old_ob = ob
        return ob, 0, False, {}
        
    def _build_env(self):
        import gym
        import gym_sokoban
        self._current_version = gym.__version__
        _env_name = {
            'gym_sokoban':'Sokoban-v2',
            'gym_sokoban_small':'Sokoban-small-v1',
            'gym_sokoban_large':'Sokoban-large-v2',
            'gym_sokoban_huge':'Sokoban-huge-v0',
            'gym_sokoban_tiny_world': 'TinyWorld-Sokoban-v2',
            'gym_sokoban_small_tiny_world': 'TinyWorld-Sokoban-small-v1',
            'gym_sokoban_small_tiny_world_easy': 'TinyWorld-Sokoban-small-v1',
        }
        
        # make the environments
        self._env = gym.make(_env_name[self._env_name])
        self._env_info = env_register.get_env_info(self._env_name)
    
    def _one_hot(self, ob):
        one_hot_ob = (np.arange(ob.max()+1) == ob[...,None]).astype(int)
        
        return one_hot_ob
    
    def get_supervised_goal(self):
        return None
    
    def _keep_n_boxes(self, num_boxes):
        targets = np.where(self._env.env.room_fixed == 2)
        boxes = np.where(
            (self._env.env.room_state == 3)|(self._env.env.room_state == 4)
        )
        
        for i in range(num_boxes, len(targets[0])):
            self._env.env.room_fixed[targets[0][i], targets[1][i]] = 1
            if self._env.env.room_state[targets[0][i], targets[1][i]] == 2:
                self._env.env.room_state[targets[0][i], targets[1][i]] = 1
            self._env.env.room_state[boxes[0][i], boxes[1][i]] = 1

    def _reward(self, data_dict):
        total_targets = np.sum(
            np.where(self._env.env.room_fixed == 2, 1, 0)
        )

        start_state = self._undo_one_hot(
            data_dict['start_state']
        )
        end_state = self._undo_one_hot(
            data_dict['end_state']
        )

        boxes_before = np.sum(
            np.where(start_state == 3, 1, 0)
        )

        boxes_after = np.sum(
            np.where(end_state == 3, 1, 0)
        )

        reward = .1 * (boxes_after - boxes_before)

        if (boxes_after == total_targets) and \
            (boxes_before != total_targets):
            reward += 1.0

        elif (boxes_after != total_targets) and \
            (boxes_before == total_targets):
            reward -= 1.0

        return reward

    def _undo_one_hot(self, state):
        _room_dim_shape = self._env.env.dim_room
        geo_state_array = np.reshape(
            state, [*_room_dim_shape] + [-1]
        )

        ground_truth_state = np.argmax(geo_state_array, axis=-1)
        return ground_truth_state

    def fdynamics(self, data_dict):
        action = float(data_dict['action'])
        
        state = _undo_one_hot(data_dict['start_state'])
        
        self._env.env.room_state = ground_truth_state
        
        # no act, creates bugs

    def set_info(self, info):
        if 'fixed_state' in info:
            self._env.env.room_fixed = np.copy(info['fixed_state'])

        if 'init_state' in info:
            self._env.env.room_state = np.copy(info['init_state'])

    def get_info(self):
        return {'fixed_state': np.copy(self._env.env.room_fixed),
            'init_state': np.copy(self._env.env.room_state)}

    def get_obs_from_info(self, info):
        return np.reshape(self._one_hot(info['init_state']), [-1])

    def shuffle(self):
        temp_state = np.copy(self._env.env.room_state)
        floor = np.where(temp_state == 1)
        player = np.where(temp_state == 5)

        if len(floor) > 0:
            new_player_ind = self._npr.randint(len(floor))
            temp_state[player] = self._env.env.room_fixed[player]
            temp_state[floor[0][new_player_ind]][floor[1][new_player_ind]] = \
                5

        floor = np.where(temp_state == 1)
        boxes = np.where(temp_state == 4|temp_state == 3)

        if len(floor) > 0:
            maximum_length = min(len(floor), len(boxes))
            new_box_inds = self._npr.randint(0, len(floor), maximum_length)
            for box in range(maximum_length):
                temp_state[boxes[0][box]][boxes[1][box]] = \
                    self._env.env.room_fixed[boxes[0][box]][[boxes][1][box]]

                ix = new_box_inds[box]

                temp_state[floor[0][ix]][floor[1][ix]] = 4

    def a_star_cost(self, info):
        room_state = info['init_state']
        room_fixed = info['fixed_state']

        boxes = np.where(
            (room_state == 4)|(room_state == 3)
        )

        player = np.where(
            room_state == 5
        )

        targets = np.where(
            (room_fixed == 2)
        )

        total_manhattan_distances = 0.0

        # box term
        for i in range(len(boxes[0])):
            min_manhattan_dist = 1000.0
            manhattan_dist_temp = 0.0
            for j in range(len(targets[0])):
                manhattan_dist_temp = \
                    np.abs(boxes[0][i] - targets[0][j]) + \
                    np.abs(boxes[1][i] - targets[1][j])

                if manhattan_dist_temp < min_manhattan_dist:
                    min_manhattan_dist = manhattan_dist_temp
            total_manhattan_distances += min_manhattan_dist

        # player term
        min_manhattan_dist = 1000.0
        for i in range(len(boxes[0])):
            manhattan_dist_temp = \
                np.abs(boxes[0][i] - player[0][0]) + \
                np.abs(boxes[0][i] - player[1][0])

            if manhattan_dist_temp < min_manhattan_dist:
                min_manhattan_dist = manhattan_dist_temp

        total_manhattan_distances += min_manhattan_dist

        return total_manhattan_distances
