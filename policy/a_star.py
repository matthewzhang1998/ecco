#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 21 14:15:34 2018

@author: matthewszhang

This file implements a version of A* that is also tractable in continuous observation spaces
"""
import numpy as np
from . import dqn_base

class model(dqn_base.model):
    def __init__(self, *args, **kwargs):
        super(model, self).__init__(*args, **kwargs)

    def _generate_trajectories(self):
        total_trajectory_list = []
        for env in self.environments:
            env_trajectory_list = []
            open_list = {}
            closed_list = {}

            state = env.get_info()
            base_name = 'node'
            node_num = 0
            name = base_name + str(node_num)
            node_num += 1
            num_appended = 0

            node = {'parent': None, 'info': state,
                'trajectory_length': 1, 'action': 0,
                'cost': env.a_star_cost(state)}
            open_list[name] = node

            maximum_traj = self.args.lookahead_increment

            while True:
                name = base_name + str(node_num)
                min_cost = 10000.0
                min_node = None
                construct_trajectory = False

                for node_ix in open_list:
                    node = open_list[node_ix]
                    if node['trajectory_length'] >= maximum_traj:
                        closed_list[node_ix] = node
                        del open_list[node_ix]
                        traj = self.construct_trajectory(
                            closed_list, node_ix, env
                        )
                        env_trajectory_list.append(traj)
                        construct_trajectory = True
                        num_appended += 1
                        break

                    elif node['cost'] < min_cost:
                        min_cost = node['cost']
                        min_node = node_ix

                if construct_trajectory:
                    if len(env_trajectory_list) >= self.args.transfer_num_traj:
                        break
                    elif num_appended % self.args.transfer_shuffle_freq == 0:
                        env.shuffle()
                    continue

                node = open_list[min_node]
                closed_list[min_node] = node
                del open_list[min_node]

                for action in range(self.action_size):
                    env.set_info(node['info'])
                    action = np.array(action)
                    env.step(action)
                    next_state = env.get_info()

                    is_seen = False
                    for next_node_ix in closed_list:
                        same_state = np.array_equal(
                                closed_list[next_node_ix]['info']['init_state'],
                                next_state['init_state']
                            ) & np.array_equal(
                                closed_list[next_node_ix]['info']['fixed_state'],
                                next_state['fixed_state']
                            )

                        if same_state:
                            is_seen = True
                            break

                    if not is_seen:
                        name = base_name + str(node_num)
                        trajectory_length = node['trajectory_length'] + 1

                        new_node = {'parent': node_ix, 'info': next_state,
                            'trajectory_length': trajectory_length,
                            'action': action, 'cost': env.a_star_cost(next_state)
                        }

                        open_list[name] = new_node
                        node_num += 1
            total_trajectory_list.extend(env_trajectory_list)
        return total_trajectory_list

    def construct_trajectory(self, closed_list, start_ix, env):
        trajectory = []

        final_action = 0 # dummy action for final state
        final_node = closed_list[start_ix]

        obs = env.get_obs_from_info(final_node['info'])
        trajectory.append({'start_state':obs, 'action': final_action})

        final_action = final_node['action']
        parent = final_node['parent']

        while parent is not None:
            node = closed_list[parent]
            obs = env.get_obs_from_info(final_node['info'])

            trajectory.append({
                'start_state':obs, 'action': final_action, 'reward': 0
            })
            final_action = final_node['action']
            parent = node['parent']

        return list(reversed(trajectory))

    def set_weights(self, weights):
        return

    def get_weights(self):
        return {}

    def set_environments(self, environments):
        self.environments = environments