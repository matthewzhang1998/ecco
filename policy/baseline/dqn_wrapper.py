#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 17:17:48 2018

@author: matthewszhang
"""

# import all baseline dependencies
import os
import tempfile

import tensorflow as tf
import numpy as np

import baselines.common.tf_util as U
from baselines.common.tf_util import load_state, save_state
from baselines.common.schedules import LinearSchedule

from baselines import deepq
from baselines.deepq.deepq import ActWrapper
from baselines.deepq.replay_buffer import ReplayBuffer, \
    PrioritizedReplayBuffer
from baselines.deepq.utils import ObservationInput

from baselines.deepq.models import build_q_func

def init_wrapper(env,
    network_type,
    lr=1e-4,
    gamma=1.0,
    param_noise=True,
    buffer_size=int(1e5),
    prioritized_replay_alpha=.6,
    prioritized_replay=True,
    prioritized_replay_beta_iters=None,
    prioritized_replay_beta=.4,
    exploration_fraction=.1,
    total_timesteps=int(1e6),
    exploration_final_eps=0.02,
    **network_kwargs):
    # decomposes baseline deepq into initialize and inference components
    # basically copied from deepqn repository
    
    # see baselines repo for concise param documentation
    
    q_func = build_q_func(network_type, **network_kwargs)
    
    observation_space = env.observation_space
    def make_obs_ph(name):
        return ObservationInput(observation_space, name=name)

    act, train, update_target, debug = deepq.build_train(
        make_obs_ph=make_obs_ph,
        q_func=q_func,
        num_actions=env.action_space.n,
        optimizer=tf.train.AdamOptimizer(learning_rate=lr),
        gamma=gamma,
        grad_norm_clipping=10,
        param_noise=param_noise
    )

    act_params = {
        'make_obs_ph': make_obs_ph,
        'q_func': q_func,
        'num_actions': env.action_space.n,
    }

    act = ActWrapper(act, act_params)
  
    # Create the replay buffer
    
    # WARNING: do not use internal replay buffer, use baselines only for
    # stability reasons
    
    if prioritized_replay:
        replay_buffer = PrioritizedReplayBuffer(
            buffer_size, alpha=prioritized_replay_alpha
        )
        if prioritized_replay_beta_iters is None:
            prioritized_replay_beta_iters = total_timesteps
        beta_schedule = LinearSchedule(prioritized_replay_beta_iters,
                                       initial_p=prioritized_replay_beta,
                                       final_p=1.0)
    else:
        replay_buffer = ReplayBuffer(buffer_size)
        beta_schedule = None
    # Create the schedule for exploration starting from 1.
    exploration = LinearSchedule(
        schedule_timesteps=int(exploration_fraction * total_timesteps),
        initial_p=1.0,
        final_p=exploration_final_eps
    )

    # Initialize the parameters and copy them to the target network.
    U.initialize()
    update_target()
    
    # return hashed objects
    return {
    'train_function': train,
    'act_function': act,
    'replay_buffer': replay_buffer, 
    'update_target_function': update_target,
    'exploration_scheme': exploration,
    'beta_schedule': beta_schedule
    }   

def act_wrapper(env, action_function,
    exploration, obs,
    replay_buffer, timestep, 
    reset=False, param_noise=True):
    kwargs = {}
    if not param_noise:
        update_eps = exploration.value(timestep)
        update_param_noise_threshold = 0.
    else:
        update_eps = 0.
        update_param_noise_threshold = -np.log(
            1. - exploration.value(timestep) + exploration.value(timestep)
            / float(env.action_space.n)
        )
        kwargs['reset'] = reset
        kwargs['update_param_noise_threshold'] = update_param_noise_threshold
        kwargs['update_param_noise_scale'] = True
    action = action_function(
        np.array(obs)[None], update_eps=update_eps, **kwargs
    )[0]
    
    return action
    
def train_wrapper(replay_buffer, beta_schedule, prioritized_replay,
    prioritized_replay_eps, timestep, update_target_function,
    batch_size, train=True, target_update=False):
    if train:
        # Minimize the error in Bellman's equation on a batch 
        # sampled from replay buffer.
        if prioritized_replay:
            experience = replay_buffer.sample(
                batch_size, beta=beta_schedule.value(timestep)
            )
            (obses_t, actions, rewards, obses_tp1,
             dones, weights, batch_idxes) = experience
        else:
            obses_t, actions, rewards, obses_tp1, dones = \
                replay_buffer.sample(batch_size)
            weights, batch_idxes = np.ones_like(rewards), None
        td_errors = train(obses_t, actions, rewards, obses_tp1, dones, weights)
        if prioritized_replay:
            new_priorities = np.abs(td_errors) + prioritized_replay_eps
            replay_buffer.update_priorities(batch_idxes, new_priorities)

    if target_update:
        # Update target network periodically.
        update_target_function()
        
    
def save_wrapper(checkpoint_path=None):
    with tempfile.TemporaryDirectory() as td:
        td = checkpoint_path or td
    
        model_file = os.path.join(td, "model")
        save_state(model_file)
        
def load_wrapper(load_path=None, checkpoint_path=None):
    with tempfile.TemporaryDirectory() as td:
        td = checkpoint_path or td
    
        if tf.train.latest_checkpoint(td) is not None:
            model_file = os.path.join(td, "model")
            load_state(model_file)
            
        elif load_path is not None:
            load_state(load_path)
            
        else:
            raise Warning("Baselines DQN: no model file found")
        