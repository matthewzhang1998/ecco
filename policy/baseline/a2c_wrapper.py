#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 10:49:01 2018

@author: matthewszhang
"""
import functools
import tensorflow as tf
import numpy as np

from baselines.common import tf_util
from baselines.common.policies import build_policy

from baselines.a2c.utils import Scheduler

from tensorflow import losses

class Model(object):

    def __init__(self, policy, env, nsteps,
            ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5, lr=7e-4,
            alpha=0.99, epsilon=1e-5, total_timesteps=int(80e6),
            lrschedule='linear'):

        sess = tf_util.get_session()
        nenvs = env.num_envs
        nbatch = nsteps


        with tf.variable_scope('a2c_model', reuse=tf.AUTO_REUSE):
            step_model = policy(nenvs, 1, sess)
            train_model = policy(nbatch, nsteps, sess)

        A = tf.placeholder(train_model.action.dtype, train_model.action.shape)
        ADV = tf.placeholder(tf.float32, [nbatch])
        R = tf.placeholder(tf.float32, [nbatch])
        LR = tf.placeholder(tf.float32, [])

        neglogpac = train_model.pd.neglogp(A)
        entropy = tf.reduce_mean(train_model.pd.entropy())

        pg_loss = tf.reduce_mean(ADV * neglogpac)
        vf_loss = losses.mean_squared_error(tf.squeeze(train_model.vf), R)

        loss = pg_loss - entropy*ent_coef + vf_loss * vf_coef

        params = tf.trainable_variables()
        grads = tf.gradients(loss, params)
        if max_grad_norm is not None:
            grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads = list(zip(grads, params))
        trainer = tf.train.RMSPropOptimizer(learning_rate=LR, decay=alpha, epsilon=epsilon)
        _train = trainer.apply_gradients(grads)

        lr = Scheduler(v=lr, nvalues=total_timesteps, schedule=lrschedule)

        def train(obs, states, rewards, masks, actions, values):
            advs = rewards - values
            for step in range(len(obs)):
                cur_lr = lr.value()

            td_map = {train_model.X:obs, A:actions,
                      ADV:advs, R:rewards, LR:cur_lr}
            if states is not None:
                td_map[train_model.S] = states
                td_map[train_model.M] = masks
            policy_loss, value_loss, policy_entropy, _ = sess.run(
                [pg_loss, vf_loss, entropy, _train],
                td_map
            )
            return policy_loss, value_loss, policy_entropy


        self.train = train
        self.train_model = train_model
        self.step_model = step_model
        self.step = step_model.step
        self.value = step_model.value
        self.initial_state = step_model.initial_state
        self.save = functools.partial(tf_util.save_variables, sess=sess)
        self.load = functools.partial(tf_util.load_variables, sess=sess)
        tf.global_variables_initializer().run(session=sess)

def init_wrapper(environment, network_type, number_steps,
                 entropy_coefficient, vf_coefficient, gradient_clipping,
                 learning_rate, alpha, epsilon, total_timesteps,
                 learning_rate_schedule = 'constant',
                 **network_kwargs):
    policy = build_policy(environment, network_type, **network_kwargs)
    
    model = Model(
        policy=policy, env=environment, nsteps=number_steps,
        ent_coef=entropy_coefficient, vf_coef=vf_coefficient,
        max_grad_norm=gradient_clipping, lr=learning_rate,
        alpha=alpha, epsilon=epsilon, total_timesteps=total_timesteps,
        lrschedule=learning_rate_schedule
    )
    
    return {
        'policy': policy,
        'model': model        
    }
    
def act_wrapper(model, data_dict, masks):
    actions, values, states, _ = model.step(
        data_dict['start_state'],
        S = data_dict['hidden_states'],
        M = masks
    )
    
    if states is None:
        states = np.zeros_like(data_dict['start_state'])
    
    return {
        'actions': actions,
        'value': values,
        'hidden_states': states     
    }
    
def train_wrapper(model, data_dict, masks):
    policy_loss, value_loss, policy_entropy = \
        model.train(data_dict['start_state'],
            data_dict['hidden_states'],
            data_dict['advantage'],
            masks,
            data_dict['actions'],
            data_dict['value']
        )
        
    return {
        'a2c_policy_loss': policy_loss,
        'a2c_vf_loss': value_loss,
        'a2c_policy_entropy': policy_entropy
    }
    
def save_wrapper(baselines_model, save_path=None):
    if save_path is not None:
        baselines_model.save(save_path)
    
def load_wrapper(baselines_model, load_path=None):
    if load_path is not None:
        baselines_model.load(load_path)
    