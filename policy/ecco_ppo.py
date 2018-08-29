#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 20 20:47:46 2018

@author: matthewszhang
"""
import numpy as np
import tensorflow as tf
from . import ecco_base
from collections import defaultdict
import copy

class model(ecco_base.base_model):
    def _build_loss(self):
        # PPO clipping
        self._clipratio_list = np.array(
            [self.args.clip_manager] * \
            (self.args.maximum_hierarchy_depth - 1) + \
            [self.args.clip_actor]
        )
            
        self._value_clipratio_list = np.array(
            [self.args.value_clip_manager] * \
            (self.args.maximum_hierarchy_depth - 1) + \
            [self.args.value_clip_actor]
        )
            
        self._input_ph['log_oldp_n'] = tf.placeholder(
            tf.float32, [None, self.args.maximum_hierarchy_depth]        
        )
        
        self._input_ph['old_values'] = tf.placeholder(
            tf.float32, [None, self.args.maximum_hierarchy_depth]
        )
        
        self._input_ph['cliprange'] = tf.placeholder(
            tf.float32, [self.args.maximum_hierarchy_depth]
        )
        
        self._input_ph['value_cliprange'] = tf.placeholder(
            tf.float32, [self.args.maximum_hierarchy_depth]
        )
        
        # reduce last dim of value output
        self._tensor['output_values'] = \
            tf.reduce_mean(self._tensor['output_values'], axis=-1)
        
        # Building policy loss
        self._tensor['ratio'] = \
            tf.exp(self._tensor['output_log_p_n'] - \
                   self._input_ph['log_oldp_n'])
        
        __batch_tile_matrix = tf.cast(
            [self._input_ph['batch_size']], tf.int32
        )
        
        self._tensor['cliprange_matrix'] = \
            tf.reshape(tf.tile(self._input_ph['cliprange'],
                    __batch_tile_matrix,
                ), tf.concat([__batch_tile_matrix, [-1]], axis=0)
            )
            
        self._tensor['clipped_ratio'] = tf.clip_by_value(
            self._tensor['ratio'],
            1. - self._tensor['cliprange_matrix'],
            1. + self._tensor['cliprange_matrix']
        )
        
        # weird name so that it is recognized by both actor and manager
        self._update_operator['approximate_KL_actor_manager'] = .5 * \
            tf.reduce_mean(
                tf.square(
                     self._tensor['output_log_p_n'][:,-1] - \
                     self._input_ph['log_oldp_n'][:,-1]
                )        
            )
        
        self._update_operator['pol_loss_unclipped'] = \
            -self._tensor['ratio'] * \
            self._input_ph['advantage']
        
        self._update_operator['pol_loss_clipped'] = \
            -self._tensor['clipped_ratio'] * \
            self._input_ph['advantage']
            
        self._update_operator['pol_loss'] = tf.clip_by_value(
            tf.maximum(
                self._update_operator['pol_loss_unclipped'],
                self._update_operator['pol_loss_clipped']
            ), -self.args.pol_loss_clip, +self.args.pol_loss_clip
        )
        
        self._update_operator['surr_loss'] = tf.reduce_mean(
            self._update_operator['pol_loss']
        )
            
        self._update_operator['entropy_loss'] = -tf.reduce_mean(
            self._input_ph['entropy_coefficients'] * \
            self._tensor['output_entropy'] 
        )
        
        # Value loss
        self._tensor['value_cliprange_matrix'] = \
            tf.reshape(tf.tile(self._input_ph['value_cliprange'],
                __batch_tile_matrix
            ), tf.concat([__batch_tile_matrix, [-1]], axis=-1)
        )
        
#        self._tensor['value_clipped'] = \
#            self._input_ph['old_values'] + tf.clip_by_value(
#                self._tensor['output_values'] -
#                self._input_ph['old_values'],
#                -self._tensor['value_cliprange_matrix'],
#                self._tensor['value_cliprange_matrix']
#            )
#            
#        self._update_operator['val_loss_clipped'] = tf.square(
#            self._tensor['value_clipped'] - self._input_ph['value_target']
#        )
#        
#        self._update_operator['val_loss_unclipped'] = tf.square(
#            self._tensor['output_values'] - self._input_ph['value_target']
#        )
#        
#        self._update_operator['vf_loss'] = .5 * tf.reduce_mean(
#            tf.maximum(self._update_operator['val_loss_clipped'],
#                       self._update_operator['val_loss_unclipped'])
#        ) 
        
        self._update_operator['vf_loss'] = .5 * tf.reduce_mean(
            tf.square(
                self._tensor['output_values'] - \
                self._input_ph['value_target']
            )
        )
        
        # Aggregate and update
        self._update_operator['loss'] = \
            self._update_operator['surr_loss'] + \
            self._update_operator['entropy_loss']
        
        if self.args.joint_value_update:
            self._update_operator['loss'] += self._update_operator['vf_loss']
            
        else:
            self._update_operator['vf_update_op'] = tf.train.AdamOptimizer(
                learning_rate=self.args.value_lr,
                beta1=0.5, beta2=0.99, epsilon=1e-4
            ).minimize(self._update_operator['vf_loss'])
        
        if self.args.clip_gradients:
            self._tensor['update_op_proto'] = \
            tf.train.AdamOptimizer(
                learning_rate=self._input_ph['learning_rate']
            )
            _params = tf.trainable_variables()
            self._tensor['update_op_gradients'] = \
                tf.gradients(
                        self._update_operator['loss'], _params
                )
            self._tensor['update_op_gradients'], _ = \
                tf.clip_by_global_norm(
                    self._tensor['update_op_gradients'],
                    self.args.clip_gradient_threshold
                )
                
            self._tensor['update_op_gradients'] = list(zip(
                self._tensor['update_op_gradients'], _params
            ))
            
            self._update_operator['update_op'] = \
                self._tensor['update_op_proto'].apply_gradients(
                    self._tensor['update_op_gradients']
                )
        
        else:
            self._update_operator['update_op'] = tf.train.AdamOptimizer(
                learning_rate=self._input_ph['learning_rate']
            ).minimize(self._update_operator['loss'])
        
        # build additional ops for decoupled networks 
        self._build_decoupled_loss()
        
    def _build_decoupled_loss(self):
        
        self._update_operator['surr_loss_manager'] = tf.reduce_mean(
            self._update_operator['pol_loss'][:,:-1]
        )
            
        self._update_operator['surr_loss_actor'] = tf.reduce_mean(
            self._update_operator['pol_loss'][:,-1]
        )
            
        self._update_operator['entropy_manager'] = tf.reduce_mean(
            self._tensor['output_entropy'][:-1]
        )
        
        self._update_operator['entropy_actor'] = tf.reduce_mean(
            self._tensor['output_entropy'][-1]
        )
            
        self._update_operator['entropy_loss_manager'] = -tf.reduce_mean(
            self._input_ph['entropy_coefficients'][:-1] * \
            self._tensor['output_entropy'][:-1]
        )
        
        self._update_operator['entropy_loss_actor'] = -tf.reduce_mean(
            self._input_ph['entropy_coefficients'][-1] * \
            self._tensor['output_entropy'][-1]
        )
        
        self._update_operator['vf_loss_manager'] = tf.reduce_mean(
            tf.square(
                self._tensor['output_values'][:,:-1] - \
                self._input_ph['value_target'][:,:-1]
            )
        )
            
        self._update_operator['vf_loss_actor'] = tf.reduce_mean(
            tf.square(
                self._tensor['output_values'][:,-1] - \
                self._input_ph['value_target'][:,-1]
            )
        )
            
        # Aggregate and update
        self._update_operator['loss_manager'] = \
            self._update_operator['surr_loss_manager'] + \
            self._update_operator['entropy_loss_manager']
            
        self._update_operator['loss_actor'] = \
            self._update_operator['surr_loss_actor'] + \
            self._update_operator['entropy_loss_actor']
        
        if self.args.joint_value_update:
            self._update_operator['loss_manager'] += \
                self._update_operator['vf_loss_manager']
                
            self._update_operator['loss_actor'] += \
                self._update_operator['vf_loss_actor']
            
        else:
            self._update_operator['vf_update_op_manager'] = \
            tf.train.AdamOptimizer(
                learning_rate=self.args.value_lr,
                beta1=0.5, beta2=0.99, epsilon=1e-4
            ).minimize(self._update_operator['vf_loss'])
            
            self._update_operator['vf_update_op_actor'] = \
            tf.train.AdamOptimizer(
                learning_rate=self.args.value_lr,
                beta1=0.5, beta2=0.99, epsilon=1e-4
            ).minimize(self._update_operator['vf_loss'])
            
        if self.args.clip_gradients:
            self._tensor['update_op_proto_manager'] = \
            tf.train.AdamOptimizer(
                learning_rate=self._input_ph['learning_rate']
            )
            _params = tf.trainable_variables()
            self._tensor['update_op_gradients_manager'] = \
                tf.gradients(
                        self._update_operator['loss_manager'], _params
                )
            self._tensor['update_op_gradients_manager'], _ = \
                tf.clip_by_global_norm(
                    self._tensor['update_op_gradients_manager'],
                    self.args.clip_gradient_threshold
                )
                
            self._tensor['update_op_gradients_manager'] = list(zip(
                self._tensor['update_op_gradients_manager'], _params
            ))
            
            self._update_operator['update_op_manager'] = \
                self._tensor['update_op_proto_manager'].apply_gradients(
                    self._tensor['update_op_gradients_manager']
                )
            
        else:
           self._update_operator['update_op_manager'] = \
            tf.train.AdamOptimizer(
                learning_rate=self._input_ph['learning_rate']
            ).minimize(self._update_operator['loss_manager'])
            
        self._update_operator['update_op_actor'] = tf.train.AdamOptimizer(
            learning_rate=self._input_ph['learning_rate']
        ).minimize(self._update_operator['loss_actor'])
        
    def train(self, data_dict, replay_buffer, train_net = None):
        replay_dict = replay_buffer.get_data(self.args.replay_batch_size)
        
        # safety
        return_dict = copy.deepcopy(data_dict)
        self._generate_prelim_outputs(return_dict)
        return_dict = self._generate_advantages(return_dict)
        
        _num_minibatches = self.args.num_minibatches
        
        if replay_dict is not None:
            if self.args.use_manager_replay_only and train_net is 'manager':
                data_dict = replay_dict
            
            elif train_net is 'manager' or train_net is None:
                for key in replay_dict:
                    _num_minibatches *= \
                        len(replay_dict['start_state'])/ \
                        len(data_dict['start_state']) + 1
                        
                    # safety cast
                    _num_minibatches = int(_num_minibatches)
                    data_dict[key] = np.concatenate(
                        (data_dict[key], replay_dict[key]), axis=0
                    )       
                    
        update_hindsight = False
        
        if self.args.use_hindsight_replay and train_net in ['actor', None]:
            update_hindsight = True
            _num_minibatches *= 2 
            
        self._generate_prelim_outputs(data_dict, update_hindsight)
        
        data_dict = self._generate_advantages(data_dict)
        self._timesteps_so_far += self.args.batch_size
        
        if self.args.joint_value_update:
            _update_epochs = self.args.policy_epochs
            
        else:
            _update_epochs = max(self.args.policy_epochs,
                                 self.args.value_epochs)
            
        _log_stats = defaultdict(list)
            
        for epoch in range(_update_epochs):
            total_batch_len = len(data_dict['start_state'])
            total_batch_inds = np.arange(total_batch_len)
            if self.args.use_recurrent:
                # ensure we can factorize into episodes
                assert total_batch_len/_num_minibatches \
                    % self.args.episode_length == 0
            else:
                self._npr.shuffle(total_batch_inds)
            minibatch_size = total_batch_len//_num_minibatches
            
            for start in range(_num_minibatches):
                start = start * minibatch_size
                end = min(start + minibatch_size, total_batch_len)
                batch_inds = total_batch_inds[start:end]
                feed_dict = {
                        self._input_ph[key]: data_dict[key][batch_inds] \
                        for key in ['start_state', 'actions', 'advantage',
                         'goal', 'log_oldp_n', 'lookahead_state',
                         'initial_goals']
                }
                
                num_episodes = int((end - start) / self.args.episode_length)
                states_dict = {
                    self._input_ph['net_states'][layer['name']]:
                    np.reshape(np.tile(
                        self._dummy_states[layer['name']], [num_episodes]),
                        [num_episodes, 
                         *self._dummy_states[layer['name']].shape]
                    )
                    for layer in self._recurrent_layers
                }
                
                # concat variables with recurrent states
                feed_dict = {**feed_dict, **states_dict}
                
                feed_dict[self._input_ph['batch_size']] = \
                        np.array(float(end - start))  
                feed_dict[self._input_ph['episode_length']] = \
                        np.array(self.args.episode_length)
                feed_dict[self._input_ph['learning_rate']] = \
                        self._learning_rate    
                feed_dict[self._input_ph['cliprange']] = \
                        self._clipratio_list    
                feed_dict[self._input_ph['value_cliprange']] = \
                        self._value_clipratio_list   
                feed_dict[self._input_ph['entropy_coefficients']] = \
                        self._entropy_coefficients  
                feed_dict[self._input_ph['old_values']]= \
                        data_dict['value'][batch_inds]
                feed_dict[self._input_ph['value_target']] = \
                        data_dict['value_target'][batch_inds]
                        
                _value_keys = ['vf_loss', 'vf_update_op']
                _sub_nets = ['actor', 'manager', 'vae']
                if epoch < self.args.policy_epochs:
                    if self.args.joint_value_update:
                        _prelim_update_keys = \
                            [key for key in self._update_operator]
                    else:
                        _prelim_update_keys = [key for key in
                            self._update_operator if key not in _value_keys]
                        
                    if train_net is not None:
                        _update_keys = [key for key in _prelim_update_keys if
                            train_net in key]
                        
                    else:
                        _update_keys = []
                        for key in _prelim_update_keys:
                            if not any([net in key for net in _sub_nets]):
                                _update_keys.append(key)
                                
                    temp_stats_dict = self._session.run(
                        {key: self._update_operator[key]
                        for key in _update_keys},
                        feed_dict
                    )
                    
                # update the whitening variables
                self._set_whitening_var(dict(data_dict['whitening_stats']))
                    
                if epoch < self.args.value_epochs and \
                    not self.args.joint_value_update:
                    
                    if train_net is not None:
                        _update_value_keys = \
                            [key for key in _value_keys if
                             (train_net in key)]
                        
                    else:
                        _update_value_keys = []
                        for key in _value_keys:
                            if not any([net in key for net in _sub_nets]):
                                _update_value_keys.append(key)
                        
                    temp_stats_dict.update(self._session.run(
                        {key: self._update_operator[key] 
                        for key in _update_value_keys},
                        feed_dict=feed_dict)
                    )
                
                for key in temp_stats_dict:
                    _log_stats[key].append(temp_stats_dict[key])
        
        
        _final_stats = {}
        
        for key in _log_stats:
            if 'update_op' not in key:                
                _final_stats[key] = np.mean(np.array(_log_stats[key]))
        
        for hierarchy in range(self.args.maximum_hierarchy_depth):
            _final_stats["motivations_" + str(hierarchy)] = np.mean(
                data_dict["motivations"], axis=0         
            )[hierarchy]
            
        self._update_parameters(_final_stats)
        
        return _final_stats, return_dict
    
    def _update_parameters(self, statistics):
        if self.args.lr_schedule == 'adaptive' and \
            'approximate_KL_actor_manager' in statistics:
            mean_kl = statistics['approximate_KL_actor_manager']
            if mean_kl > self.args.target_kl_high * self.args.target_kl_ppo:
                self._learning_rate /= self.args.lr_alpha
            if mean_kl < self.args.target_kl_low * self.args.target_kl_ppo:
                self._learning_rate *= self.args.kl_alpha

            self._learning_rate = max(self._learning_rate, 
                                      self.args.adaptive_lr_min)
            self._learning_rate = min(self._learning_rate, 
                                      self.args.adaptive_lr_max)
        
        else:
            self._learning_rate = self.args.policy_lr * max(
                1.0 - float(self._timesteps_so_far) / self.args.max_timesteps,
                0.0
            )  
        self._entropy_coefficients = \
            self._initial_entropy_coefficients * (
            1.0 - float(self._timesteps_so_far) / \
            self.args.max_timesteps
            )
    
    def _generate_prelim_outputs(self, data_dict, update_hindsight=False):
        '''
        obtains key statistics, such as motivations and baseline values
        '''
        
        # TODO: generate dummy goal for final state to prevent errors
        feed_dict = {
            self._input_ph['start_state']: data_dict['start_state'],
            self._input_ph['goal']: data_dict['goal'],
        }
        
        batch_size = data_dict['start_state'].shape[0]
        
        # TODO: replace this with dynamic initial goals
        _dummy_goals = np.reshape(np.tile(
            self._dummy_initial_goals,[batch_size]),
            [batch_size] + list(self._dummy_initial_goals.shape)
        )
        
        feed_dict[self._input_ph['initial_goals']] = _dummy_goals
        
        feed_dict[self._input_ph['episode_length']] = \
            self.args.episode_length + 1
        
        num_episodes = int(len(data_dict['start_state']) / \
            (self.args.episode_length + 1))
        
        states_dict = {
            self._input_ph['net_states'][layer['name']]:
            np.reshape(np.tile(self._dummy_states[layer['name']],
            [num_episodes]), [num_episodes, -1])
            for layer in self._recurrent_layers
        }
        
        # concat variables with recurrent states
        feed_dict = {**feed_dict, **states_dict}
        
        data_dict.update(
            self._session.run(
                {'lookahead_state': self._tensor['output_lookahead'],
                 'motivations': self._tensor['output_motivations'],
                 'value': self._tensor['output_values']},
                 feed_dict
            )
        )
            
        if update_hindsight:
           self._update_hindsight_transitions(data_dict)
           
           # TODO: use np.tile instead of remaking these variables
           num_episodes = int(len(data_dict['start_state']) / \
                (self.args.episode_length + 1))
        
           states_dict = {
                self._input_ph['net_states'][layer['name']]:
                np.reshape(np.tile(self._dummy_states[layer['name']],
                [num_episodes]), [num_episodes, -1])
                for layer in self._recurrent_layers
            }
            
        # cannot compute logp for last state (no action exists)
        data_dict = self._discard_final_states(data_dict)
        
        feed_dict = {
            self._input_ph['start_state']: data_dict['start_state'],
            self._input_ph['actions']: data_dict['actions'],
            self._input_ph['initial_goals']: data_dict['initial_goals'],
            self._input_ph['lookahead_state']: data_dict['lookahead_state'],
            self._input_ph['goal']: data_dict['goal'],
            self._input_ph['episode_length']: self.args.episode_length,
        }
        
        feed_dict = {**feed_dict, **states_dict}

        data_dict.update(
            self._session.run(
                {'log_oldp_n': self._tensor['output_log_p_n']},
                feed_dict
            )        
        )
            
    def _update_hindsight_transitions(self, data_dict):
        feed_dict = {
            self._input_ph['start_state']: data_dict['start_state'],
            self._input_ph['initial_goals']: data_dict['initial_goals'],
            self._input_ph['lookahead_state']: data_dict['lookahead_state'],
            self._input_ph['episode_length']: self.args.episode_length+1,
        }
        
        num_episodes = int(len(data_dict['start_state']) / \
            (self.args.episode_length + 1))
        
        states_dict = {
            self._input_ph['net_states'][layer['name']]:
            np.reshape(np.tile(self._dummy_states[layer['name']],
            [num_episodes]), [num_episodes, -1])
            for layer in self._recurrent_layers
        }
    
        feed_dict = {**feed_dict, **states_dict}

        _hindsight_goal = \
            self._session.run(
                self._tensor['hindsight_goal'],
                feed_dict
            )
        
        feed_dict[self._input_ph['goal']] = _hindsight_goal
        
        _hindsight_motivations = \
            self._session.run(
                self._tensor['output_motivations'],
                feed_dict
            )
        
        data_dict['goal'] = np.concatenate(
            [data_dict['goal'], _hindsight_goal],
            axis=0
        )
        
        data_dict['motivations'] = np.concatenate(
            [data_dict['motivations'], _hindsight_motivations],
            axis=0
        )
        
        _duplicate_keys = ['start_state', 'actions', 'lookahead_state',
                           'initial_goals', 'value']
        
        for key in _duplicate_keys:
                data_dict[key] = np.tile(data_dict[key],
                    [2] + [1]*(data_dict[key].ndim-1)
                ) 