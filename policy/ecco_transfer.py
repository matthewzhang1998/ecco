#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 12:49:30 2018

@author: matthewszhang
"""
import numpy as np
import tensorflow as tf

from collections import defaultdict

from .ecco_base import base_model

class model(base_model):
    def build_model(self):
        with tf.variable_scope(self._name_scope, reuse=tf.AUTO_REUSE):
            self._build_proto()
            self._build_placeholders()
            self._build_preprocess()
            self._build_networks()
            self._build_loss()             
            self._set_var_list()
    
    def _build_transfer_agent(self):
        # useful manager feedback
        _manager = self._agents['manager1']
        
        state_by_episode = \
                tf.reshape(self._input_ph['start_state'],
                       [-1, self._input_ph['episode_length'],
                        self._observation_size])
        _lookaheads = \
            tf.concat(
                [state_by_episode[:,self._lookahead_by_level(1):],
                tf.tile(tf.expand_dims(state_by_episode[:,-1], 1),
                    [1,self._lookahead_by_level(1),1]
                )], axis=1
            )
        _lookahead_manager_final = tf.reshape(
            _lookaheads, [-1, self._observation_size]        
        )
        
        self._tensor['lookahead_manager_final'] = _lookahead_manager_final
        self._tensor['hindsight_goal'] = _manager.outputs['hindsight_goal']
        
        # building actor instance
        level = 0
        name = 'actor' + str(level)
        input_tensors = {
            'net_input': self._tensor['net_input'],
            'lookahead_input': tf.identity(self._tensor['net_input']),
            'action_input': self._input_ph['actions'],
            'goal_input': 
                tf.stop_gradient(_manager.outputs['hindsight_goal']),
            'recurrent_input': self._input_ph['net_states'][name],
            'old_goal_input': 
                tf.stop_gradient(_manager.outputs['hindsight_goal'])
        }
        
        self._transfer_agent = self._actor_net_proto(
            self.args, input_tensors,
            self._action_distribution,
            self._network_input_size,
            self._goal_size_by_level(level+1),
            self._action_size,
            self._maximum_dim,
            self._npr,
            self._input_ph['episode_length'],
            self._input_ph['batch_size'],
            self._lookahead_by_level(level), name,
            is_manager = False,
            reuse = tf.AUTO_REUSE
        )

        self._input_ph['test_goal'] = tf.placeholder(
            tf.float32, [None, self._maximum_dim]
        )

        test_input_tensors = {
            'net_input': self._tensor['net_input'],
            'lookahead_input': tf.identity(self._tensor['net_input']),
            'action_input': self._input_ph['actions'],
            'goal_input':
                tf.stop_gradient(self._input_ph['test_goal']),
            'recurrent_input': self._input_ph['net_states'][name],
            'old_goal_input':
                tf.stop_gradient(_manager.outputs['hindsight_goal'])
        }

        self._test_agent = self._actor_net_proto(
            self.args, test_input_tensors,
            self._action_distribution,
            self._network_input_size,
            self._goal_size_by_level(level + 1),
            self._action_size,
            self._maximum_dim,
            self._npr,
            self._input_ph['episode_length'],
            self._input_ph['batch_size'],
            self._lookahead_by_level(level), name,
            is_manager=False,
            reuse=tf.AUTO_REUSE
        )

        self._tensor['test_action'] = self._test_agent.action
        
    def _build_loss(self):        
        self.required_keys = [
            'start_state', 'end_state', 'rewards', 'actions'
        ]
        
        self._build_transfer_agent()
        
        self._tensor['supervised_actor_logprobs'] = \
            self._transfer_agent.outputs['output_log_p_n']
            
        self._tensor['supervised_actor_values'] = \
            self._transfer_agent.outputs['output_values']
            
        self._update_operator['transfer_policy_loss'] = \
            tf.reduce_mean(
                tf.exp(self._tensor['supervised_actor_logprobs']) * \
                -self._input_ph['advantage'][:,-1]
            )
        
        self._update_operator['transfer_value_loss'] = \
            .5 * tf.reduce_mean(
                tf.square(
                    self._tensor['output_values'] - \
                    self._input_ph['value_target'][:,-1]
                )
            )
        self._update_operator['transfer_loss'] = \
            self._update_operator['transfer_policy_loss']
        
        if self.args.transfer_joint_value_update:
            self._update_operator['transfer_loss'] += \
                self._update_operator['transfer_value_loss']
                
        else:
            self._update_operator['transfer_vf_update_op'] = \
                tf.train.AdamOptimizer(
                    learning_rate=self.args.transfer_value_lr,
                    beta1=0.5, beta2=0.99, epsilon=1e-4
                ).minimize(self._update_operator['transfer_value_loss'])
                
        if self.args.transfer_clip_gradients:
            self._tensor['transfer_update_op_proto'] = \
            tf.train.AdamOptimizer(
                learning_rate=self._input_ph['learning_rate']
            )
            _params = tf.trainable_variables()
            self._tensor['transfer_update_op_gradients'] = \
                tf.gradients(
                        self._update_operator['transfer_loss'], _params
                )
            self._tensor['transfer_update_op_gradients'], _ = \
                tf.clip_by_global_norm(
                    self._tensor['transfer_update_op_gradients'],
                    self.args.transfer_clip_gradient_threshold
                )
                
            self._tensor['transfer_update_op_gradients'] = list(zip(
                self._tensor['transfer_update_op_gradients'], _params
            ))
            
            self._update_operator['transfer_update_op'] = \
                self._tensor['transfer_update_op_proto'].apply_gradients(
                    self._tensor['transfer_update_op_gradients']
                )
        
        else:
            self._update_operator['transfer_update_op'] = \
                tf.train.AdamOptimizer(
                    learning_rate=self._input_ph['learning_rate']
                ).minimize(self._update_operator['transfer_loss'])
        
    def train(self, data_dict, replay_buffer, train_net):
        self._generate_prelim_outputs(data_dict)
        self._generate_advantages(data_dict)
        
        _num_minibatches = self.args.transfer_minibatches
        
        _log_stats = defaultdict(list)
        
        for epoch in range(max(self.args.transfer_policy_epochs,
                               self.args.transfer_value_epochs)):
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
                         'goal', 'initial_goals', 'value_target', 
                         'lookahead_state']
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
                        self.args.transfer_policy_lr
                feed_dict[self._input_ph['value_target']] = \
                        data_dict['value_target'][batch_inds]
                        
                _value_keys = ['vf_loss', 'vf_update_op']
                
                if epoch < self.args.transfer_policy_epochs:
                    if self.args.transfer_joint_value_update:
                        _update_keys = \
                            [key for key in self._update_operator if 
                             'transfer' in key]
                    else:
                        _update_keys = [key for key in
                            self._update_operator if key not in _value_keys
                            and 'transfer' in key]
                    
                    temp_stats_dict = self._session.run(
                        {key: self._update_operator[key]
                        for key in _update_keys},
                        feed_dict
                    )
                    
                # update the whitening variables
                self._set_whitening_var(dict(data_dict['whitening_stats']))
                    
                if epoch < self.args.transfer_value_epochs and \
                    not self.args.transfer_joint_value_update:
                    
                    _update_value_keys = ['transfer_value_loss',
                                          'transfer_vf_update_op']
                        
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
        
        _final_stats["motivations"] = np.mean(
            data_dict["motivations"][:,-1])
        
        return _final_stats, data_dict

    def _generate_prelim_outputs(self, data_dict):
        feed_dict = {
            self._input_ph['start_state']: data_dict['start_state'],
        }
        
        feed_dict[self._input_ph['episode_length']] = \
            self.args.episode_length + 1
        
        batch_size = data_dict['start_state'].shape[0]
        
        # TODO: replace this with dynamic initial goals
        _dummy_goals = np.reshape(np.tile(
            self._dummy_initial_goals,[batch_size]),
            [batch_size] + list(self._dummy_initial_goals.shape)
        )
        
        feed_dict[self._input_ph['initial_goals']] = \
            data_dict['initial_goals'] = _dummy_goals
        
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
        
        final_lookaheads = self._session.run(
            self._tensor['lookahead_manager_final'], feed_dict
        )
        
        _dummy_lookaheads = np.zeros(
            (batch_size, self.args.maximum_hierarchy_depth - 1,
             self._observation_size), dtype=np.float32
        )
            
        _dummy_lookaheads[:, -1] = final_lookaheads
            
        feed_dict[self._input_ph['lookahead_state']] = \
            data_dict['lookahead_state'] =  _dummy_lookaheads
        
        hindsight_goals = self._session.run(
            self._tensor['hindsight_goal'], feed_dict       
        )
        
        _dummy_input_goals = np.zeros(
            (batch_size, self.args.maximum_hierarchy_depth - 1,
             self._maximum_dim), dtype=np.float32
        )
        _hindsight_goals_by_episode = np.reshape(
            hindsight_goals, (num_episodes, self.args.episode_length + 1, -1)
        )
        correct_ratio = int(num_episodes * self.args.hindsight_correct_eps)
        incorrect_ratio = num_episodes - correct_ratio
        
        correct_goals = self._npr.choice(
            np.arange(num_episodes), (correct_ratio), replace=False
        )
        incorrect_goals = self._npr.randint(
            low=0, high=num_episodes, size=(incorrect_ratio,)     
        )
        final_goals = []
        
        # j is the ind in the incorrect_goal array
        j = 0
        for i in range(num_episodes):
            if i in correct_goals:
                final_goals.append(_hindsight_goals_by_episode[i])
            else:
                final_goals.append(
                    _hindsight_goals_by_episode[incorrect_goals[j]]
                )
                j += 1
                
        final_goals = np.reshape(np.array(final_goals), [batch_size, -1])
        _dummy_input_goals[:,-1] = final_goals
        data_dict['goal'] = feed_dict[self._input_ph['goal']] = \
            _dummy_input_goals
            
        data_dict.update(
            self._session.run(
            {'motivations': self._tensor['output_motivations'],
             'value': self._tensor['output_values'][:,:,0]}, feed_dict
            )
        )
            
        data_dict = self._discard_final_states(data_dict)

    def act(self, data_dict, control_infos):
        __last_ob = np.array(data_dict['start_state'])
        batch_size = len(__last_ob)

        feed_dict = {
            self._input_ph['start_state']: \
                np.reshape(__last_ob, (batch_size, -1)),
            self._input_ph['test_goal']: data_dict['test_goal']
        }

        feed_dict[self._input_ph['episode_length']] = \
            np.array(1, dtype=np.int32)

        _timestep = control_infos['step_index']
        _hash = {}
        states_dict = {}

        for layer in self._recurrent_layers:
            if self.args.use_dilatory_network:
                _modulo = _timestep % layer['lookahead']
                _hash[layer['name']] = layer['name'] + '_' + str(_modulo)
            else:
                _hash[layer['name']] = layer['name']

            if _hash[layer['name']] in data_dict:
                states_dict[self._input_ph['net_states'][layer['name']]] = \
                    np.array(data_dict[_hash[layer['name']]])

            else:
                states_dict[self._input_ph['net_states'][layer['name']]] = \
                    np.reshape(np.tile(
                        self._dummy_states[layer['name']], [batch_size]),
                        [batch_size] +
                        list(self._dummy_states[layer['name']].shape)
                    )

        run_dict = {
            'actions': self._tensor['output_action']
        }

        states_dict = {}

        for layer in self._recurrent_layers:
            states_dict[_hash[layer['name']]] = self._tensor[layer['name']]

        run_dict = {**run_dict, **states_dict}

        return self._session.run(run_dict, feed_dict)

    def _generate_prelim_test(self, data_dict):
        # target goal generation

        feed_dict = {
            self._input_ph['start_state']: data_dict['start_state'],
        }

        feed_dict[self._input_ph['episode_length']] = \
            self.args.transfer_test_length

        batch_size = data_dict['start_state'].shape[0]

        _dummy_goals = np.reshape(np.tile(
            self._dummy_initial_goals, [batch_size]),
            [batch_size] + list(self._dummy_initial_goals.shape)
        )

        feed_dict[self._input_ph['initial_goals']] = \
            data_dict['initial_goals'] = _dummy_goals

        # states are not used for hindsight goal generation
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

        final_lookaheads = self._session.run(
            self._tensor['lookahead_manager_final'], feed_dict
        )

        _dummy_lookaheads = np.zeros(
            (batch_size, self.args.maximum_hierarchy_depth - 1,
             self._observation_size), dtype=np.float32
        )

        _dummy_lookaheads[:, -1] = final_lookaheads

        feed_dict[self._input_ph['lookahead_state']] = \
            data_dict['lookahead_state'] = _dummy_lookaheads

        data_dict['test_goal'] = self._session.run(
            self._tensor['hindsight_goal'], feed_dict
        )

    def _test(self, data_dict):
        feed_dict = {
            self._input_ph['start_state']: data_dict['start_state'],
            self._input_ph['test_goal']: data_dict['test_goal']
        }

        feed_dict[self._input_ph['episode_length']] = \
            self.args.transfer_test_length

        batch_size = data_dict['start_state'].shape[0]

        _dummy_goals = np.reshape(np.tile(
            self._dummy_initial_goals, [batch_size]),
            [batch_size] + list(self._dummy_initial_goals.shape)
        )

        feed_dict[self._input_ph['initial_goals']] = \
            data_dict['initial_goals'] = _dummy_goals

        num_episodes = int(len(data_dict['start_state']) / \
                           (self.args.transfer_test_length))

        states_dict = {
            self._input_ph['net_states'][layer['name']]:
                np.reshape(np.tile(self._dummy_states[layer['name']],
                                   [num_episodes]), [num_episodes, -1])
            for layer in self._recurrent_layers
        }

        # concat variables with recurrent states

        _dummy_input_goals = np.zeros(
            (batch_size, self.args.maximum_hierarchy_depth - 1,
             self._maximum_dim), dtype=np.float32
        )

        _dummy_input_goals[:,-1] = data_dict['test_goal']

        feed_dict[self._input_ph['goal']] = _dummy_input_goals

        feed_dict = {**feed_dict, **states_dict}

        _motivations = self._session.run(
            self._tensor['output_motivations'],
        )

        return {'motivations': np.sum(_motivations) / self.args.transfer_test_length}

    def _get_hash_states(self, data_dict, start_inds):
        states_dict = {}

        for layer in self._recurrent_layers:
            if self.args.use_dilatory_network:
                for x in range(self.args.test_transfer_nenvs):
                    for t in range(layer['lookahead']):
                        _modulos = (start_inds - t) % layer['lookahead']
                        _hash[layer['name']] = layer['name'] + '_' + str(_modulo)
            else:
                _hash[layer['name']] = layer['name']


