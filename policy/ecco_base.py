#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 10:18:48 2018

@author: matthewszhang
"""
from collections import defaultdict

import init_path
import tensorflow as tf
import numpy as np
from policy.networks import ecco_network, fixed_manager_network, \
    fixed_actor_network
from util import tf_networks, tf_utils, whitening_util

class base_model(object):
    '''
    General class for organizing feudal networks
    Can train all networks in one sess.run call
    '''
    def __init__(self, args, session, name_scope,
                 observation_size, action_size,
                 action_distribution):
        if session is not None:
            self._session = session
        else:
            self._session = tf.get_default_session()
        
        self.args = args
        self._name_scope = name_scope
        self._observation_size = observation_size
        self._action_size = action_size
        self._action_distribution = action_distribution
        self._base_dir = init_path.get_base_dir()
        
        self._whitening_operator = {}
        self._whitening_variable = []
        
        self._npr = np.random.RandomState(args.seed)
        self._input_ph = {}
        self._tensor = {}
        self._agents = {}
        self._update_operator = {}
            
    def build_model(self):
        with tf.variable_scope(self._name_scope, reuse=tf.AUTO_REUSE):
            self._build_proto()
            self._build_placeholders()
            self._build_preprocess()
            self._build_networks()
            self._build_loss()             
            self._set_var_list()
        
    def _build_loss(self):
        raise NotImplementedError
        
    def train(self, *args, **kwargs):
        raise NotImplementedError
        
    def _build_proto(self):
        '''
        Builds lambda parameters and initializes network protos
        '''
        
        # Goal vector output dim
        if self.args.use_fixed_manager:
            self._goal_size_by_level = \
                lambda x: self._observation_size
        else:
            self._goal_size_by_level = \
                lambda x: self.args.goals_dim_min * (
                    self.args.goals_dim_increment ** (
                        self.args.maximum_hierarchy_depth - x
                )
            )
                
        # Discounting factor
        self._gamma_by_level = \
            lambda x: 1 - self.args.gamma_max * (
                self.args.gamma_increment ** (
                    self.args.maximum_hierarchy_depth - 1 - x
                )
            )
                
        # Lookahead weighting
        self._lookahead_by_level = \
            lambda x: self.args.lookahead_increment ** x
            
        # Intrinsic Reward weighting
        self._beta_by_level = \
            lambda x: self.args.beta_min + \
            (self.args.beta_max - self.args.beta_min) / \
            max(self.args.maximum_hierarchy_depth - 1, 1) * x

        # Variable initialize
        self._learning_rate = self.args.policy_lr
        self._initial_entropy_coefficients = np.array(
            [self.args.manager_entropy_coefficient] \
            * (self.args.maximum_hierarchy_depth - 1) + \
            [self.args.actor_entropy_coefficient]
        )
        
        self._entropy_coefficients = self._initial_entropy_coefficients
            
        self._gamma_list = np.array(list(map(
            self._gamma_by_level, 
            list(range(self.args.maximum_hierarchy_depth))
        )))
        
        self._beta_list = np.array(list(map(
            self._beta_by_level, 
            list(range(self.args.maximum_hierarchy_depth))
        )))
        
        # Network prototypes
        if self.args.use_fixed_manager:
            self._manager_net_proto = fixed_manager_network.network
            self._maximum_dim = self._observation_size
            
        elif not(self.args.use_state_embedding
            or self.args.use_state_preprocessing):
            self._manager_net_proto = ecco_network.network
            self._maximum_dim = self._observation_size
            
        else:
            self._manager_net_proto = ecco_network.network
            self._maximum_dim = self._goal_size_by_level(1)
            
        if self.args.use_fixed_agent:
            self._actor_net_proto = fixed_actor_network.network
        else:
            self._actor_net_proto = ecco_network.network
            
        if self.args.recurrent_cell_type in ['gru', 'basic']:
            _number_hidden_units = self.args.joint_embed_dimension
            
        else:
            _number_hidden_units = 2 * self.args.joint_embed_dimension
            
        self._recurrent_layers = [
            {'name': 'manager' + str(i),
             'number_hidden_units': _number_hidden_units,
             'lookahead':self._lookahead_by_level(i)
            } for i in range(1, self.args.maximum_hierarchy_depth)
        ] + [
            {'name': 'actor' + str(0),
             'number_hidden_units': _number_hidden_units,
             'lookahead':self._lookahead_by_level(0)
            }
        ]
        self._timesteps_so_far = 0
        self.required_keys = [
            'start_state', 'end_state', 'rewards',
            'goal', 'initial_goals', 'actions',
        ]
        
        # return zero 
        self._dummy_states = {
            layer['name']: np.zeros([layer['number_hidden_units']])
            for layer in self._recurrent_layers
        }

        self._dummy_initial_goals = np.ones(self._maximum_dim)
            
    def _build_preprocess(self):
        whitening_util.add_whitening_operator(
            self._whitening_operator, self._whitening_variable,
            'state', self._observation_size
        )
        
        if self.args.use_state_normalization:
            self._tensor['normalized_start_state'] = (
                self._input_ph['start_state'] -
                self._whitening_operator['state_mean']
            ) / self._whitening_operator['state_std']
            
            self._tensor['normalized_lookahead'] = (
                self._input_ph['lookahead_state'] -
                self._whitening_operator['state_mean']
            ) / self._whitening_operator['state_std']
            
            self._tensor['net_input'] = \
                self._tensor['normalized_start_state']
            self._tensor['net_lookahead'] = \
                self._tensor['normalized_lookahead']
        
        else:
            self._tensor['net_input'] = self._input_ph['start_state']
            self._tensor['normalized_start_state'] = \
                self._input_ph['start_state']
            self._tensor['net_lookahead'] = self._input_ph['lookahead_state']
        
        self._network_input_size = self._observation_size
        
        # state is preprocessed with an MLP
        if self.args.use_state_preprocessing:
            shared_processing_network_size = [self._observation_size] + \
                self.args.preprocess_network_shape
            num_layer = len(shared_processing_network_size) - 1
            
            activation_type = \
                [self.args.preprocess_activation_type] * num_layer
            norm_type = \
                [self.args.preprocess_normalizer_type] *  num_layer
            init_data = []
            for _ in range(num_layer):
                init_data.append(
                    {'w_init_method': 'normc', 'w_init_para': {'stddev': 1.0},
                    'b_init_method': 'constant', 'b_init_para': {'val': 0.0}}
                )
                            
            self._preprocessing_mlp = tf_networks.MLP(
                dims = shared_processing_network_size, 
                scope = 'preprocessing',
                activation_type = activation_type,
                normalizer_type = norm_type,
                train=True, init_data = init_data
            )
            
            self._tensor['net_input'] = self._preprocessing_mlp(
                self._tensor['normalized_start_state']
            )
            
            self._tensor['net_lookahead'] = self._preprocessing_mlp(
                self._tensor['normalized_lookahead']
            )
                
            self._network_input_size = \
                self.args.preprocess_network_shape[-1]
    
    def _build_placeholders(self):
        self._input_ph['net_states'] = {
            layer['name']:
            tf.placeholder(
                tf.float32, [
                    None,
                    layer['number_hidden_units']
                ]
            ) for layer in self._recurrent_layers
        }
            
        self._input_ph['start_state'] = tf.placeholder(
            tf.float32, [None, self._observation_size]
        )
        
        self._input_ph['initial_goals'] = tf.placeholder(
            tf.float32, [None, self._maximum_dim]
        )
        
        self._input_ph['advantage'] = tf.placeholder(
            tf.float32, [None, self.args.maximum_hierarchy_depth]
        )
        
        self._input_ph['actions'] = tf.placeholder(
            tf.int32, [None]        
        )
        
        self._input_ph['value_target'] = tf.placeholder(
            tf.float32, [None, self.args.maximum_hierarchy_depth]
        )
        
        self._input_ph['lookahead_state'] = tf.placeholder(
            tf.float32, [
                None, 
                self.args.maximum_hierarchy_depth - 1,
                self._observation_size]
        )
        
        self._input_ph['goal'] = tf.placeholder(
            tf.float32, [
                None, 
                self.args.maximum_hierarchy_depth - 1,
                self._maximum_dim]
        )
    
        self._input_ph['learning_rate'] = tf.placeholder(tf.float32, [])
        
        self._input_ph['episode_length'] = tf.placeholder(tf.int32, [])
        
        self._input_ph['batch_size'] = tf.placeholder(tf.float32, [])
        
        self._input_ph['entropy_coefficients'] = tf.placeholder(
            tf.float32, [self.args.maximum_hierarchy_depth]
        )
        
    
    def _build_networks(self):
        proto_distribution = 'continuous'
        
        old_goal_input = tf.concat(
            [tf.expand_dims(self._input_ph['initial_goals'], 1), 
             self._input_ph['goal']], axis = 1
        )
        
        # list for storing hierarchy variables before concatenation
        list_tensor = defaultdict(list)
        
        list_tensor['output_goal'].append(self._input_ph['initial_goals'])
        
        # build all master agents
        for i in range(self.args.maximum_hierarchy_depth - 1):
            level = self.args.maximum_hierarchy_depth - i - 1
            name = 'manager' + str(level)
            
            input_tensors = {
                'net_input': self._tensor['net_input'],
                'lookahead_input':self._tensor['net_lookahead'][:, i],
                'goal_input': list_tensor['output_goal'][i],
                'recurrent_input': self._input_ph['net_states'][name],
                'old_goal_output': self._input_ph['goal'][:, i],
                'old_goal_input': old_goal_input[:, i]
            }
            
            # building manager instance
            agent = self._manager_net_proto(
                self.args, input_tensors,
                proto_distribution,
                self._network_input_size,
                self._goal_size_by_level(level+1),
                self._goal_size_by_level(level),
                self._maximum_dim,
                self._npr,
                self._input_ph['episode_length'],
                self._input_ph['batch_size'],
                self._lookahead_by_level(level), name,
                is_manager = True
            )
            self._agents[name] = agent
            
            # appending instance variables to list
            self._tensor.update(agent.states)
            for key in agent.outputs:
                if self.args.debug_end_to_end:
                    list_tensor[key].append(tf.stop_gradient(
                        agent.outputs[key])
                    )
                else:
                    list_tensor[key].append(agent.outputs[key])
                
            state_by_episode = \
                tf.reshape(self._input_ph['start_state'],
                       [-1, self._input_ph['episode_length'],
                        self._observation_size])
            _lookaheads = \
                tf.concat(
                    [state_by_episode[:,self._lookahead_by_level(level):],
                    tf.tile(tf.expand_dims(state_by_episode[:,-1], 1),
                        [1,self._lookahead_by_level(level),1]
                    )], axis=1
                )
            list_tensor['output_lookahead'].append(tf.reshape(
                _lookaheads, [-1, self._observation_size]        
            ))
            
        # building actor instance
        level = 0
        name = 'actor' + str(level)
        input_tensors = {
            'net_input': self._tensor['net_input'],
            'lookahead_input': tf.identity(self._tensor['net_input']),
            'action_input': self._input_ph['actions'],
            'goal_input': list_tensor['output_goal'][-1],
            'recurrent_input': self._input_ph['net_states'][name],
            'old_goal_input': old_goal_input[:, -1]
        }
        
        agent = self._actor_net_proto(
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
            is_manager = False
        )
        self._agents[name] = agent
        self._tensor['output_action'] = agent.action
        self._tensor.update(agent.states)
        
        # same update as the managers, but without output goal
        for key in agent.outputs:
            list_tensor[key].append(agent.outputs[key])
            
        list_tensor['output_goal'] = list_tensor['output_goal'][1:]
        
        # converting lists to tensors
        for key in list_tensor:
            if key is 'output_entropy':
                self._tensor[key] = tf.transpose(tf.stack(list_tensor[key]))
            else:
                self._tensor[key] = tf.transpose(
                    tf.stack(list_tensor[key]), 
                    tf.concat([[1, 0,], 
                    tf.range(2, tf.rank(list_tensor[key]))],
                    axis=0)
                )
                    
    def act(self, data_dict, control_infos):
        
        __last_ob = np.array(data_dict['start_state'])
        batch_size = len(__last_ob)
        
        if 'get_dummy_goals' in control_infos and \
            control_infos['get_dummy_goals']:
            return {'goal':
                np.zeros((batch_size, self.args.maximum_hierarchy_depth - 1,
                self._maximum_dim), dtype=np.float32)}
        
        feed_dict = {
            self._input_ph['start_state']: \
            np.reshape(__last_ob, 
                       (batch_size, -1))
        }
        feed_dict[self._input_ph['episode_length']] = \
            np.array(1, dtype=np.int32)
            
        initial_goal_dict = {}
            
        if 'use_default_goal' in control_infos and \
            control_infos['use_default_goal']:
            _dummy_goals = np.reshape(np.tile(
                self._dummy_initial_goals,[batch_size]),
                [batch_size] + list(self._dummy_initial_goals.shape)
            )
            
            feed_dict[self._input_ph['initial_goals']] = _dummy_goals
            
            initial_goal_dict['initial_goals'] = _dummy_goals
                
        else:
            feed_dict[self._input_ph['initial_goals']] = \
                data_dict['initial_goals']
        
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
            
        # concat dictionaries
        feed_dict = {**feed_dict, **states_dict}
        
        run_dict = {
            'actions': self._tensor['output_action'],
            'goal': self._tensor['output_goal']
        }
        
        states_dict = {}
        
        for layer in self._recurrent_layers:
            states_dict[_hash[layer['name']]] =  self._tensor[layer['name']]
        
        run_dict = {**run_dict, **states_dict}
        
        return_dict = self._session.run(run_dict, feed_dict)
        
        return_dict.update(initial_goal_dict)
        
        return return_dict
        
    def _discard_final_states(self, data_dict):
        # remove terminated states from dictionary
        for key in data_dict:
            _temp_item = np.array(data_dict[key])
            
            if _temp_item.ndim != 0:
                # if the shape divides by the episode length + 1
                if (_temp_item.shape[0] \
                    % (self.args.episode_length + 1)) == 0:
                    
                    _temp_data = np.reshape(_temp_item,
                        [-1, self.args.episode_length + 1,
                        *_temp_item.shape[1:]])
                        
                    if key in ['motivations']:
                        _temp_data = _temp_data[:,1:]
                    else:
                        _temp_data = _temp_data[:,:-1]
                    data_dict[key] = np.reshape(_temp_data, 
                        [-1, *_temp_item.shape[1:]]
                    )
                    
        return data_dict
            
    def _generate_advantages(self, data_dict):
        # fix indices
        data_dict['advantage'] = \
            np.zeros(data_dict['value'].shape)
            
        _batch_size = len(data_dict['start_state'])
            
        _tile_reward = np.reshape(np.repeat(data_dict['rewards'],
            self.args.maximum_hierarchy_depth, -1),
            (_batch_size, -1)
        )
                        
        # fast shifting of motivations
        _motivations = np.zeros((data_dict['motivations'].shape[0],
            data_dict['motivations'].shape[1] + 1))
        _motivations[:,1:] = data_dict['motivations']
        data_dict['motivations'] = _motivations
        
        data_dict['joint_reward'] = \
            _tile_reward * (1 - self._beta_list) + \
            data_dict['motivations'] * self._beta_list
        
        # assume that the states are arranged as a flatten vector of episodes,
        # each of fixed length
        
        for episode in range(int(_batch_size/self.args.episode_length)):
            episode_start = episode * self.args.episode_length
            episode_end = episode_start + self.args.episode_length
            for i_step in reversed(range(episode_start, episode_end)):
                if i_step < episode_end - 1:
                    delta = data_dict['joint_reward'][i_step] \
                        + self._gamma_list * \
                        data_dict['value'][i_step + 1] \
                        - data_dict['value'][i_step]
                    data_dict['advantage'][i_step] = \
                        delta + self._gamma_list * self.args.gae_lam \
                        * data_dict['advantage'][i_step]
                        
                else:
                    delta = data_dict['joint_reward'][i_step] \
                        - data_dict['value'][i_step]
                    data_dict['advantage'][i_step] = delta
                    
                    
        data_dict['value_target'] = \
            data_dict['advantage'] + data_dict['value']
            
        data_dict['advantage'] -= \
            np.mean(data_dict['advantage'], axis=0, keepdims=True)
        data_dict['advantage'] /= \
            np.std(data_dict['advantage'], axis=0, keepdims=True) + 1e-8
        
        return data_dict
            
    def _set_whitening_var(self, whitening_stats):
        whitening_util.set_whitening_var(
            self._session, self._whitening_operator, whitening_stats, ['state']
        )
        
    def get_weights(self):
        return self._get_network_weights()
    
    def set_weights(self, weights):
        self._set_network_weights(weights)
        
    def _set_var_list(self):
        # collect the tf variable and the trainable tf variable
        self._trainable_var_list = [var for var in tf.trainable_variables()
                                    if self._name_scope in var.name]

        self._all_var_list = [var for var in tf.global_variables()
                              if self._name_scope in var.name]

        # the weights that actually matter
        self._network_var_list = \
            self._trainable_var_list + self._whitening_variable

        self._set_network_weights = tf_utils.set_network_weights(
            self._session, self._network_var_list, self._name_scope
        )

        self._get_network_weights = tf_utils.get_network_weights(
            self._session, self._network_var_list, self._name_scope
        )