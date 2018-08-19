#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 11 19:59:16 2018

@author: matthewszhang
"""
import init_path
import numpy as np
import tensorflow as tf
from policy.networks import base_network
from util import tf_networks, tf_distributions

class network(base_network.base_network):
    def __init__(self, *args, **kwargs):
        super(network, self).__init__(*args, **kwargs)
        self._base_dir = init_path.get_base_dir()
        
        with tf.variable_scope(self.name):
            self._build_preprocess()
            self._build_outputs()
        
    def _build_preprocess(self):
        # Build state embedding layer
        if self._is_manager:
            self._embed_state_size = self._output_size
            
        else:
            self._embed_state_size = self._input_state_size
        
        state_embedding_size = \
            [self._input_state_size] + \
            [self._embed_state_size]
            
        norm_type = [self.args.state_embed_norm_type]
        init_data = [
            {'w_init_method': 'normc', 'w_init_para': {'stddev': 1.0},
             'b_init_method': 'constant', 'b_init_para': {'val': 0.0}}
        ]
                        
        self._state_embedding_layer = tf_networks.Linear(
            dims = state_embedding_size, 
            scope = 'state_embedding_0',
            normalizer_type = norm_type,
            train=True, init_data = init_data
        )
        
        # Build goal embedding layer
        if self.args.embed_goal_type == 'linear':
            goal_embedding_size = \
                [self._input_goal_size] + \
                [self._embed_state_size]
        
        elif self.args.embed_goal_type == 'matrix':
            goal_embedding_size = \
                [self._input_goal_size] + \
                [self._embed_state_size *
                 self._embed_state_size]
                
        else:
            raise ValueError("Invalid embed type.")
            
        norm_type = [self.args.goal_embed_norm_type]
        init_data = [
            {'w_init_method': 'normc', 'w_init_para': {'stddev': 1.0},
             'b_init_method': 'constant', 'b_init_para': {'val': 0.0}}
        ]
                        
        self._goal_embedding_layer = tf_networks.Linear(
            dims = goal_embedding_size, 
            scope = 'goal_embedding_0',
            normalizer_type = norm_type,
            train=True, init_data = init_data
        )
        
        joint_embedding_size = [self._embed_state_size] + \
            [self.args.joint_embed_dimension]
        
        act_type = [self.args.joint_embed_act_type]
        norm_type = [self.args.joint_embed_norm_type]
        
        if self.args.use_recurrent:
            # Borrows tensorflow implementation, takes too much time to recode
            if self._is_manager and self.args.use_dilatory_network:
                self._joint_embedding_layer = \
                    tf_networks.Dilated_Recurrent_Network(
                        hidden_size = self.args.joint_embed_dimension,
                        scope = 'joint_embedding_0',
                        activation_type = act_type,
                        normalizer_type = norm_type, train = True,
                        recurrent_cell_type = self.args.recurrent_cell_type,
                        dilation = self._lookahead_range
                    )
                    
            else:
                self._joint_embedding_layer = \
                    tf_networks.Recurrent_Network(
                        hidden_size = self.args.joint_embed_dimension,
                        scope = 'joint_embedding_0',
                        activation_type = act_type,
                        normalizer_type = norm_type, train = True,
                        recurrent_cell_type = self.args.recurrent_cell_type
                    )
            
        else:
            init_data = [
                {'w_init_method': 'normc', 'w_init_para': {'stddev': 1.0},
                 'b_init_method': 'constant', 'b_init_para': {'val': 0.0}}
            ]
            self._joint_embedding_layer = tf_networks.MLP(
                dims = joint_embedding_size,
                scope = 'joint_embedding_0',
                activation_type = act_type,
                normalizer_type = norm_type,
                train=True, init_data = init_data
            )
        
        # Build policy and value output networks
        self._build_policy_and_value_networks()
        
    def _build_policy_and_value_networks(self):
        if self._distribution == 'continuous':
            self._proto_distribution = tf_distributions.gaussian
            
        elif self._distribution == 'discrete':
            self._proto_distribution = tf_distributions.categorical
            
        policy_output_size =  self._output_size
        
        policy_network_shape = [self.args.joint_embed_dimension] + \
            self.args.policy_network_shape + [policy_output_size]
                
        num_layer = len(policy_network_shape) - 1
        act_type = \
            [self.args.policy_activation_type] * (num_layer - 1) + [None]
        norm_type = \
            [self.args.policy_normalizer_type] * (num_layer - 1) + [None]
        init_data = [
            {'w_init_method': 'normc', 'w_init_para': {'stddev': 1.0},
             'b_init_method': 'constant', 'b_init_para': {'val': 0.0}}
        ] * num_layer
        
        init_data[-1]['w_init_para']['stddev'] = 0.01  # the output layer std
        
        self._policy_MLP = tf_networks.MLP(
            dims=policy_network_shape, scope= 'policy_mlp',
            activation_type=act_type, normalizer_type=norm_type,
            train=True, init_data=init_data
        )
        
        value_network_shape = [self._embed_state_size] + \
            self.args.value_network_shape + [1]
        num_layer = len(value_network_shape) - 1
        act_type = \
            [self.args.value_activation_type] * (num_layer - 1) + [None]
        norm_type = \
            [self.args.value_normalizer_type] * (num_layer - 1) + [None]
        init_data = [
            {'w_init_method': 'normc', 'w_init_para': {'stddev': 1.0},
             'b_init_method': 'constant', 'b_init_para': {'val': 0.0}}
        ] * num_layer
        
        self._value_MLP = tf_networks.MLP(
            dims=value_network_shape, scope='value_mlp', train=True,
            activation_type=act_type, normalizer_type=norm_type,
            init_data=init_data
        )
        
    def _build_outputs(self):
        self._input_tensor['net_input'] = tf.reshape(
            self._input_tensor['net_input'],
            tf.concat([tf.constant([-1], dtype=tf.int32),
            [self._batch_dimension],
            tf.constant([self._input_state_size])], axis=0)
        )
            
        self._input_tensor['lookahead_input'] = tf.reshape(
            self._input_tensor['lookahead_input'],
            tf.concat([tf.constant([-1], dtype=tf.int32),
            [self._batch_dimension],
            tf.constant([self._input_state_size])], axis=0)
        )
            
        self._tensor['goal_input'] = tf.reshape(
            self._input_tensor['goal_input'][:,:self._input_goal_size],
            tf.concat([tf.constant([-1], dtype=tf.int32),
            [self._batch_dimension],
            tf.constant([self._input_goal_size])], axis=0)
        )
            
        if self._is_manager:
            self._tensor['old_goal_output'] = tf.reshape(
                self._input_tensor['old_goal_output'][:,:self._output_size],
                tf.concat([tf.constant([-1], dtype=tf.int32),
                [self._batch_dimension],
                tf.constant([self._output_size])], axis=0)
            )
        
        self._tensor['embed_state'] = \
            self._state_embedding_layer(self._input_tensor['net_input'])
            
        self._tensor['embed_lookahead'] = \
            self._state_embedding_layer(self._input_tensor['lookahead_input'])
        
        if self._is_manager:            
            self._tensor['label_action'] = \
                tf.nn.l2_normalize(
                    self._tensor['embed_lookahead'] - \
                    self._tensor['embed_state'], axis=-1
                )
                
        else:
            self._tensor['label_action'] = \
                self._input_tensor['action_input']
            
        self._tensor['embed_goal'] = \
            self._goal_embedding_layer(self._tensor['goal_input'])
            
        if self.args.embed_goal_type == 'linear':
            self._tensor['mixture_joint'] = \
                self._tensor['embed_state'] * self._tensor['embed_goal']
            
        elif self.args.embed_goal_type == 'matrix':
            self._tensor['embed_goal'] = \
                tf.reshape(self._tensor['embed_goal'],
                   tf.concat([tf.constant([-1], dtype=tf.int32),
                   [self._batch_dimension],
                   tf.constant([self._embed_state_size,
                                self._embed_state_size])]), axis=0)
            
            self._tensor['mixture_joint'] = \
                tf.einsum('ijkl,ijk->ijl', self._tensor['embed_goal'],
                          self._tensor['embed_state'])
                
        if self.args.use_recurrent:
            self._tensor['embed_joint'], self.states[self.name] = \
                self._joint_embedding_layer(self._tensor['mixture_joint'],
                    hidden_states = self._input_tensor['recurrent_input'])
                
        else:
            self._tensor['embed_joint'] = \
                self._joint_embedding_layer(self._tensor['mixture_joint'])
                
            if self.args.recurrent_cell_type in ['gru', 'basic']:
                _number_hidden_units = self.args.joint_embed_dimension
                
            else:
                _number_hidden_units = 2 * self.args.joint_embed_dimension    
                
            self.states[self.name] = tf.zeros([self._batch_size,
                _number_hidden_units
            ])
        
        if self._distribution == 'continuous':
            # mu from network
            self._tensor['action_dist_mu'] = \
                self._policy_MLP(self._tensor['embed_joint'])
            
            # logstd cheaply estimated
            self._tensor['action_logstd'] = tf.Variable(
                (0 * self._npr.randn(1, self._output_size)).astype(np.float32),
                name="action_logstd", trainable=True
            )
            
            self._tensor['action_dist_logstd'] = tf.reshape(
                tf.tile(
                    self._tensor['action_logstd'],
                    [1, tf.shape(self._tensor['action_dist_mu'])[0] \
                    * self._batch_dimension]
                ),
                tf.concat(
                    [[tf.shape(self._tensor['action_dist_mu'])[0]],
                    [self._batch_dimension], [-1]], axis=-1
                )
            )
        
            self._policy_distribution = \
                self._proto_distribution(mu = self._tensor['action_dist_mu'],
                                logsigma = self._tensor['action_dist_logstd'])
                            
            self.outputs['output_log_p_n'] = self._reshape_batch(
                tf_distributions.gauss_log_prob(
                    self._tensor['action_dist_mu'],
                    self._tensor['action_dist_logstd'],
                    self._tensor['label_action']
                )
            )
            
        elif self._distribution == 'discrete':
            self._tensor['logits'] = \
                self._policy_MLP(self._tensor['embed_joint'])
                
            self._policy_distribution = \
                self._proto_distribution(logits = self._tensor['logits'])
                
            self.outputs['output_log_p_n'] = self._reshape_batch(
                tf_distributions.categorical_log_prob(
                    self._tensor['logits'],
                    self._tensor['label_action']
                )
            )
                
        self.outputs['output_values'] = self._reshape_batch(
            self._value_MLP(self._tensor['embed_state']) 
        )
        
        self.outputs['output_entropy'] = tf.reduce_mean(
            self._policy_distribution.entropy
        )

        if self._is_manager:
            self.outputs['output_goal'] = self._reshape_batch(tf.pad(
                self._policy_distribution.sample,
                tf.constant([[0,0],[0,0],
                [0,self._maximum_dimension - self._output_size]],
                dtype=tf.int32), "CONSTANT"
                )
            )
                
        else:
            self.action = self._reshape_batch(
                self._policy_distribution.sample
            )
            
        if self._is_manager:
            self._build_motivations()
            
        
    def _build_motivations(self):
        self._tensor['backward_padding'] = tf.constant(
            [[0,0],[self._lookahead_range,0], [0,0]]
        )
        
        self._tensor['backward_padded_state'] = tf.pad(
            self._tensor['embed_state'],
            self._tensor['backward_padding'], "CONSTANT"
        )
        
        self._tensor['backward_padded_goal'] = tf.pad(
            self._tensor['old_goal_output'],
            self._tensor['backward_padding'], "CONSTANT"
        )
        
        # helper for history matrix
        
        # padded state = [batch, ep_length+lookahead, size]
        self._tensor['backward_padded_state'] = tf.expand_dims(
            self._tensor['backward_padded_state'], 3
        )
        
        self._tensor['backward_padded_goal'] = tf.expand_dims(
            self._tensor['backward_padded_goal'], 3
        )
        
        # use extract_image_patches for strided slice
        self._tensor['history_matrix'] = tf.reshape(
            tf.squeeze(
                tf.extract_image_patches(
                    self._tensor['backward_padded_state'],
                    ksizes = [1, self.args.episode_length+1, 
                        self._output_size, 1],
                    strides = [1, 1, 1, 1],
                    rates = [1, 1, 1, 1],
                    padding = "VALID"
                )
            ),[-1, self.args.episode_length+1,
            self._lookahead_range+1, self._output_size]
        )
        
        # past goals concatenated gives order matrix
        self._tensor['order_matrix'] = tf.reshape(tf.squeeze(
                tf.extract_image_patches(
                    self._tensor['backward_padded_goal'],
                    ksizes = [1, self.args.episode_length+1,
                        self._output_size, 1],
                    strides = [1, 1, 1, 1],
                    rates = [1, 1, 1, 1],
                    padding = "VALID"
                )
            ), [-1, self.args.episode_length+1,
            self._lookahead_range+1, self._output_size]
        )
            
        # one redundant dimension gives 0 output
        self._tensor['delta_matrix'] = \
            tf.expand_dims(self._tensor['embed_state'], 2) - \
            self._tensor['history_matrix'][:,:,:-1,:]
        
        self._tensor['norm_delta'] = tf.nn.l2_normalize(
            self._tensor['delta_matrix'], axis=-1
        )
        
        self._tensor['norm_order'] = tf.nn.l2_normalize(
            self._tensor['order_matrix'][:,:,:-1,:], axis=-1
        )
        
        # reduce sum along last two dimensions
        # shifted by 1 index as rewards are computed based on currents
        self.outputs['output_motivations'] = self._reshape_batch(
            tf.reduce_sum(
            self._tensor['norm_order'] * self._tensor['norm_delta'], [-2, -1]
            )[:,1:] / self._lookahead_range
        )
        
    def _reshape_batch(self, tensor):
        return tf.reshape(tensor, tf.concat(
            [[-1], tf.shape(tensor)[2:]], axis=0))
        