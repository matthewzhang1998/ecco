#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 20 14:31:23 2018

@author: matthewszhang
"""
from policy.networks import autoencoder
from util import tf_networks, tf_distributions

import tensorflow as tf

class network(autoencoder.base_network):
    def __init__(self, *args, **kwargs):
        super(network, self).__init__(*args, **kwargs)
        
        with tf.variable_scope(self.name):
            self._build_network()
            self._build_outputs()
        
    def _build_network(self):
        decoder_network_shape = [self._latent_size] + \
            self._network_shape + [self._input_size]
                    
        
        num_layer = len(decoder_network_shape) - 1
        act_type = [self._network_act] * num_layer
        norm_type = [self._network_norm] * num_layer
        init_data = []
        for _ in range(num_layer):
            init_data.append(
                {'w_init_method': 'normc', 'w_init_para': {'stddev': 1.0},
                 'b_init_method': 'constant', 'b_init_para': {'val': 0.0}}
            )
        self._decoder_MLP = tf_networks.MLP(
            dims=decoder_network_shape, scope= 'decoder_mlp',
            activation_type=act_type, normalizer_type=norm_type,
            train=True, init_data=init_data
        )
        
    def _build_outputs(self):
        self._tensor['decoded_outputs'] = self._decoder_MLP(
            tf.expand_dims(self._input_tensor['encoded_inputs'], axis=0)
        )[0]  
        
        self.output_tensor['cross_entropy_loss'] = tf.reduce_sum(
            tf.zeros_like(
                self._input_tensor['raw_inputs']
            ), axis=-1
        )
        
        self._tensor['positive_inputs'] = self._input_tensor['raw_inputs'] - \
            tf.reduce_min(self._input_tensor['raw_inputs'],
            axis=-1, keepdims=True)
            
        self._tensor['norm_inputs'] = self._tensor['positive_inputs'] / \
            tf.reduce_sum(self._tensor['positive_inputs'],
                axis=-1, keepdims=True
            )
            
        for i in range(self.args.vae_num_samples):
            self.output_tensor['cross_entropy_loss'] += \
                tf.reduce_sum(
                    tf.nn.softmax_cross_entropy_with_logits(
                        logits = self._tensor['decoded_outputs'][:,i],
                        labels = self._tensor['norm_inputs'], dim=-1
                    ), axis=-1
                )
                    
