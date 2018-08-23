#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 20 15:38:34 2018

@author: matthewszhang
"""
from policy.networks import autoencoder
from util import tf_networks, tf_distributions

import numpy as np
import tensorflow as tf

class network(autoencoder.base_network):
    def __init__(self, *args, **kwargs):
        super(network, self).__init__(*args, **kwargs)
        
        with tf.variable_scope(self.name):
            self._build_network()
            self._build_outputs()
        
    def _build_network(self):
        encoder_network_shape = [self._input_size] + \
            self._network_shape + [self._latent_size]
                
        num_layer = len(encoder_network_shape) - 1
        norm_type = [self._network_norm]
        init_data = []
        for _ in range(num_layer):
            init_data.append(
                {'w_init_method': 'normc', 'w_init_para': {'stddev': 1.0},
                 'b_init_method': 'constant', 'b_init_para': {'val': 0.0}}
            )
        
        self._encoder_mu_MLP = tf_networks.Linear(
            dims=encoder_network_shape, scope='state_embedding_0',
            normalizer_type=norm_type,
            train=True, init_data=init_data,
            reuse=tf.AUTO_REUSE
        )
        
        self._encoder_sigma_MLP = tf_networks.Linear(
            dims=encoder_network_shape, scope='encoder_sigma_0',
            normalizer_type=norm_type,
            train=True, init_data=init_data,
            reuse=tf.AUTO_REUSE
        )
        
    def _build_outputs(self):
        self._tensor['encoded_means'] = self._encoder_mu_MLP(
            tf.expand_dims(self._input_tensor['raw_inputs'], axis=0)
        )[0]
        
        #logstd separate variable
        self._tensor['encoded_logstds'] = self._encoder_sigma_MLP(
            tf.expand_dims(self._input_tensor['raw_inputs'], axis=0)
        )[0]
        
        # mean probability of output given gaussian distribution
        self._encoder_distribution = \
            tf_distributions.gaussian(
                self._tensor['encoded_means'], self._tensor['encoded_logstds']
            )
        
        _sample_outputs = []
        
        for _ in range(self.args.vae_num_samples):
            _sample_outputs.append(self._encoder_distribution.sample)
        
        # latent vectors will be one-dimensional
        self.output_tensor['sampled_outputs'] = tf.transpose(
            tf.stack(_sample_outputs), [1,0,2]
        )
        
        self.output_tensor['kl_loss'] = \
            tf_distributions.gauss_kl(
                self._tensor['encoded_means'], self._tensor['encoded_logstds']
            )