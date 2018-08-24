#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 21 16:28:20 2018

@author: matthewszhang
"""
from collections import defaultdict
from . import ecco_ppo
from .networks import decoder_network, encoder_network
import tensorflow as tf
import numpy as np

class model(ecco_ppo.model):
    def __init__(self, *args, **kwargs):
        super(model, self).__init__(*args, **kwargs)
        
    def _build_autoencoder_networks(self):
        self._decoder_proto = decoder_network.network
        self._encoder_proto = encoder_network.network
        _return_tensor = defaultdict(list)
        
        for inverse_level in range(self.args.maximum_hierarchy_depth - 1):
            level = self.args.maximum_hierarchy_depth - inverse_level - 1
            encoder_input_tensors = {
                'raw_inputs': self._tensor['net_input']
            }
            
            # different 'input' sizes, one passed through preconstructed MLP
            _input_size = self._network_input_size
            _output_size = self._observation_size
            
            _latent_size = self._goal_size_by_level(level)
            
            name = 'manager' + str(level)
            
            # single layer encoder, rest done by state_preprocessing
            self._encoder = self._encoder_proto(
                self.args,
                encoder_input_tensors,
                _input_size,
                _latent_size,
                network_shape = [],
                network_act = 'none',
                network_norm = self.args.state_embed_norm_type,
                name = name,
            )
            
            _sample_outputs = self._encoder.output_tensor['sampled_outputs']
            
            decoder_input_tensors = {
                'raw_inputs': self._tensor['normalized_start_state'],
                'encoded_inputs': _sample_outputs,
            }
            self._decoder = self._decoder_proto(
                self.args,
                decoder_input_tensors,
                _output_size,
                _latent_size,
                network_shape = self.args.decoder_network_shape,
                network_act = self.args.decoder_act_type,
                network_norm = self.args.decoder_norm_type,
                name = 'decoder' + str(level)
            )
            
            _return_tensor['kl_loss'].append(tf.reduce_mean(
                self._encoder.output_tensor['kl_loss']))
            _return_tensor['cross_entropy_loss'].append(tf.reduce_mean(
                self._decoder.output_tensor['cross_entropy_loss']))
        
        self._tensor['kl_loss'] = tf.transpose(tf.stack(
            _return_tensor['kl_loss']))
        self._tensor['cross_entropy_loss'] = tf.transpose(tf.stack(
            _return_tensor['cross_entropy_loss'])) / self.args.vae_num_samples        
                
    def _build_loss(self):
        super(model, self)._build_loss()
        
        self._build_autoencoder_networks()
        
        self._update_operator['vae_cross_entropy_loss'] = tf.reduce_mean(
            self._tensor['cross_entropy_loss']       
        )
        
        self._update_operator['vae_kl_loss'] = tf.reduce_mean(
            self._tensor['kl_loss']        
        )
        
        self._update_operator['vae_loss'] = \
            self.args.kl_beta * self._update_operator['vae_kl_loss'] + \
            self._update_operator['vae_cross_entropy_loss']
        
        if self.args.clip_gradients:
            self._tensor['update_op_proto_vae'] = \
            tf.train.AdamOptimizer(
                learning_rate=self.args.vae_lr
            )
            
            _params = tf.trainable_variables()
            self._tensor['update_op_gradients_vae'] = \
                tf.gradients(
                        self._update_operator['vae_loss'], _params
                )
            self._tensor['update_op_gradients_vae'], _ = \
                tf.clip_by_global_norm(
                    self._tensor['update_op_gradients_vae'],
                    self.args.clip_gradient_threshold
                )
                
            self._tensor['update_op_gradients_vae'] = list(zip(
                self._tensor['update_op_gradients_vae'], _params
            ))
            
            self._update_operator['update_op_vae'] = \
                self._tensor['update_op_proto_vae'].apply_gradients(
                    self._tensor['update_op_gradients_vae']
                )
            
        else:
           self._update_operator['update_op_vae'] = \
            tf.train.AdamOptimizer(
                learning_rate=self.args.vae_lr
            ).minimize(self._update_operator['vae_loss'])
        
        
    def train(self, data_dict, replay_buffer, train_net = None):
        return_stats = defaultdict(list)
        _temp_stats = defaultdict(list)
       
        if train_net == 'vae':
            for epoch in range(self.args.vae_epochs):
                total_batch_len = len(data_dict['start_state'])
                total_batch_inds = np.arange(total_batch_len)
                self._npr.shuffle(total_batch_inds)
                minibatch_size = total_batch_len//self.args.num_minibatches
                
                for start in range(self.args.num_minibatches):
                    start = start * minibatch_size
                    end = min(start + minibatch_size, total_batch_len)
                    batch_inds = total_batch_inds[start:end]
                    feed_dict = {
                        self._input_ph['start_state']: \
                            data_dict['start_state'][batch_inds]      
                    }
                   
                    _update_keys = [key for key in self._update_operator if
                            train_net in key]
                    
                    _temp_stats = self._session.run(
                        {key: self._update_operator[key]
                        for key in _update_keys}, feed_dict     
                    )
                    
                    for key in _temp_stats:
                        if 'update_op' not in key:
                            return_stats[key].append(_temp_stats[key])
                      
            for key in return_stats:
                return_stats[key] = np.mean(np.array(return_stats[key]))
                    
            # train actor as well
            _actor_dict, return_dict = (
                super(model, self).train(data_dict, replay_buffer, 'actor')
            )
            for key in _actor_dict:
                return_stats[key] = _actor_dict[key]
            
        else:
            return_stats, return_dict = \
                super(model, self).train(data_dict, replay_buffer, train_net)
                
        return return_stats, return_dict
        
        