#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 20 20:32:23 2018

@author: matthewszhang
"""

class base_network(object):
    def __init__(self, args, input_tensor, input_size, latent_size,
                 network_shape, network_act, network_norm, name):
        self.args = args
        self._input_tensor = input_tensor
        self._input_size = input_size
        self._latent_size = latent_size
        self.name = name
        self._tensor = {}
        self.output_tensor = {}
        
        self._network_shape = network_shape
        self._network_act = network_act
        self._network_norm = network_norm
    
    def _build_network(self):
        raise NotImplementedError
        
    def _build_outputs(self):
        raise NotImplementedError