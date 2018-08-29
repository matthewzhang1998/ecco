#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 22:01:44 2018

@author: matthewszhang
"""
from .tf_networks import get_activation_func

def get_dqn_network_kwargs_from_namespace(args):
    if args.dqn_network_type is 'mlp':
        kwargs = {
            'num_layers': args.dqn_num_mlp_layers,
            'num_hidden': args.dqn_num_mlp_hidden,
            'activation': get_activation_func(args.dqn_mlp_activation)
        }
        
    elif args.dqn_network_type is 'conv_only':
        _convolutions = []
        for i in range(0, len(args.dqn_convolution_sizes), 3):
            _convolutions.append(
                tuple(args.dqn_convolution_sizes[i:i+3])
            )
            
        kwargs = {
            'convs': _convolutions
        }
        
    elif args.dqn_network_type is 'lstm':
        pass
    
    return kwargs

def get_a2c_network_kwargs_from_namespace(args):
    if args.a2c_network_type is 'mlp':
        kwargs = {
            'num_layers': args.a2c_num_mlp_layers,
            'num_hidden': args.a2c_num_mlp_hidden,
            'activation': get_activation_func(args.a2c_mlp_activation)
        }
        
    elif args.a2c_network_type is 'conv_only':
        _convolutions = []
        for i in range(0, len(args.a2c_convolution_sizes), 3):
            _convolutions.append(
                tuple(args.a2c_convolution_sizes[i:i+3])
            )
            
        kwargs = {
            'convs': _convolutions
        }
        
    elif args.a2c_network_type is 'lstm':
        pass
    
    return kwargs