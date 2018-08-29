#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 17:16:12 2018

@author: matthewszhang
"""
import init_path

def get_dqn_transfer_config(parser):
    parser.add_argument("--base_policy", type=str, default='dqn'),
    # dqn, a2c
                
    parser.add_argument("--a2c_network_type", type=str, default='mlp')
    
    parser.add_argument("--a2c_entropy_coefficient", type=float, default=0.01)
    parser.add_argument("--a2c_vf_coefficient", type=float, default=0.5)
    parser.add_argument("--a2c_gradient_max", type=float, default=0.5)
    
    parser.add_argument("--a2c_lr", type=float, default=1e-4)
    parser.add_argument("--a2c_lr_schedule", type=str, default="linear")
    
    parser.add_argument("--a2c_gamma", type=float, default=0.99)
    # linear, constant
    parser.add_argument("--a2c_rms_decay", type=float, default=0.99)
    parser.add_argument("--a2c_rms_epsilon", type=float, default=1e-5)
    parser.add_argument("--a2c_iterations", type=int, default=1000)
    
    # mlp args
    parser.add_argument("--a2c_num_mlp_layers", type=int, default=2)
    parser.add_argument("--a2c_num_mlp_hidden", type=int, default=64)
    parser.add_argument("--a2c_mlp_activation", type=str, default='relu')
    
    # conv
    parser.add_argument("--a2c_convolution_network_shape", type=str,
                        default='32,8,4,64,4,2,64,3,1')
    
    # dqn args
    parser.add_argument("--dqn_network_type", type=str, default='mlp')
    # 'conv_only', 'mlp', 'lstm'
    
    # mlp args
    parser.add_argument("--dqn_num_mlp_layers", type=int, default=3)
    parser.add_argument("--dqn_num_mlp_hidden", type=int, default=128)
    parser.add_argument("--dqn_mlp_activation", type=str, default='relu')
    
    # conv
    parser.add_argument("--dqn_convolution_network_shape", type=str,
                        default='32,8,4,64,4,2,64,3,1')
    
    parser.add_argument("--dqn_buffer_size", type=int, default=20000)
    parser.add_argument("--dqn_batch_size", type=int, default=32)
    parser.add_argument("--dqn_update_epochs", type=int, default=100)
    
    parser.add_argument("--use_dqn_prioritized_replay", type=int, default=0)
    parser.add_argument("--dqn_prioritized_alpha", type=float, default=0.6)
    parser.add_argument("--dqn_beta_iters", type=int, default=None)
    parser.add_argument("--dqn_prioritized_beta", type=float, default=0.4)
    parser.add_argument("--dqn_prioritized_replay_eps",
                        type=float, default=1e-6)
    
    parser.add_argument("--dqn_epsilon", type=float, default=0.9)
    parser.add_argument("--dqn_min_epsilon", type=float, default=0.02)
    
    parser.add_argument("--train_dqn_iterations", type=int, default=1000)
    parser.add_argument("--dqn_update_target_steps", type=int, default=20000)
    parser.add_argument("--dqn_lr", type=float, default=1e-5)
    
    parser.add_argument("--dqn_gamma", type=float, default=1.0)
    parser.add_argument("--use_dqn_param_noise", type=int, default=0)
    
    parser.add_argument("--train_transfer_iterations", type=int, 
                        default=1000)

    parser.add_argument("--hindsight_correct_eps", type=float, default=.5)
    
    parser.add_argument("--transfer_joint_value_update", type=int,
                        default=0)
    
    parser.add_argument("--transfer_clip_gradients", type=int,
                        default=1)
    parser.add_argument("--transfer_clip_gradient_threshold", type=float,
                        default=0.1)
    
    parser.add_argument("--transfer_minibatches", type=int,
                        default=10)
    parser.add_argument("--transfer_policy_epochs", type=int,
                        default=5)
    parser.add_argument("--transfer_value_epochs", type=int,
                        default=10)
    
    parser.add_argument("--transfer_policy_lr", type=float,
                        default=1e-4)
    parser.add_argument("--transfer_value_lr", type=float,
                        default=1e-4)
    
    
    
    
    return parser