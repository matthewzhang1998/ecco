#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 17:46:18 2018

@author: matthewszhang
"""
import init_path

def get_ecco_config(parser):
    # use special networks
    parser.add_argument("--use_fixed_manager", type=int, default=0)
    parser.add_argument("--use_fixed_agent", type=int, default=0)
    parser.add_argument("--use_recurrent", type=int, default=1)
    
    parser.add_argument("--use_state_preprocessing", type=int, default=0)
    parser.add_argument("--state_preprocessing_network_shape",
                        type=str, default='64,64')
    parser.add_argument("--preprocess_activation_type",
                        type=str, default='tanh')
    parser.add_argument("--preprocess_normalizer_type",
                        type=str, default='layer_norm')
    
    parser.add_argument("--use_dilatory_network", type=int, default=1)
    
    parser.add_argument("--decoupled_managers", type=int, default=1)
    
    parser.add_argument("--manager_updates", type=int, default=5)
    parser.add_argument("--actor_updates", type=int, default=5)
    
    parser.add_argument("--use_manager_replay_only", type=int, default=1)
    
    parser.add_argument("--joint_embed_dimension", type=int, default=64)
    parser.add_argument("--joint_embed_act_type", type=str, default='tanh')
    parser.add_argument("--joint_embed_norm_type",type=str, default='none')
    parser.add_argument("--recurrent_cell_type", type=str, 
                        default='gru')
    
    parser.add_argument("--embed_goal_type", type=str, default="linear")
    parser.add_argument("--embed_goal_size", type=int, default=64)
    parser.add_argument("--state_embed_norm_type", type=str, default='none')
    parser.add_argument("--goal_embed_norm_type", type=str, default='none')
    
    parser.add_argument('--maximum_hierarchy_depth', type=int, default=2)
    
    parser.add_argument('--gae_lam', type=float, default=0.995)
    
    parser.add_argument('--manager_entropy_coefficient', type=float,
                        default=1e-4)
    parser.add_argument('--actor_entropy_coefficient', 
                        type=float, default=1e-2)
    
    parser.add_argument('--goals_dim_min', type=int, default=64)
    parser.add_argument('--goals_dim_increment', type=int, default=2)
    
    parser.add_argument('--gamma_increment', type=float, default=0.5)
    
    parser.add_argument('--lookahead_increment', type=int, default=20)
    
    parser.add_argument('--beta_min', type=float, default=0)
    parser.add_argument('--beta_max', type=float, default=1)
    
    parser.add_argument('--clip_manager', type=float, default=0.05)
    parser.add_argument('--clip_actor', type=float, default=0.05)
    
    parser.add_argument('--value_clip_manager', type=float, default=0.1)
    parser.add_argument('--value_clip_actor', type=float, default=0.1)

    return parser
