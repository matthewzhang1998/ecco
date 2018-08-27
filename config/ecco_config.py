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
    parser.add_argument("--preprocess_network_shape",
                        type=str, default='64,64')
    parser.add_argument("--preprocess_activation_type",
                        type=str, default='tanh')
    parser.add_argument("--preprocess_normalizer_type",
                        type=str, default='layer_norm')
    
    parser.add_argument("--use_dilatory_network", type=int, default=1)
    
    parser.add_argument("--debug_end_to_end", type=int, default=0)
    
    parser.add_argument("--decoupled_managers", type=int, default=1)
    
    parser.add_argument("--manager_updates", type=int, default=5)
    parser.add_argument("--actor_updates", type=int, default=5)
    
    parser.add_argument("--use_manager_replay_only", type=int, default=1)
    parser.add_argument("--use_hindsight_replay", type=int, default=1)
    
    parser.add_argument("--joint_embed_dimension", type=int, default=64)
    parser.add_argument("--joint_embed_act_type", type=str, default='tanh')
    parser.add_argument("--joint_embed_norm_type",type=str, default='none')
    parser.add_argument("--recurrent_cell_type", type=str, 
                        default='gru')

    parser.add_argument("--decoder_network_shape", type=str, default='64,64')
    parser.add_argument("--decoder_act_type", type=str, default='tanh')
    parser.add_argument("--decoder_norm_type",type=str, default='layer_norm')
    
    parser.add_argument("--vae_lr", type=float, default=1e-5)
    parser.add_argument("--vae_epochs", type=int, default=5)
    parser.add_argument("--kl_beta", type=float, default=0.001)
    parser.add_argument("--vae_num_samples", type=int, default=10)
    parser.add_argument("--pretrain_vae", type=int, default=1)
    parser.add_argument("--pretrain_iterations", type=int, default=50)
    
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
    
    parser.add_argument("--lr_schedule", type=str, default='adaptive')
    
    parser.add_argument("--adaptive_lr_max", type=float, default=1e-6)
    parser.add_argument("--adaptive_lr_min", type=float, default=1e-10)
    # adaptive, linear, constant
    parser.add_argument("--target_kl_high", type=float, default=2)
    parser.add_argument("--target_kl_low", type=float, default=.5)
    parser.add_argument("--target_kl_ppo", type=float, default=0.01)
    parser.add_argument("--kl_alpha", type=float, default=1.5)
    parser.add_argument("--lr_alpha", type=int, default=2)
    
    parser.add_argument('--lookahead_increment', type=int, default=20)
    
    parser.add_argument('--beta_min', type=float, default=0)
    parser.add_argument('--beta_max', type=float, default=1)
    
    parser.add_argument('--clip_manager', type=float, default=0.05)
    parser.add_argument('--clip_actor', type=float, default=0.05)
    
    parser.add_argument('--pol_loss_clip', type=float, default=5.)
    
    parser.add_argument('--value_clip_manager', type=float, default=0.1)
    parser.add_argument('--value_clip_actor', type=float, default=0.1)

    return parser
