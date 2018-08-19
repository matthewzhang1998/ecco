#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 11 19:22:12 2018

@author: matthewszhang
"""
import init_path

class base_network(object):
    def __init__(self, args, input_tensor_dict,
                 output_distribution, input_state_size,
                 input_goal_size,
                 output_goal_size, maximum_dimension,
                 random_state, batch_length,
                 batch_size,
                 lookahead, name,
                 is_manager=False):
        
        self.args = args
        self._input_tensor = input_tensor_dict
        self._distribution = output_distribution
        
        self._input_state_size = input_state_size
        self._input_goal_size = input_goal_size
        self._output_size = output_goal_size
        self._maximum_dimension = maximum_dimension
        self._batch_dimension = batch_length
        self._batch_size = batch_size
        self._lookahead_range = lookahead
        self._base_dir = init_path.get_base_dir()
        
        self._npr = random_state
        self.name = name
        self._is_manager = is_manager
        
        self._tensor = {}
        self.outputs = {}
        self.states = {}
        
    def _build_preprocess(self):
        raise NotImplementedError()
        
    def _build_outputs(self):
        raise NotImplementedError()