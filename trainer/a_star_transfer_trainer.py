#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon September 10 06:52:00 2018

@author: matthewszhang
"""
'''
CREDIT TO TINGWU WANG FOR THIS CODE
'''

import numpy as np
import multiprocessing
import init_path
import os.path as osp
import tensorflow as tf
from util import parallel_util
from util import whitening_util
from util import replay_buffer
from util import logger
from util import misc_utils
from env import env_register

class trainer(object):
    def __init__(self, models, args, scope='trainer'):



    def run(self):
        replay_buffer = self._generate_trajectories()

        while True:
            if self._timesteps_so_far > self.args.transfer_iterations:


    def _generate_trajectories(self):
        trajectories = self.models['explore']._generate_trajectories()

