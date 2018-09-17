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
import init_path
import time
import copy
import os.path as osp
from collections import OrderedDict
import tensorflow as tf
from . import dqn_transfer_trainer
from util import replay_buffer
from util import logger
from util import misc_utils
from main.base_main import log_results
from env import env_register

class trainer(dqn_transfer_trainer.trainer):
    def __init__(self, models, args, scope = 'trainer',
        environment_cache=None):
        self.args = args
        self._name_scope = scope
        self._network_type = models

        # the base agent
        self._base_path = init_path.get_abs_base_dir()

        # used to save the checkpoint files
        self._npr = np.random.RandomState(args.seed)
        self.data_dict = {}
        self._environments_cache = environment_cache
        self.current_iteration = 0
        self.weights = None
        if environment_cache is None:
            self._environments_cache = []

    def run(self):
        self._set_io_size()
        self._build_session()

        timer_dict = OrderedDict()
        timer_dict['Program Start'] = time.time()

        self._build_environments()

        with self._session as sess:
            self._build_models()
            if self.weights is not None:
                self._set_weights(self.weights)
            self._network['explore'].set_environments(self._environments_cache)
            self._network['explore'].action_size = self._action_size
            buffer = self._generate_trajectories()
            self._init_whitening_stats()

            while True:
                data_dict = replay_buffer.sample(self.args.transfer_sample_traj)
                episode_length = self.args.lookahead_increment

                training_info = {'episode_length': episode_length}

                stats, _ = self.models['transfer'].train(
                    data_dict, buffer, training_info
                )

                timer_dict['** Program Total Time **'] = time.time()
                log_results(stats, timer_dict)

                if self.current_iteration > self.args.transfer_iterations:
                    _log_path = logger._get_path()
                    _save_root = 'transfer'
                    _save_extension = _save_root + \
                      "_{}_{}.ckpt".format(
                          self._name_scope, self._timesteps_so_far
                      )

                    _save_dir = osp.join(_log_path, _save_extension)
                    self._saver.save(self._session, _save_dir)
                    break

                else:
                    self.current_iteration += 1
        return self._get_weights()

    def _generate_trajectories(self):
        trajectories = self._network['explore']._generate_trajectories()
        buffer = replay_buffer.make_dummy_buffer(trajectories, self.args.seed)
        return buffer

    def _set_weights(self, network_weights):
        self.weights = network_weights

    def _build_environments(self):
        # have at least 1 environment

        if len(self._environments_cache) < 1:
            _env, self._env_info = env_register.make_env(
                self.args.task, self._npr.randint(0, 9999),
                self.args.episode_length,
                {'allow_monitor': self.args.monitor \
                                  and self._worker_id == 0}
            )
            _env.reset()

            self._environments_cache.append(copy.deepcopy(_env))