# -----------------------------------------------------------------------------
#   @author:
#       Tingwu Wang
#   @brief:
#       The environment wrapper
# -----------------------------------------------------------------------------
'''
Credit to Tingwu Wang for this code
'''

import numpy as np


class base_env(object):

    def __init__(self, env_name, rand_seed, maximum_length, misc_info={}):
        self._env_name = env_name
        self._seed = rand_seed
        self._npr = np.random.RandomState(self._seed)
        self._maximum_length = maximum_length
        self._misc_info = misc_info
        
        self._current_step = 0

        # build the environment
        self._build_env()

    def step(self, action):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def _build_env(self):
        raise NotImplementedError
        