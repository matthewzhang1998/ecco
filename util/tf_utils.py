#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 12 16:41:48 2018

@author: matthewszhang
"""
'''
CREDIT TO TINGWU WANG FOR THIS CODE
'''

import tensorflow as tf

class get_network_weights(object):
    """ @brief:
            call this function to get the weights in the policy network
    """

    def __init__(self, session, var_list, base_namescope):
        self._session = session
        self._base_namescope = base_namescope
        # self._op is a dict, note that the base namescope is removed, as the
        # worker and the trainer has different base_namescope
        self._op = {
            var.name.replace(self._base_namescope, ''): var
            for var in var_list
        }

    def __call__(self):
        return self._session.run(self._op)


class set_network_weights(object):
    """ @brief:
            Call this function to set the weights in the policy network
    """

    def __init__(self, session, var_list, base_namescope):
        self._session = session
        self._base_namescope = base_namescope

        self._var_list = var_list
        self._placeholders = {}
        self._assigns = []

        with tf.get_default_graph().as_default():
            for var in self._var_list:
                var_name = var.name.replace(self._base_namescope, '')[1:]
                self._placeholders[var_name] = tf.placeholder(
                    tf.float32, var.get_shape()
                )
                self._assigns.append(
                    tf.assign(var, self._placeholders[var_name])
                )

    def __call__(self, weight_dict):
        assert len(weight_dict) == len(self._var_list)
        feed_dict = {}
        for var in self._var_list:
            var_name = var.name.replace(self._base_namescope, '')
            assert var_name in weight_dict
            feed_dict[self._placeholders[var_name[1:]]] = \
                weight_dict[var_name]
            # print(var.name, var_name, self._session.run(var))

        self._session.run(self._assigns, feed_dict)

