#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 12 23:35:19 2018

@author: matthewszhang
"""
'''
CREDIT TO TINGWU WANG
'''

import init_path
from util import parallel_util
from util import logger
import multiprocessing
import time
from collections import OrderedDict

def make_trainer(trainer, network_type, args, scope="trainer"):
    # initialized the weights for policy networks and dynamics network

    trainer_tasks = multiprocessing.JoinableQueue()
    trainer_results = multiprocessing.Queue()
    trainer_agent = trainer(args, network_type,
                                    trainer_tasks, trainer_results,
                                    scope)
    trainer_agent.start()
    trainer_tasks.put((parallel_util.START_SIGNAL, None))
    print(trainer_tasks)
    
    trainer_tasks.join()

    # init_weights: {'policy': list of weights, 'dynamics': ..., 'reward': ...}
    init_weights = trainer_results.get()
    return trainer_tasks, trainer_results, trainer_agent, init_weights


def make_sampler(sampler, worker_type, network_type, args):
    sampler_agent = sampler.sampler(args, worker_type, network_type)
    return sampler_agent


def log_results(results, timer_dict, start_timesteps=0):
    logger.info("-" * 15 + " Iteration %d " % results['iteration'] + "-" * 15)

    for i_id in range(len(timer_dict) - 1):
        start_key, end_key = list(timer_dict.keys())[i_id: i_id + 2]
        time_elapsed = (timer_dict[end_key] - timer_dict[start_key]) / 60.0

        logger.info("Time elapsed for [{}] is ".format(end_key) +
                    "%.4f mins" % time_elapsed)

    logger.info("{} total steps have happened".format(results['totalsteps']))

    # the stats
    from tensorboard_logger import log_value
    for key in results['stats']:
        logger.info("[{}]: {}".format(key, results['stats'][key]))
        if results['stats'][key] is not None:
            log_value(key, results['stats'][key], start_timesteps +
                      results['totalsteps'])


def train(trainer, sampler, worker, dynamics, policy, reward, args=None):
    logger.info('Training starts at {}'.format(init_path.get_abs_base_dir()))
    network_type = {'policy': policy, 'dynamics': dynamics, 'reward': reward}

    # make the trainer and sampler
    sampler_agent = make_sampler(sampler, worker, network_type, args)
    trainer_tasks, trainer_results, trainer_agent, init_weights = \
        make_trainer(trainer, network_type, args)
    sampler_agent.set_weights(init_weights)

    timer_dict = OrderedDict()
    timer_dict['Program Start'] = time.time()
    totalsteps = 0
    current_iteration = 0

    while True:
        timer_dict['** Program Total Time **'] = time.time()

        rollout_data = sampler_agent._rollout_with_workers()

        timer_dict['Generate Rollout'] = time.time()

        # step 2: train the weights for dynamics and policy network
        training_info = {}
        trainer_tasks.put(
            (parallel_util.TRAIN_SIGNAL,
             {'data': rollout_data['data'], 'training_info': training_info})
        )
        trainer_tasks.join()
        training_return = trainer_results.get()
        timer_dict['Train Weights'] = time.time()

        # step 4: update the weights
        sampler_agent.set_weights(training_return['network_weights'])
        timer_dict['Assign Weights'] = time.time()

        # log and print the results
        log_results(training_return, timer_dict)

        if totalsteps > args.max_timesteps:
            break
        else:
            current_iteration += 1

    # end of training
    sampler_agent.end()
    trainer_tasks.put((parallel_util.END_SIGNAL, None))
