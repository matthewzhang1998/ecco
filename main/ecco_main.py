#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 12 23:36:56 2018

@author: matthewszhang
"""
'''
Credit to tingwu wang for implementation
'''
import init_path

from config import base_config
from config import ecco_config
from main.base_main import make_sampler, make_trainer, log_results
from util import logger
from util import parallel_util
import time
from collections import OrderedDict


def train(trainer, sampler, worker, network_type, args=None):
    logger.info('Training starts at {}'.format(init_path.get_abs_base_dir()))
    
    # make the trainer and sampler
    sampler_agent = make_sampler(sampler, worker, network_type, args)
    trainer_tasks, trainer_results, trainer_agent, init_weights = \
        make_trainer(trainer, network_type, args)
    sampler_agent.set_weights(init_weights)

    timer_dict = OrderedDict()
    timer_dict['Program Start'] = time.time()
    current_iteration = 0

    while True:
        timer_dict['** Program Total Time **'] = time.time()

        # step 1: collect rollout data
        rollout_data = \
            sampler_agent._rollout_with_workers()

        timer_dict['Generate Rollout'] = time.time()

        # step 2: train the weights for dynamics and policy network
        training_info = {}
        
        if args.pretrain_vae and current_iteration < args.pretrain_iterations:
            training_info['train_net'] = 'vae'
        
        elif args.decoupled_managers:
            if (current_iteration % \
                (args.manager_updates + args.actor_updates)) \
                < args.manager_updates:
                training_info['train_net'] = 'manager'
            
            else:
                training_info['train_net'] = 'actor'
        
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

        #if totalsteps > args.max_timesteps:
        if training_return['totalsteps'] > args.max_timesteps:
            break
        else:
            current_iteration += 1

    # end of training
    sampler_agent.end()
    trainer_tasks.put((parallel_util.END_SIGNAL, None))

def main():
    parser = base_config.get_base_config()
    parser = ecco_config.get_ecco_config(parser)
    args = base_config.make_parser(parser)

    if args.write_log:
        logger.set_file_handler(path=args.output_dir,
                                prefix='ecco_ecco' + args.task,
                                time_str=args.exp_id)

    print('Training starts at {}'.format(init_path.get_abs_base_dir()))
    from trainer import ecco_trainer
    from runners import task_sampler
    from runners.workers import worker
    from policy import ecco_pretrain

    train(ecco_trainer.trainer, task_sampler, worker,
          ecco_pretrain.model, args)
    
if __name__ == '__main__':
    main()
