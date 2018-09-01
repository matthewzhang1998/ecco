#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 22:54:47 2018

@author: matthewszhang
"""

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
from config import dqn_transfer_config
from main.base_main import make_sampler, make_trainer, log_results
from util import logger
from util import parallel_util
import time
from collections import OrderedDict

def make_single_threaded_agent(worker_trainer_proto, models, args=None):
    return worker_trainer_proto(models, args, scope='pretrain')

def pretrain(worker_trainer, models, args=None):
    logger.info('Pretraining starts at {}'.format(
        init_path.get_abs_base_dir()))

    single_threaded_agent = make_single_threaded_agent(
        worker_trainer.trainer, models, args
    )
    
    weights, environments = single_threaded_agent.run()

    return weights, environments

def train(trainer, sampler, worker, models,
          args=None, pretrain_dict = None):
    logger.info('Training starts at {}'.format(init_path.get_abs_base_dir()))
    
    # make the trainer and sampler
    sampler_agent = make_sampler(sampler, worker, models, args)
    trainer_tasks, trainer_results, trainer_agent, init_weights = \
        make_trainer(trainer, models, args)

    if pretrain_dict is not None:
        pretrain_weights, environments_cache = \
            pretrain_dict['pretrain_fnc'](
                pretrain_dict['pretrain_thread'], models, args
            )

    else:
        pretrain_weights = environments_cache = None
        
    init_weights = init_weights \
        if pretrain_weights is None else pretrain_weights

    trainer_tasks.put(
        parallel_util.TRAINER_SET_WEIGHTS,
        init_weights
    )
    sampler_agent.set_weights(init_weights)
    if environments_cache is not None:
        sampler_agent.set_environments(environments_cache)

    timer_dict = OrderedDict()
    timer_dict['Program Start'] = time.time()
    current_iteration = 0

    while True:
        timer_dict['** Program Total Time **'] = time.time()

        training_info = {}
        rollout_info = {}
        
        if current_iteration < args.train_transfer_iterations:
            training_info['train_model'] = 'transfer'
            rollout_info['rollout_model'] = 'transfer'
        
        else:
            training_info['train_model'] = 'final'
            rollout_info['rollout_model'] = 'final'
            
            if args.freeze_actor_final:
                training_info['train_net'] = 'manager'
            
            elif args.decoupled_managers:
                if (current_iteration % \
                    (args.manager_updates + args.actor_updates)) \
                    < args.manager_updates:
                    training_info['train_net'] = 'manager'
                
                else:
                    training_info['train_net'] = 'actor'
                    
            else:
                training_info['train_net'] = None
            
        rollout_data = \
            sampler_agent._rollout_with_workers(rollout_info)

        timer_dict['Generate Rollout'] = time.time()

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
        
        if current_iteration == args.train_dqn_iterations:
            trainer_tasks.put(
                (parallel_util.SAVE_SIGNAL,
                {'net': 'base'})
            )
            
        elif current_iteration == \
            (args.train_dqn_iterations + args.train_transfer_iterations):
            trainer_tasks.put(
                parallel_util.SAVE_SIGNAL,
                {'net': 'transfer'}
            )

        elif training_return['totalsteps'] > args.max_timesteps:
            trainer_tasks.put(
                parallel_util.SAVE_SIGNAL,
                {'net': 'final'}
            )

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
    parser = dqn_transfer_config.get_dqn_transfer_config(parser)
    args = base_config.make_parser(parser)

    if args.write_log:
        logger.set_file_handler(path=args.output_dir,
                                prefix='ecco_ecco' + args.task,
                                time_str=args.exp_id)

    from trainer import dqn_transfer_trainer, dqn_transfer_jwt
    from runners import dqn_transfer_task_sampler
    from runners.workers import dqn_transfer_worker
    from policy import ecco_pretrain
    from policy import dqn_base, a2c_base
    from policy import ecco_transfer
    
    base_model = {
        'dqn': dqn_base, 'a2c':a2c_base        
    }[args.base_policy]
    
    models = {'final': ecco_pretrain.model, 'transfer': ecco_transfer.model,
           'base': base_model.model}

    train(dqn_transfer_trainer.trainer, dqn_transfer_task_sampler, 
          dqn_transfer_worker, models, args,
          {'pretrain_fnc':pretrain, 'pretrain_thread': dqn_transfer_jwt})
    
if __name__ == '__main__':
    main()
