python ecco_main.py --task gym_sokoban_small_tiny_world --batch_size 2500 --episode_length 100 --seed 1 --num_cache 20 --gamma_max 0.1 --vae_lr 0.0001 --kl_beta 0.01 --pretrain_iterations 50 --replay_buffer_size 20000 --goals_dim_min 128 --decoupled_managers 0 --use_state_preprocessing 1 --use_replay_buffer 1 --actor_entropy_coefficient 0.1 --clip_actor 0.05 --adaptive_lr_max 1e-08 --num_minibatches 25 --replay_batch_size 2500 --output_dir log/11;
