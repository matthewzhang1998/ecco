python ecco_main.py --task gym_sokoban_small_tiny_world --batch_size 5000 --episode_length 100 --seed 1 --num_cache 20 --gamma_max 0.1 --vae_lr 1e-06 --pretrain_iterations 100 --replay_buffer_size 20000 --goals_dim_min 128 --decoupled_managers 1 --use_replay_buffer 1 --actor_entropy_coefficient 0.1 --clip_actor 0.05 --num_minibatches 50 --output_dir log/1;
python ecco_main.py --task gym_sokoban_small_tiny_world --batch_size 5000 --episode_length 100 --seed 1 --num_cache 20 --gamma_max 0.1 --vae_lr 1e-06 --pretrain_iterations 500 --replay_buffer_size 20000 --goals_dim_min 128 --decoupled_managers 0 --use_replay_buffer 0 --actor_entropy_coefficient 0.1 --clip_actor 0.05 --num_minibatches 50 --output_dir log/1;
python ecco_main.py --task gym_sokoban_small_tiny_world --batch_size 5000 --episode_length 100 --seed 1 --num_cache 20 --gamma_max 0.1 --vae_lr 1e-06 --pretrain_iterations 500 --replay_buffer_size 20000 --goals_dim_min 128 --decoupled_managers 0 --use_replay_buffer 1 --actor_entropy_coefficient 0.1 --clip_actor 0.05 --num_minibatches 50 --output_dir log/1;
