import argparse


def get_config():

    parser = argparse.ArgumentParser(
        description='mappo_lagrangian', formatter_class=argparse.RawDescriptionHelpFormatter)

    # INFO: ===========================================================================
    parser.add_argument("--seed", type=int, default=1, help="Random seed for numpy/torch")
    parser.add_argument("--cuda", action='store_false', default=False,
                        help="by default True, will use GPU to train; or else will use CPU;")
    parser.add_argument("--cuda_deterministic",
                        action='store_false', default=True, help="by default, make sure random seed effective. if set, bypass such function.")

    # INFO: ===========================================================================
    parser.add_argument('--scenario', type=str, default='Ant-v2', help="Which mujoco task to run on")
    parser.add_argument("--alg", type=str,
                        default='mappo_lagr', choices=["mappo_lagr"])
    parser.add_argument("--experiment_name", type=str, default="check",
                        help="an identifier to distinguish different experiment.")
    parser.add_argument("--env_name", type=str, default='mujoco', help="specify the name of environment")
    parser.add_argument('--agent_obsk', type=int, default=1)  # agent-specific state should be designed carefully
    parser.add_argument("--use_single_network", action='store_true', default=False)
    parser.add_argument("--user_name", type=str, default='marl',
                        help="[for wandb usage], to specify user's name for simply collecting training data.")
    parser.add_argument("--use_wandb", action='store_false', default=False,
                        help="[for wandb usage], by default True, will log date to wandb server. or else will use tensorboard to log data.")
    parser.add_argument("--use_obs_instead_of_state", action='store_true',
                        default=False, help="Whether to use global state or concatenated obs")
    parser.add_argument("--safety_bound", type=float, default=1, help="constraint upper bound")

    # INFO: ===========================================================================
    parser.add_argument("--n_rollout_threads", type=int, default=36,
                        help="Number of parallel envs for training rollouts")
    parser.add_argument("--n_training_threads", type=int,
                        default=1, help="Number of torch threads for training")

    parser.add_argument("--n_eval_rollout_threads", type=int, default=1,
                        help="Number of parallel envs for evaluating rollouts")
    parser.add_argument("--num_env_steps", type=int, default=10e6,
                        help='Number of environment steps to train (default: 10e6)')
    parser.add_argument("--eps_limit", type=int, default=200,
                        help="The gain # of last action layer")

    # INFO: ============================ Environment parameters. =============================
    parser.add_argument('--n_agents', type=int, default=10, help="Number of base stations.")
    parser.add_argument("--N", type=int,
                        default=2, help="Number of ants in each base station.")
    parser.add_argument("--K", type=int,
                        default=30, help="Number of users")
    parser.add_argument("--tau_p", type=int,
                        default=10, help="Number of pilots.")
    parser.add_argument("--n_chs", type=int,
                        default=50, help="Number of channels to be generated for every simulation")
    parser.add_argument("--width", type=float,
                        default=500, help="The maximum power for users.")
    parser.add_argument("--width_dim", type=float,
                        default=250, help="The maximum power for users.")
    parser.add_argument("--p_max", type=float,
                        default=200, help="The maximum power for users.")
    parser.add_argument("--se_inc_thr", type=float,
                        default=0.01, help="The threshold for se improvement whether surve user.")
    parser.add_argument("--se_thr", type=float,
                        default=0.5, help="The rate threshold for every user.")

    # INFO: ============================= network parameters =============================
    parser.add_argument("--share_policy", action='store_false',
                        default=False, help='Whether agent share the same policy')
    parser.add_argument("--use_centralized_V", action='store_false',
                        default=True, help="Whether to use centralized V function")
    parser.add_argument("--stacked_frames", type=int, default=1,
                        help="Dimension of hidden layers for actor/critic networks")
    parser.add_argument("--use_stacked_frames", action='store_true',
                        default=False, help="Whether to use stacked_frames")
    parser.add_argument("--hidden_size", type=int, default=64,
                        help="Dimension of hidden layers for actor/critic networks")
    parser.add_argument("--layer_N", type=int, default=1,
                        help="Number of layers for actor/critic networks")
    parser.add_argument("--use_ReLU", action='store_false',
                        default=True, help="Whether to use ReLU")
    parser.add_argument("--use_popart", action='store_false', default=True,
                        help="by default True, use running mean and std to normalize rewards.")
    parser.add_argument("--use_valuenorm", action='store_false', default=True,
                        help="by default True, use running mean and std to normalize rewards.")
    parser.add_argument("--use_feature_normalization", action='store_false',
                        default=True, help="Whether to apply layernorm to the inputs")
    parser.add_argument("--use_orthogonal", action='store_false', default=True,
                        help="Whether to use Orthogonal initialization for weights and 0 initialization for biases")
    parser.add_argument("--gain", type=float, default=0.01,
                        help="The gain # of last action layer")

    # INFO: ============================= recurrent parameters =============================
    parser.add_argument("--use_naive_recurrent_policy", action='store_true',
                        default=False, help='Whether to use a naive recurrent policy')
    parser.add_argument("--use_recurrent_policy", action='store_true',
                        default=False, help='use a recurrent policy')
    parser.add_argument("--recurrent_N", type=int, default=1, help="The number of recurrent layers.")
    parser.add_argument("--data_chunk_length", type=int, default=10,
                        help="Time length of chunks used to train a recurrent_policy")

    # INFO: ============================= optimizer parameters =============================
    parser.add_argument("--lr", type=float, default=5e-4,
                        help='learning rate (default: 5e-4)')
    parser.add_argument("--critic_lr", type=float, default=5e-4,
                        help='critic learning rate (default: 5e-4)')
    parser.add_argument("--opti_eps", type=float, default=1e-5,
                        help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--std_x_coef", type=float, default=1)
    parser.add_argument("--std_y_coef", type=float, default=0.5)

    # INFO: ================================ ppo parameters ==================================
    parser.add_argument("--ppo_epoch", type=int, default=15,
                        help='number of ppo epochs (default: 15)')
    parser.add_argument("--use_clipped_value_loss",
                        action='store_false', default=True, help="by default, clip loss value. If set, do not clip loss value.")
    parser.add_argument("--clip_param", type=float, default=0.2,
                        help='ppo clip parameter (default: 0.2)')
    parser.add_argument("--num_mini_batch", type=int, default=1,
                        help='number of batches for ppo (default: 1)')
    parser.add_argument("--entropy_coef", type=float, default=0.01,
                        help='entropy term coefficient (default: 0.01)')

    # TODO: lagrangian_coef is the lagrangian coefficient for mappo_lagrangian
    parser.add_argument("--lamda_lagr", type=float, default=0.78,
                        help='lagrangrian coef coefficient (default: 0.78)')
    parser.add_argument("--lagrangian_coef_rate", type=float, default=5e-4,
                        help='lagrangrian coef learning rate (default: 5e-4)')

    parser.add_argument("--lagrangian_coef", type=float, default=0.01,
                        help='entropy term coefficient (default: 0.01)')
    parser.add_argument("--value_loss_coef", type=float,
                        default=1, help='value loss coefficient (default: 0.5)')
    parser.add_argument("--use_max_grad_norm",
                        action='store_false', default=True, help="by default, use max norm of gradients. If set, do not use.")
    parser.add_argument("--max_grad_norm", type=float, default=10.0,
                        help='max norm of gradients (default: 0.5)')
    parser.add_argument("--use_gae", action='store_false',
                        default=True, help='use generalized advantage estimation')
    parser.add_argument("--gamma", type=float, default=0.99,
                        help='discount factor for rewards (default: 0.99)')
    parser.add_argument("--gae_lambda", type=float, default=0.95,
                        help='gae lambda parameter (default: 0.95)')
    parser.add_argument("--use_proper_time_limits", action='store_true',
                        default=False, help='compute returns taking into account time limits')
    parser.add_argument("--use_huber_loss", action='store_false', default=True,
                        help="by default, use huber loss. If set, do not use huber loss.")
    parser.add_argument("--use_value_active_masks",
                        action='store_false', default=True, help="by default True, whether to mask useless data in value loss.")
    parser.add_argument("--use_policy_active_masks",
                        action='store_false', default=True, help="by default True, whether to mask useless data in policy loss.")
    parser.add_argument("--huber_delta", type=float, default=10.0, help=" coefficience of huber loss.")
    parser.add_argument("--use_linear_lr_decay", action='store_true',
                        default=False, help='use a linear schedule on the learning rate')

    # INFO:  ========================= save parameters =============================
    parser.add_argument("--save_interval", type=int, default=1,
                        help="time duration between contiunous twice models saving.")
    parser.add_argument("--log_interval", type=int, default=5,
                        help="time duration between contiunous twice log printing.")
    parser.add_argument("--use_eval", action='store_true', default=False,
                        help="by default, do not start evaluation. If set`, start evaluation alongside with training.")
    parser.add_argument("--eval_interval", type=int, default=25,
                        help="time duration between contiunous twice evaluation progress.")
    parser.add_argument("--eval_episodes", type=int, default=32, help="number of episodes of a single evaluation.")

    # pretrained parameters
    parser.add_argument("--model_dir", type=str, default=None,
                        help="by default None. set the path to pretrained model.")
    parser.add_argument("--env_dir", type=str, default="./data/env/",
                        help="by default None. set the path to pretrained model.")

    return parser
