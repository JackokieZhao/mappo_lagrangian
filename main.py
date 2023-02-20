#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File        :main.py
@Data        :2023/02/20 10:21:04
@Version     :1.0
@Author      :Jackokie
@Contact     :jackokie@gmail.com
'''


import os
import socket
import sys
from pathlib import Path

import numpy as np
import setproctitle
import torch
import wandb

from config import get_config
from envs.env_wrappers import ShareDummyVecEnv, ShareSubprocVecEnv
from envs.mujo_env import MujoEnv as Environment
from runner import MujocoRunner as Runner

# from envs.fdran_env import FdranEnv as Environment


os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def make_env(configs, n_thr):
    def get_env_fn(rank):
        def init_env():
            env = Environment(configs, rank+1)
            env.seed(configs.seed + rank * 1000)
            return env
        return init_env

    if n_thr == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(n_thr)])


def main(args):
    parser = get_config()
    configs = parser.parse_known_args(args)[0]
    print("mumu config: ", configs)

    if configs.alg == "mappo_lagr":
        configs.share_policy = False
    else:
        raise NotImplementedError

    # cuda
    # configs.cuda = True
    if configs.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(configs.n_training_threads)
        if configs.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("cuda flag: ", configs.cuda, "Torch: ", torch.cuda.is_available())
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(configs.n_training_threads)

    run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[
        0] + "/results") / configs.env_name / configs.scenario / configs.alg / configs.experiment_name
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    if configs.use_wandb:
        run = wandb.init(config=configs,
                         project=configs.env_name,
                         entity=configs.user_name,
                         notes=socket.gethostname(),
                         name=str(configs.alg) + "_" +
                         str(configs.experiment_name) +
                         "_seed" + str(configs.seed),
                         group=configs.map_name,
                         dir=str(run_dir),
                         job_type="training",
                         reinit=True)
    else:
        if not run_dir.exists():
            curr_run = 'run1'
        else:
            exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in run_dir.iterdir() if
                             str(folder.name).startswith('run')]
            if len(exst_run_nums) == 0:
                curr_run = 'run1'
            else:
                curr_run = 'run%i' % (max(exst_run_nums) + 1)
        run_dir = run_dir / curr_run
        if not run_dir.exists():
            os.makedirs(str(run_dir))

    setproctitle.setproctitle(
        str(configs.alg) + "-" + str(configs.env_name) + "-" + str(configs.experiment_name) + "@" + str(
            configs.user_name))

    # seed
    torch.manual_seed(configs.seed)
    torch.cuda.manual_seed_all(configs.seed)
    np.random.seed(configs.seed)

    # env
    envs = make_env(configs, configs.n_rollout_threads)
    eval_envs = make_env(configs, configs.n_eval_rollout_threads) if configs.use_eval else None

    config = {
        "configs": configs,
        "envs": envs,
        "eval_envs": eval_envs,
        "n_agents": configs.n_agents,
        "device": device,
        "run_dir": run_dir
    }

    runner = Runner(config)
    runner.run()

    # post process
    envs.close()
    if configs.use_eval and eval_envs is not envs:
        eval_envs.close()

    if configs.use_wandb:
        run.finish()
    else:
        runner.writter.export_scalars_to_json(str(runner.log_dir + '/summary.json'))
        runner.writter.close()


if __name__ == "__main__":
    main(sys.argv[1:])
