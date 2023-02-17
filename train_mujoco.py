#!/usr/bin/env python
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
from envs.multi_envs import MujocoMulti

# from envs.fd_ran.environment import Environment as MujocoMulti

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def make_env(all_args, n_thr):
    def get_env_fn(rank):
        def init_env():
            env = MujocoMulti(env_args=all_args)
            env.seed(all_args.seed + rank * 1000)
            return env
        return init_env

    if n_thr == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(n_thr)])


def main(args):
    parser = get_config()
    all_args = parser.parse_known_args(args)[0]
    print("mumu config: ", all_args)

    if all_args.alg == "mappo_lagr":
        all_args.share_policy = False
    else:
        raise NotImplementedError

    # cuda
    # all_args.cuda = True
    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("cuda flag: ", all_args.cuda, "Torch: ", torch.cuda.is_available())
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[
        0] + "/results") / all_args.env_name / all_args.scenario / all_args.alg / all_args.experiment_name
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    if all_args.use_wandb:
        run = wandb.init(config=all_args,
                         project=all_args.env_name,
                         entity=all_args.user_name,
                         notes=socket.gethostname(),
                         name=str(all_args.alg) + "_" +
                         str(all_args.experiment_name) +
                         "_seed" + str(all_args.seed),
                         group=all_args.map_name,
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
        str(all_args.alg) + "-" + str(all_args.env_name) + "-" + str(all_args.experiment_name) + "@" + str(
            all_args.user_name))

    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # env
    envs = make_env(all_args, all_args.n_rollout_threads)
    eval_envs = make_env(all_args, all_args.n_eval_rollout_threads) if all_args.use_eval else None

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "n_agents": all_args.n_agents,
        "device": device,
        "run_dir": run_dir
    }

    # TODO: run experiments
    if all_args.share_policy:
        from runner.shared.mujoco_runner import MujocoRunner as Runner
    else:
        # in origin code not implement this method
        if all_args.alg == "mappo_lagr":
            from runner.mujoco_runner_mappo_lagr import MujocoRunner as Runner
        else:
            from runner.mujoco_runner import MujocoRunner as Runner

    runner = Runner(config)
    runner.run()

    # post process
    envs.close()
    if all_args.use_eval and eval_envs is not envs:
        eval_envs.close()

    if all_args.use_wandb:
        run.finish()
    else:
        runner.writter.export_scalars_to_json(str(runner.log_dir + '/summary.json'))
        runner.writter.close()


if __name__ == "__main__":
    main(sys.argv[1:])
