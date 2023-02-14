#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File        :environment.py
@Data        :2023/01/19 22:41:12
@Version     :1.0
@Author      :Jackokie
@Contact     :jackokie@gmail.com
'''

import mat73 as mat
import numpy as np
import torch
import numpy as np
from gym import utils
import mujoco_py as mjp
from collections import OrderedDict
import os


from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path
import gym
from .associate import access_pilot, semvs_associate
from .channel import channel_statistics, chl_estimate
from .compute import compute_se_lsfd_mmse
from .positions import gen_bs_pos, gen_ues_pos


def convert_observation_to_space(observation):
    if isinstance(observation, dict):
        space = spaces.Dict(OrderedDict([
            (key, convert_observation_to_space(value))
            for key, value in observation.items()
        ]))
    elif isinstance(observation, np.ndarray):
        low = np.full(observation.shape, -float('inf'), dtype=np.float32)
        high = np.full(observation.shape, float('inf'), dtype=np.float32)
        space = spaces.Box(low, high, dtype=observation.dtype)
    else:
        raise NotImplementedError(type(observation), observation)

    return space


class Environment():
    def __init__(self, sce_idx, device,  env_dir='./data/env/', eps_limit=1e4, se_imp_thr=0.01, M=16, N=4,
                 K=50, N_chs=50, width=500.0, p_max=200.0, tau_p=10) -> None:
        """__init__: Initi function of the class.

        Args:
            M (int, optional): The number of base station. Defaults to 16.
            K (int, optional): Number of users. Defaults to 50.
            N_ant (int, optional): Number of antennas for a base station. Defaults to 4.
            N_chs (int, optional): Number of channels for every scenario. Defaults to 50.
            width (float, optional): Width of simulation area. Defaults to 500.0.
            p_max (float, optional): Maximization power. Defaults to 200.0.
            tau (float, optional): Number of pilots. Defaults to 10.0.
        """

        # INFO: Multiple
        self._device = device
        self._sce_idx = sce_idx
        self._steps = 0
        self._ratio = 0.1

        self._W_dim = 500
        self._pos_inc = 1000 / self._W_dim

        self._width = width
        self._M = M
        self._K = K
        self._N = N
        self._N_chs = N_chs
        self._p_max = p_max
        self._tau_p = tau_p
        self._se_imp_thr = se_imp_thr

        # INFO:  load scenario from scriptz
        self.eps_lim = eps_limit

        self._se = 0
        self._state_la = []
        self._state = []

        self._reward = []
        self._reward_la = []

        self._cost = []
        self._cost_la = []

        # Environment:
        self._ues_pos = []
        self._bs_pos = []

        self._Gain = []
        self._SE = []
        self._D = []
        self._R = []
        self._R_sqrt = []

        self._state = []

        self._R_dict = []
        self._R_sqrt_dict = []
        self._Gain_dict = []
        self._acts_cont = 0

        self._load_envs(env_dir, sce_idx)

        # self._set_action_space()
        self.action_space = spaces.Box(low=np.full(8, -1, dtype=np.float32),
                                       high=np.full(8, 1, dtype=np.float32))
        self.observation_space = spaces.Box(np.full(29, -float('inf'), dtype=np.float32),
                                            np.full(29, -float('inf'), dtype=np.float32))

        self.seed()

        # reset for the environment.
        self.reset()

    @classmethod
    def from_args(cls, sce_idx, device, cfg, **kwargs):
        defaults = dict(sce_idx=sce_idx, device=device, env_dir=cfg.env_dir,
                        se_imp_thr=cfg.se_imp_thr, M=cfg.M, N=cfg.N, K=cfg.K,
                        N_chs=cfg.N_chs, width=cfg.width, p_max=cfg.p_max, tau_p=cfg.tau_p)
        defaults.update(**kwargs)
        return cls(**defaults)

    def _load_envs(self, env_dir, env_idx):

        # Load environment.
        env_file = env_dir + 'scenario_' + str(env_idx) + '.mat'
        data = mat.loadmat(env_file)    # ues_pos, bs_pos, R, R_sqrt, gain

        self._ues_pos = torch.tensor(data['ues_pos'])
        self._R_dict = torch.tensor(data['R'])
        self._R_sqrt_dict = torch.tensor(data['R_sqrt'])
        self._Gain_dict = torch.tensor(data['gain'])

    def reset(self, ):
        """reset reset the environment.

        """
        # position reset.
        self._steps = 0
        self._bs_pos = gen_bs_pos(self._M, self._width, True, 0, 0)

        # Update the reset state.
        self.step(np.zeros([self._M, 2]))
        return self._get_obs()

    def pos2idx(self, ):
        pos_convert = np.floor(self._bs_pos / 2)
        pos_idx = self._W_dim * pos_convert[:, 1] + pos_convert[:, 0]
        return pos_idx

    def step(self, acts):
        """step Environment change accoring to acts.
        Args:
            acts (float array): acts for different agents.
        """

        # INFO: Store the last state.
        self._state_la = self._state
        self._reward_la = self._reward
        self._cost_la = self._cost

        # INFO: Environment transfer.
        self._state_transfer(acts)

        # Store the current state.
        self._steps = self._steps + 1

        ob = self.get_obs()
        done = self.check_terminate(acts)
        reward = self._reward

        return ob, reward, done, self._cost

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def check_terminate(self, acts):
        if np.sum(acts) == 0:
            self._acts_cont += 1
        else:
            self._acts_cont = 0

        return self._acts_cont >= 3

    def _state_transfer(self, acts):

        # INFO: Update base station positions.
        self._bs_pos = self._bs_pos + acts

        # Acquire Gain, R, and R_sqrt.
        # bs_pos_ax = torch.floor(self._bs_pos[i])
        pos_idx = self.pos2idx()

        gain = self._Gain_dict[pos_idx, :]
        R = self._R_dict[:, :, pos_idx, :]
        R_sqrt = self._R_sqrt_dict[:, :, pos_idx, :]

        [se, se_inc, D, D_C] = self._env_transfer(gain, R, R_sqrt)

        self._state = [gain, D, D_C, self._bs_pos]

        # INFO: Reward.
        self._reward = np.sum(se_inc, 1)

        # INFO: Cost
        self._cost = self._ratio * np.sum(np.abs(acts), 1)

    def _env_transfer(self, Gain, R, R_sqrt):
        # candidate ubs and pilot allocation.
        [D_C, pilot] = access_pilot(self._M, self._K, Gain, self._tau_p)
        Hhat, H, C = chl_estimate(R, R_sqrt, self._N_chs, self._M, self._N,
                                  self._K, self._tau_p, pilot, self._p_max)

        # Statistics for channels.
        [gki_stat, gki2_stat, F_stat] = channel_statistics(
            Hhat, H, D_C, C, self._N_chs, self._M, self._N, self._K, self._p_max)

        # # Determine the access matrix for FD-RAN.
        [D, se_inc] = semvs_associate(self._se_imp_thr, self._M, self._K, self._tau_p,
                                      D_C, gki_stat, gki2_stat, F_stat, self._p_max)

        # Compute spectrum efficiency.
        se = compute_se_lsfd_mmse(self._K, D, gki_stat, gki2_stat, F_stat, self._p_max, True)

        return se, se_inc, D, D_C

    def get_obs_glb(self, state):
        """ Returns all agent observations in a list """
        [gain, D, glb_info] = state
        obs_n = []
        for a in range(self.n_agents):
            agent_id_feat = np.zeros(self._M, dtype=np.float32)
            agent_id_feat[a] = 1.0

            obs_a = torch.concat([gain[:, a], D[:, a]], dim=1)
            obs_g = torch.concat([glb_info, agent_id_feat], dim=1)

            obs_n.append([obs_a, obs_g])

        return obs_n

    def _get_obs(self, ):

        [gain, D, D_C, glb_info] = self._state

        obs_n = []
        for a in range(self.n_agents):
            # INFO: Local observe for agent a.
            obs_a = torch.concat([gain[a, :]*D_C[a, :], [D[a, :]]], axis=0)

            # INFO: Global observe --> user positions.
            agent_id_fea = torch.zeros(self._M, dtyp=torch.float32)
            agent_id_fea[a] = 1
            obs_g = torch.concat([agent_id_fea, glb_info])

            obs_n.append([obs_a, obs_g])

        return obs_n

    def get_record(self, ):
        return self.get_obs(), self._reward, self._cost, self._end

    def get_state(self,):
        """get_state: Get current state.

        """
        return self._state

    def get_state_la(self,):
        """get_state_la: Get last state.

        """
        return self._state_la

    def get_reward(self,):
        """get_reward Get current reward.

        """
        return self._se

    def get_reward_la(self,):
        """get_reward_la: Get last reward.
        """
        return self._se_la

    def terminate(self,):
        """terminate Terminate the simulation.

        """
        self._eng.quit()

    def get_bs_pos(self, idx=None):
        """get_bs_pos Get bs's positions.
        Args:
            idx (_type_): bs index.

        Returns:
            _type_: _description_
        """
        if idx is None:
            return self._bs_pos
        else:
            return self._bs_pos[idx]

    def update_bs_pos(self, pos, idx=None):
        """update_bs_pos Update the bs's positions.

        """
        if idx is None:
            self._bs_pos = pos
        else:
            self._bs_pos[idx] = pos

    def get_ues_pos(self, idx=None):
        """get_ues_pos Get ues' positions.

        """
        if idx is None:
            return self._ues_pos
        else:
            return self._ues_pos[idx]

    def update_ues_pos(self, pos, idx=None):
        """update_ues_pos Update uese' positions.

        """
        if idx is None:
            self._ues_pos = pos
        else:
            self._ues_pos[idx] = pos
