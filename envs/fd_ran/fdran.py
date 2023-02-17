#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File        :environment.py
@Data        :2023/01/19 22:41:12
@Version     :1.0
@Author      :Jackokie
@Contact     :jackokie@gmail.com
'''

from collections import OrderedDict

import gym
import mat73 as mat
import numpy as np
import torch
from gym import spaces, utils
from gym.utils import seeding

from .associate import access_pilot, semvs_associate
from .channel import channel_statistics, chl_estimate
from .compute import compute_se_lsfd_mmse
from .positions import gen_bs_pos


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


class FDRAN(gym.Env, utils.EzPickle):
    def __init__(self, configs, sce_idx=1, device='cpu', ) -> None:
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

        self._pos_inc = configs.width / configs.width_dim

        self._width = configs.width
        self._width_dim = configs.width_dim
        self._M = configs.n_agents
        self._K = configs.K
        self._N = configs.N
        self._N_chs = configs.n_chs
        self._p_max = configs.p_max
        self._tau_p = configs.tau_p
        self._se_inc_thr = configs.se_inc_thr

        # INFO:  load scenario from scriptz
        self.eps_lim = configs.eps_limit
        self._R_thr = configs.r_thr

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
        self._actions_cont = 0

        self._load_envs(configs.env_dir, sce_idx)

        # self._set_action_space()
        self.action_space = spaces.Box(low=np.full(self._M * 2, -10, dtype=np.float32),
                                       high=np.full(self._M * 2, 10, dtype=np.float32))

        # bs <---> user channel, bs index.
        self.obs_space = spaces.Box(low=np.full(self._K + self._M, -float('inf'), dtype=np.float32),
                                    high=np.full(self._K + self._M, float('inf'), dtype=np.float32))
        self.obs_glb_space = spaces.Box(low=np.full(self._M * 2, -float('inf'), dtype=np.float32),
                                        high=np.full(self._M * 2, float('inf'), dtype=np.float32))

        self.seed()

        # reset for the environment.
        self.reset()

    def _load_envs(self, env_dir, env_idx):

        # Load environment.
        env_file = env_dir + 'result_' + str(env_idx) + '.mat'
        data = mat.loadmat(env_file)    # ues_pos, bs_pos, R, R_sqrt, gain

        self._ues_pos = torch.tensor(data['ues_pos'])
        self._R_dict = torch.tensor(data['R'])
        self._R_sqrt_dict = torch.tensor(data['R_sqrt'])
        self._Gain_dict = torch.tensor(data['gain'])

    def reset(self, ):
        """
        The function resets the environment by generating a new set of base station positions and
        updating the reset state.

        Returns:
          The observation of the environment.
        """
        # position reset.
        self._steps = 0
        self._bs_pos = gen_bs_pos(self._M, self._width, True, 0, 0)

        # Update the reset state.
        self.step(np.zeros([self._M, 2]))
        return self._get_obs()

    def pos2idx(self, ):
        """
        It takes the position of the bounding box and converts it to an index.

        Returns:
          the index of the position of the bounding box.
        """
        pos_convert = np.floor(self._bs_pos / 2)
        pos_idx = self._width_dim * pos_convert[:, 1] + pos_convert[:, 0]
        return pos_idx

    def step(self, actions):
        """step Environment change accoring to actions.
        Args:
            actions (float array): actions for different agents.
        """

        # INFO: Store the last state.
        self._state_la = self._state
        self._reward_la = self._reward
        self._cost_la = self._cost

        # INFO: Environment transfer.
        self._state_transfer(actions)

        # Store the current state.
        self._steps = self._steps + 1

        obs = self._get_obs()
        obs_glb = self._get_obs_glb()
        done = self.check_terminate(actions)
        reward = self._reward

        return obs, obs_glb, reward, self._cost.sum(), done

    def seed(self, seed=None):
        """
        The function takes in a seed and returns a seed

        Args:
          seed: Seed for the random number generator (if None, a random seed will be used).

        Returns:
          The seed is being returned.
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def check_terminate(self, actions):
        """
        If the sum of the actions is 0, then increment the actions_cont variable by 1. If the sum of the
        actions is not 0, then set the actions_cont variable to 0. If the actions_cont variable is greater
        than or equal to 3, then return True

        Args:
          actions: the actions of the agents

        Returns:
          The number of times the agent has not taken an action.
        """
        if np.sum(actions) == 0:
            self._actions_cont += 1
        else:
            self._actions_cont = 0

        return self._actions_cont >= 3

    def _state_transfer(self, actions):

        # INFO: Update base station positions.
        self._bs_pos = self._bs_pos + actions

        # Acquire Gain, R, and R_sqrt.
        # bs_pos_ax = torch.floor(self._bs_pos[i])
        pos_idx = self.pos2idx()

        gain = self._Gain_dict[pos_idx, :]
        R = self._R_dict[:, :, pos_idx, :]
        R_sqrt = self._R_sqrt_dict[:, :, pos_idx, :]

        [se, se_inc, D, D_C] = self._env_transfer(gain, R, R_sqrt)

        self._state = [gain, D, D_C, self._bs_pos]

        # TODO: Reward: =========================================
        # self._reward = torch.sum(se_inc, 1)
        self._reward = se.sum()  # Share the reward

        # TODO: Cost: ===========================================
        # self._cost = self._ratio * np.sum(np.abs(actions))
        # self._cost = self._ratio * np.sum(np.abs(actions), 1)
        se_dif = self._R_thr - se
        self._cost = np.sum(se_dif[se_dif > 0])

    def _env_transfer(self, Gain, R, R_sqrt):
        # candidate ubs and pilot allocation.
        [D_C, pilot] = access_pilot(self._M, self._K, Gain, self._tau_p)
        Hhat, H, C = chl_estimate(R, R_sqrt, self._N_chs, self._M, self._N,
                                  self._K, self._tau_p, pilot, self._p_max)

        # Statistics for channels.
        [gki_stat, gki2_stat, F_stat] = channel_statistics(
            Hhat, H, D_C, C, self._N_chs, self._M, self._N, self._K, self._p_max)

        # # Determine the access matrix for FD-RAN.
        [D, se_inc] = semvs_associate(self._se_inc_thr, self._M, self._K, self._tau_p,
                                      D_C, gki_stat, gki2_stat, F_stat, self._p_max)

        # Compute spectrum efficiency.
        se = compute_se_lsfd_mmse(self._K, D, gki_stat, gki2_stat, F_stat, self._p_max, True)

        return se, se_inc, D, D_C

    def _get_obs_glb(self, ):
        """ Returns all agent observations in a list """
        return self._bs_pos / self._width

    def _get_obs(self, ):
        return self._state[0]

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
