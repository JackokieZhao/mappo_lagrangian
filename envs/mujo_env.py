
import os
from collections import OrderedDict
from os import path

import gym
import mujoco_py
import mujoco_py as mjp
import numpy as np
from gym import spaces, utils
from gym.spaces import Box
from gym.utils import seeding


class NormalizedActions(gym.ActionWrapper):

    def _action(self, action):
        action = (action + 1) / 2
        action *= (self.action_space.high - self.action_space.low)
        action += self.action_space.low
        return action

    def action(self, action_):
        return self._action(action_)

    def _reverse_action(self, action):
        action -= self.action_space.low
        action /= (self.action_space.high - self.action_space.low)
        action = action * 2 - 1
        return action


class MujoEnv(object):

    def __init__(self, env_args, n_agents=2, n_actions=4, n_obs=31, **kwargs):
        self.scenario = env_args.scenario  # e.g. Ant-v2
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.agent_obsk = env_args.agent_obsk  # if None, fully observable else k>=0 implies observe nearest k agents or
        self.eps_limit = env_args.eps_limit

        self.obs_size = n_obs
        self.share_obs_size = n_obs

        frame_skip = 5
        utils.EzPickle.__init__(self)
        self.n_agents = 2
        fullpath = os.path.join(os.path.dirname(__file__), 'mujoco/ant.xml')
        if not path.exists(fullpath):
            raise IOError("File %s does not exist" % fullpath)
        self.frame_skip = frame_skip
        self.model = mujoco_py.load_model_from_path(fullpath)
        self.sim = mujoco_py.MjSim(self.model)
        self.data = self.sim.data

        self.init_qpos = self.sim.data.qpos.ravel().copy()
        self.init_qvel = self.sim.data.qvel.ravel().copy()

        # self._set_action_space()
        self.action_space = spaces.Box(low=np.full(8, -1, dtype=np.float32),
                                       high=np.full(8, 1, dtype=np.float32))
        self.observation_space = spaces.Box(np.full(29, -float('inf'), dtype=np.float32),
                                            np.full(29, -float('inf'), dtype=np.float32))
        self.share_observation_space = [Box(low=-10, high=10, shape=(self.share_obs_size,)) for _ in
                                        range(self.n_agents)]

        # COMPATIBILITY
        # self.observation_space = [spaces.Box(np.full(29, -float('inf'), dtype=np.float32),
        #                                      np.full(29, -float('inf'), dtype=np.float32)) for _ in range(self.n_agents)]
        # self.share_observation_space = [spaces.Box(np.full(29, -float('inf'), dtype=np.float32),
        #                                            np.full(29, -float('inf'), dtype=np.float32)) for _ in
        #                                 range(self.n_agents)]
        # self.action_space = tuple([spaces.Box(low=np.full(8, -1, dtype=np.float32),
        #                                       high=np.full(8, 1, dtype=np.float32)) for a in
        #                            range(self.n_agents)])


        self.observation_space = [Box(low=-10, high=10, shape=(self.obs_size,)) for _ in range(self.n_agents)]
        self.share_observation_space = [Box(low=-10, high=10, shape=(self.share_obs_size,)) for _ in
                                        range(self.n_agents)]
        self.action_space = tuple([spaces.Box(low=np.full(self.n_actions, -1, dtype=np.float32),
                                              high=np.full(self.n_actions, 1, dtype=np.float32)) for a in
                                   range(self.n_agents)])

        
        
        self.steps = 0
        self.seed()
        self.reset()

    def step(self, actions):
        # need to remove dummy actions that arise due to unequal action vector sizes across agents
        actions = np.concatenate([actions[i][:self.action_space[i].low.shape[0]] for i in range(self.n_agents)])
        
        xposbefore = self.data.get_body_xpos("torso")[0]
        self.sim.data.ctrl[:] = actions
        for _ in range(self.frame_skip):
            self.sim.step()

        mjp.functions.mj_rnePostConstraint(self.sim.model,
                                           self.sim.data)  # calc contacts, this is a mujoco py version mismatch issue with mujoco200
        xposafter = self.data.get_body_xpos("torso")[0]
        forward_reward = (xposafter - xposbefore) / (self.model.opt.timestep * self.frame_skip)
        ctrl_cost = .5 * np.square(actions).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        survive_reward = 1.0

        # safety stuff
        yposafter = self.data.get_body_xpos("torso")[1]
        ywall = np.array([-5, 5])
        if xposafter < 20:
            y_walldist = yposafter - xposafter * np.tan(30 / 360 * 2 * np.pi) + ywall
        elif xposafter > 20 and xposafter < 60:
            y_walldist = yposafter + (xposafter - 40) * np.tan(30 / 360 * 2 * np.pi) - ywall
        elif xposafter > 60 and xposafter < 100:
            y_walldist = yposafter - (xposafter - 80) * np.tan(30 / 360 * 2 * np.pi) + ywall
        else:
            y_walldist = yposafter - 20 * np.tan(30 / 360 * 2 * np.pi) + ywall

        obj_cost = (abs(y_walldist) < 1.8).any() * 1.0
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward

        body_quat = self.data.get_body_xquat('torso')
        z_rot = 1 - 2 * (
            body_quat[1] ** 2 + body_quat[2] ** 2)  # normally xx-rotation, not sure what axes mujoco uses
        state = np.concatenate([self.sim.data.qpos.flat, self.sim.data.qvel.flat])
        notdone = np.isfinite(state).all() \
            and state[2] >= 0.2 and state[2] <= 1.0 \
            and z_rot >= -0.7
        done = not notdone
        done_cost = done * 1.0
        cost = np.clip(obj_cost + done_cost, 0, 1)
        ob = self._get_obs()

        rewards = [[reward]] * self.n_agents
        costs = [[cost]] * self.n_agents
        dones = [done] * self.n_agents

        self.steps += 1

        if self.steps >= self.eps_limit:
            self.reset()
        return ob, ob, rewards, costs, dones, self.get_avail_actions()

    def get_avail_actions(self):  # all actions are always available
        return np.ones(shape=(self.n_agents, self.n_actions,))

    def _get_obs(self):

        x = self.sim.data.qpos.flat[0]
        y = self.sim.data.qpos.flat[1]
        if x < 20:
            y_off = y - x * np.tan(30 / 360 * 2 * np.pi)
        elif x > 20 and x < 60:
            y_off = y + (x - 40) * np.tan(30 / 360 * 2 * np.pi)
        elif x > 60 and x < 100:
            y_off = y - (x - 80) * np.tan(30 / 360 * 2 * np.pi)
        else:
            y_off = y - 20 * np.tan(30 / 360 * 2 * np.pi)

        state = np.concatenate([
            self.sim.data.qpos.flat[2:-42],
            self.sim.data.qvel.flat[:-36],
            [x / 5],
            [y_off]])

        obs = []
        for a in range(self.n_agents):
            agent_id_feats = np.zeros(self.n_agents, dtype=np.float32)
            agent_id_feats[a] = 1.0
            obs_i = np.concatenate([state, agent_id_feats])
            obs_i = (obs_i - np.mean(obs_i)) / np.std(obs_i)
            obs.append(obs_i)
        return obs
    
    def reset(self):
        self.steps = 0
        self.sim.reset()
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
        qpos[-42:] = self.init_qpos[-42:]
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        qvel[-36:] = self.init_qvel[-36:]

        assert qpos.shape == (self.model.nq,) and qvel.shape == (self.model.nv,)
        old_state = self.sim.get_state()
        new_state = mujoco_py.MjSimState(old_state.time, qpos, qvel,
                                         old_state.act, old_state.udd_state)
        self.sim.set_state(new_state)
        self.sim.forward()

        return self._get_obs(), self._get_obs(), self.get_avail_actions()
    
    def close(self):
        pass


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    