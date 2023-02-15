import numpy as np
from gym import utils
import mujoco_py as mjp
from collections import OrderedDict
import os

import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path

import mujoco_py


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


class AntEnv(gym.Env, utils.EzPickle):
    def __init__(self, kwargs):
        model_path = 'ant.xml'
        frame_skip = 5
        utils.EzPickle.__init__(self)

        fullpath = os.path.join(os.path.dirname(__file__), "./assets", model_path)
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
        self.seed()

    def step(self, a):
        xposbefore = self.data.get_body_xpos("torso")[0]
        self.sim.data.ctrl[:] = a
        for _ in range(self.frame_skip):
            self.sim.step()

        mjp.functions.mj_rnePostConstraint(self.sim.model,
                                           self.sim.data)  # calc contacts, this is a mujoco py version mismatch issue with mujoco200
        xposafter = self.data.get_body_xpos("torso")[0]
        forward_reward = (xposafter - xposbefore) / (self.model.opt.timestep * self.frame_skip)
        ctrl_cost = .5 * np.square(a).sum()
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
        return ob, reward, done, cost

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

        return np.concatenate([
            self.sim.data.qpos.flat[2:-42],
            self.sim.data.qvel.flat[:-36],
            [x / 5],
            [y_off],
            # np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
        ])

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
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

        return self._get_obs()
