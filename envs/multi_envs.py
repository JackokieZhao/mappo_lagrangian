
import gym
import numpy as np
from gym.spaces import Box

from .mujoco.ant import AntEnv as Environment


class TimeLimit(gym.Wrapper):
    def __init__(self, env, max_episode_steps=None):
        super(TimeLimit, self).__init__(env)
        if max_episode_steps is None and self.env.spec is not None:
            max_episode_steps = env.spec.max_episode_steps
        if self.env.spec is not None:
            self.env.spec.max_episode_steps = max_episode_steps
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = None

    def step(self, action):
        assert self._elapsed_steps is not None, "Cannot call env.step() before calling reset()"
        observe, reward, done, cost = self.env.step(action)
        [obs, obs_glb] = observe
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            done = True
        return obs, obs_glb, reward, done, cost

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        return self.env.reset(**kwargs)


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


class EnvironmentMulti(object):

    def __init__(self, env_args, n_agents=2, n_actions=4, n_obs=31, **kwargs):
        self.scenario = env_args.scenario  # e.g. Ant-v2
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.eps_limit = env_args.eps_limit

        self.wrapped_env = NormalizedActions(
            TimeLimit(Environment(env_args), max_episode_steps=self.eps_limit))
        self.timelimit_env = self.wrapped_env.env
        self.timelimit_env._max_episode_steps = self.eps_limit
        self.env = self.timelimit_env.env
        self.timelimit_env.reset()
        self.obs_size = n_obs
        self.share_obs_size = n_obs

        # COMPATIBILITY
        self.observation_space = [Box(low=-10, high=10, shape=(self.obs_size,)) for _ in range(self.n_agents)]
        self.share_observation_space = [Box(low=-10, high=10, shape=(self.share_obs_size,)) for _ in
                                        range(self.n_agents)]
        self.action_space = tuple([Box(self.env.action_space.low[:self.n_actions],
                                       self.env.action_space.high[:self.n_actions]) for a in
                                   range(self.n_agents)])

    def step(self, actions):

        # need to remove dummy actions that arise due to unequal action vector sizes across agents
        flat_actions = np.concatenate([actions[i][:self.action_space[i].low.shape[0]] for i in range(self.n_agents)])
        obs, obs_glb, reward, done, cost = self.wrapped_env.step(flat_actions)
        self.steps += 1

        rewards = [[reward]] * self.n_agents
        costs = [[cost]] * self.n_agents
        dones = [done] * self.n_agents
        return obs, obs_glb, rewards, dones, costs, self.get_avail_actions()

    def get_avail_actions(self):  # all actions are always available
        return np.ones(shape=(self.n_agents, self.n_actions,))

    def reset(self, **kwargs):
        """ Returns initial observations and states"""
        self.steps = 0
        self.timelimit_env.reset()
        return self.env.get_obs(), self.env.get_obs(), self.get_avail_actions()

    def close(self):
        pass

    def seed(self, args):
        pass
