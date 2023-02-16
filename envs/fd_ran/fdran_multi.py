
import gym
import numpy as np
from gym.spaces import Box
from gym.wrappers import TimeLimit

from .fdran import FDRAN


# using code from https://github.com/ikostrikov/pytorch-ddpg-naf
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


class FdranMulti(object):

    def __init__(self, env_args, n_agents=2,n_actions=4, n_obs=31, **kwargs):
        self.scenario = env_args["scenario"]  # e.g. Ant-v2
        self.agent_conf = env_args["agent_conf"]  # e.g. '2x3'

        self.n_agents = n_agents
        self.n_actions = n_actions
        self.agent_obsk = env_args.get("agent_obsk",
                                                 None)  # if None, fully observable else k>=0 implies observe nearest k agents or joints
        self.agent_obsk_agents = env_args.get("agent_obsk_agents",
                                                        False)  # observe full k nearest agents (True) or just single joints (False)

                                
        self.eps_limit = env_args["eps_limit"]
        
        self.wrapped_env = NormalizedActions(
            TimeLimit(FDRAN(**env_args), max_episode_steps=self.eps_limit))
        self.timelimit_env = self.wrapped_env.env
        self.timelimit_env._max_episode_steps = self.eps_limit
        self.env = self.timelimit_env.env
        self.timelimit_env.reset()
        self.obs_size = n_obs
        self.share_obs_size = n_obs

        # COMPATIBILITY
        self.n = self.n_agents
        # self.observation_space = [Box(low=np.array([-10]*self.n_agents), high=np.array([10]*self.n_agents)) for _ in range(self.n_agents)]
        self.observation_space = [Box(low=-10, high=10, shape=(self.obs_size,)) for _ in range(self.n_agents)]
        self.share_observation_space = [Box(low=-10, high=10, shape=(self.share_obs_size,)) for _ in
                                        range(self.n_agents)]

        self.action_space = tuple([Box(self.env.action_space.low[:self.n_actions],
                                       self.env.action_space.high[:self.n_actions]) for a in
                                   range(self.n_agents)])



    def step(self, actions):

        # need to remove dummy actions that arise due to unequal action vector sizes across agents
        flat_actions = np.concatenate([actions[i][:self.action_space[i].low.shape[0]] for i in range(self.n_agents)])
        obs_n, reward_n, done_n, info_n = self.wrapped_env.step(flat_actions)
        self.steps += 1

        info = {}
        info.update(info_n)

        # if done_n:
        #     if self.steps < self.eps_limit:
        #         info["eps_limit"] = False   # the next state will be masked out
        #     else:
        #         info["eps_limit"] = True    # the next state will not be masked out
        if done_n:
            if self.steps < self.eps_limit:
                info["bad_transition"] = False  # the next state will be masked out
            else:
                info["bad_transition"] = True  # the next state will not be masked out

        # return reward_n, done_n, info
        rewards = [[reward_n]] * self.n_agents
        # print("self.n_agents", self.n_agents)
        info["cost"] = [[info["cost"]]] * self.n_agents
        dones = [done_n] * self.n_agents
        infos = [info for _ in range(self.n_agents)]
        return self.get_obs(), self.get_state(), rewards, dones, infos, self.get_avail_actions()

    def get_obs(self):
        """ Returns all agent observat3ions in a list """
        state = self.env._get_obs()
        obs_n = []
        for a in range(self.n_agents):
            agent_id_feats = np.zeros(self.n_agents, dtype=np.float32)
            agent_id_feats[a] = 1.0
            # obs_n.append(self.get_obs_agent(a))
            # obs_n.append(np.concatenate([state, self.get_obs_agent(a), agent_id_feats]))
            # obs_n.append(np.concatenate([self.get_obs_agent(a), agent_id_feats]))
            obs_i = np.concatenate([state, agent_id_feats])
            obs_i = (obs_i - np.mean(obs_i)) / np.std(obs_i)
            obs_n.append(obs_i)
        return obs_n


    def get_obs_size(self):
        """ Returns the shape of the observation """
        if self.agent_obsk is None:
            return self.get_obs_agent(0).size
        else:
            return len(self.get_obs()[0])
            # return max([len(self.get_obs_agent(agent_id)) for agent_id in range(self.n_agents)])

    def get_state(self, team=None):
        # TODO: May want global states for different teams (so cannot see what the other team is communicating e.g.)
        state = self.env._get_obs()
        share_obs = []
        for a in range(self.n_agents):
            agent_id_feats = np.zeros(self.n_agents, dtype=np.float32)
            agent_id_feats[a] = 1.0
            # share_obs.append(np.concatenate([state, self.get_obs_agent(a), agent_id_feats]))
            state_i = np.concatenate([state, agent_id_feats])
            state_i = (state_i - np.mean(state_i)) / np.std(state_i)
            share_obs.append(state_i)
        return share_obs


    def get_avail_actions(self):  # all actions are always available
        return np.ones(shape=(self.n_agents, self.n_actions,))

    def get_avail_agent_actions(self, agent_id):
        """ Returns the available actions for agent_id """
        return np.ones(shape=(self.n_actions,))

    def get_stats(self):
        return {}

    # TODO: Temp hack
    def get_agg_stats(self, stats):
        return {}

    def reset(self, **kwargs):
        """ Returns initial observations and states"""
        self.steps = 0
        self.timelimit_env.reset()
        return self.get_obs(), self.get_state(), self.get_avail_actions()

    def close(self):
        pass

    def seed(self, args):
        pass

    def get_env_info(self):

        env_info = {"state_shape": self.n_obs,
                    "obs_shape": self.n_obs,
                    "n_actions": self.n_actions(),
                    "n_agents": self.n_agents,
                    "eps_limit": self.eps_limit,
                    "action_spaces": self.action_space,
                    "actions_dtype": np.float32,
                    "normalise_actions": False
                    }
        return env_info
    