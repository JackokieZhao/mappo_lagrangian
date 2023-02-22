#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File        :runner.py
@Data        :2023/02/20 10:21:24
@Version     :1.0
@Author      :Jackokie
@Contact     :jackokie@gmail.com
'''


import copy
import os
import time

import numpy as np
import torch
import wandb
from tensorboardX import SummaryWriter

from algorithms.r_mappo.algorithm.MACPPOPolicy import MACPPOPolicy as Policy
from algorithms.r_mappo.r_mappo_lagr import R_MAPPO_Lagr as Trainalg
from mappo_lagrangian.utils.separated_buffer import SeparatedReplayBuffer


def _t2n(x):
    return x.detach().cpu().numpy()


class MujocoRunner(object):
    """Runner class to perform training, evaluation. and data collection for SMAC. See parent class for details."""

    def __init__(self, config):
        self.configs = config['configs']
        self.envs = config['envs']
        self.eval_envs = config['eval_envs']
        self.device = config['device']
        self.n_agents = config['n_agents']

        # parameters
        self.env_name = self.configs.env_name
        self.alg = self.configs.alg
        self.experiment_name = self.configs.experiment_name
        self.use_centralized_V = self.configs.use_centralized_V
        self.use_obs_instead_of_state = self.configs.use_obs_instead_of_state
        self.num_env_steps = self.configs.num_env_steps
        self.eps_limit = self.configs.eps_limit
        self.n_rollout_threads = self.configs.n_rollout_threads
        self.n_eval_rollout_threads = self.configs.n_eval_rollout_threads
        self.use_linear_lr_decay = self.configs.use_linear_lr_decay
        self.hidden_size = self.configs.hidden_size
        self.use_wandb = self.configs.use_wandb
        self.recurrent_N = self.configs.recurrent_N
        self.use_single_network = self.configs.use_single_network
        # interval
        self.save_interval = self.configs.save_interval
        self.use_eval = self.configs.use_eval
        self.eval_interval = self.configs.eval_interval
        self.log_interval = self.configs.log_interval
        self.gamma = self.configs.gamma
        self.use_popart = self.configs.use_popart

        self.safety_bound = self.configs.safety_bound

        # dir
        self.model_dir = self.configs.model_dir

        if self.use_wandb:
            self.save_dir = str(wandb.run.dir)
        else:
            self.run_dir = config["run_dir"]
            self.log_dir = str(self.run_dir / 'logs')
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)
            self.writter = SummaryWriter(self.log_dir)
            self.save_dir = str(self.run_dir / 'models')
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)

        print("obs_glb_space: ", self.envs.obs_glb_space)
        print("obs_space: ", self.envs.obs_space)
        print("action_space: ", self.envs.action_space)

        self.policy = []
        for agent_id in range(self.n_agents):
            obs_glb_space = self.envs.obs_glb_space[agent_id] if self.use_centralized_V else \
                self.envs.obs_space[agent_id]
            # policy network
            po = Policy(self.configs,
                        self.envs.obs_space[agent_id],
                        obs_glb_space,
                        self.envs.action_space[agent_id],
                        device=self.device)
            self.policy.append(po)

        if self.model_dir is not None:
            self.restore()

        self.trainer = []
        self.buffer = []
        # todo: revise this for trpo
        for agent_id in range(self.n_agents):
            # algorithm
            tr = Trainalg(self.configs, self.policy[agent_id], device=self.device)
            # buffer
            obs_glb_space = self.envs.obs_glb_space[agent_id] if self.use_centralized_V else \
                self.envs.obs_space[agent_id]
            bu = SeparatedReplayBuffer(self.configs,
                                       self.envs.obs_space[agent_id],
                                       obs_glb_space,
                                       self.envs.action_space[agent_id])
            self.buffer.append(bu)
            self.trainer.append(tr)

    def run(self):
        self.warmup()

        start = time.time()
        episodes = int(self.num_env_steps) // self.eps_limit // self.n_rollout_threads

        train_episode_rewards = [0 for _ in range(self.n_rollout_threads)]
        train_episode_costs = [0 for _ in range(self.n_rollout_threads)]

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)

            done_episodes_rewards = []
            done_episodes_costs = []

            for step in range(self.eps_limit):
                # Sample actions
                values, actions, action_log_probs, rnn_states, rnn_states_critic, cost_preds, \
                    rnn_states_cost = self.collect(step)

                # Obser reward cost and next obs
                obs, obs_glb, rewards, costs, dones = self.envs.step(actions)
                dones_env = np.all(dones, axis=1)
                reward_env = np.mean(rewards, axis=1).flatten()
                cost_env = np.mean(costs, axis=1).flatten()
                train_episode_rewards += reward_env
                train_episode_costs += cost_env
                for t in range(self.n_rollout_threads):

                    if dones_env[t]:
                        done_episodes_rewards.append(train_episode_rewards[t])
                        train_episode_rewards[t] = 0
                        done_episodes_costs.append(train_episode_costs[t])
                        train_episode_costs[t] = 0

                data = obs, obs_glb, rewards, costs, dones, \
                    values, actions, action_log_probs, \
                    rnn_states, rnn_states_critic, cost_preds, rnn_states_cost  # fixme: it's important!!!

                # insert data into buffer
                self.insert(data)

            # compute return and update network
            self.compute()
            train_infos = self.train()

            # post process
            total_num_steps = (episode + 1) * self.eps_limit * self.n_rollout_threads
            # save model
            if (episode % self.save_interval == 0 or episode == episodes - 1):
                self.save()

            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                # print("\n Scenario {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                #       .format(self.configs.scenario,
                #               self.alg,
                #               self.experiment_name,
                #               episode,
                #               episodes,
                #               total_num_steps,
                #               self.num_env_steps,
                #               int(total_num_steps / (end - start))))

                self.log_train(train_infos, total_num_steps)

                if len(done_episodes_rewards) > 0:
                    aver_episode_rewards = np.mean(done_episodes_rewards)
                    aver_episode_costs = np.mean(done_episodes_costs)
                    self.return_aver_cost(aver_episode_costs)
                    print("some episodes done, average rewards: {}, average costs: {}".format(aver_episode_rewards,
                                                                                              aver_episode_costs))
                    self.writter.add_scalars("train_episode_rewards", {"aver_rewards": aver_episode_rewards},
                                             total_num_steps)
                    self.writter.add_scalars("train_episode_costs", {"aver_costs": aver_episode_costs},
                                             total_num_steps)

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

    def return_aver_cost(self, aver_episode_costs):
        for agent_id in range(self.n_agents):
            self.buffer[agent_id].return_aver_insert(aver_episode_costs)

    def warmup(self):
        # reset env
        obs, obs_glb = self.envs.reset()
        # replay buffer
        if not self.use_centralized_V:
            obs_glb = obs

        for agent_id in range(self.n_agents):
            # print(obs_glb[:, agent_id])
            self.buffer[agent_id].obs_glb[0] = obs_glb[:, agent_id].copy()
            self.buffer[agent_id].obs[0] = obs[:, agent_id].copy()

    @torch.no_grad()
    def collect(self, step):
        # values, actions, action_log_probs, rnn_states, rnn_states_critic, cost_preds, \
        # rnn_states_cost = self.collect(step)

        value_collector = []
        action_collector = []
        action_log_prob_collector = []
        rnn_state_collector = []
        rnn_state_critic_collector = []
        cost_preds_collector = []
        rnn_states_cost_collector = []

        for agent_id in range(self.n_agents):
            self.trainer[agent_id].prep_rollout()
            value, action, action_log_prob, rnn_state, rnn_state_critic, cost_pred, rnn_state_cost \
                = self.trainer[agent_id].policy.get_actions(self.buffer[agent_id].obs_glb[step],
                                                            self.buffer[agent_id].obs[step],
                                                            self.buffer[agent_id].rnn_states[step],
                                                            self.buffer[agent_id].rnn_states_critic[step],
                                                            self.buffer[agent_id].masks[step],
                                                            rnn_states_cost=self.buffer[agent_id].rnn_states_cost[step]
                                                            )
            value_collector.append(_t2n(value))
            action_collector.append(_t2n(action))
            action_log_prob_collector.append(_t2n(action_log_prob))
            rnn_state_collector.append(_t2n(rnn_state))
            rnn_state_critic_collector.append(_t2n(rnn_state_critic))
            cost_preds_collector.append(_t2n(cost_pred))
            rnn_states_cost_collector.append(_t2n(rnn_state_cost))
        # [self.envs, agents, dim]
        values = np.array(value_collector).transpose(1, 0, 2)
        actions = np.array(action_collector).transpose(1, 0, 2)
        action_log_probs = np.array(action_log_prob_collector).transpose(1, 0, 2)
        rnn_states = np.array(rnn_state_collector).transpose(1, 0, 2, 3)
        rnn_states_critic = np.array(rnn_state_critic_collector).transpose(1, 0, 2, 3)
        cost_preds = np.array(cost_preds_collector).transpose(1, 0, 2)
        rnn_states_cost = np.array(rnn_states_cost_collector).transpose(1, 0, 2, 3)

        return values, actions, action_log_probs, rnn_states, rnn_states_critic, cost_preds, rnn_states_cost

    def train(self):
        # have modified for SAD_PPO
        train_infos = []
        cost_train_infos = []
        # random update order
        action_dim = self.buffer[0].actions.shape[-1]
        factor = np.ones((self.eps_limit, self.n_rollout_threads, action_dim), dtype=np.float32)
        for agent_id in torch.randperm(self.n_agents):
            self.trainer[agent_id].prep_training()
            self.buffer[agent_id].update_factor(factor)
            old_actions_logprob, _ = self.trainer[agent_id].policy.actor.evaluate_actions(
                self.buffer[agent_id].obs[:-1].reshape(-1, *self.buffer[agent_id].obs.shape[2:]),
                self.buffer[agent_id].rnn_states[0:1].reshape(-1, *self.buffer[agent_id].rnn_states.shape[2:]),
                self.buffer[agent_id].actions.reshape(-1, *self.buffer[agent_id].actions.shape[2:]),
                self.buffer[agent_id].masks[:-1].reshape(-1, *self.buffer[agent_id].masks.shape[2:]),
                self.buffer[agent_id].active_masks[:-1].reshape(-1, *self.buffer[agent_id].active_masks.shape[2:]))

            # safe_buffer, cost_adv = self.buffer_filter(agent_id)
            # train_info = self.trainer[agent_id].train(safe_buffer, cost_adv)

            train_info = self.trainer[agent_id].train(self.buffer[agent_id])

            new_actions_logprob, _ = self.trainer[agent_id].policy.actor.evaluate_actions(
                self.buffer[agent_id].obs[:-1].reshape(-1, *self.buffer[agent_id].obs.shape[2:]),
                self.buffer[agent_id].rnn_states[0:1].reshape(-1, *self.buffer[agent_id].rnn_states.shape[2:]),
                self.buffer[agent_id].actions.reshape(-1, *self.buffer[agent_id].actions.shape[2:]),
                self.buffer[agent_id].masks[:-1].reshape(-1, *self.buffer[agent_id].masks.shape[2:]),
                self.buffer[agent_id].active_masks[:-1].reshape(-1, *self.buffer[agent_id].active_masks.shape[2:]))
            factor = factor * _t2n(torch.exp(new_actions_logprob - old_actions_logprob).reshape(self.eps_limit,
                                                                                                self.n_rollout_threads,
                                                                                                action_dim))
            train_infos.append(train_info)

            self.buffer[agent_id].after_update()

        return train_infos, cost_train_infos

    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_episode = 0
        eval_episode_rewards = []
        one_episode_rewards = []
        eval_episode_costs = []
        one_episode_costs = []

        for eval_i in range(self.n_eval_rollout_threads):
            one_episode_rewards.append([])
            eval_episode_rewards.append([])

            one_episode_costs.append([])
            eval_episode_costs.append([])

        eval_obs, eval_obs_glb, _ = self.eval_envs.reset()

        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, self.n_agents, self.recurrent_N, self.hidden_size),
                                   dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.n_agents, 1), dtype=np.float32)

        while True:
            eval_actions_collector = []
            eval_rnn_states_collector = []
            for agent_id in range(self.n_agents):
                self.trainer[agent_id].prep_rollout()
                eval_actions, temp_rnn_state = \
                    self.trainer[agent_id].policy.act(eval_obs[:, agent_id],
                                                      eval_rnn_states[:, agent_id],
                                                      eval_masks[:, agent_id],
                                                      deterministic=True)
                eval_rnn_states[:, agent_id] = _t2n(temp_rnn_state)
                eval_actions_collector.append(_t2n(eval_actions))

            eval_actions = np.array(eval_actions_collector).transpose(1, 0, 2)

            # Obser reward and next obs
            eval_obs, eval_obs_glb, eval_rewards, eval_costs, eval_dones, _ = self.eval_envs.step(
                eval_actions)
            for eval_i in range(self.n_eval_rollout_threads):
                one_episode_rewards[eval_i].append(eval_rewards[eval_i])
                one_episode_costs[eval_i].append(eval_costs[eval_i])

            eval_dones_env = np.all(eval_dones, axis=1)

            eval_rnn_states[eval_dones_env == True] = np.zeros(
                ((eval_dones_env == True).sum(), self.n_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)

            eval_masks = np.ones((self.configs.n_eval_rollout_threads, self.n_agents, 1), dtype=np.float32)
            eval_masks[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.n_agents, 1),
                                                          dtype=np.float32)

            for eval_i in range(self.n_eval_rollout_threads):
                if eval_dones_env[eval_i]:
                    eval_episode += 1
                    eval_episode_rewards[eval_i].append(np.sum(one_episode_rewards[eval_i], axis=0))
                    one_episode_rewards[eval_i] = []

            if eval_episode >= self.configs.eval_episodes:
                eval_episode_rewards = np.concatenate(eval_episode_rewards)
                eval_env_infos = {'eval_average_episode_rewards': eval_episode_rewards,
                                  'eval_max_episode_rewards': [np.max(eval_episode_rewards)]}
                self.log_env(eval_env_infos, total_num_steps)
                # print("eval_average_episode_rewards is {}.".format(np.mean(eval_episode_rewards)))
                break

    def insert(self, data):
        obs, obs_glb, rewards, costs, dones, \
            values, actions, action_log_probs, rnn_states, rnn_states_critic, cost_preds, rnn_states_cost = data  #

        dones_env = np.all(dones, axis=1)

        rnn_states[dones_env == True] = np.zeros(
            ((dones_env == True).sum(), self.n_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        rnn_states_critic[dones_env == True] = np.zeros(
            ((dones_env == True).sum(), self.n_agents, *self.buffer[0].rnn_states_critic.shape[2:]), dtype=np.float32)

        rnn_states_cost[dones_env == True] = np.zeros(
            ((dones_env == True).sum(), self.n_agents, *self.buffer[0].rnn_states_cost.shape[2:]), dtype=np.float32)

        masks = np.ones((self.n_rollout_threads, self.n_agents, 1), dtype=np.float32)
        masks[dones_env == True] = np.zeros(((dones_env == True).sum(), self.n_agents, 1), dtype=np.float32)

        active_masks = np.ones((self.n_rollout_threads, self.n_agents, 1), dtype=np.float32)
        active_masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)
        active_masks[dones_env == True] = np.ones(((dones_env == True).sum(), self.n_agents, 1), dtype=np.float32)

        if not self.use_centralized_V:
            obs_glb = obs

        for agent_id in range(self.n_agents):
            self.buffer[agent_id].insert(obs_glb[:, agent_id], obs[:, agent_id], rnn_states[:, agent_id],
                                         rnn_states_critic[:, agent_id], actions[:, agent_id],
                                         action_log_probs[:, agent_id],
                                         values[:, agent_id], rewards[:, agent_id], masks[:, agent_id], None,
                                         active_masks[:, agent_id], costs=costs[:, agent_id],
                                         cost_preds=cost_preds[:, agent_id],
                                         rnn_states_cost=rnn_states_cost[:, agent_id])

    @torch.no_grad()
    def compute(self):
        for agent_id in range(self.n_agents):
            self.trainer[agent_id].prep_rollout()
            next_value = self.trainer[agent_id].policy.get_values(self.buffer[agent_id].obs_glb[-1],
                                                                  self.buffer[agent_id].rnn_states_critic[-1],
                                                                  self.buffer[agent_id].masks[-1])
            next_value = _t2n(next_value)
            self.buffer[agent_id].compute_returns(next_value, self.trainer[agent_id].value_normalizer)

            next_costs = self.trainer[agent_id].policy.get_cost_values(self.buffer[agent_id].obs_glb[-1],
                                                                       self.buffer[agent_id].rnn_states_cost[-1],
                                                                       self.buffer[agent_id].masks[-1])
            next_costs = _t2n(next_costs)
            self.buffer[agent_id].compute_cost_returns(next_costs, self.trainer[agent_id].value_normalizer)

    # episode length of envs is exactly equal to buffer size, that is, num_thread = num_episode
    def buffer_filter(self, agent_id):
        eps_limit = len(self.buffer[0].rewards)
        # J constraints for all agents, just a toy example
        J = np.zeros((self.n_rollout_threads, 1), dtype=np.float32)
        for t in reversed(range(eps_limit)):
            J = self.buffer[agent_id].costs[t] + self.gamma * J

        factor = self.buffer[agent_id].factor

        if self.use_popart:
            cost_adv = self.buffer[agent_id].cost_returns[:-1] - \
                self.trainer[agent_id].value_normalizer.denormalize(self.buffer[agent_id].cost_preds[:-1])
        else:
            cost_adv = self.buffer[agent_id].cost_returns[:-1] - self.buffer[agent_id].cost_preds[:-1]

        expectation = np.mean(factor * cost_adv, axis=(0, 2))

        constraints_value = J + np.expand_dims(expectation, -1)

        del_id = []
        print("===================================================")
        print("safety_bound: ", self.safety_bound)
        for i in range(self.n_rollout_threads):
            if constraints_value[i][0] > self.safety_bound:
                del_id.append(i)

        buffer_filterd = self.remove_episodes(agent_id, del_id)
        return buffer_filterd, cost_adv

    def remove_episodes(self, agent_id, del_ids):
        buffer = copy.deepcopy(self.buffer[agent_id])
        buffer.obs_glb = (buffer.obs_glb, del_ids, 1)
        buffer.obs = (buffer.obs, del_ids, 1)
        buffer.rnn_states = (buffer.rnn_states, del_ids, 1)
        buffer.rnn_states_critic = (buffer.rnn_states_critic, del_ids, 1)
        buffer.rnn_states_cost = (buffer.rnn_states_cost, del_ids, 1)
        buffer.value_preds = (buffer.value_preds, del_ids, 1)
        buffer.returns = (buffer.returns, del_ids, 1)
        buffer.actions = (buffer.actions, del_ids, 1)
        buffer.action_log_probs = (buffer.action_log_probs, del_ids, 1)
        buffer.rewards = (buffer.rewards, del_ids, 1)
        # todo: cost should be calculated entirely
        buffer.costs = (buffer.costs, del_ids, 1)
        buffer.cost_preds = (buffer.cost_preds, del_ids, 1)
        buffer.cost_returns = (buffer.cost_returns, del_ids, 1)
        buffer.masks = (buffer.masks, del_ids, 1)
        buffer.bad_masks = (buffer.bad_masks, del_ids, 1)
        buffer.active_masks = (buffer.active_masks, del_ids, 1)
        if buffer.factor is not None:
            buffer.factor = (buffer.factor, del_ids, 1)
        return buffer

    def save(self):
        for agent_id in range(self.n_agents):
            if self.use_single_network:
                policy_model = self.trainer[agent_id].policy.model
                torch.save(policy_model.state_dict(), str(self.save_dir) + "/model_agent" + str(agent_id) + ".pt")
            else:
                policy_actor = self.trainer[agent_id].policy.actor
                torch.save(policy_actor.state_dict(), str(self.save_dir) + "/actor_agent" + str(agent_id) + ".pt")
                policy_critic = self.trainer[agent_id].policy.critic
                torch.save(policy_critic.state_dict(), str(self.save_dir) + "/critic_agent" + str(agent_id) + ".pt")

    def restore(self):
        for agent_id in range(self.n_agents):
            if self.use_single_network:
                policy_model_state_dict = torch.load(str(self.model_dir) + '/model_agent' + str(agent_id) + '.pt')
                self.policy[agent_id].model.load_state_dict(policy_model_state_dict)
            else:
                policy_actor_state_dict = torch.load(str(self.model_dir) + '/actor_agent' + str(agent_id) + '.pt')
                self.policy[agent_id].actor.load_state_dict(policy_actor_state_dict)
                policy_critic_state_dict = torch.load(str(self.model_dir) + '/critic_agent' + str(agent_id) + '.pt')
                self.policy[agent_id].critic.load_state_dict(policy_critic_state_dict)

    def log_train(self, train_infos, total_num_steps):
        print("average_step_rewards is {}.".format(np.mean(self.buffer[0].rewards)))
        train_infos[0][0]["average_step_rewards"] = 0
        for agent_id in range(self.n_agents):
            train_infos[0][agent_id]["average_step_rewards"] = np.mean(self.buffer[agent_id].rewards)
            for k, v in train_infos[0][agent_id].items():
                agent_k = "agent%i/" % agent_id + k
                if self.use_wandb:
                    wandb.log({agent_k: v}, step=total_num_steps)
                else:
                    self.writter.add_scalars(agent_k, {agent_k: v}, total_num_steps)

    def log_env(self, env_infos, total_num_steps):
        for k, v in env_infos.items():
            if len(v) > 0:
                if self.use_wandb:
                    wandb.log({k: np.mean(v)}, step=total_num_steps)
                else:
                    self.writter.add_scalars(k, {k: np.mean(v)}, total_num_steps)
