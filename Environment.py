
import time
import random
from collections import deque
import numpy as np
import os

import gym
from gym import envs

import tensorflow as tf

def log_metric(log, name, value, step=0):
    summary = tf.Summary(value=[tf.Summary.Value(tag=name, simple_value=value)])
    log.add_summary(summary, global_step=step)

def find_new_path(base_path):
    if not os.path.exists(base_path):
        return base_path
    else:
        for i in range(1, 1000):
            path = base_path + '_' + str(i)
            if not os.path.exists(path):
                return path

def get_obj_parameter_values(obj):
    return [
        (a, getattr(obj,a)) 
        for a in dir(obj) 
        if not a.startswith('__') and (type(getattr(obj,a)) == int or type(getattr(obj,a)) == float)]

class Environment:
    def __init__(self, problem, run_name=None):
        self.problem = problem
        self.env = gym.make(problem)
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        self.timestep_limit = self.env.spec.timestep_limit
        # How much reward assume we keep getting per step if we get cutoff by timestep_limit
        self.cutoff_reward = 1 if self.env.spec.id.startswith("CartPole") else 0

        self.episode = 0
        self.run_name = run_name
        self.log = None
        if run_name is not None:
            log_path = find_new_path(LOG_DIR + '/' + run_name)
            self.log = tf.summary.FileWriter(log_path)

    def run(self, agent, render=False, train=True):
        state = self.env.reset()
        self.episode += 1
        step = 0
        agent.episode_start(train)
        metrics = self.EpisodeMetrics(self, agent, self.episode, agent.gamma)

        while True:
            step += 1
            if render:
                self.env.render()

            (action, Q) = agent.act(state)

            next_state, reward, done, info = self.env.step(action)
            reward_plus = reward
            if done:
                next_state = None
                if step == self.timestep_limit:
                    reward_plus += self.cutoff_reward * agent.gamma / (1 - agent.gamma)

            agent.observe((state, action, reward_plus, next_state), train)
            metrics.observe_step(step, done, reward, reward_plus, Q)

            if done:
                break
            state = next_state

        if render:
            self.env.render(close=True)

        metrics.log_episode_finish(self.log, step)

    class EpisodeMetrics:
        def __init__(self, env, agent, episode, gamma):
            self.env = env
            self.agent = agent
            self.episode = episode
            self.gamma = gamma

            self.total_reward = 0
            self.Qs = np.array([])
            self.cum_rewards = np.array([])
            self.start_time = time.time()

        def observe_step(self, step, done, reward, reward_plus, Q):
            self.total_reward += reward
            self.Qs = np.concatenate((self.Qs, [Q]))
            self.cum_rewards = np.concatenate((
                self.cum_rewards +
                reward_plus * np.exp(np.arange(self.cum_rewards.size, 0, -1)*np.log(self.gamma)),
                [reward_plus]))

        def log_episode_finish(self, log, steps):
            elapsed = time.time() - self.start_time
            agent = self.agent

            (first_Q, last_Q) = (self.Qs[0], self.Qs[-1])
            avg_Q = np.average(self.Qs)

            dQ = self.cum_rewards - self.Qs
            (first_dQ, last_dQ) = (dQ[0], dQ[-1])
            rms_dQ = np.sqrt(np.average(np.square(dQ)))

            if log is None or self.episode % 100 == 1:
                print("{:4.0f} /{:7.0f} :: reward={:3.0f}, Q=({:5.2f}, {:5.2f}, {:5.2f}), eps={:.3f}, fps={:4.0f}".format(
                    self.episode, 
                    agent.steps, 
                    self.total_reward,
                    first_Q, 
                    avg_Q, 
                    last_Q, 
                    agent.epsilon, 
                    steps/(elapsed+0.000001)))

            if log is not None:
                # first row
                log_metric(log, 'metrics/_Reward', self.total_reward, agent.steps)
                log_metric(log, 'metrics/_epsilon', agent.epsilon, agent.steps)
                log_metric(log, 'metrics/_fps', steps/elapsed, agent.steps)
                # second row
                log_metric(log, 'metrics/Q_avg', avg_Q, agent.steps)
                log_metric(log, 'metrics/Q_first', first_Q, agent.steps)
                if steps < self.env.timestep_limit:
                    log_metric(log, 'metrics/Q_last', last_Q, agent.steps)
                # third row
                log_metric(log, 'metrics/dQ_arms', rms_dQ, agent.steps)
                log_metric(log, 'metrics/dQ_first', first_dQ, agent.steps)
                log_metric(log, 'metrics/dQ_last', last_dQ, agent.steps)
                
                log_metric(log, 'metrics/episodes', self.episode, agent.steps)

                if self.episode == 1:
                    for (p, v) in get_obj_parameter_values(agent):
                        log_metric(log, 'params/' + p, v)
                    for (p, v) in get_obj_parameter_values(agent.brain):
                        log_metric(log, 'params/' + p, v)
