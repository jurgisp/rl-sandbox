
import time
import random
from collections import deque
import numpy as np
import os
import threading
import json

import gym
from gym import envs

import tensorflow as tf

# TODO: how to define LOG_DIR so that it's global?
#LOG_DIR = None

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

class Counter(object):
    def __init__(self, value=0):
        self.v = value
        self.lock = threading.Lock()

    def inc(self, step=1):
        with self.lock:
            self.v += step
            return self.v

    def val(self):
        with self.lock:
            return self.v

class Environment:
    _env_cache = []
    _env_lock = threading.Lock()

    def __init__(self, problem, repeat_steps=1, gamma=0.99, run_name=None):
        self.problem = problem
        self.repeat_steps = repeat_steps
        self.gamma = gamma

        env = gym.make(problem)
        env.seed(random.randint(1, 999))
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.n
        self.timestep_limit = env.spec.timestep_limit
        # How much reward assume we keep getting per step if we get cutoff by timestep_limit
        self.cutoff_reward = self._get_cutoff_reward(env) * repeat_steps
        self._env_cache.append(env)
        self._last_env = None

        self.episode = Counter()
        self.episode_rewards = []
        self.total_steps = Counter()
        self.start_time = time.time()
        self.agent_parameters = None

        self.run_name = None
        self.log = None
        if run_name is not None and LOG_DIR is not None:
            log_path = find_new_path(LOG_DIR + '/' + run_name)
            self.run_name = log_path.split('/')[-1]
            self.log = tf.summary.FileWriter(log_path)
            self.log.close()

    def _get_cutoff_reward(self, env):
        if env.spec.id.startswith("CartPole-"): return 1
        if env.spec.id.startswith("MountainCar-"): return -1
        return 0

    def close(self):
        if self.log:
            self.log.close()
        if self._last_env:
            self._last_env.close()

    def log_summary(self, summary_file):
        if summary_file is not None and self.episode.val() > 0:
            elapsed = time.time() - self.start_time
            with open(summary_file, "a") as f:
                summary = {
                    'problem': self.problem,
                    'run_name': self.run_name,
                    'episodes': self.episode.val(),
                    'steps': self.total_steps.val(),
                    'sum_reward': np.sum(self.episode_rewards),
                    'avg_reward': np.mean(self.episode_rewards),
                    'avg_reward_last100': np.mean(self.episode_rewards[-100:]),
                    'elapsed': elapsed,
                    'fps': self.total_steps.val() / (elapsed + 0.000001),
                    'repeat_steps': self.repeat_steps,
                    **self.agent_parameters
                }
                print(json.dumps(summary), file = f)

    def _get_available_env(self):
        with self._env_lock:
            if self._env_cache:
                env = self._env_cache.pop()
            else:
                env = gym.make(self.problem)
            self._last_env = env
            return env
    
    def _return_available_env(self, env):
        with self._env_lock:
            self._env_cache.append(env)

    def _act_random(self, state):
        return (random.randint(0, self.action_size-1), 0, None)

    def run(self, 
        agent, 
        train=True, # If false, will not train brain (but still gather DQN experience)
        explore=True, # If true, will use epsilon-exploration, if false will be greedy
        random=False, # If true, will act random, not according to brain
        log_tensorboard=False,
        log_print=True,
        render=False, # If true, will render
        render_delay=0 # If rendering, will put delay
        ):

        episode = self.episode.inc()
        step = 0
        metrics = self.EpisodeMetrics(self, agent, episode, self.gamma)
        if self.agent_parameters is None:
            self.agent_parameters = agent.get_parameters()

        env = self._get_available_env()
        state = env.reset()
        agent.episode_start(train)
        done = False

        while not done:
            step += 1
            global_step = self.total_steps.inc()

            [action, value, eps, info] = (
                agent.act(state, global_step, explore) 
                if not random 
                else self._act_random(state))

            reward = 0
            for _ in range(self.repeat_steps):
                next_state, r, done, _ = env.step(action)
                reward += r
                if render:
                    env.render()
                    time.sleep(render_delay)
                if done:
                    break

            reward_plus = reward
            if done:
                if step < self.timestep_limit / self.repeat_steps:
                    # Actual game-over
                    next_state = None
                else:
                    # Episode interrupted because of time - don't treat it as final state in Q-learning
                    # reward_plus is with all expected future-rewards if we keep running
                    reward_plus += self.cutoff_reward * self.gamma / (1 - self.gamma)

            agent.observe([state, action, reward, next_state, info], train, done)
            metrics.observe_step(step, done, reward, reward_plus, value, eps)

            state = next_state

        if render:
            env.close()

        self._return_available_env(env)
        self.episode_rewards.append(metrics.total_reward)
        metrics.log_episode_finish(self.log, step, next_state is not None, explore, log_tensorboard, log_print)

    class EpisodeMetrics:
        def __init__(self, env, agent, episode, gamma):
            self.env = env
            self.agent = agent
            self.episode = episode
            self.gamma = gamma

            self.total_reward = 0
            self.Qs = []
            self.rewards = []
            self.epsilons = []
            self.start_time = time.time()
            self.start_steps = env.total_steps.val()

        def observe_step(self, step, done, reward, reward_plus, Q, eps):
            self.total_reward += reward
            self.Qs.append(Q)
            self.rewards.append(reward_plus)
            self.epsilons.append(eps)

        def log_episode_finish(self, log, steps, is_timestep_limit, explore, log_tensorboard, log_print):
            elapsed = time.time() - self.start_time
            agent = self.agent
            env = self.env

            (first_Q, last_Q) = (self.Qs[0], self.Qs[-1])
            avg_Q = np.average(self.Qs)

            n = len(self.rewards)
            cum_rewards = np.zeros(n)
            r = 0
            for i in range(n-1, -1, -1):
                r = r * self.gamma + self.rewards[i]
                cum_rewards[i] = r

            dQ = cum_rewards - self.Qs
            (first_dQ, last_dQ) = (dQ[0], dQ[-1])
            rms_dQ = np.sqrt(np.average(np.square(dQ)))

            total_steps = env.total_steps.val()
            steps_diff = total_steps - self.start_steps
            fps = steps_diff/(elapsed+0.000001)

            if log_print:
                print("{:4.0f} /{:7.0f} :: time_limit={} reward1={:3.0f}, reward10={:3.0f}, reward100={:3.0f}, eps={:.3f}, fps={:4.0f}".format(
                    self.episode, 
                    total_steps,
                    is_timestep_limit,
                    self.total_reward,
                    np.mean(self.env.episode_rewards[-10:]),
                    np.mean(self.env.episode_rewards[-100:]),
                    np.mean(self.epsilons), 
                    fps))

            if log_tensorboard and log is not None:
                log.reopen()

                prefix = 'metrics/' if explore else 'metrics_test/'
                
                # first row
                log_metric(log, prefix + '_Reward1', self.total_reward, total_steps)
                log_metric(log, prefix + '_Reward10', np.mean(self.env.episode_rewards[-10:]), total_steps)
                log_metric(log, prefix + '_epsilon', np.mean(self.epsilons), total_steps)                
                # second row
                log_metric(log, prefix + 'Q_avg', avg_Q, total_steps)
                log_metric(log, prefix + 'Q_first', first_Q, total_steps)
                if steps < self.env.timestep_limit:
                    log_metric(log, prefix + 'Q_last', last_Q, total_steps)
                # third row
                log_metric(log, prefix + 'dQ_arms', rms_dQ, total_steps)
                log_metric(log, prefix + 'dQ_first', first_dQ, total_steps)
                log_metric(log, prefix + 'dQ_last', last_dQ, total_steps)
                
                log_metric(log, 'monitor/z_episodes', self.episode, total_steps)
                log_metric(log, 'monitor/z_fps', fps, total_steps)

                if self.episode == 1:
                    for (p, v) in agent.get_parameters().items():
                        log_metric(log, 'params/' + p, v)
