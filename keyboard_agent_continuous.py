#!/usr/bin/env python
from __future__ import print_function

import sys, gym, time
import numpy as np

#
# Test yourself as a learning agent! Pass environment name as a command-line argument, for example:
#
# python keyboard_agent.py SpaceInvadersNoFrameskip-v4
#

env = gym.make('LunarLanderContinuous-v2' if len(sys.argv)<2 else sys.argv[1])

if hasattr(env.action_space, 'n'):
    raise Exception('Expecting continuous action space')
ACTIONS = env.action_space.shape[0]
SKIP_CONTROL = 0 
DELAY = 0.05

human_agent_action = np.zeros(ACTIONS)
human_wants_restart = False
human_sets_pause = False
human_escape = False

def key_press(key, mod):
    global human_agent_action, human_wants_restart, human_sets_pause, human_escape

    if key==0xff0d: # Enter
        human_wants_restart = True
    if key==32: # Space
        human_sets_pause = not human_sets_pause
    if key==65307: # Escape
        human_escape = True

    if key==65362: # Down
        human_agent_action[0] -= 1.
    if key==65364: # Up
        human_agent_action[0] += 1.
    if ACTIONS >= 2:
        if key==65361: # Left
            human_agent_action[1] -= 1.
        if key==65363: # Right
            human_agent_action[1] += 1.

def key_release(key, mod):
    global human_agent_action
    if key==65362: # Down
        human_agent_action[0] -= -1.
    if key==65364: # Up
        human_agent_action[0] += -1.
    if ACTIONS >= 2:
        if key==65361: # Left
            human_agent_action[1] -= -1.
        if key==65363: # Right
            human_agent_action[1] += -1.

env.reset()
env.render()
env.unwrapped.viewer.window.on_key_press = key_press
env.unwrapped.viewer.window.on_key_release = key_release

def rollout(env):
    global human_agent_action, human_wants_restart, human_sets_pause
    human_wants_restart = False
    obser = env.reset()
    skip = 0
    total_reward = 0
    total_timesteps = 0
    while 1:
        if not skip:
            a = human_agent_action
            total_timesteps += 1
            skip = SKIP_CONTROL
        else:
            skip -= 1

        obser, r, done, info = env.step(a)
        print("reward %0.3f, action %s" % (r, a))
        total_reward += r
        window_still_open = env.render()
        if window_still_open==False: return False
        if done: break
        if human_wants_restart or human_escape: break
        while human_sets_pause:
            env.render()
            time.sleep(DELAY)
        time.sleep(DELAY)
    print("timesteps %i reward %0.2f" % (total_timesteps, total_reward))

print("ACTIONS={}".format(ACTIONS))
print("Press keys 1 2 3 ... to take actions 1 2 3 ...")
print("No keys pressed is taking action 0")

while 1:
    window_still_open = rollout(env)
    if window_still_open==False: break
    if human_escape: break

