import torch
from env import get_map_array
import numpy as np
import gymnasium as gym


def get_action_by_wasd(input):
    if input == "w":
        return 0
    elif input == "s":
        return 1
    elif input == "a":
        return 2
    elif input == "d":
        return 3
    else:
        raise Exception("invalid input")

map_array = get_map_array()
env = gym.make("PathFinding-v0", map_array = map_array)
o, _ = env.reset()
env.render()
done = False
for i in range(20):
    _input = input()
    action = get_action_by_wasd(_input)
    o, r, done, _, _ = env.step(action)
    env.render()
env.close()