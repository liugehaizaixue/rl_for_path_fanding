from util import utils
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from load_map import MAPS_REGISTRY
import random
from copy import deepcopy

COLORS = ['#FFFFFF', '#000000', '#6600CC', '#FF0000']

class Env:
    def __init__(self, map_array, seed=40, if_render = False):
        self.pos = None
        self.target = None
        self.original_map = deepcopy(map_array)
        self.map_array = map_array
        self.map_size = self.map_array.shape
        self.if_render = if_render
        self.action_size = 4
        self.obs_size = 65
        self.seed = seed
        self.max_episode_steps = 512
        self.current_step = 0
        """ 
        ---------------> y
        |
        |
        |
        ↓
        x 
        """

    def get_action(self, action):
        if action == 0: # 上
            return [-1,0]
        elif action == 1: #下
            return [1,0]
        elif action == 2: #左
            return [0,-1]
        else: #右
            return [0,1]

    def step(self, action):
        self.current_step = self.current_step+1

        _action = self.get_action(action)
        
        if self.pos[0]+_action[0] >= 0 and self.pos[0]+_action[0] < self.map_size[0] and \
            self.pos[1]+_action[1] >= 0 and self.pos[1]+_action[1] < self.map_size[1]:
            if self.map_array[self.pos[0]+_action[0]][self.pos[1]+_action[1]] != 1:
                self.map_array[self.pos[0]][self.pos[1]] = 0
                self.pos = (self.pos[0]+_action[0] , self.pos[1]+_action[1])
                self.map_array[self.pos[0]][self.pos[1]] = 2
                reward = -0.001
            else:
                # 撞障碍
                reward = -0.002
        else:
            # 撞边界
            reward = -0.002
        
        if self.pos == self.target:
            reward = 1
            done = True
        else:
            done = False
        
        if self.current_step > self.max_episode_steps:
            truncated = True
        else:
            truncated = False

        info = ""
        obs = np.expand_dims(self.map_array, axis=0)
        if self.if_render:
            self.step_render()
        return obs , reward , done , truncated, info


    def reset(self,seed=0, if_render=False):
        self.current_step = 0
        self.seed = seed
        self.if_render = if_render
        start , target = utils.generate_start_target(self.map_array,self.seed)
        self.map_array = deepcopy(self.original_map)
        self.map_array[start[0]][start[1]] = 2
        self.map_array[target[0]][target[1]] = 3
        self.pos = start
        self.target = target

        obs = np.expand_dims(self.map_array, axis=0)
        info = ""
        if self.if_render:
            self.init_render()
        return obs , info
    
    def init_render(self):
        plt.ion()
        cmap = ListedColormap(COLORS)
        self.rendering = plt.imshow(self.map_array, cmap=cmap, interpolation='none')
        plt.axis('off')
        plt.show()
    
    def step_render(self):
        self.rendering.set_array(self.map_array)
        # 刷新图形
        plt.pause(1)
    
    def close(self):
        pass


def make_env(seed=0, if_render = False):
    random_key = random.choice(list(MAPS_REGISTRY.keys()))
    map_str = MAPS_REGISTRY[random_key]
    map_array = np.array([[1 if c == '#' else 0 for c in line] for line in map_str.strip().split('\n')])
    return Env(map_array=map_array,seed=seed, if_render=if_render)