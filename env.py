from util import utils
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

COLORS = ['#FFFFFF', '#000000', '#6600CC', '#FF0000']

class Env:
    def __init__(self, map_array, if_render = True):
        self.pos = None
        self.target = None
        self.map_array = map_array
        self.map_size = self.map_array.shape
        self.if_render = if_render
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
        _action = self.get_action(action)
        
        if self.pos[0]+_action[0] >= 0 and self.pos[0]+_action[0] < self.map_size[0] and \
            self.pos[1]+_action[1] >= 0 and self.pos[1]+_action[1] < self.map_size[1]:
            if self.map_array[self.pos[0]+_action[0]][self.pos[1]+_action[1]] != 1:
                self.map_array[self.pos[0]][self.pos[1]] = 0
                self.pos = (self.pos[0]+_action[0] , self.pos[1]+_action[1])
                self.map_array[self.pos[0]][self.pos[1]] = 2
                print("success")
                reward = -0.001
            else:
                print("obstacles")
                # 撞障碍
                reward = -0.002
        else:
            # 撞边界
            print("border")
            reward = -0.002
        
        if self.pos == self.target:
            done = True
        else:
            done = False
        
        info = ""
        obs = self.map_array
        self.step_render()
        return obs , reward , done , info


    def reset(self):
        start , target = utils.generate_start_target(self.map_array)
        self.map_array[start[0]][start[1]] = 2
        self.map_array[target[0]][target[1]] = 3
        self.pos = start
        self.target = target

        obs = self.map_array
        info = ""
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