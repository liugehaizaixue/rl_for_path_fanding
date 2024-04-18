import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from load_map import MAPS_REGISTRY
import random
from util import utils
from env import Env

# 将地图表示为二维数组
random_key = random.choice(list(MAPS_REGISTRY.keys()))
map_str = MAPS_REGISTRY[random_key]


# 解析地图字符串并创建二维数组
map_array = np.array([[1 if c == '#' else 0 for c in line] for line in map_str.strip().split('\n')])
colors = ['#FFFFFF', '#000000', '#6600CC', '#FF0000']  # 白色表示可通行区域，黑色表示障碍物,蓝色当前位置，红色终点
cmap = ListedColormap(colors)

env = Env(map_array)
obs , info = env.reset()


while True:
    action = random.randint(0, 3)
    obs , reward , done , info = env.step(action)


