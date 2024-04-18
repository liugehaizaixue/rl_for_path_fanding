import numpy as np
from load_map import MAPS_REGISTRY
import random
from env import Env

# 将地图表示为二维数组
random_key = random.choice(list(MAPS_REGISTRY.keys()))
map_str = MAPS_REGISTRY[random_key]


# 解析地图字符串并创建二维数组
map_array = np.array([[1 if c == '#' else 0 for c in line] for line in map_str.strip().split('\n')])

env = Env(map_array)
obs , info = env.reset()


while True:
    action = random.randint(0, 3)
    obs , reward , done , info = env.step(action)


