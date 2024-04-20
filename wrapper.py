import gymnasium as gym

class MultiEnv():
    def __init__(self,envs):
        self.envs = envs
        self.single_action_space = gym.spaces.Discrete(4)
        self.single_observation_space = gym.spaces.Box(0, 3.0, shape=(1, 64, 64))
        pass

    def step(self, actions):
        obs, rewards, dones, truncated, infos = [], [], [], [], []
        for idx , env in enumerate(self.envs):
            action = actions[idx]
            o, r, d, t, i = env.step(action)
            obs.append(o)
            rewards.append(r)
            dones.append(d)
            truncated.append(t)
            infos.append(i)
        return obs, rewards, dones, truncated, infos

    def reset(self, seed=0, **kwargs):
        obs = []
        for idx, env in enumerate(self.envs):
            inner_seed = seed + idx
            o, _ = env.reset(seed=inner_seed, **kwargs)
            obs.append(o)
        return obs, {}

    def close(self):
        for idx, env in enumerate(self.envs):
            env.close()

    def render(self):
        for q in self.envs:
            q.render()