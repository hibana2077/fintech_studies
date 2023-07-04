'''
Author: hibana2077 hibana2077@gmail.com
Date: 2023-06-25 19:05:09
LastEditors: hibana2077 hibana2077@gmail.com
LastEditTime: 2023-07-04 21:43:32
FilePath: \fintech_studies\ai_tpsl\src\env.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import gym
from gym import spaces
import numpy as np

class StockTradingEnv(gym.Env):
    metadata = {'render.modes': ['human']}
  
    def __init__(self, df):
        super(StockTradingEnv, self).__init__()

        self.df = df
        self.reward_range = (0, np.inf)
        self.action_space = spaces.Discrete(3)  # buy, sell, hold
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(6,))

        # Initialize state
        self.reset()

    def step(self, action):
        self.current_step += 1

        if action == 0:  # buy
            self.balance -= self.df.loc[self.current_step, "Close"]
            self.position += 1
        elif action == 1:  # sell
            self.balance += self.df.loc[self.current_step, "Close"]
            self.position -= 1
        # else: hold

        obs = self._next_observation()
        reward = self.balance
        done = self.balance < 0 or self.current_step >= len(self.df) - 1
        info = {}

        return obs, reward, done, info

    def reset(self):
        self.balance = 10000  # Initial balance
        self.position = 0  # Initial position
        self.current_step = 0
        return self._next_observation()

    def render(self, mode='human'):
        print(f'Step: {self.current_step}, Balance: {self.balance}, Position: {self.position}')

    def _next_observation(self):
        # Normalize values (this is just an example, normally you would need more features and they could be in different ranges)
        obs = np.array([
            self.df.loc[self.current_step, "Open"],
            self.df.loc[self.current_step, "High"],
            self.df.loc[self.current_step, "Low"],
            self.df.loc[self.current_step, "Close"],
            self.df.loc[self.current_step, "Volume"],
            self.position,
        ])
        return obs
