'''
Author: hibana2077 hibana2077@gmail.com
Date: 2023-06-25 19:05:09
LastEditors: hibana2077 hibana2077@gmail.com
LastEditTime: 2023-07-05 09:54:31
FilePath: \fintech_studies\ai_tpsl\src\env.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''

import gym
from gym import spaces
import numpy as np
import pandas as pd

class StockTradingEnv(gym.Env):
    metadata = {'render.modes': ['human']}
  
    def __init__(self, df:pd.DataFrame, commission=0.0001, init_balance=10000, windows_size=10):
        super(StockTradingEnv, self).__init__()

        self.df = df
        self.commsion = commission
        self.init_balance = init_balance
        self.windows_size = windows_size
        self.reward_range = (0, np.inf)
        self.action_space = spaces.Discrete(3)  # buy, sell, hold
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(windows_size,df.shape[1]), dtype=np.float32)

        # Initialize state
        self.reset()

    def step(self, action):
        self.current_step += 1

        if action == 0:  # buy
            self.balance -= self.df.loc[self.current_step, "close"]
            self.position += 1
        elif action == 1:  # sell
            self.balance += self.df.loc[self.current_step, "close"]
            self.position -= 1
        # else: hold

        obs = self._next_observation()
        reward = self.balance
        done = self.balance < 0 or self.current_step >= len(self.df) - 1
        info = {}

        return obs, reward, done, info

    def reset(self):
        self.balance = self.init_balance
        self.position = 0  # Initial position
        self.current_step = 0
        return self._next_observation()

    def render(self, mode='human'):
        print(f'Step: {self.current_step}, Balance: {self.balance}, Position: {self.position}') if mode == 'human' else None

    def _next_observation(self):
        # Normalize values (this is just an example, normally you would need more features and they could be in different ranges)
        obs = np.array([
            self.df.loc[self.current_step, "open"],
            self.df.loc[self.current_step, "high"],
            self.df.loc[self.current_step, "low"],
            self.df.loc[self.current_step, "close"],
            self.df.loc[self.current_step, "volume"],
            self.position,
        ])
        return obs
