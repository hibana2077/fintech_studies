import gym
from gym import spaces

class TradingEnv(gym.Env):
    def __init__(self, df):
        super(TradingEnv, self).__init__()

        self.df = df
        self.reward_range = (-float('inf'), float('inf'))

        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions, Box for continuous action
        self.action_space = spaces.Discrete(10)
        self.observation_space = spaces.Box(low=-float('inf'), high=float('inf'), shape=(n,))

    def step(self, action):
        '''
        Parameters
        ----------
        action : int
            The action to be executed.
        
        Returns
        -------
        observation : numpy.array
            Agent's observation of the current environment.
        reward : float
            Amount of reward returned after previous action.
        done : bool
            Whether the episode has ended, in which case further step() calls will return undefined results.
        info : dict
            Contains auxiliary diagnostic information (helpful for debugging, and sometimes learning).
        '''
        return observation, reward, done, info

    def reset(self):
        # Reset the state of the environment to an initial state
        ...
        return observation  # reward, done, info can't be included

    def render(self, mode='human'):
        # Render the environment to the screen
        ...
