import torch
import numpy as np
import gym
from gym import spaces
import collections

class StockTradingEnv(gym.Env):
    """
    StockTradingEnv is a custom environment for simulating stock trading. 
    
    Args:
        data (dict): Input data for the environment.
        input_size (int): Number of input features.
        feature_seq (int): Length of the feature sequence.
        transaction_cost (float): Cost incurred during transactions.
        risk_sensitivity (float): Risk-sensitivity parameter for the reward function. 
        weight_normalization (bool, optional): Whether to normalize the weights. Defaults to False.
        short_bias (bool, optional): Whether to have a bias towards short selling. Defaults to False.
        extra_reward_weight (float, optional): Additional weight for the reward. Defaults to 0.1.
        weight_punishment (float, optional): Punishment for weight adjustment. Defaults to 0.1.
        std_lookback (int, optional): Lookback period for calculating standard deviation. Defaults to 60.
    """
    def __init__(self, data, input_size, feature_seq, transaction_cost, risk_sensitivity, 
                    weight_normalization = False, short_bias = False, extra_reward_weight = 0.1, 
                    weight_punishment = 0.1, std_lookback = 60):
        super(StockTradingEnv, self).__init__()

        self.data = data
        self.current_step = 0
        self.input_size = input_size
        self.feature_seq = feature_seq
        self.n_steps = len(self.data) 
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32) 
        self.transaction_cost = transaction_cost 
        self.risk_sensitivity = risk_sensitivity
        self.short_bias = short_bias
        self.extra_reward_weight = extra_reward_weight
        self.weight_normalization = weight_normalization
        self.weight_punishment = weight_punishment
        self.last_action = torch.zeros(1, 1)
        self.std_factor = risk_sensitivity
        self.std_lookback = std_lookback
        self.return_history = collections.deque(maxlen=60)
        self.returns = np.zeros((len(self.data),))
        self.position = np.zeros((1,))

    def step(self, action):
        
        self.current_step += 1 
        action = action.detach().numpy().flatten()
        reward = self.reward_function(action)
        self.last_action = action              
        self.position = action 
        
        if self.current_step >= (self.n_steps - 1):   
            done = True
            next_state = torch.zeros((self.input_size, self.feature_seq), dtype=torch.float32) 
        else:
            done = False
            next_state = self.data[self.current_step][0]  

        return next_state, reward, done, {}


    def rekey_dictionary(self, original_dict):
        return {i: v for i, (k, v) in enumerate(original_dict.items())}

    def reset(self):
        self.current_step = 0
        self.last_action = torch.zeros(1, 1)
        self.last_action = torch.zeros(*self.action_space.shape)
        self.position.fill(0)
        return self.data[self.current_step][0]

    def asset_return(self, position_new, position_old, price_new, price_old, transaction_fraction=0.0002) -> np.ndarray:
        """ R_t = a_{t-1} * log(p_t / p_{t-1}) - transaction costs """
        price_changes = (price_new + float(np.finfo(np.float32).eps)) / (price_old + float(np.finfo(np.float32).eps))
        gross_return = position_new * np.log(price_changes)

        weight_as_result_of_price_change = (position_old * price_changes) / ((position_old * (price_changes - 1)) + 1)
        required_rebalance = np.abs(position_new - weight_as_result_of_price_change)
        t_costs = transaction_fraction * required_rebalance

        net_return = gross_return - t_costs
        return net_return

    def reward_function(self, action) -> float: 
        """ Returns reward signal based on the environments chosen reward function. """
        ret = self.asset_return(position_new=action, 
                        position_old=self.position, 
                        price_new=self.data[self.current_step][1].numpy(),
                        price_old=self.data[self.current_step-1][1].numpy(),
                        transaction_fraction=self.transaction_cost)
        self.returns[self.current_step-1] = ret  # Add returns to array of old returns
        risk_punishment = np.std(self.returns[max(0, (self.current_step-self.std_lookback)):self.current_step])
        return np.sum(ret) - (self.std_factor * risk_punishment)

   

