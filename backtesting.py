import pandas as pd
import numpy as np 
import torch 
from sklearn.preprocessing import MinMaxScaler
from agents import AgentDDPG, AgentREINFORCE
from networks import PolicyNetworkCNN, QNetworkCNN
from trading_env import StockTradingEnv

class Backtester:
    """
    Backtester class for running backtests on stock trading strategies using DDPG and REINFORCE agents.

    Args:
        df (pd.DataFrame): DataFrame containing historical stock price data.
        window_size (int): Size of the rolling window for creating input sequences.
        train_start_date (str): Start date for the training period. Example: "2010-01-01"
        train_end_date (str): End date for the training period. Example: "2011-01-01"
        test_duration (int): Period by which to increase the training data for full refit of the models (in months)
        input_size (int): Number of input features.
        feature_seq (int): Length of the feature sequence.
        hidden_size (int): Number of hidden units in the networks.
        action_space (int): Number of possible actions.
        kernel_size (int): Size of the convolutional kernel.
        stride (int): Stride for the convolutional layers.
        relu_slope (float): Negative slope for the LeakyReLU activation function.
        dropout (float): Dropout rate for regularization.
        maxpool_kernel (int): Kernel size for the max pooling layer.
        maxpool_stride (int): Stride for the max pooling layer.
        lr_policy (float): Learning rate for the policy network.
        lr_critic (float): Learning rate for the critic network.
        criterion (nn.Module): Loss function used for training the critic network.
        batch_size (int): Number of samples in each mini batch.
        memory_size (int): Maximum size of the replay memory.
        update_freq (int): Frequency of updating the weights for DDPG networks.
        exploration_rate (float): Initial exploration rate for action selection.
        exploration_decay (float): Decay rate for the exploration rate.
        exploration_min (float): Minimum exploration rate.
        max_episodes (int): Maximum number of training episodes.
        min_train_episodes (int): Minimum number of episodes before validation and early stop.
        early_stop (int): Number of episodes with no improvement after which training stops.
        transaction_cost (float): Cost incurred during transactions.
        risk_sensitivity (float): Risk sensitivity for reward function.
        weight_normalization (bool, optional): Whether to normalize the weights. Defaults to False.
        short_bias (bool, optional): Whether to have a bias towards short selling. Defaults to False.
        extra_reward_weight (float, optional): Additional weight for the reward. Defaults to 0.0.
        weight_punishment (float, optional): Punishment for weight adjustment. Defaults to 0.0.
        val_returns (np.ndarray, optional): Validation returns used for calculating performance metrics. Defaults to None.
        val_env (gym.Env, optional): Validation environment. Defaults to None.
        train_returns (np.ndarray, optional): Training returns used for tracking training progress. Defaults to None.
        geom_ret_period (int, optional): Period for calculating geometric returns. Defaults to 252.
        normalize_rewards (bool, optional): Whether to normalize rewards. Defaults to False.
        train_mode (bool, optional): Whether the agent is in training mode. Defaults to True.
        clip_grad (bool, optional): Whether to clip gradients. Defaults to False.
        plot_loss (bool, optional): Whether to plot the loss convergence. Defaults to False.
    """
    def __init__(self, df, window_size, train_start_date, train_end_date, test_duration, input_size, 
                    feature_seq, hidden_size, action_space, kernel_size, stride, relu_slope, dropout,
                    maxpool_kernel, maxpool_stride, lr_policy, batch_size, 
                    exploration_rate, exploration_decay, exploration_min, max_episodes, 
                    min_train_episodes, early_stop, transaction_cost, risk_sensitivity,lr_critic=1e-3, 
                    criterion = torch.nn.MSELoss(), memory_size = 10000, update_freq = 100, 
                    weight_normalization=False, short_bias=False, extra_reward_weight=0.0, weight_punishment=0.0, 
                    val_returns=None, val_env = None, train_returns=None, geom_ret_period=252, normalize_rewards=False, 
                    train_mode=True, clip_grad=False, plot_loss=False):

        self.df = df
        self.window_size = window_size
        self.train_start_date = pd.to_datetime(train_start_date)
        self.train_end_date = pd.to_datetime(train_end_date)
        self.test_duration = test_duration
        self.input_size = input_size
        self.feature_seq = feature_seq
        self.transaction_cost = transaction_cost
        self.risk_sensitivity = risk_sensitivity
        self.weight_normalization = weight_normalization
        self.short_bias = short_bias
        self.extra_reward_weight = extra_reward_weight
        self.weight_punishment = weight_punishment
        self.hidden_size = hidden_size
        self.action_space = action_space
        self.relu_slope = relu_slope
        self.kernel_size = kernel_size
        self.stride = stride
        self.dropout = dropout
        self.maxpool_kernel = maxpool_kernel
        self.maxpool_stride = maxpool_stride
        self.lr = lr_policy
        self.lr_critic = lr_critic
        self.memory_size = memory_size
        self.update_freq = update_freq
        self.criterion = criterion
        self.batch_size = batch_size
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.exploration_min = exploration_min
        self.max_episodes = max_episodes
        self.min_train_episodes = min_train_episodes
        self.early_stop = early_stop
        self.val_returns = val_returns
        self.val_env = val_env
        self.train_returns = train_returns
        self.geom_ret_period = geom_ret_period
        self.normalize_rewards = normalize_rewards
        self.train_mode = train_mode
        self.clip_grad = clip_grad
        self.plot_loss = plot_loss
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self.main_dict = self.preprocess_data()
        self.splits = self.generate_train_test_splits()
    
    
    def preprocess_data(self):
        """
        Preprocess the historical stock price data to generate input features for the trading environment.

        Returns:
            dict: A dictionary with dates as keys and input features as values.
        """
        df = self.df.copy()
        L = 60
        ma_window = 50

        df['Log_Return_Close'] = np.log(df['Close'] / df['Close'].shift(1)) 
        df['Log_Return_High'] = np.log(df['High'] / df['Close'].shift(1))
        df['Log_Return_Low'] = np.log(df['Low'] / df['Close'].shift(1))

        df['Volatility'] = df['Log_Return_Close'].rolling(window=L).std() * np.sqrt(L)

        df['Normalized_Close'] = df['Log_Return_Close'] / df['Volatility']
        df['Normalized_High'] = df['Log_Return_High'] / df['Volatility']
        df['Normalized_Low'] = df['Log_Return_Low'] / df['Volatility']
        df['close_to_ma'] = (df['Close'] - df['Close'].rolling(window=ma_window).mean()) / df['Close'].rolling(window=ma_window).mean()

        df['p_close'] = df['Normalized_Close'].clip(lower=-1, upper=1)
        df['p_high'] = df['Normalized_High'].clip(lower=-1, upper=1)
        df['p_low'] = df['Normalized_Low'].clip(lower=-1, upper=1)

        df['Open'] = df['Open'].shift(-1)

        df.dropna(inplace=True)
        df.reset_index(drop=True, inplace=True)

        input_data = [
            torch.tensor(df[['p_close', 'p_high', 'p_low', 'close_to_ma']].iloc[i:i+self.window_size].values.T, dtype=torch.float32)
            for i in range(len(df) - self.window_size + 1)
        ]

        x = torch.stack(input_data)
        y = torch.tensor(df[self.window_size-1:]['Open'].tolist(), dtype=torch.float32)

        main_dict = {
            df['Date'].iloc[i + self.window_size - 1]: {'x': x[i], 'y': y[i]}
            for i in range(len(y))
        }
        
        return main_dict

    def split_and_convert(self, start_date, end_date=None):
        """
        Split the preprocessed data into training and testing sets based on the provided dates.

        Args:
            start_date (str): The start date for the split.
            end_date (str, optional): The end date for the split. Defaults to None.

        Returns:
            dict: A dictionary with indices as keys and tuples of input features and target values as values.
        """
        if end_date:
            filtered = {date: data for date, data in self.main_dict.items() if start_date <= date < end_date}
        else:
            filtered = {date: data for date, data in self.main_dict.items() if date >= start_date}

        x_tensors = torch.stack([data['x'] for data in filtered.values()])
        y_tensors = torch.tensor([data['y'] for data in filtered.values()])
        return {i: (x, y) for i, (x, y) in enumerate(zip(x_tensors, y_tensors))}

    def generate_train_test_splits(self):
        """
        Generate training and testing splits based on the specified train and test durations.

        Returns:
            list: A list of tuples, each containing training and testing dictionaries.
        """
        current_train_end = self.train_end_date
        test_splits = []
        max_date = pd.to_datetime(self.df['Date'].max())

        while current_train_end + pd.DateOffset(months=self.test_duration) <= max_date:
            train_dict = self.split_and_convert(self.train_start_date.strftime('%Y-%m-%d'), current_train_end.strftime('%Y-%m-%d'))
            test_start_date = current_train_end.strftime('%Y-%m-%d')
            test_end_date = (current_train_end + pd.DateOffset(months=self.test_duration)).strftime('%Y-%m-%d')
            test_dict = self.split_and_convert(test_start_date, test_end_date)

            test_splits.append((train_dict, test_dict))

            current_train_end += pd.DateOffset(months=self.test_duration)

        return test_splits

    
    def runREINFORCE(self):
        """
        Run the REINFORCE algorithm on the generated train and test splits.

        Returns:
            tuple: Lists of actions and asset returns for each test period.
        """

        actions = []
        asset_return = []

        for i in range(len(self.splits)):
            train_dict_i, test_dict_i = self.splits[i]
            

            train_env_i = StockTradingEnv(data=train_dict_i, input_size=self.input_size, feature_seq=self.feature_seq,
                                        transaction_cost=self.transaction_cost, risk_sensitivity=self.risk_sensitivity,
                                        weight_normalization=self.weight_normalization, short_bias=self.short_bias,
                                        extra_reward_weight=self.extra_reward_weight, weight_punishment=self.weight_punishment)
            test_env_i = StockTradingEnv(data=test_dict_i, input_size=self.input_size, feature_seq=self.feature_seq,
                                        transaction_cost=self.transaction_cost, risk_sensitivity=self.risk_sensitivity,
                                        weight_normalization=self.weight_normalization, short_bias=self.short_bias,
                                        extra_reward_weight=self.extra_reward_weight, weight_punishment=self.weight_punishment)


            policy_i = PolicyNetworkCNN(input_size=self.input_size, hidden_size=self.hidden_size, 
                                    action_space=self.action_space, feature_seq=self.feature_seq,
                                    relu_slope=self.relu_slope, kernel_size=self.kernel_size, 
                                    stride=self.stride, dropout=self.dropout, 
                                    maxpool_kernel=self.maxpool_kernel, maxpool_stride=self.maxpool_stride)
            
            agent_i = AgentREINFORCE(env=train_env_i, policy_network=policy_i, lr=self.lr, batch_size=self.batch_size, 
                                    exploration_rate=self.exploration_rate, exploration_decay=self.exploration_decay, 
                                    exploration_min=self.exploration_min, max_episodes=self.max_episodes, 
                                    min_train_episodes=self.min_train_episodes, early_stop=self.early_stop, 
                                    val_returns=self.val_returns, train_returns=self.train_returns, 
                                    geom_ret_period=self.geom_ret_period, val_env=self.val_env, 
                                    normalize_rewards=self.normalize_rewards, train_mode=True, 
                                    clip_grad=self.clip_grad, plot_loss=self.plot_loss)
            
            print(f'Train starting for split {i}')
            agent_i.train()
            print(f'Train finished for split {i}')
            
            agent_i.env = test_env_i
            agent_i.train_mode = False
            test_actions_i = agent_i.test()

            actions.extend(test_actions_i)
            asset_return.extend(pd.Series([data[1].item() for data in test_dict_i.values()]).pct_change()[1:])

            print(f'Split {i} finished')

            del policy_i, agent_i, train_env_i, test_env_i
        
        return actions, asset_return


    def runDDPG(self):
        """
        Run the DDPG algorithm on the generated train and test splits.

        Returns:
            tuple: Lists of actions and asset returns for each test period.
        """
        actions = []
        asset_return = []

        for i in range(len(self.splits)):
            train_dict_i, test_dict_i = self.splits[i]
            
            train_env_i = StockTradingEnv(data=train_dict_i, input_size=self.input_size, feature_seq=self.feature_seq,
                                        transaction_cost=self.transaction_cost, risk_sensitivity=self.risk_sensitivity,
                                        weight_normalization=self.weight_normalization, short_bias=self.short_bias,
                                        extra_reward_weight=self.extra_reward_weight, weight_punishment=self.weight_punishment)
            test_env_i = StockTradingEnv(data=test_dict_i, input_size=self.input_size, feature_seq=self.feature_seq,
                                        transaction_cost=self.transaction_cost, risk_sensitivity=self.risk_sensitivity,
                                        weight_normalization=self.weight_normalization, short_bias=self.short_bias,
                                        extra_reward_weight=self.extra_reward_weight, weight_punishment=self.weight_punishment)

            policy_i = PolicyNetworkCNN(input_size=self.input_size, hidden_size=self.hidden_size, 
                                    action_space=self.action_space, feature_seq=self.feature_seq,
                                    relu_slope=self.relu_slope, kernel_size=self.kernel_size, 
                                    stride=self.stride, dropout=self.dropout, 
                                    maxpool_kernel=self.maxpool_kernel, maxpool_stride=self.maxpool_stride)

            critic_i = QNetworkCNN(action_size = self.action_space, input_size=self.input_size, feature_seq=self.feature_seq, 
                                    hidden_size=self.feature_seq, conv_out_channels=self.hidden_size, kernel_size=self.kernel_size, 
                                    stride=self.stride, maxpool_kernel=self.maxpool_kernel, maxpool_stride=self.maxpool_stride, 
                                    relu_slope=self.relu_slope, dropout=self.dropout)
            
            agent_i = AgentDDPG(env=train_env_i, actor=policy_i, critic=critic_i, criterion = self.criterion, lr_actor=self.lr,
                                    lr_critic=self.lr_critic, batch_size=self.batch_size, memory_size=self.memory_size, 
                                    update_freq=self.update_freq, exploration_rate=self.exploration_rate, 
                                    exploration_decay=self.exploration_decay, exploration_min=self.exploration_min,
                                    min_train_episodes=self.min_train_episodes, early_stop=self.early_stop, 
                                    max_episodes=self.max_episodes, plot_loss=self.plot_loss)
            
            print(f'Train starting for split {i}')
            agent_i.train()
            print(f'Train finished for split {i}')
            
            agent_i.env = test_env_i
            agent_i.train_mode = False
            test_actions_i = agent_i.test()

            actions.extend(test_actions_i)
            asset_return.extend(pd.Series([data[1].item() for data in test_dict_i.values()]).pct_change()[1:])

            print(f'Split {i} finished')

            del policy_i, critic_i, agent_i, train_env_i, test_env_i
        
        return actions, asset_return
