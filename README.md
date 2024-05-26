
# Deep Reinforcement Learning for Stock Trading

## Overview

This project implements a deep reinforcement learning approach for trading one or more financial instruments. The agent learns to trade autonomously using two different reinforcement learning algorithms: REINFORCE and Actor-Critic (DDPG). The implementation includes data preprocessing, agent training and testing, all orchestrated by the `Backtester` class.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
  - [Data Preparation](#data-preparation)
  - [Running Backtests](#running-backtests)
- [Implementation Details](#implementation-details)
  - [Data Preprocessing](#data-preprocessing)
  - [Agents](#agents)
  - [Training and Validation](#training-and-validation)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Installation

To install the necessary dependencies, run:

```bash
pip install -r requirements.txt
```

Ensure you have `torch`, `gym`, `numpy`, `pandas`, and other necessary libraries installed. You can list these dependencies in the `requirements.txt` file.

## Usage

### Data Preparation

1. Prepare your historical stock price data as a pandas DataFrame with columns: `Date`, `Open`, `High`, `Low`, `Close`.
2. Ensure your DataFrame is sorted by date in ascending order.
3. Save the DataFrame as a CSV file, or load it directly into the `Backtester` class.

### Running Backtests

1. **Initialize the Backtester:**
   ```python
   from backtesting import Backtester

   df = pd.read_csv('your_data.csv')
   backtester = Backtester(
       df=df,
       window_size=60,
       train_start_date='2010-01-01',
       train_end_date='2011-01-01',
       test_duration=6,  # in months
       input_size=4,
       feature_seq=60,
       hidden_size=256,
       action_space=1,
       kernel_size=3,
       stride=1,
       relu_slope=0.01,
       dropout=0.2,
       maxpool_kernel=2,
       maxpool_stride=2,
       lr_policy=1e-3,
       batch_size=128,
       exploration_rate=1.0,
       exploration_decay=0.99,
       exploration_min=0.01,
       max_episodes=500,
       min_train_episodes=50,
       early_stop=10,
       transaction_cost=0.001,
       risk_sensitivity=0.1
   )
   ```

2. **Run REINFORCE Backtest:**
   ```python
   actions, asset_returns = backtester.runREINFORCE()
   ```

3. **Run DDPG Backtest:**
   ```python
   actions, asset_returns = backtester.runDDPG()
   ```

## Implementation Details

### Data Preprocessing

The `Backtester` class preprocesses the historical stock price data to generate input features for the trading environment. It calculates log returns, volatility, normalized prices, and relative distances to a moving average. These features are then used to create input sequences for the agent.

### Agents

Two types of agents are implemented:

- **REINFORCE Agent**: Uses a policy network to determine actions based on states. It learns through the policy gradient method.
- **DDPG Agent**: Uses both an actor network (policy) and a critic network (value function) to learn deterministic policies.

Both agents are defined in the `agents.py` file.

### Training and Validation

- **Training**: Agents interact with the environment, select actions, observe rewards, and update their networks. The environment provides feedback, and the agents adjust their strategies accordingly.
- **Testing**: The best-performing policy networks are used to evaluate the agents' actions on test data, ensuring robust and generalizable strategies.

## Results

Results of the backtests include the actions taken by the agents and the corresponding asset returns. Detailed performance metrics are calculated and plotted, showcasing the effectiveness of the implemented trading strategies.

## Contributing

Contributions are welcome! Please submit a pull request or open an issue to discuss changes or improvements.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.
