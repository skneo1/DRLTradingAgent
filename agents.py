import torch 
import torch.optim as optim
import torch.distributions as distributions
from replay_memory import Transition, ReplayMemory
from utils import OUNoise
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from itertools import accumulate

torch.manual_seed(42)
np.random.seed(42)

#------------------ REINFORCE
class AgentREINFORCE:
    """
    AgentREINFORCE implements the REINFORCE algorithm.

    Args:
        env (gym.Env): The environment in which the agent interacts.
        policy_network (nn.Module): Mean-generating parametric function approximator.
        lr (float): Learning rate for the Adam optimizer.
        batch_size (int): The number of samples in each mini-batch.
        exploration_rate (float): Initial standart deviation for Gaussian distribution.
        exploration_decay (float): Decay rate for the exploration rate.
        exploration_min (float): Minimum exploration rate.
        max_episodes (int): Maximum number of training episodes.
        min_train_episodes (int): Minimum number of episodes before validation and early stop.
        early_stop (int): Number of episodes with no improvement after which training stops.
        val_returns (np.ndarray): Validation returns used for calculating performance metrics.
        train_returns (np.ndarray): Training returns used for tracking training progress.
        geom_ret_period (int): Period for calculating geometric returns.
        discount_factor (float, optional): Discount factor for future rewards. Defaults to 0.
        val_env (gym.Env, optional): Validation environment. Defaults to None.
        plot_val_results (bool, optional): Whether to plot validation results. Defaults to False.
        normalize_rewards (bool, optional): Whether to normalize rewards. Defaults to False.
        train_mode (bool, optional): Whether the agent is in training mode. Defaults to True.
        clip_grad (bool, optional): Whether to clip gradients. Defaults to False.
        plot_loss (bool, optional): Whether to plot the loss convergence. Defaults to False.
    """
    def __init__(self, env, policy_network, lr, batch_size, exploration_rate, 
                    exploration_decay, exploration_min, max_episodes, min_train_episodes, early_stop, 
                    val_returns, train_returns, geom_ret_period, discount_factor = 0, val_env = None, plot_val_results = False,
                    normalize_rewards = False, train_mode = True, clip_grad = False, plot_loss = False):

        self.env = env
        self.val_env = val_env
        self.policy_network = policy_network
        self.best_policy_network = deepcopy(policy_network)
        self.val_policy_network = None 
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=lr)
        self.batch_size = batch_size
        self.d_factor = discount_factor
        self.epsilon = exploration_rate
        self.val_epsilon = 1e-45
        self.epsilon_decay = exploration_decay
        self.epsilon_min = exploration_min
        self.train_mode = train_mode
        self.early_stop = early_stop
        self.val_return = val_returns
        self.train_returns = train_returns
        self.plot_val_results = plot_val_results
        self.geom_ret_period = geom_ret_period
        self.max_episodes = max_episodes
        self.clip_grad = clip_grad
        self.plot_loss = plot_loss
        self.normalize_rewards = normalize_rewards
        self.min_train_episodes = min_train_episodes

        if not self.train_mode:
            self.policy_network.eval()
        else:
            self.policy_network.train()
    
    @staticmethod
    def check_gradients(policy_network):
        """
        Check gradients of the policy network parameters.

        Args:
            policy_network (nn.Module): The policy network whose gradients are checked.
        """
        for name, param in policy_network.named_parameters():
            if param.grad is None:
                print(f"No gradient computed for {name}")
            else:
                print(f"Gradient computed for {name} with value {param.grad.norm().item()}")


    def plot_loss_convergence(self, losses):
        """
        Plot the loss convergence over episodes.

        Args:
            losses (list): List of loss values.
        """
        plt.figure(figsize=(8, 6))
        plt.plot(losses, label='Training Loss')
        plt.xlabel('Episode')
        plt.ylabel('Loss')
        plt.title('Loss Convergence Over Episodes')
        plt.legend()
        plt.show()
    
    def plot_validation_performance(self, val_ret, actions):
        """
        Plot the validation performance: asset returns vs policy returns.

        Args:
            val_ret (np.ndarray): Validation returns.
            actions (list): List of actions taken.
        """
        plt.figure(figsize=(10, 5))  
        plt.plot((1+val_ret.cumprod())-1 , label='Asset Returns')  
        plt.plot((1 + val_ret * actions).cumprod()-1, label='Policy Returns') # было actions[:-1]
        plt.title('Validation Performance: Asset Returns vs Policy Returns')
        plt.legend()
        plt.show()

    
    def sharpe_ratio(self, actions = None, algorithm = False):
        """
        Calculate the Sharpe ratio.

        Args:
            actions (list, optional): List of actions taken. Defaults to None.
            algorithm (bool, optional): If True, calculates the Sharpe ratio for the policy. Defaults to False.

        Returns:
            float: Sharpe ratio.
        """
        risk_free = 0.01
        asset_return = self.val_return
        traiding_days_per_year = 252

        if not algorithm:
            daily_risk_free_rate = (1 + risk_free) ** (1 / traiding_days_per_year) - 1
            expected_return = asset_return.mean()
            std_return = asset_return.std()
            sharpe_ratio = (expected_return - daily_risk_free_rate) / std_return * np.sqrt(traiding_days_per_year)
            return sharpe_ratio

        if algorithm and actions is not None:
            alg_ret = asset_return * actions 
            daily_risk_free_rate = (1 + risk_free) ** (1 / traiding_days_per_year) - 1
            expected_return = alg_ret.mean()
            std_return = alg_ret.std()
            sharpe_ratio = (expected_return - daily_risk_free_rate) / std_return * np.sqrt(traiding_days_per_year)
            return sharpe_ratio


    def geom_return(self, actions = None, algorithm = False):
        """
        Calculate the geometric return.

        Args:
            actions (list, optional): List of actions taken. Defaults to None.
            algorithm (bool, optional): If True, calculates the geometric return for the policy. Defaults to False.

        Returns:
            float: Geometric return for the specified period.
        """
        if not algorithm:
            ret = self.val_return # should be .pct_change() and take [1:] because NaN at the first place
            avg_geom_ret = (np.prod(1 + ret) ** (1 / len(ret))) - 1
            period = self.geom_ret_period
            geom_ret_for_period = (1 + avg_geom_ret) ** period - 1

            return geom_ret_for_period
        else:
            ret = self.val_return * actions  
            avg_geom_ret = (np.prod(1 + ret) ** (1 / len(ret))) - 1
            period = self.geom_ret_period
            geom_ret_for_period = (1 + avg_geom_ret) ** period - 1

            return geom_ret_for_period

    def maximum_drawdown(self, actions = None, algorithm = False):
        """
        Calculate the maximum drawdown.

        Args:
            actions (list, optional): List of actions taken. Defaults to None.
            algorithm (bool, optional): If True, calculates the maximum drawdown for the policy. Defaults to False.

        Returns:
            float: Maximum drawdown.
        """
        if not algorithm:
            ret = self.val_return
        else:
            ret = self.val_return * actions 

        cumulative_return = (1 + ret).cumprod()
        cumulative_max = cumulative_return.cummax()
        drawdown = cumulative_return / cumulative_max - 1
        max_drawdown = drawdown.min()
            
        return max_drawdown


    def select_action(self, policy, state, prev_action, validation = False):
        """
        Select an action based on the policy network.

        Args:
            policy (nn.Module): The policy network.
            state (torch.Tensor): The current state.
            prev_action (torch.Tensor): The previous action taken.
            validation (bool, optional): If True, uses validation exploration rate. Defaults to False.

        Returns:
            tuple: The selected action and its log probability.
        """
        mu = policy(state, prev_action)
        sigma = self.epsilon_min if validation else self.epsilon
        dist = distributions.Normal(mu, sigma)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return torch.clamp(action, -1, 1), log_prob

    def get_policy_loss(self, rewards: list, log_probs: list, normalize=False):
        """
        Calculate the policy loss.

        Args:
            rewards (list): List of rewards.
            log_probs (list): List of log probabilities of the actions.
            normalize (bool, optional): Whether to normalize rewards. Defaults to False.

        Returns:
            torch.Tensor: Policy loss.
        """
        r = torch.FloatTensor(rewards)
        if len(r) > 1 and normalize:
            r = (r - r.mean()) / (r.std() + float(np.finfo(np.float32).eps))
        log_probs = torch.stack(log_probs).squeeze()
        policy_loss = torch.mul(log_probs, r).mul(-1).sum()
        
        return policy_loss

    def learn(self, policy_loss):
        """
        Perform a learning step using the policy loss.

        Args:
            policy_loss (torch.Tensor): The policy loss.
        """
        loss = policy_loss
        self.optimizer.zero_grad()
        loss.backward()
        if self.clip_grad:
            torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), max_norm=1) 
        self.optimizer.step()

    def validate(self, num_episodes=1):
        """
        Validate the policy network.

        Args:
            num_episodes (int, optional): Number of validation episodes. Defaults to 1.

        Returns:
            tuple: Validation metrics including cumulative rewards, Sharpe ratios, geometric returns, and maximum drawdowns.
        """
        self.val_policy_network = deepcopy(self.policy_network)
        self.val_policy_network.eval()
        validation_reward = 0
        cum_rewards = []
        actions = []
        
        with torch.no_grad():
            for _ in range(num_episodes):
                state = self.val_env.reset()
                prev_action = None
                rewards = []
                log_probs = []

                while True:
                    action, log_prob = self.select_action(self.val_policy_network, state.unsqueeze(0), prev_action, validation = True) 
                    next_state, reward, done, _ = self.val_env.step(action)

                    actions.append(action.detach().item())
                    rewards.append(reward)
                    cum_rewards.append(reward.item() if isinstance(reward, torch.Tensor) else reward)
                    log_probs.append(log_prob)
                    prev_action = action.detach()
                    state = next_state

                    if done:
                        break

        validation_reward /= num_episodes

        sharpe_ratio_asset = self.sharpe_ratio()
        sharpe_ratio_policy = self.sharpe_ratio(actions=actions, algorithm=True)
        avg_geom_return_asset = self.geom_return()
        avg_geom_return_policy = self.geom_return(actions=actions, algorithm=True)
        max_drawdown_asset = self.maximum_drawdown()
        max_drawdown_policy = self.maximum_drawdown(actions=actions, algorithm=True)
        median_dayly_ret = (self.val_return * actions).mean()
        total_ret = ((1 + self.val_return * actions).cumprod() - 1).to_numpy()[-1] 
        performance = (self.val_return.cumsum() <= ((1 + self.val_return * actions).cumprod()-1)).mean()
        
        return (
            sum(cum_rewards), sharpe_ratio_asset, sharpe_ratio_policy, avg_geom_return_asset, 
            avg_geom_return_policy, max_drawdown_asset, max_drawdown_policy, median_dayly_ret, 
            total_ret, actions, performance
        )

    def train(self):
        """
        Train the policy network 
        """
        num_episodes = self.max_episodes if self.max_episodes else 100
        episode_losses = []
        validation_rewards = []
        best_val_reward = float('-inf')
        reward_history = []
        actions = []
        log_probs = []
        rewards = []
        best_policy_weights = None
        early_stop_counter = 0

        for i in range(num_episodes):
            if (i+1) % 10 == 0 or (i+1) == 1:
                print(f"Starting episode {i+1}")

            state = self.env.reset()
            prev_action = None
            done = False
            episode_loss = 0
            val_rewards = []
            train_actions = []

            while True:
                action, log_prob = self.select_action(self.policy_network, state.unsqueeze(0), prev_action, validation = False)
                next_state, reward, done, _ = self.env.step(action)

                actions.append(action.detach().item())
                log_probs.append(log_prob)
                rewards.append(reward)
                train_actions.append(action.detach().item())
                val_rewards.append(reward.item() if isinstance(reward, torch.Tensor) else reward)

                if done:
                    break

                if self.train_mode and len(rewards) >= self.batch_size:

                    if self.d_factor != 0:
                        weighted_rewards = list(accumulate(reversed(rewards), lambda x,y: x * self.d_factor + y))[::-1]
                        policy_loss = self.get_policy_loss(weighted_rewards, log_probs, self.normalize_rewards)
                    else:
                        policy_loss = self.get_policy_loss(rewards, log_probs, self.normalize_rewards)

                    episode_loss += policy_loss.detach().item()
                    self.learn(policy_loss)
                    actions.clear()
                    log_probs.clear()
                    rewards.clear()
                   
                prev_action = action.detach()
                state = next_state

            if done and len(rewards) > 0:
                if self.d_factor != 0:
                    weighted_rewards = list(accumulate(reversed(rewards), lambda x, y: x * self.d_factor + y))[::-1]
                    policy_loss = self.get_policy_loss(weighted_rewards, log_probs, self.normalize_rewards)
                else:
                    policy_loss = self.get_policy_loss(rewards, log_probs, self.normalize_rewards)
                episode_loss += policy_loss.detach().item()
                self.learn(policy_loss)
                actions.clear()
                log_probs.clear()
                rewards.clear()
          
            episode_losses.append(episode_loss)
            reward_history.append(sum(val_rewards))
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

            if done and (i+1) >= self.min_train_episodes:
                if sum(val_rewards) > best_val_reward:
                    best_val_reward = sum(val_rewards)
                    early_stop_counter = 0
                else:
                    early_stop_counter += 1
                    
                if early_stop_counter >= self.early_stop:
                    print(f"Early stopping after {i+1} episodes")
                    break

            best_policy_weights = self.policy_network.state_dict()

        if best_policy_weights is not None:
            self.best_policy_network.load_state_dict(best_policy_weights)
            print(f"Best policy weights saved and loaded into the test model after ep:{i+1}")
        else:
            print("Problem with saving weights")
            
        if self.plot_loss:
            self.plot_loss_convergence(reward_history)
           
    def test(self):
        """
        Test the trained policy network.
        """
        num_episodes = 1
        self.best_policy_network.eval()
        prev_action = None
        actions = []

        for i in range(num_episodes):
            print(f"Starting test ...")
            state = self.env.reset()

            while True:
                action, _ = self.select_action(self.best_policy_network, state.unsqueeze(0), prev_action, validation=True)
                next_state, reward, done, _ = self.env.step(action.detach())
                actions.append(action.detach().item())

                prev_action = action.detach()
                state = next_state

                if done:
                    print('Test finished')
                    break
                
        return actions


#------------------ Deep Deterministic Policy Gradients (Actor-Critic)
class AgentDDPG:
    """
    AgentDDPG implements the Deep Deterministic Policy Gradient (DDPG) algorithm.

    Args:
        env (gym.Env): The environment in which the agent interacts.
        actor (nn.Module): The actor network (Deterministic Policy).
        critic (nn.Module): The critic network (Action-value function).
        criterion (nn.Module): The loss function used for training the critic network.
        lr_actor (float): Learning rate for the actor network.
        lr_critic (float): Learning rate for the critic network.
        batch_size (int): The number of samples in each batch.
        memory_size (int): The maximum size of the replay memory.
        update_freq (int): The frequency of updating the networks.
        exploration_rate (float): Initial exploration rate for action selection.
        exploration_decay (float): Decay rate for the exploration rate.
        exploration_min (float): Minimum exploration rate.
        max_episodes (int): Maximum number of training episodes.
        min_train_episodes (int): Minimum number of episodes before validation and early stop.
        early_stop (int): Number of episodes with no improvement after which training stops.
        add_ou_noise (bool, optional): Whether to add Ornstein-Uhlenbeck noise to actions. Defaults to False.
        train_mode (bool, optional): Whether the agent is in training mode. Defaults to True.
        plot_loss (bool, optional): Whether to plot the loss convergence. Defaults to False.
    """
    def __init__(self, env, actor, critic, criterion, lr_actor, lr_critic,
        batch_size, memory_size, update_freq, exploration_rate, exploration_decay, exploration_min,
        max_episodes, min_train_episodes, early_stop, add_ou_noise = False, train_mode = True, plot_loss = False):

        self.env = env
        self.actor_net = actor
        self.critic_net = critic
        self.memory_size = memory_size
        self.best_actor = deepcopy(actor)
        self.best_critic = deepcopy(critic)
        self.batch_size = batch_size
        self.replay_buffer = ReplayMemory(memory_size=self.memory_size, batch_size=self.batch_size, reccurent=False)
        self.update_freq = update_freq
        self.criterion = criterion
        self.alpha_critic = lr_critic
        self.alpha_actor = lr_actor
        self.optimizer_actor = optim.Adam(self.actor_net.parameters(), lr=self.alpha_actor, weight_decay=1e-3) 
        self.optimizer_critic = optim.Adam(self.critic_net.parameters(), lr=self.alpha_critic, weight_decay=1e-3)
                                                
        self.epsilon = exploration_rate
        self.epsilon_decay = exploration_decay
        self.epsilon_min = exploration_min
        self.max_episodes = max_episodes
        self.min_train_episodes = min_train_episodes
        self.train_mode = train_mode
        self.early_stop = early_stop
        self.plot_loss = plot_loss

        self.ou_noise = OUNoise(action_dim=self.env.action_space.shape[0])
        self.add_ou_noise = add_ou_noise

        if not self.train_mode:
            self.actor_net.eval()
            self.critic_net.eval()
        else:
            self.critic_net.train()
            self.actor_net.train()
    
    def plot_loss_convergence(self, losses):
        """
        Plot the loss convergence over episodes.

        Args:
            losses (list): List of loss values.
        """
        plt.figure(figsize=(8, 6))
        plt.plot(losses, label='Training Loss')
        plt.xlabel('Episode')
        plt.ylabel('Loss')
        plt.title('Loss Convergence Over Episodes')
        plt.legend()
        plt.show()


    def select_action(self, policy, state, prev_action, validation = False):
        """
        Select an action based on the actor network.

        Args:
            policy (nn.Module): The policy network (actor).
            state (torch.Tensor): The current state.
            prev_action (torch.Tensor): The previous action taken.
            validation (bool, optional): If True, uses validation exploration rate. Defaults to False.

        Returns:
            torch.Tensor: The selected action.
        """
        if validation:
            eps = self.epsilon
        else:
            eps = self.epsilon_min

        if self.add_ou_noise:
            mu = policy(state, prev_action)
            noise = self.ou_noise.noise() if self.train_mode else torch.zeros_like(mu)
            return torch.clamp(mu + eps * noise, -1, 1)
        else:
            mu = self.actor_net(state, prev_action)
            W = torch.rand(1) * 2 -1
            return torch.clamp(mu + eps * W, -1, 1)

    def learn(self):
        """
        Perform a learning step, updating both the actor and critic networks.
        """

        if len(self.replay_buffer.memory) < self.batch_size:
            return
        
        states, actions, rewards, prev_actions = self.replay_buffer.get_batch()
        # update critic
        self.optimizer_critic.zero_grad()
        q_sa = self.critic_net(states, actions, prev_actions)
        rewards = rewards.view(-1, 1) # изменил чтобы было 128, 1
        critic_loss = self.criterion(q_sa, rewards) #self.criterion
        critic_loss.backward()
        self.optimizer_critic.step()
        # freeze params
        for p in self.critic_net.parameters():
            p.requires_grad = False

        self.optimizer_actor.zero_grad()
        actor_loss = -self.critic_net(states, self.select_action(self.actor_net, states, prev_actions), prev_actions).mean()
        actor_loss.backward()
        self.optimizer_actor.step()

        #unfreeze params
        for p in self.critic_net.parameters():
            p.requires_grad = True
        
        return critic_loss, actor_loss


    def train(self):
        """
        Train the actor and critic networks.
        """
        num_episodes = self.max_episodes if self.max_episodes else 100
        critic_losses = []
        actor_losses = []
        step_counter = 0
        early_stop_counter = 0
        best_val_reward = float('-inf')
        best_actor_weights = None
        best_critic_weights = None

        for i in range(num_episodes):
            
            if (i + 1) % 10 == 0 or (i+1) == 1:
                print(f"Starting episode {i + 1}")

            prev_action = None
            done = False
            state = self.env.reset()  
            critic_episode_loss = 0
            actor_episode_loss = 0
            val_rewards = []

            while True:
                action = self.select_action(self.actor_net, state.unsqueeze(0), prev_action, validation=False)
                next_state, reward, done, _ = self.env.step(action.detach())
                step_counter += 1
                val_rewards.append(reward.item() if isinstance(reward, torch.Tensor) else reward)

                if done:
                    step_counter = 0
                    break
                           
                self.replay_buffer.push(
                    state.unsqueeze(0), action.detach(), torch.FloatTensor([reward]),
                    torch.zeros(action.detach().shape[0], 1) if prev_action is None else prev_action
                )

                prev_action = action.detach()
                state = next_state

                if step_counter % self.update_freq == 0 and self.train_mode and len(self.replay_buffer.memory) >= self.batch_size:
                    critic_loss, actor_loss = self.learn()
                    critic_episode_loss += critic_loss.detach().item()
                    actor_episode_loss += actor_loss.detach().item()

            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
            critic_losses.append(critic_episode_loss)
            actor_losses.append(actor_episode_loss)

            if done and (i+1) >= self.min_train_episodes:
                if sum(val_rewards) > best_val_reward:
                    best_val_reward = sum(val_rewards)
                    early_stop_counter = 0
                else:
                    early_stop_counter += 1
                    
                if early_stop_counter >= self.early_stop:
                    print(f"Early stopping after {i+1} episodes")
                    break

            best_actor_weights = self.actor_net.state_dict()
            best_critic_weights = self.critic_net.state_dict()

        if best_actor_weights is not None and best_critic_weights is not None:
            self.best_actor.load_state_dict(best_actor_weights)
            self.best_critic.load_state_dict(best_critic_weights)
            print(f"Best actor/critic weights saved and loaded into the test model after ep:{i+1}")
        else:
            print("Problem with saving weights")     

        if self.plot_loss:
            self.plot_loss_convergence(critic_losses)
            self.plot_loss_convergence(actor_losses)


    def test(self):
        """
        Test the trained actor network.
        """
        num_episodes = 1
        prev_action = None
        actions = []
        self.best_actor.eval()

        for i in range(num_episodes):
            print(f"Starting test ...")
            state = self.env.reset()

            while True:
                action = self.select_action(self.best_actor, state.unsqueeze(0), prev_action, validation=True)
                next_state, reward, done, _ = self.env.step(action.detach())
                actions.append(action.detach().item())

                if done:
                    print('Test finished')
                    break
                prev_action = action.detach()
                state = next_state

        return actions