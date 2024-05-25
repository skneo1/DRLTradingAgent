import torch 
import torch.nn as nn
torch.manual_seed(42)


# ------------------ Policy Network (Convolutional)
## ------------------ REINFORCE (Policy), Actor-Critic (Deterministic policy for Actor)
class PolicyNetworkCNN(nn.Module):
    """
    PolicyNetworkCNN: The deterministic policy for Actor-Critic 
    and the mean-generating parametric function approximator in the stochastic policy REINFOFRCE 
    
    Args:
        input_size (int): Number of input features.
        hidden_size (int): Number of hidden units in the network.
        action_space (int): Number of possible actions (Action space).
        feature_seq (int): Length of the feature sequence.
        relu_slope (float): Negative slope for the LeakyReLU activation function.
        kernel_size (int): Size of the convolutional kernel.
        stride (int): Stride for the convolutional layers.
        dropout (float): Dropout rate for regularization.
        maxpool_kernel (int): Kernel size for the max pooling layer.
        maxpool_stride (int): Stride for the max pooling layer.

    """
    def __init__(self, input_size, hidden_size, action_space, feature_seq,
                    relu_slope, kernel_size, stride, dropout, maxpool_kernel, maxpool_stride):
        super(PolicyNetworkCNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.relu_slope = relu_slope
        self.kernel_size = kernel_size
        self.stride = stride 
        self.dropout = dropout
        self.maxpool_kernel = maxpool_kernel
        self.maxpool_stride = maxpool_stride
        self.num_actions = action_space
        self.feature_seq = feature_seq

        # Sequential Inforamtion Layer 
        self.information_layer = nn.Sequential(
            nn.Conv1d(in_channels=self.input_size, out_channels=self.hidden_size,
            kernel_size = self.kernel_size, stride = self.stride),
            nn.BatchNorm1d(num_features = self.hidden_size),
            nn.LeakyReLU(negative_slope = self.relu_slope),
            nn.Dropout(p=self.dropout),
            nn.Conv1d(in_channels=self.hidden_size, out_channels=self.hidden_size, 
            kernel_size=self.kernel_size, stride=self.stride),
            nn.BatchNorm1d(num_features=self.hidden_size),
            nn.LeakyReLU(negative_slope=self.relu_slope),
            nn.MaxPool1d(kernel_size=self.maxpool_kernel, stride=self.maxpool_stride),
            nn.Dropout(p=self.dropout)
        )

        self._to_linear = None
        self._num_flat_features()
        
        # Decision-Making Layer
        self.decision_layer = nn.Linear(self._to_linear + self.num_actions, 1)  
        self.register_buffer('prev_action', torch.zeros(1, 1)) 

    def _num_flat_features(self):
        sample_input = torch.randn(1, self.input_size, self.feature_seq)  
        output = self.information_layer(sample_input)
        self._to_linear = output.numel()  

    def forward(self, x, prev_action):
        """
        Args:
            x (torch.Tensor): Current state.
            prev_action (torch.Tensor): Previous action taken.
        
        Returns:
            torch.Tensor: independent mean for the Gaussian distribution (REINFORCE),
                            Determenistic policy (Actor-Critic)
        """
        x = self.information_layer(x)
        x = x.view(x.size(0), -1)
        
        if prev_action is None:
            prev_action = torch.zeros(x.size(0), 1) 

        x_with_prev_action = torch.cat([x, prev_action], dim=1)
        decision = self.decision_layer(x_with_prev_action)
        action = torch.tanh(decision)

        return action

# ------------------ Q-Network (Convolutional)
## ------------------ Actor-Critic (Critic)
class QNetworkCNN(nn.Module):
    """
    QNetworkCNN is a CNN  action-value function that assigns the value of performing
    a specific action in a specific state
    
    Args:
        action_size (int): Number of possible actions (Action space).
        input_size (int): Number of input features.
        feature_seq (int): Length of the feature sequence.
        hidden_size (int): Number of hidden units in the network.
        conv_out_channels (int): Number of output channels for the convolutional layers.
        kernel_size (int): Size of the convolutional kernel.
        stride (int): Stride for the convolutional layers.
        maxpool_kernel (int): Kernel size for the max pooling layer.
        maxpool_stride (int): Stride for the max pooling layer.
        relu_slope (float): Negative slope for the LeakyReLU activation function.
        dropout (float): Dropout rate for regularization.
    """


    def __init__(self, action_size, input_size, feature_seq, hidden_size, conv_out_channels,
            kernel_size, stride, maxpool_kernel, maxpool_stride, relu_slope, dropout):
        super(QNetworkCNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.conv_out_channels = conv_out_channels
        self.stride = stride
        self.maxpool_kernel = maxpool_kernel
        self.maxpool_stride = maxpool_stride
        self.kernel_size = kernel_size
        self.relu_slope = relu_slope
        self.dropout = dropout
        self.feature_seq = feature_seq
        self.action_space = action_size

        # First Fully-Connected layer
        self.fc_in = nn.Sequential(
            nn.Linear(in_features=(self.feature_seq + self.action_space), 
                out_features = self.hidden_size),
                nn.LeakyReLU(negative_slope=self.relu_slope),
                nn.Dropout(p=self.dropout)
        )
        # Sequential Information Layer
        self.information_layer = nn.Sequential(
            nn.Conv1d(in_channels=self.input_size, out_channels=self.conv_out_channels,
            kernel_size = self.kernel_size, stride = self.stride),
            nn.BatchNorm1d(num_features = self.conv_out_channels),
            nn.LeakyReLU(negative_slope = self.relu_slope),
            nn.Dropout(p=self.dropout),
            nn.Conv1d(in_channels=self.conv_out_channels, out_channels=self.conv_out_channels, 
            kernel_size=self.kernel_size, stride=self.stride),
            nn.BatchNorm1d(num_features=self.conv_out_channels),
            nn.LeakyReLU(negative_slope=self.relu_slope),
            nn.MaxPool1d(kernel_size=self.maxpool_kernel, stride=self.maxpool_stride),
            nn.Dropout(p=self.dropout)
        )

        self._to_linear = None
        self._num_flat_features()

        # Decision-Making Layer
        self.decision_layer = nn.Linear(self._to_linear + self.action_space, 1)  
        self.register_buffer('prev_action', torch.zeros(1, 1)) 

    def _num_flat_features(self):
        sample_input = torch.randn(1, self.input_size, self.hidden_size) 
        output = self.information_layer(sample_input)
        self._to_linear = output.numel()  


    def forward(self, state, action, prev_action):
        """ 
        Args:
            state (torch.Tensor): Current state.
            action (torch.Tensor): Action taken.
            prev_action (torch.Tensor): Previous action taken.
        
        Returns:
            torch.Tensor: Q-value of the action taken (Actor-Critic)
        """
        x = torch.cat([state, action.unsqueeze(-1).expand(-1, self.input_size, -1)], dim=2)
        x = self.fc_in(x)
        x = self.information_layer(x)
        x = x.view(x.size(0), -1)
        
        if prev_action is None:
            prev_action = torch.zeros(x.size(0), 1)  

        x = torch.cat([x, prev_action], dim=1)
        q_value = self.decision_layer(x)

        return q_value




