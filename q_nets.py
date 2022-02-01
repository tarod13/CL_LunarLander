import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from custom_layers import Linear_noisy, parallel_Linear, parallel_Linear_noisy
from vision_nets import vision_Net
from net_utils import *

use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")


# Continuous action space
#-------------------------------------------------
class q_Net(nn.Module):
    def __init__(self, s_dim, a_dim, noisy=False, lr=3e-4):
        super().__init__() 
        self.s_dim = s_dim
        self.a_dim = a_dim

        if noisy:
            layer = Linear_noisy
        else:
            layer = nn.Linear

        self.l1 = layer(s_dim+a_dim, 256)
        self.l2 = layer(256, 256)
        self.lQ = layer(256, 1)
        
        if not noisy:
            self.apply(weights_init_rnd)
            torch.nn.init.orthogonal_(self.lQ.weight, 0.01)
            self.lQ.bias.data.zero_()
        else:
            torch.nn.init.orthogonal_(self.lQ.mean_weight, 0.01)
            self.lQ.mean_bias.data.zero_()

        self.optimizer = Adam(self.parameters(), lr=lr)
        
    def forward(self, s, a):
        sa = torch.cat([s,a], dim=1)        
        x = F.relu(self.l1(sa))
        x = F.relu(self.l2(x))
        Q = self.lQ(x)        
        return Q


# Discrete action space
#-------------------------------------------------
class DuelingQNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, lr=1e-4, hidden_dim = 64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """        
        super(DuelingQNetwork, self).__init__()
        self.num_actions = action_size
        fc3_1_size = fc3_2_size = 32
        self.fc1 = nn.Linear(state_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        ## Here we separate into two streams
        # The one that calculate V(s)
        self.fc3_1 = nn.Linear(hidden_dim, fc3_1_size)
        self.fc4_1 = nn.Linear(fc3_1_size, 1)

        # The one that calculate A(s,a)
        self.fc3_2 = nn.Linear(hidden_dim, fc3_2_size)
        self.fc4_2 = nn.Linear(fc3_2_size, action_size)

        self.optimizer = Adam(self.parameters(), lr=lr)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        val = F.relu(self.fc3_1(x))
        val = self.fc4_1(val)
        
        adv = F.relu(self.fc3_2(x))
        adv = self.fc4_2(adv)
        # Q(s,a) = V(s) + (A(s,a) - 1/|A| * sum A(s,a'))
        action = val + adv - adv.mean(1).unsqueeze(1).expand(state.size(0), self.num_actions)
        return action


class dueling_q_Net(nn.Module):
    def __init__(
        self, 
        s_dim: int, 
        n_actions: int,
        lr: float = 1e-4, 
        dueling_layers: int = 2,
        hidden_dim: int = 256,
        hidden_dim_2: int = 32,
        n_heads: int = 1,
        noisy: bool = False,
        init_noise: float = 0.5
        ):
        super().__init__() 
        
        self.s_dim = s_dim
        self.n_actions = n_actions
        self.n_heads = n_heads

        if noisy:
            linear_layer = parallel_Linear_noisy
        else:
            linear_layer = parallel_Linear

        self.common_pipe = nn.Sequential(
                linear_layer(n_heads, s_dim, hidden_dim, init_noise=init_noise),
                nn.ReLU(),
                linear_layer(n_heads, hidden_dim, hidden_dim, init_noise=init_noise)
            )
        
        if dueling_layers == 2:
            self.V_pipe = nn.Sequential(
                linear_layer(n_heads, hidden_dim, hidden_dim_2, init_noise=init_noise),
                nn.ReLU(),
                linear_layer(n_heads, hidden_dim_2, 1, init_noise=init_noise)
            )
            self.A_pipe = nn.Sequential(
                linear_layer(n_heads, hidden_dim, hidden_dim_2, init_noise=init_noise),
                nn.ReLU(),
                linear_layer(n_heads, hidden_dim_2, n_actions, init_noise=init_noise)
            )

        elif dueling_layers == 1:
            self.V_pipe = linear_layer(n_heads, hidden_dim, 1, init_noise=init_noise)
            self.A_pipe = linear_layer(n_heads, hidden_dim, n_actions, init_noise=init_noise)
            
        else:
            raise ValueError("Invalid number of dueling layers")
       
        self.optimizer = Adam(self.parameters(), lr=lr)
        
    def forward(self, s):        
        x = F.relu(self.common_pipe(s))
        V = self.V_pipe(x)
        A = self.A_pipe(x)
        Q = V + A - A.mean(-1, keepdim=True) 
        return Q


class multihead_dueling_q_Net(nn.Module):
    def __init__(
        self, s_dim, n_actions, n_heads, 
        lr=1e-4, int_heads=False, hidden_dim=256
        ):
        super().__init__() 
        
        self.s_dim = s_dim
        self.n_actions = n_actions

        self._n_heads = n_heads
        self._int_heads = int_heads
        self._total_n_heads = n_heads * (2 if int_heads else 1) 

        self.l1 = parallel_Linear(self._total_n_heads, s_dim, hidden_dim)
        self.l2 = parallel_Linear(self._total_n_heads, hidden_dim, hidden_dim)
        self.lV = parallel_Linear(self._total_n_heads, hidden_dim, 1)
        self.lA = parallel_Linear(self._total_n_heads, hidden_dim, n_actions)
        
        self.apply(weights_init_rnd)
        torch.nn.init.orthogonal_(self.lV.weight, 0.01)
        self.lV.bias.data.zero_()
        torch.nn.init.orthogonal_(self.lA.weight, 0.01)
        self.lA.bias.data.zero_()

        self.optimizer = Adam(self.parameters(), lr=lr)
        
    def forward(self, s):        
        x = F.relu(self.l1(s))
        x = F.relu(self.l2(x))
        V = self.lV(x)        
        A = self.lA(x)
        Q = V + A - A.mean(2, keepdim=True) 
        return Q


class vision_multihead_dueling_q_Net(multihead_dueling_q_Net):
    def __init__(
        self, feature_dim, n_actions, n_heads, 
        input_channels=1, height=42, width=158,
        lr=1e-4, int_heads=False
        ):

        self.vision_nets = nn.ModuleList([
            vision_Net(
                feature_dim, input_channels, height, width,
                noisy=False, 
            ) for i in range(0, n_heads * (2 if int_heads else 1))
            ]
        )

        super().__init__(
            feature_dim, n_actions, n_heads, 
            lr, int_heads
        )
        
    def forward(self, pixels):    
        state = []
        for head in range(0, self._total_n_heads):
            head_features = self.vision_nets[head](pixels)
            state.append(head_features)
        state = torch.stack(state, dim=1)
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        V = self.lV(x)        
        A = self.lA(x)
        Q = V + A - A.mean(2, keepdim=True)
        Q = Q.reshape(
            pixels.shape[0], -1, self._n_heads, self.n_actions
        )
        return Q