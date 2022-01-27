import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch.nn.parameter import Parameter
from torch.optim import Adam

from policy_nets import softmax_policy_Net, vision_softmax_policy_Net
from q_nets import multihead_dueling_q_Net, vision_multihead_dueling_q_Net
from net_utils import *

use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")


class discrete_actor_critic_Net(nn.Module):
    def __init__(
        self, n_actions, feature_dim, use_pixels=False, n_heads=8, 
        init_log_alpha=0.0, input_channels=1, 
        height=42, width=158, parallel=True, 
        lr=1e-4, lr_alpha=1e-4, lr_actor=1e-4, int_heads=False
        ):
        super().__init__()   

        self.n_actions = n_actions     
        self._parallel = parallel    
        
        if use_pixels:
            self.q = vision_multihead_dueling_q_Net(
                feature_dim, n_actions, n_heads, input_channels, 
                height, width, lr, int_heads
            )        
            self.q_target = vision_multihead_dueling_q_Net(
                feature_dim, n_actions, n_heads, input_channels, 
                height, width, lr, int_heads
            )

            self.actor = vision_softmax_policy_Net(
                feature_dim, n_actions, input_channels, 
                height, width, noisy=False, lr=lr_actor
            ) 

        else:
            self.q = multihead_dueling_q_Net(
                feature_dim, n_actions, n_heads, lr, int_heads
            )        
            self.q_target = multihead_dueling_q_Net(
                feature_dim, n_actions, n_heads, lr, int_heads
            )

            self.actor = softmax_policy_Net(
                feature_dim, n_actions, noisy=False, lr=lr_actor
            )
        
        self.update(rate=1.0)
        
        self.log_alpha = Parameter(torch.Tensor(1))
        nn.init.constant_(self.log_alpha, init_log_alpha)
        self.alpha_optimizer = Adam([self.log_alpha], lr=lr_alpha)
    
    def forward(self, obs):
        q = self.q(obs)
        q_target = self.q_target(obs)
        pi, log_pi = self.actor(obs)
        log_alpha = self.log_alpha.view(-1,1)
        return q, q_target, pi, log_pi, log_alpha

    def evaluate_critic(self, obs, next_obs):
        q = self.q(obs)
        next_q = self.q_target(next_obs)
        next_pi, next_log_pi = self.actor(next_obs)
        log_alpha = self.log_alpha.view(-1,1)
        return q, next_q, next_pi, next_log_pi, log_alpha
    
    def evaluate_actor(self, obs):
        with torch.no_grad():
            q = self.q(obs)            
        pi, log_pi = self.actor(obs)
        return q, pi, log_pi
    
    def sample_action(self, obs, explore=True):
        PA_s = self.actor(obs.unsqueeze(0))[0].squeeze(0).view(-1)
        assert torch.all(PA_s == PA_s), 'Boom. Capoot.'
        if explore:
            A = Categorical(probs=PA_s).sample().item()
        else:
            tie_breaking_dist = torch.isclose(PA_s, PA_s.max()).float()
            tie_breaking_dist /= tie_breaking_dist.sum()
            A = Categorical(probs=tie_breaking_dist).sample().item()  
        return A, PA_s.detach().cpu().numpy()
    
    def update(self, rate=5e-3):
        updateNet(self.q_target, self.q, rate)
    
    def get_alpha(self):
        return self.log_alpha.exp().item()
