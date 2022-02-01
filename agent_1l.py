import numpy as np

import torch
import torch.nn as nn

from actor_critic_nets import discrete_actor_critic_Net_v2
from rnd import RND_Module
from net_utils import freeze
from utils import numpy2torch as np2torch
from utils import time_stamp


def create_first_level_agent(
    n_actions: int = 8, 
    feature_dim: int = 256, 
    init_log_alpha: float = 0.0, 
    noop_action: bool = False, 
    device: str = 'cuda', 
    lr: float = 1e-4, 
    lr_alpha: float = 1e-4, 
    lr_actor: float = 1e-4, 
    hidden_dim: int = 256,
    hidden_dim_2: int = 256,
    dueling_layers: int = 2,
    n_heads: int = 2,
    noisy_q_nets: bool = False,
    init_noise: float = 0.5
    ):
    
    actor_critic = discrete_actor_critic_Net_v2(
        n_actions = n_actions+int(noop_action), 
        feature_dim = feature_dim, 
        init_log_alpha = init_log_alpha, 
        lr = lr, 
        lr_alpha = lr_alpha, 
        lr_actor = lr_actor, 
        dueling_layers = dueling_layers, 
        hidden_dim = hidden_dim,
        hidden_dim_2 = hidden_dim_2,
        n_heads = n_heads,
        noisy_q_nets = noisy_q_nets,
        init_noise = init_noise
    )

    first_level_agent = First_Level_Agent(
        n_actions, actor_critic, 
        noop_action
    ).to(device)

    return first_level_agent


class First_Level_Agent(nn.Module):
    def __init__(
        self, 
        n_actions: int, 
        actor_critic: discrete_actor_critic_Net_v2, 
        noop_action: bool
        ):  
        super().__init__()    
        
        self.actor_critic = actor_critic
         
        self._n_actions = n_actions + int(noop_action)
        self._noop = noop_action
        self._id = time_stamp()
    
    def forward(self, states):
        pass 

    def sample_action(self, state, explore=True, use_actor=True, eps=0.0):
        with torch.no_grad():
            action, dist = self.actor_critic.sample_action(
                state, explore, use_actor, eps)            
            return action, dist
    
    def save(self, save_path, best=False):
        if best:
            model_path = save_path + 'best_agent_2l_' + self._id
        else:
            model_path = save_path + 'last_agent_2l_' + self._id
        torch.save(self.state_dict(), model_path)
    
    def load(self, load_directory_path, model_id, device='cuda'):
        dev = torch.device(device)
        self.load_state_dict(
            torch.load(
                load_directory_path + 'agent_2l_' + model_id, map_location=dev
            )
        )

    def get_id(self):
        return self._id


if __name__ == "__main__":
    agent = create_first_level_agent()
    print("Successful first level agent creation")