import numpy as np

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_

from actor_critic_nets import discrete_actor_critic_Net
from rnd import RND_Module
from net_utils import freeze
from utils import numpy2torch as np2torch
from utils import time_stamp


def create_second_level_agent(
    use_pixels: bool = False, n_actions: int = 8, 
    feature_dim: int = 256, n_heads: int = 8, 
    init_log_alpha: float = 0.0, 
    noop_action: bool = False, 
    device: str = 'cuda', noisy: bool = True, 
    parallel: bool = True, lr: float = 1e-4, 
    lr_alpha: float = 1e-4, lr_actor: float = 1e-4, 
    rnd_out_dim: int = 128, int_heads: bool = False,
    input_channels: int = 1, height: int = 21, 
    width: int = 79
    ):
    
    second_level_architecture = discrete_actor_critic_Net(
        n_actions+int(noop_action), feature_dim, use_pixels, n_heads, 
        init_log_alpha, input_channels, height, width,
        parallel, lr, lr_alpha, lr_actor, int_heads
    )

    if use_pixels:
        rnd_input_shape = (1, input_channels, height, width)
    else:
        rnd_input_shape=(1, feature_dim)        
    
    rnd_module = RND_Module(
            input_shape=rnd_input_shape,
            out_dim=rnd_out_dim
        ).to(device)

    second_level_agent = Second_Level_Agent(
        n_actions, second_level_architecture, 
        rnd_module, noop_action, use_pixels
    ).to(device)
    return second_level_agent


class Second_Level_Agent(nn.Module):
    def __init__(
        self, 
        n_actions: int, 
        second_level_architecture: discrete_actor_critic_Net, 
        rnd_module: RND_Module, 
        noop_action: bool,
        use_pixels: bool
        ):  
        super().__init__()    
        
        self.second_level_architecture = second_level_architecture
        self.rnd_module = rnd_module
        
        self._n_actions = n_actions + int(noop_action)
        self._noop = noop_action
        self._use_pixels = use_pixels
        self._id = time_stamp()
    
    def forward(self, states):
        pass 

    def calc_novelty(self, obs):
        return self.rnd_module.calc_novelty(obs)
    
    def sample_action(self, state, explore=True):
        if self._use_pixels:
            obs = self.observe_second_level_state(state)
        else:
            obs = np2torch(state)
        with torch.no_grad():
            action, dist = self.second_level_architecture.sample_action(
                obs, explore=explore)            
            return action, dist

    def observe_second_level_state(self, state):
        pixel_np = state['pixel'].astype(np.float)/255.        
        pixel = np2torch(pixel_np)        
        return pixel

    def train_rnd_module(self, obs):
        int_rewards = self.rnd_module(obs)
        rnd_loss = int_rewards.mean()
        self.rnd_module.predictor.optimizer.zero_grad()
        rnd_loss.backward()
        clip_grad_norm_(self.rnd_module.predictor.parameters(), 1.0)
        self.rnd_module.predictor.optimizer.step()
        return rnd_loss.item(), int_rewards.detach()
    
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
    agent = create_second_level_agent()
    print("Successful second level agent creation")