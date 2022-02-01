import numpy as np

import torch
import torch.nn as nn

from policy_nets import s_Net
from concept_nets import concept_Net
from net_utils import freeze
from utils import numpy2torch as np2torch
from utils import time_stamp


class Conceptual_Agent(concept_Net):
    def __init__(self, s_dim, latent_dim, n_concepts, noisy=True, lr=1e-4):  
        super().__init__(s_dim, latent_dim, n_concepts, noisy, lr)    
        self._id = time_stamp()
    
    def save(self, save_path, v=None):
        path = save_path + 'agent_c_S_' + self._id
        if v is not None:
            path += '_v' + str(v)
        torch.save(self.state_dict(), path)
    
    def load(self, load_directory_path, model_id, v=None, device='cuda'):
        dev = torch.device(device)
        path = load_directory_path + 'agent_c_S_' + model_id
        if v is not None:
            path += '_v' + str(v)
        self.load_state_dict(torch.load(path, map_location=dev))



if __name__ == "__main__":
    agent = Conceptual_Agent(33, 64, 20)
    print("Successful conceptual agent creation")