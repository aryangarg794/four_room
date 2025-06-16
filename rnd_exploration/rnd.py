import gymnasium as gym
import dill
import torch 
import torch.nn as nn
import numpy as np

from typing import Self, List
from torch import Tensor

from four_room.arch import CNN
from four_room.env import FourRoomsEnv
from four_room.wrappers import gym_wrapper


class BaseNetwork(nn.Module):
    
    def __init__(
        self: Self,
        use_action: bool, 
        obs_space: gym.spaces.Box,
        action_space: gym.spaces.Discrete,
        feature_units: int = 64, 
        hidden_layers: List = list([512, 1024, 512]), 
        *args, 
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        
        self.cnn = CNN(obs_space, features_dim=feature_units)
        
        self.net = nn.Sequential()
        if use_action:
            self.net.append(nn.Linear(feature_units + np.prod(action_space.shape, dtype=np.int64), hidden_layers[0]))
        else:
            self.net.append(nn.Linear(feature_units, hidden_layers[0]))
        
        for next_dim, prev_dim in zip(hidden_layers[1:], hidden_layers[:-1]):
            self.net.extend([
                nn.Linear(prev_dim, next_dim),
                nn.ReLU()
            ])
        self.use_action = use_action
        
    def forward(self: Self, state: Tensor, action: Tensor = None) -> Tensor:
        features = self.cnn(state)
        if self.use_action: 
            inp = torch.cat([features, action], dim=-1)
        else:
            inp = features
        return self.net(inp)
        

class RNDNetwork:
    
    def __init__(
        self: Self,
        env: gym.Env,
        use_actions: bool = False,
        scale: float = 3.5, 
        lr: float = 1e-3,
        device: str = 'cpu',
        *args, 
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        
        self.target_net = BaseNetwork(use_actions, env.observation_space, env.action_space).to(device)
        self.rnd_net = BaseNetwork(use_actions, env.observation_space, env.action_space).to(device)
        
        for param in self.target_net.parameters():
            param.requires_grad = False
        
        self.optimizer = torch.optim.Adam(self.rnd_net.parameters(), lr=lr)
        self.device = device
        self.scale = scale
        self.env = env
        self.loss = nn.MSELoss(reduction='none')
        self.use_actions = use_actions
        
    def observe(self: Self, states: Tensor, actions: Tensor = None) -> None:
        states = self.sanitize(states)
        if self.use_actions: 
            actions = self.sanitize(actions)
            
        preds = self.rnd_net(states, actions)
        targets = self.target_net(states, actions)
        loss = self.loss(preds, targets).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
    
    def get_error(self: Self, state: Tensor, action: Tensor = None) -> float:
        states = self.sanitize(state).reshape(1, *state.shape)
        if self.use_actions: 
            action = self.sanitize(action)
            
        with torch.no_grad():
            return self.scale * self.loss(self.rnd_net(states, action), 
                                          self.target_net(states, action)).sum().item()
        
    
    def sanitize(self: Self, tensor: Tensor) -> Tensor:
        if not isinstance(tensor, Tensor):
            tensor = torch.as_tensor(tensor, device=self.device) 
        if len(tensor.shape) < 4: 
            tensor = tensor.unsqueeze(dim=0)
        return tensor
        
        
        
        
        
        
        
        
if __name__ == "__main__":
    gym.register('MiniGrid-FourRooms-v1', FourRoomsEnv)
    size = 19
    with open('configs/train.pl', 'rb') as file:
        train_config = dill.load(file)

    with open('configs/test_reachable.pl', 'rb') as file:
        test_config = dill.load(file)

    with open('configs/validation_unreachable.pl', 'rb') as file:
        val_config = dill.load(file)
        
    env = gym_wrapper(gym.make(
        'MiniGrid-FourRooms-v1', 
        agent_pos= train_config['agent positions'],
        goal_pos = train_config['goal positions'],
        doors_pos = train_config['topologies'],
        agent_dir = train_config['agent directions'],
        size=size, 
    ),
    original_obs=True
)
    model = BaseNetwork(env.observation_space)
    print(model)