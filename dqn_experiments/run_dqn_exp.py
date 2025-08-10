import gymnasium as gym
import numpy as np 
import torch 
import torch.nn as nn

from rnd_exploration.utils import RunningAverage
from rnd_exploration.rnd import RNDNetwork
from dqn.model import DQN

def train_dqn_rnd(
    agent: DQN, 
    rnd_net: RNDNetwork,
    env: gym.Env,
    batch_size: int = 512, 
    gamma: float = 0.99, 
    num_timesteps: int = int(2e5), 
    eval_iter: int = 5000, 
    grad_norm: float = 1.0,
    device: str = 'cuda',
    seed: int = 0 
): 
    metrics = RunningAverage(window_size=25)
    val_rewards = []
    mse_loss = nn.MSELoss()
    
    
    return metrics, agent, rnd_net, val_rewards