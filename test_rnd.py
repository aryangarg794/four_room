import dill 
import torch 
import torch.nn as nn
import numpy as np
from four_room.shortest_path import find_all_action_values
from four_room.env import FourRoomsEnv
from four_room.wrappers import gym_wrapper
import numpy as np
import imageio
import argparse
import gymnasium as gym
from four_room.utils import obs_to_state


import warnings
warnings.filterwarnings(action='once')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

from rnd_exploration.rnd import RNDNetwork

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
        max_steps=1200, 
    ),
    original_obs=True
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-lr', '--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('-s', '--scale', type=float, default=1, help='learning rate')
            
    args = parser.parse_args()
    net = RNDNetwork(env, lr=args.lr, device=device, scale=args.scale)

    # for i in range(2 * len(train_config['topologies'])):
    for i in [1, 1, 1, 2, 2, 2, 1, 3, 4, 3]:
        
        env.get_wrapper_attr('set_context')(i)
        obs, _ = env.reset()
        done = False 
        valid_pos = env.get_wrapper_attr('valid_pos')
        
        for idx in range(len(valid_pos)):
            env.get_wrapper_attr('move_valid_pos')(idx)
            
            for _ in range(4): 
                obs, _, _, _, _ = env.step(1)
                state = obs_to_state(obs)
                agent_pos = state[:2]
                q = find_all_action_values(state[:2], state[2], state[3:5], state[5:], 0.99, size)
                action = np.array([1])
                rnd_value = net.get_error(obs, action)
                net.observe(obs, action)
                
                print(f'Context is {(i)%200:04d} | agent x: {agent_pos[0]:03d} | agent y: {agent_pos[1]:03d} | RND Val: {rnd_value:.5f}', end='\r')