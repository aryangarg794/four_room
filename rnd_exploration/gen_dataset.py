import numpy as np 
import gymnasium as gym
import dill 
import argparse

from four_room.env import FourRoomsEnv
from four_room.utils import obs_to_state
from four_room.shortest_path import find_all_action_values
from four_room.wrappers import gym_wrapper
from rnd_exploration.explore_go import ExploreGoDataset, Transition
from rnd_exploration.rnd import RNDNetwork

gym.register('MiniGrid-FourRooms-v1', FourRoomsEnv)
size = 19
with open('configs/train.pl', 'rb') as file:
    train_config = dill.load(file)

with open('configs/test_reachable.pl', 'rb') as file:
    test_config = dill.load(file)

with open('configs/validation_unreachable.pl', 'rb') as file:
    val_config = dill.load(file)


def create_explogostar_dataset(dataset_size, save_dir, threshhold, render=False, device='cpu'):
    # storing exporego* dataset
    overlapdataset = ExploreGoDataset()
    
    if render:
        env = gym_wrapper(gym.make(
                'MiniGrid-FourRooms-v1', 
                agent_pos= train_config['agent positions'],
                goal_pos = train_config['goal positions'],
                doors_pos = train_config['topologies'],
                agent_dir = train_config['agent directions'],
                size=size, 
                render_mode="rgb_array",
            ),
            original_obs=True
        )
    else:
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
    try:
        
        rnd_net = RNDNetwork(env, device=device)
        explorego = ExploreGoDataset()
        imgs = []
        
        ep_highlight_mask = np.zeros((len(train_config['agent positions']), 
                                        env.get_wrapper_attr('width'), env.get_wrapper_attr('height')), dtype=bool)
        ep_colors = np.empty_like(ep_highlight_mask, dtype=object)

        while len(explorego) <= dataset_size:
            
            obs, _ = env.reset()
            done = False
            
            # emulate the (very good) pure exploration of explorego
            max_k = len(env.get_wrapper_attr('valid_pos'))
            k = np.random.randint(low=0, high=max_k)
            env.unwrapped.move_valid_pos(k)
        
            current_context = env.unwrapped.context
            print(f'Current size of dataset: {len(explorego):08d} | Current Context {current_context} | Current Uniqueness {explorego.ratio_unique_trans:.4f}', end='\r')
            
            # find optimal trajectory
            past_pos = []
            while not done:
                agent_pos = env.get_wrapper_attr('agent_pos')

                state = obs_to_state(obs)
                q = find_all_action_values(state[:2], state[2], state[3:5], state[5:], 0.99, size)
                q = np.array(q)
                action = q.argmax()
                rnd_val = rnd_net.get_error(obs, action)
                
                if rnd_val <= threshhold:
                    explorego.add_trans(np.array(obs), q)
                    explorego.add(np.array(obs), q, np.array(state))
                    if render: 
                        ep_colors[current_context, agent_pos[0], agent_pos[1]] = (0, 0, 255)
                        ep_highlight_mask[current_context, agent_pos[0], agent_pos[1]] = True
                        past_pos.append(agent_pos)
                        
                obs_prime, _, terminated, truncated, _ = env.step(action)
                obs = obs_prime
                done = terminated or truncated
                if render and len(explorego) >= dataset_size - 1000: imgs.append(env.render(highlight_mask=ep_highlight_mask[current_context], 
                                                colors=ep_colors[current_context]))
            if render:
                for pos in past_pos:
                    ep_colors[current_context, pos[0], pos[1]] = (51, 0, 102)
    

    except KeyboardInterrupt:
        with open(f'action_values/{save_dir}.pl', 'wb') as file:
            dill.dump(explorego, file)
        
    # save the obj       
    with open(f'action_values/{save_dir}.pl', 'wb') as file:
        dill.dump(explorego, file)
        
    return explorego, imgs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--size', type=int, default=10000, help='size of dataset')
    parser.add_argument('-f', '--dir', type=str, default='untitled', help='name of dataset')
    parser.add_argument('-d', '--device', type=str, default='cuda', help='device')
    parser.add_argument('-r', '--render', action='store_true', help='render mode')
    parser.add_argument('-t', '--thresh', type=float, default=float('-inf'), help='threshhold to add')

    args = parser.parse_args()

    dataset, img = create_explogostar_dataset(
        args.size, 
        args.dir, 
        args.threshhold, 
        args.render,
        args.device
    )