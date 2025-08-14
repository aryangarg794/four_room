import gymnasium as gym
import numpy as np 
import torch 
import torch.nn as nn
import argparse
import random
import imageio
import os

from copy import deepcopy
from dataclasses import dataclass
from tqdm import tqdm
from collections import deque

from rnd_exploration.rnd import RNDNetwork
from four_room.env import FourRoomsEnv
from four_room.utils import obs_to_state
from four_room.shortest_path import find_all_action_values
from four_room.wrappers import gym_wrapper
from rnd_exploration.rnd import RNDNetwork
from rnd_exploration.utils import RunningAverage, train_config, val_config, test_config, size
from dqn_experiments.regression_exp_utils import run_experiment
from dqn.model import DQN

gym.register('MiniGrid-FourRooms-v1', FourRoomsEnv)

@dataclass
class Args:
    env: gym.Env
    val_env: gym.Env 
    lr_agent: float = 5e-4
    use_cnn: bool = True
    capacity: int = int(1e5)
    tau: float = 0.005
    lr_rnd: float = 1e-5
    use_actions: bool = False
    device: str = 'cuda'

    

def train_dqn_rnd(
    args: Args, 
    batch_size: int = 512, 
    gamma: float = 0.99, 
    num_timesteps: int = int(2e5), 
    grad_norm: float = 1.0,
    regression_freq: int = 50000,
    seed: int = 0,
    alpha: float = 1.5, 
    window: int = 250, 
    update_freq: int = 1, 
    warmupsteps: int = 4000,
    render: bool = False 
): 
    """
    """
    rms = RunningAverage(window_size=window)
    mse_loss = nn.MSELoss()
    os.makedirs('dqn_results', exist_ok=True)
    imgs = []
    learning_curves = []
    scores = []
    uniqueness = []
    
    torch.backends.cudnn.deterministic = True
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    agent = DQN(
        env=args.env,
        val_env=args.val_env,
        capacity=args.capacity,
        tau=args.tau,
        lr=args.lr_agent,
        device=args.device,
        use_cnn=args.use_cnn
    )
    
    rnd_net = RNDNetwork(
        env=args.env, 
        lr=args.lr_rnd,
        device=args.device
    )
    
    env = deepcopy(args.env)
    items_added = 0
    
    obs, _ = env.reset(seed=seed)
    record = False
    state = obs_to_state(obs)
    goal_pos = state[3:5]
    target_pos = state[3:5] # first phase is warmup
    
    max_k = len(env.get_wrapper_attr('valid_pos'))
    k = np.random.randint(low=0, high=max_k)
    aux_pos = env.get_wrapper_attr('valid_pos')[k]
    env.get_wrapper_attr('move_valid_pos')(k)
    
    ep_highlight_mask = np.zeros((len(train_config['agent positions']), 
                                        env.get_wrapper_attr('width'), env.get_wrapper_attr('height')), dtype=bool)
    ep_colors = np.empty_like(ep_highlight_mask, dtype=object)
    
    current_context = env.unwrapped.context
    past_pos = []
    visit_history = deque(maxlen=args.capacity+1)
    
    for step in (pbar := tqdm(range(1, num_timesteps+1))): 
        obs_torch = torch.as_tensor(obs, device=args.device).view(1, *obs.shape)
        state = obs_to_state(obs)
    
        agent_pos = env.get_wrapper_attr('agent_pos')
        
        q = find_all_action_values(state[:2], state[2], target_pos, state[5:], 0.99, size)
        action = np.array(q).argmax()
        dqn_val = agent(obs_torch).max().item()
        
        obs_prime, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        if step < warmupsteps or record:
            assert np.array_equal(target_pos, goal_pos)
            agent.buffer.update(obs, action, reward, obs_prime, int(done), q_value=q)
            if render: 
                ep_colors[current_context, agent_pos[0], agent_pos[1]] = (0, 0, 255)
                ep_highlight_mask[current_context, agent_pos[0], agent_pos[1]] = True
                past_pos.append(agent_pos)
                visit_history.append((current_context, *agent_pos))
                
                if agent.buffer.size >= agent.buffer.capacity:
                    to_remove = visit_history[0]
                    ep_highlight_mask[to_remove[0], to_remove[1], to_remove[2]] = False
                    ep_colors[to_remove[0], to_remove[1], to_remove[2]] = None
               
            rnd_net.observe(obs)
            items_added += 1

        elif dqn_val - rms.avg >= alpha * rms.std or np.array_equal(agent_pos, aux_pos): # swap to record mode 
            record = True
            target_pos = goal_pos
            
        if render and step >= num_timesteps - 1000:
            env.get_wrapper_attr('set_aux')(aux_pos) # cannot add beforehand or else included in obs
            agent_col = (255, 0, 0) if np.array_equal(target_pos, goal_pos) else (0, 0, 255) 
            
            imgs.append(env.unwrapped.render(highlight_mask=ep_highlight_mask[current_context], 
                                        colors=ep_colors[current_context], agent_col=agent_col))
            env.get_wrapper_attr('remove_aux')(aux_pos)
            
        obs = obs_prime
        rms.update(dqn_val)
        
        if done:
            if render:
                for pos in past_pos:
                    ep_colors[current_context, pos[0], pos[1]] = (51, 0, 102)
                
            past_pos = []
            
            obs, _ = env.reset(seed=seed)
            done = False
            state = obs_to_state(obs)
            goal_pos = state[3:5]
            
            max_k = len(env.get_wrapper_attr('valid_pos'))
            k = np.random.randint(low=0, high=max_k)
            aux_pos = env.get_wrapper_attr('valid_pos')[k]
        
            if step < warmupsteps:
                target_pos = goal_pos # goal state
                env.get_wrapper_attr('move_valid_pos')(k)
            else: 
                target_pos = aux_pos
                
            record = False
            current_context = env.unwrapped.context
            
        if step % update_freq == 0: 
            batch_obs, batch_actions, _, batch_primes, batch_dones = agent.buffer.sample(batch_size=batch_size)
            batch_rewards = rnd_net.get_error(batch_obs)
            
            with torch.no_grad():
                target_vals = agent.target_net(batch_primes).max(dim=1, keepdim=True)[0]
                targets = batch_rewards + gamma * target_vals * (1 - batch_dones)
                
            q_values = agent.net(batch_obs).gather(dim=1, index=batch_actions)
            loss = mse_loss(q_values, targets)
            
            agent.optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(agent.net.parameters(), grad_norm)
            agent.optimizer.step()
            
            rnd_net.observe(batch_obs)
            
        agent.soft_update()
        
        if step % regression_freq == 0 and agent.buffer.size >= agent.buffer.capacity:
            lc, test_score = run_experiment(agent.buffer, device=args.device)
            learning_curves.append(lc)
            scores.append(test_score)
            
            results = {
                'agent': agent.net.state_dict(),
                'buffer': agent.buffer, 
                'rnd_net': rnd_net.rnd_net.state_dict(),
                'lcs': learning_curves, 
                'reg_test_scores' : scores,
                'uniqueness': uniqueness, 
                'images': imgs, 
            } 
            torch.save(results, f'dqn_results/{args.dir}_seed_{args.seed}.pt')
        
        uniqueness.append(agent.buffer.ratio_unique_trans)
        pbar.set_description(f"Training RND DQN | Uniqueness: {agent.buffer.ratio_unique_trans:.4f} | Last Regression Exp: {(scores[-1] if len(scores) > 0 else 0):.4f} | Total Items added: {items_added} | Current Context: {current_context}")
    
    return {
        'agent': agent.net.state_dict(),
        'buffer': agent.buffer, 
        'rnd_net': rnd_net.rnd_net.state_dict(),
        'lcs': learning_curves, 
        'reg_test_scores' : scores,
        'uniqueness': uniqueness, 
        'images': imgs, 
    } 
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--timesteps', type=int, default=int(2e5), help='timesteps')
    parser.add_argument('-f', '--dir', type=str, default='test', help='save name')
    parser.add_argument('-a', '--alpha', type=float, default=1.5, help='alpha')
    parser.add_argument('-rnd', '--lr_rnd', type=float, default=1e-5, help='lr for rnd')
    parser.add_argument('-ag', '--lr_agent', type=float, default=5e-4, help='lr for dqn agent')
    parser.add_argument('-d', '--device', type=str, default='cuda', help='device')
    parser.add_argument('-r', '--render', action='store_true', help='render mode')
    parser.add_argument('-s', '--replaysize', type=int, default=int(25000), help='size of replay buffer')
    parser.add_argument('-seed', '--seed', type=int, default=0, help='seed')
    parser.add_argument('-b', '--batch_size', type=int, default=512, help='batch size')
    parser.add_argument('-fr', '--freq', type=int, default=int(25000), help='freq of regression')
    parser.add_argument('-tau', '--tau', type=float, default=0.005, help='tau')
    
    args = parser.parse_args()
    
    env = gym_wrapper(gym.make(
            'MiniGrid-FourRooms-v1', 
            agent_pos= train_config['agent positions'],
            goal_pos = train_config['goal positions'],
            doors_pos = train_config['topologies'],
            agent_dir = train_config['agent directions'],
            size=size, 
            render_mode='rgb_array',
            disable_env_checker=True
        ),
        original_obs=True
    )
    
    val_env = gym_wrapper(gym.make(
            'MiniGrid-FourRooms-v1', 
            agent_pos= val_config['agent positions'],
            goal_pos = val_config['goal positions'],
            doors_pos = val_config['topologies'],
            agent_dir = val_config['agent directions'],
            size=size
        ),
        original_obs=True
    )
    
    test_env = gym_wrapper(gym.make(
            'MiniGrid-FourRooms-v1', 
            agent_pos= test_config['agent positions'],
            goal_pos = test_config['goal positions'],
            doors_pos = test_config['topologies'],
            agent_dir = test_config['agent directions'],
            size=size
        ),
        original_obs=True
    )
    
    aux_args = Args(
       env=env, 
       val_env=val_env, 
       lr_agent=args.lr_agent,
       device=args.device,
       capacity=args.replaysize, 
       tau=args.tau,
       lr_rnd=args.lr_rnd,
    )
    
    
    results = train_dqn_rnd(
        args=aux_args,
        batch_size=args.batch_size,
        num_timesteps=args.timesteps,
        seed=args.seed,
        alpha=args.alpha,
        regression_freq=args.freq,
        render=args.render
    )
    
    # with open(f'dqn_results/{args.dir}.pl', 'wb') as file:
    #     dill.dump(results, file)
    
    torch.save(results, f'dqn_results/{args.dir}_seed_{args.seed}.pt')
    if args.render:
        imageio.mimsave(f'renders/rendered_{args.dir}_seed_{args.seed}.gif', [np.array(img) for i, img in enumerate(results['images'][-500:]) if i%1 == 0], duration=150)