import network_sim
import torch
import gym
import models
import argparse
import numpy as np


def evaluate_model(args, episode_num, model=None, save_json=False):
    env = gym.make('PccNs-v0')
    env.seed(args.seed)
    torch.manual_seed(args.seed)

    if model is None:
        model = models.Policy()
        model.load_state_dict(torch.load('ac_model_%s.pkl' % args.reward))

    state = env.reset(reward=args.reward, max_bw=args.bandwidth, test=True)
    ep_reward = 0
    for t in range(1, 10000):

        # select action from policy
        state = torch.from_numpy(state).float()
        # action_mean, action_log_var, _ = model(state)
        action_mean, _ = model(state)
        # statistics.append(action_log_var.item())
        action = action_mean.item()

        # take the action
        state, reward, done, _ = env.step(action)

        model.rewards.append(reward)
        ep_reward += reward
        if done:
            break

    # log results
    # print('Evaluation results: reward = %.3f' % (ep_reward))
    if save_json:
        env.dump_events_to_file('results/test_rl_%s_%.2f.json' % (args.reward, args.bandwidth))

    return ep_reward


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
    parser.add_argument('--seed', type=int, default=543, metavar='N',
                        help='random seed (default: 543)')
    parser.add_argument('--bandwidth', '-bw', type=float, default=10.0, help='Network bandwidth in Mbps')
    parser.add_argument('--reward', type=str, default='latency', choices=['throughput', 'latency'],
                        help='RL agent\'s goal')
    args = parser.parse_args()

    ep_reward = evaluate_model(args, -1, save_json=True)
    print('Evaluate reward = %.2f' % ep_reward)
