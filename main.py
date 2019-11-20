import argparse
import gym
import numpy as np
from itertools import count
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import normal
import models
import network_sim
import evaluate

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])
eps = np.finfo(np.float32).eps.item()


def select_action(state, model):
    state = torch.from_numpy(state).float()
    action_mean, action_log_var, state_value = model(state)

    # create a categorical distribution over the list of probabilities of actions
    m = normal.Normal(action_mean, torch.exp(0.5 * action_log_var))

    # and sample an action using the distribution
    action = m.sample()

    # save to action buffer
    model.saved_actions.append(SavedAction(m.log_prob(action), state_value))

    # the action to take (left or right)
    return action.item()


def finish_episode(optimizer, model):
    """
    Training code. Calcultes actor and critic loss and performs backprop.
    """
    R = 0
    saved_actions = model.saved_actions
    policy_losses = [] # list to save actor (policy) loss
    value_losses = [] # list to save critic (value) loss
    returns = [] # list to save the true values

    # calculate the true value using rewards returned from the environment
    for r in model.rewards[::-1]:
        # calculate the discounted value
        R = r + args.gamma * R
        returns.insert(0, R)

    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)

    for (log_prob, value), R in zip(saved_actions, returns):
        advantage = R - value.item()

        # calculate actor (policy) loss
        policy_losses.append(-log_prob * advantage)

        # calculate critic (value) loss using L1 smooth loss
        value_losses.append(F.smooth_l1_loss(value, torch.tensor([R])))

    # reset gradients
    optimizer.zero_grad()

    # sum up all the values of policy_losses and value_losses
    loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()

    # perform backprop
    loss.backward()
    optimizer.step()

    # reset rewards and action buffer
    del model.rewards[:]
    del model.saved_actions[:]


def main(args):
    model = models.Policy()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    env = gym.make('PccNs-v0')
    env.seed(args.seed)
    torch.manual_seed(args.seed)
    # total_ep_rewards = []
    # total_test_rewards = []

    running_reward = 10

    # run inifinitely many episodes
    for i_episode in range(args.episodes):

        # reset environment and episode reward
        state = env.reset(reward=args.reward, max_bw=args.bandwidth, test=False)
        ep_reward = 0

        # for each episode, only run 9999 steps so that we don't
        # infinite loop while learning
        for t in range(1, 10000):

            # select action from policy
            action = select_action(state, model)

            # take the action
            state, reward, done, _ = env.step(action)

            model.rewards.append(reward)
            ep_reward += reward
            if done:
                break

        # update cumulative reward
        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward

        # perform backprop
        finish_episode(optimizer, model)
        # total_ep_rewards.append(ep_reward)

        # log results
        if i_episode % args.log_interval == 0:
            # save_json = True if i_episode % (args.log_interval * 10) == 0 else False
            test_reward = evaluate.evaluate_model(args, i_episode, model, save_json=False)
            # total_test_rewards.append(test_reward)

            torch.save(model.state_dict(), 'ac_model_%s.pkl' % args.reward)

            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}, Test reward: {:.2f}'.format(
                  i_episode, ep_reward, running_reward, test_reward))

    # torch.save(total_ep_rewards, 'total_ep_rewards.pkl')
    # torch.save(total_test_rewards, 'total_test_rewards.pkl')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor (default: 0.99)')
    parser.add_argument('--seed', type=int, default=543, metavar='N',
                        help='random seed (default: 543)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='interval between training status logs (default: 10)')
    parser.add_argument('--episodes', type=int, default=10000)
    parser.add_argument('--learning-rate', '-lr', type=float, default=1e-3)
    parser.add_argument('--bandwidth', '-bw', type=float, default=2.4, help='Network bandwidth in Mbps')
    parser.add_argument('--reward', type=str, default='latency', choices=['throughput', 'latency'],
                        help='RL agent\'s goal')

    args = parser.parse_args()

    main(args)
