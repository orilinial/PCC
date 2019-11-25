import gym
import network_sim_tcp_cc
import argparse


def main(args):
    env = gym.make('PccNsTCP-v0')
    env.seed(args.seed)

    env.reset(max_bw=args.bandwidth)
    ep_reward = 0

    for t in range(1, 10000):
        _, _, done, _ = env.step()
        if done:
            break

    env.dump_events_to_file('results/test_tcp_cc_%.2f.json' % args.bandwidth)

    return ep_reward


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
    parser.add_argument('--seed', type=int, default=543, metavar='N',
                        help='random seed (default: 543)')
    parser.add_argument('--bandwidth', '-bw', type=float, default=15.0, help='Network bandwidth in Mbps')
    args = parser.parse_args()

    main(args)