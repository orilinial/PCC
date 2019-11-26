import json
import matplotlib.pyplot as plt
import numpy as np
import torch


def parse_json(file_name):
    with open(file_name) as f:
        data = json.load(f)
    res = {
        "time_data": [float(event["Time"]) for event in data["Events"][1:]],
        "rew_data": [float(event["Reward"]) for event in data["Events"][1:]],
        "send_data": [float(event["Send Rate"]) for event in data["Events"][1:]],
        "thpt_data": [float(event["Throughput"]) for event in data["Events"][1:]],
        "latency_data": [float(event["Latency"]) for event in data["Events"][1:]],
        "loss_data": [float(event["Loss Rate"]) for event in data["Events"][1:]],
        }
    return res


def throughput_rl_vs_tcp_cc(rl, tcp, path):
    bw_list = [5.0, 10.0, 15.0, 20.0]
    for bw in rl:
        if bw in bw_list:
            fig, axs = plt.subplots(2, num=int(bw))
            plt.subplots_adjust(hspace=0.4)
            axs[0].plot(rl[bw]["time_data"], np.array(rl[bw]["thpt_data"]) / 1e6, label='RL')
            axs[0].plot(tcp[bw]["time_data"], np.array(tcp[bw]["thpt_data"]) / 1e6, label='TCP_CC')
            axs[0].plot(rl[bw]["time_data"], np.ones_like(np.array(rl[bw]["thpt_data"])) * bw, label='Max BW')
            axs[0].set(ylabel='Throughput (Mbps)')
            axs[0].legend()
            axs[0].grid()
            axs[0].set(title='Throughput: RL vs TCP, Max throughput = %d Mbps' % bw)

            plt.figure()
            axs[1].plot(rl[bw]["time_data"], np.array(rl[bw]["send_data"]) / 1e6, label='RL')
            axs[1].plot(tcp[bw]["time_data"], np.array(tcp[bw]["send_data"]) / 1e6, label='TCP_CC')
            axs[1].plot(rl[bw]["time_data"], np.ones_like(np.array(rl[bw]["send_data"])) * bw, label='Max BW')
            axs[1].set(ylabel='Send rate (Mbps)')
            axs[1].legend()
            axs[1].grid()
            axs[1].set(xlabel='Monitor Interval')
            axs[1].set(title='Send Rate: RL vs TCP, Max throughput = %d Mbps' % bw)
            fig.savefig(path + 'thpt_send_rl_tcp_%d.png' % bw)


def latency_and_loss_rl_graph(rl_dict, path):
    bw = 10.0
    rl = rl_dict[bw]
    fig, axs = plt.subplots(2)
    plt.subplots_adjust(hspace=0.4)
    axs[0].plot(rl["time_data"], rl["latency_data"])
    axs[0].set(ylabel='Latency (sec)')
    axs[0].set(title='Latency, Max Throughput = %d Mbps' % bw)
    axs[0].set(ylim=(0.0, 0.3))

    axs[1].plot(rl["time_data"], rl["loss_data"])
    axs[1].set(ylabel='Loss rate')
    axs[1].set(xlabel='Monitor Interval')
    axs[1].set(title='Loss rate, Max Throughput = %d Mbps' % bw)
    axs[1].set(ylim=(0.0, 0.2))
    fig.savefig(path + 'latency_loss_graph_%d.png' % bw)


def throughput_vs_send_rate(rl, rl_latency, path):
    res_bws = []
    res_latency = []
    max_bws = []
    for bw in rl:
        res_bws.append((np.array(rl[bw]["thpt_data"][350:]) / 1e6).mean())
        res_latency.append((np.array(rl_latency[bw]["thpt_data"][350:]) / 1e6).mean())

        max_bws.append(bw)

    plt.figure()
    plt.scatter(max_bws, res_bws, c='b', label='RL')
    plt.scatter(max_bws, res_latency, c='g', label='RL-Improved')
    plt.plot(max_bws, res_bws, '-b')
    plt.plot(max_bws, res_latency, '--g')

    plt.plot(range(24), range(24), '-r', label='Optimal')
    plt.plot(np.ones(24) * 1.2, range(24), '-k')
    plt.plot(np.ones(24) * 6.0, range(24), '-k')
    plt.grid()
    plt.xlabel('Link Capacity [Mbps]')
    plt.ylabel('Achieved Throughput [Mbps]')
    plt.legend()
    plt.title('Bandwidth sensitivity')
    plt.savefig(path + 'throughput_vs_send_rate.png')


def throughput_of_modified_reward(rl_dict, rl_latency_dict, path):
    for key in rl_latency_dict:
        bw = key
        rl = rl_dict[bw]
        rl_latency = rl_latency_dict[bw]

        fig, axs = plt.subplots(3)
        plt.subplots_adjust(hspace=0.4)
        fig.set_size_inches(11.5, 7.5)
        axs[0].plot(rl["time_data"], np.array(rl["thpt_data"])/1e6, label='Regular reward')
        axs[0].plot(rl_latency["time_data"], np.array(rl_latency["thpt_data"])/1e6, label='Latency driven reward')
        axs[0].set(ylabel='Throughput [Mbps]')
        axs[0].set(title='Throughput of different rewards')
        axs[0].grid()
        axs[0].legend(loc='lower right')

        axs[1].plot(rl["time_data"], rl["latency_data"], label='Regular reward')
        axs[1].plot(rl_latency["time_data"], rl_latency["latency_data"], label='Latency driven reward')
        axs[1].set(ylabel='Latency [sec]')
        axs[1].set(title='Latency of different rewards')
        axs[1].set(ylim=(0.15, 0.25))
        axs[1].grid()
        axs[1].legend(loc='lower right')

        axs[2].plot(rl["time_data"], rl["loss_data"], label='Regular reward')
        axs[2].plot(rl_latency["time_data"], rl_latency["loss_data"], label='Latency driven reward')
        axs[2].set(xlabel='Monitor Interval')
        axs[2].set(ylabel='Loss rate')
        axs[2].set(title='Loss rate of different rewards')
        axs[2].grid()
        axs[2].legend(loc='lower right')

        fig.savefig(path + 'throughput_of_modified_reward.png')


def analyze(path):
    test_rl = {2.0: parse_json('results/test_rl_throughput_2.00.json'),
               3.0: parse_json('results/test_rl_throughput_3.00.json'),
               5.0: parse_json('results/test_rl_throughput_5.00.json'),
               7.0: parse_json('results/test_rl_throughput_7.00.json'),
               10.0: parse_json('results/test_rl_throughput_10.00.json'),
               12.0: parse_json('results/test_rl_throughput_12.00.json'),
               15.0: parse_json('results/test_rl_throughput_15.00.json'),
               20.0: parse_json('results/test_rl_throughput_20.00.json'),
               }

    test_tcp_cc = {5.0: parse_json('results/test_tcp_cc_5.00.json'),
                   10.0: parse_json('results/test_tcp_cc_10.00.json'),
                   15.0: parse_json('results/test_tcp_cc_15.00.json'),
                   20.0: parse_json('results/test_tcp_cc_20.00.json')
                   }

    test_rl_latency = {5.0: parse_json('results/test_rl_latency_5.00.json')}

    test_rl_full_latency = {2.0: parse_json('results/test_rl_latency_2.00.json'),
                            3.0: parse_json('results/test_rl_latency_3.00.json'),
                            5.0: parse_json('results/test_rl_latency_5.00.json'),
                            7.0: parse_json('results/test_rl_latency_7.00.json'),
                            10.0: parse_json('results/test_rl_latency_10.00.json'),
                            12.0: parse_json('results/test_rl_latency_12.00.json'),
                            15.0: parse_json('results/test_rl_latency_15.00.json'),
                            20.0: parse_json('results/test_rl_latency_20.00.json'),
                            }

    throughput_vs_send_rate(test_rl, test_rl_full_latency, path)
    throughput_rl_vs_tcp_cc(test_rl, test_tcp_cc, path)
    latency_and_loss_rl_graph(test_rl, path)
    throughput_of_modified_reward(test_rl, test_rl_latency, path)  # Throughput of RL-throughput bs RL-latency


def create_reward_plot(path):
    rewards = np.array(torch.load('total_test_rewards.pkl'))
    plt.figure()
    plt.plot(np.arange(rewards.shape[0]) * 10, rewards)
    plt.grid()
    plt.xlabel('Training Episodes')
    plt.ylabel('Reward')
    plt.title('Accumulated Evaluation Reward per train episode')
    plt.savefig(path + 'reward_plot.png')


if __name__ == '__main__':
    graphs_dir = 'results/graphs/'
    analyze(graphs_dir)
    create_reward_plot(graphs_dir)
