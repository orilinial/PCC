import json
import matplotlib.pyplot as plt
import numpy as np
import sys

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


def analyze():
    graphs_dir = 'results/graphs/'

    #######
    # 200 #
    #######
    test_rl_thpt_200 = parse_json('results/test_rl_throughput_2.40.json')
    test_tcp_cc_200 = parse_json('results/test_tcp_cc_2.40.json')

    plt.figure(0)
    plt.plot(test_rl_thpt_200["time_data"], test_rl_thpt_200["rew_data"])
    plt.ylabel('Reward')
    plt.xlabel('Monitor Interval')
    plt.title('Reward Convergence, Throughput = 2.4 Mbps')
    plt.savefig(graphs_dir + 'reward_graph_200.png')

    plt.figure(1)
    plt.plot(test_rl_thpt_200["time_data"], np.array(test_rl_thpt_200["send_data"]) / 1e6, label='RL')
    plt.plot(test_tcp_cc_200["time_data"], np.array(test_tcp_cc_200["send_data"]) / 1e6, label='TCP_CC')
    plt.ylabel('Send Rate (Mbps)')
    plt.legend()
    plt.xlabel('Monitor Interval')
    plt.title('Send Rate: RL based vs Ordinary CC, Throughput = 2.4 Mbps')
    plt.savefig(graphs_dir + 'send_graph_200.png')

    plt.figure(2)
    plt.plot(test_rl_thpt_200["time_data"], np.array(test_rl_thpt_200["thpt_data"]) / 1e6, label='RL')
    plt.plot(test_tcp_cc_200["time_data"], np.array(test_tcp_cc_200["thpt_data"]) / 1e6, label='TCP_CC')
    plt.ylabel('Throughput (Mbps)')
    plt.legend()
    plt.xlabel('Monitor Interval')
    plt.title('Throughput: RL based vs Ordinary CC, Throughput = 2.4 Mbps')
    plt.savefig(graphs_dir + 'thpt_graph_200.png')

    plt.figure(3)
    plt.plot(test_rl_thpt_200["time_data"], test_rl_thpt_200["latency_data"])
    plt.ylabel('Latency (sec)')
    plt.xlabel('Monitor Interval')
    plt.title('Latency, Throughput = 2.4 Mbps')
    plt.ylim((0.0, 0.3))
    plt.savefig(graphs_dir + 'latency_graph_200.png')

    plt.figure(4)
    plt.plot(test_rl_thpt_200["time_data"], test_rl_thpt_200["loss_data"])
    plt.ylabel('Loss rate')
    plt.xlabel('Monitor Interval')
    plt.title('Loss rate, Throughput = 2.4 Mbps')
    plt.ylim((0.0, 0.2))
    plt.savefig(graphs_dir + 'loss_graph_200.png')

    ########
    # 1000 #
    ########
    test_rl_thpt_1000 = parse_json('results/test_rl_throughput_12.00.json')
    test_tcp_cc_1000 = parse_json('results/test_tcp_cc_12.00.json')

    plt.figure(5)
    plt.plot(test_rl_thpt_1000["time_data"], test_rl_thpt_1000["rew_data"])
    plt.ylabel('Reward')
    plt.xlabel('Monitor Interval')
    plt.title('Reward Convergence, Throughput = 12 Mbps')
    plt.savefig(graphs_dir + 'reward_graph_1000.png')

    plt.figure(6)
    plt.plot(test_rl_thpt_1000["time_data"], np.array(test_rl_thpt_1000["send_data"]) / 1e6, label='RL')
    plt.plot(test_tcp_cc_1000["time_data"], np.array(test_tcp_cc_1000["send_data"]) / 1e6, label='TCP_CC')
    plt.ylabel('Send Rate (Mbps)')
    plt.legend()
    plt.xlabel('Monitor Interval')
    plt.title('Send Rate: RL based vs Ordinary CC, Throughput = 12 Mbps')
    plt.savefig(graphs_dir + 'send_graph_1000.png')

    plt.figure(7)
    plt.plot(test_rl_thpt_1000["time_data"], np.array(test_rl_thpt_1000["thpt_data"]) / 1e6, label='RL')
    plt.plot(test_tcp_cc_1000["time_data"], np.array(test_tcp_cc_1000["thpt_data"]) / 1e6, label='TCP_CC')
    plt.ylabel('Throughput (Mbps)')
    plt.legend()
    plt.xlabel('Monitor Interval')
    plt.title('Throughput: RL based vs Ordinary CC, Throughput = 12 Mbps')
    plt.savefig(graphs_dir + 'th_graph_1000.png')

    plt.figure(8)
    plt.plot(test_rl_thpt_1000["time_data"], test_rl_thpt_1000["latency_data"])
    plt.ylabel('Latency (sec)')
    plt.xlabel('Monitor Interval')
    plt.title('Latency, Throughput = 12 Mbps')
    plt.ylim((0.0, 0.3))
    plt.savefig(graphs_dir + 'latency_graph_1000.png')

    plt.figure(9)
    plt.plot(test_rl_thpt_1000["time_data"], test_rl_thpt_1000["loss_data"])
    plt.ylabel('Loss rate')
    plt.xlabel('Monitor Interval')
    plt.title('Loss rate, Throughput = 12 Mbps')
    plt.ylim((0.0, 0.2))
    plt.savefig(graphs_dir + 'loss_graph_1000.png')


if __name__ == '__main__':
    analyze()
