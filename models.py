import torch
import torch.nn as nn


class Policy(nn.Module):
    """
    implements both actor and critic in one model
    """
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(30, 128)

        # actor's layer
        self.action_mean = nn.Linear(128, 1)

        # critic's layer
        self.value = nn.Linear(128, 1)

        self.relu = nn.ReLU()

        # action & reward buffer
        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        """
        forward of both actor and critic
        """
        x = self.relu(self.affine1(x))

        # actor: choses action to take from state s_t
        # by returning probability of each action
        action_mean = self.action_mean(x)
        action_log_var = self.action_mean(x)

        # critic: evaluates being in the state s_t
        state_value = self.value(x)

        # return values for both actor and critic as a tupel of 2 values:
        # 1. a list with the probability of each action over the action space
        # 2. the value from state s_t
        return action_mean, action_log_var, state_value
