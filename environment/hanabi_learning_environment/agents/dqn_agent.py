# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Simple Agent."""

from hanabi_learning_environment.rl_env import Agent

import math
import random
import matplotlib
import matplotlib.pyplot as plt

import torch

import torch.optim as optim

from replay_memory import ReplayMemory
from dqn import DQN

is_ipython = "inline" in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if gpu is to be used
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DQNAgent(Agent):
    """Agent that applies a simple heuristic."""

    def __init__(
        self, config, encoded_observation_size, *args, **kwargs
    ):  # !! There must be a way to avoid "encoded_observation_size" being a parameter
        """Initialize the agent."""
        self.config = config
        # Extract max info tokens or set default to 8.
        self.max_information_tokens = config.get("information_tokens", 8)
        # DQN Params
        self.BATCH_SIZE = 128
        self.GAMMA = 0.999
        self.EPS_START = 0.9
        self.EPS_END = 0.05
        self.EPS_DECAY = 200
        self.TARGET_UPDATE = 10
        # initialise DQN
        self.n_actions = 20  # hard coded maybe do better one day

        self.hand_size = config.get(
            "hand_size", 4 if config.get("players", 2) > 3 else 5
        )

        self.policy_net = DQN(
            input_size=encoded_observation_size, output_size=self.n_actions
        ).to(device)
        self.target_net = DQN(
            input_size=encoded_observation_size, output_size=self.n_actions
        ).to(device)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.policy_net.eval()

        self.optimizer = optim.RMSprop(self.policy_net.parameters())
        self.memory = ReplayMemory(10000)

        self.steps_done = 0

        self.action_space = self.build_action_space()

    @staticmethod
    def playable_card(card, fireworks):
        """A card is playable if it can be placed on the fireworks pile."""
        return card["rank"] == fireworks[card["color"]]

    def build_action_space(self):
        """
        returns all possible actions in an ordered list
        """
        action_space = []
        for i in range(self.hand_size):
            action_space.append({"action_type": "PLAY", "card_index": i})
            action_space.append({"action_type": "DISCARD", "card_index": i})

        for player_offset in range(1, self.config.get("players", 2)):
            for color in ["W", "B", "R", "Y", "G"]:
                action_space.append(
                    {
                        "action_type": "REVEAL_COLOR",
                        "color": color,
                        "target_offset": player_offset,
                    }
                )
            for rank in range(5):
                action_space.append(
                    {
                        "action_type": "RevealRank",
                        "rank": rank,
                        "target_offset": player_offset,
                    }
                )

        return action_space

    def select_action(self, observation, vector):
        action_space = self.action_space
        if observation["current_player_offset"] != 0:
            return None, None
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(
            -1.0 * self.steps_done / self.EPS_DECAY
        )
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.

                ordered_moves = self.policy_net(
                    vector.unsqueeze(0)
                ).argsort(descending=True)[0]
                i = 0
                action_index = ordered_moves[i].view(1, 1)
                while action_space[action_index] not in observation["legal_moves"]:
                    i += 1
                    action_index = ordered_moves[i].view(1, 1)
                return action_space[action_index], action_index.item()
        else:
            action_index = random.randrange(len(action_space))
            while action_space[action_index] not in observation["legal_moves"]:
                action_index = random.randrange(len(action_space))
            return action_space[action_index], action_index
