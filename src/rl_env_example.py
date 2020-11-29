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
"""A simple episode runner using the RL environment."""

from __future__ import print_function

import sys
import getopt
from hanabi_learning_environment import rl_env
from hanabi_learning_environment.agents.random_agent import RandomAgent
from greedy_agent import GreedyAgent
from hanabi_learning_environment.agents.simple_agent import SimpleAgent

AGENT_CLASSES = {'GreedyAgent': GreedyAgent, 'SimpleAgent': SimpleAgent, 'RandomAgent': RandomAgent}


class Runner(object):
    """Runner class."""

    def __init__(self, flags):
        """Initialize runner."""
        self.flags = flags
        self.agent_config = {'players': flags['players']}
        self.environment = rl_env.make('Hanabi-Full', num_players=flags['players'])
        self.agent_class = AGENT_CLASSES[flags['agent_class']]

    def run(self):
        """Run episodes."""
        rewards = []
        for episode in range(flags['num_episodes']):
            observations = self.environment.reset()
            agents = [self.agent_class(self.agent_config) for _ in range(self.flags['players'])]
            done = False
            episode_reward = 0
            turns = 0
            player_id = -1
            current_player_action = None
            while not done:
                for agent_id, agent in enumerate(agents):
                    agent.id = agent_id
                    observation = observations['player_observations'][agent_id]
                    if observations['current_player'] == agent_id:
                        action = agent.act(observation)
                        assert action is not None
                        current_player_action = action
                        player_id = agent_id
                observations, reward, done, unused_info = self.environment.step(current_player_action)
                if current_player_action is not None:
                    print('Agent: {} action: {}'.format(observation['current_player'], current_player_action))
                    hand = None
                    if current_player_action["action_type"].startswith("REVEAL"):
                        hand = observations['player_observations'][player_id]["observed_hands"][current_player_action["target_offset"]]
                    for agent_id, agent in enumerate(agents):
                        agent.on_action(player_id, current_player_action, hand)

                    # Make an environment step.
                episode_reward += reward
                turns += 1
            rewards.append(episode_reward)
            print('Running episode: %d' % episode)
            print('Max Reward: %.3f' % max(rewards))
        return rewards


if __name__ == "__main__":
    flags = {'players': 2, 'num_episodes': 1, 'agent_class': 'GreedyAgent'}
    options, arguments = getopt.getopt(sys.argv[1:], '',
                                       ['players=',
                                        'num_episodes=',
                                        'agent_class='])
    if arguments:
        sys.exit('usage: rl_env_example.py [options]\n'
                 '--players       number of players in the game.\n'
                 '--num_episodes  number of game episodes to run.\n'
                 '--agent_class   {}'.format(' or '.join(AGENT_CLASSES.keys())))
    for flag, value in options:
        flag = flag[2:]  # Strip leading --.
        flags[flag] = type(flags[flag])(value)
    runner = Runner(flags)
    runner.run()
