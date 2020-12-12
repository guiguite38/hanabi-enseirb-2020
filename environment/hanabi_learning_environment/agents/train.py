import numpy as np
from hanabi_learning_environment import pyhanabi
from hanabi_learning_environment import rl_env

from hanabi_learning_environment.agents.dqn_agent import DQNAgent
import torch
import torch.nn.functional as F
from itertools import count

# if gpu is to be used
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def optimize_model(model):
    if len(model.memory) < model.BATCH_SIZE:
        return
    print("Optimizing...")
    transitions = model.memory.sample(model.BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = model.memory.Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(
        tuple(map(lambda s: s is not None, batch.next_state)),
        device=device,
        dtype=torch.bool,
    )
    non_final_next_states = torch.cat(
        torch.FloatTensor([s for s in batch.next_state if s is not None])
    )
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = model.policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(model.BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = (
        model.target_net(non_final_next_states).max(1)[0].detach()
    )
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * model.GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(
        state_action_values, expected_state_action_values.unsqueeze(1)
    )

    # Optimize the model
    model.optimizer.zero_grad()
    loss.backward()
    for param in model.policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    model.optimizer.step()


def score_game(fireworks):
    """returns the game score displayed by fireworks played up to now in the game.
    for now no heuristic is used to determine which hand is the most promising for a given score"""
    score = 0
    for coloured_firework in fireworks:
        score += fireworks[coloured_firework]
    return score


def run_training(
    config, game_parameters, num_episodes=50
):  # !! config, game_parameters necessary ?
    """Play a game, selecting random actions."""
    agent1 = DQNAgent(config, encoded_observation_size=658)
    agent2 = DQNAgent(config, encoded_observation_size=658)

    agents = [agent1, agent2]
    # !! 10/12/2020 : Missing second agent
    env = rl_env.make()

    for i_episode in range(num_episodes):
        # Initialize the environment and state
        observation_all = env.reset()
        state = env.state

        for i in count():
            agent = agents[i % 2]
            observation = observation_all["player_observations"][i % 2]

            # Select and perform an action
            action = agent.select_action(observation)

            new_obs_all, reward, done, _ = env.step(action)
            reward = torch.tensor([reward], device=device)
            new_obs = new_obs_all["player_observations"][i % 2]
            # Store the transition in memory
            if done:
                agent.memory.push(observation["vectorized"], action, None, reward)
                break
            else:
                agent.memory.push(
                    observation["vectorized"], action, new_obs["vectorized"], reward
                )

            # Move to the next state
            observation_all = new_obs_all

        # Perform one step of the optimization (on the target network)
        optimize_model(agent)

        # Update the target network, copying all weights and biases in DQN
        if i_episode % agent.TARGET_UPDATE == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())

    print("Complete")

    print("")
    print("Game done. Terminal state:")
    print("")
    print(state)
    print("")
    print("score: {}".format(state.score()))


if __name__ == "__main__":
    # Check that the cdef and library were loaded from the standard paths.
    assert pyhanabi.cdef_loaded(), "cdef failed to load"
    assert pyhanabi.lib_loaded(), "lib failed to load"
    flags = {"players": 2, "num_episodes": 1, "agent_class": "DQNAgent"}
    run_training(
        {"players": flags["players"], "colors": 5, "ranks": 5, "hand_size": 5},
        {"players": 2, "random_start_player": True},
    )
