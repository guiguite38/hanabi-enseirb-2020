import numpy as np
from hanabi_learning_environment import pyhanabi
from hanabi_learning_environment import rl_env
from hanabi_learning_environment.agents.dqn_agent import DQNAgent

import torch
import torch.nn.functional as F
from itertools import count
# if gpu is to be used
import os

from torch.utils.tensorboard import SummaryWriter

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
writer = SummaryWriter()

def optimize_model(model, epoch):
    if len(model.memory) < model.BATCH_SIZE:
        return
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
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    tmp = state_batch.view((model.BATCH_SIZE, -1))
    state_action_values = model.policy_net(tmp)
    state_action_values = torch.gather(
        state_action_values, 0, action_batch.reshape((model.BATCH_SIZE, 1))
    )

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(model.BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = (
        model.target_net(non_final_next_states.reshape((-1, 658))).max(1)[0].detach()
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
    writer.add_scalar("train_loss", loss.item(), epoch)


def score_game(fireworks):
    """returns the game score displayed by fireworks played up to now in the game.
    for now no heuristic is used to determine which hand is the most promising for a given score"""
    score = 0
    for coloured_firework in fireworks:
        score += fireworks[coloured_firework]
    return score


def is_hint(action, action_space):
    dict_action = action_space[action]
    return dict_action["action_type"].startswith("REVEAL")

def backprop_reward_if_card_is_played(episode_memory, dict_action, reward, action_space, nplayers, observation):
    # Check if action is playing card
    if dict_action["action_type"] != "PLAY":
        return

    card_index = dict_action["card_index"]
    # Get card color and rank
    card = observation["observed_hands"][-1][card_index]
    color = card["color"]
    rank = card["rank"]
    
    # Iterate over over past actions and look for hint of color/rank on this card
    player = -1 % nplayers
    for memory in reversed(episode_memory):
        action_number = memory[1].item()
        dict_action = action_space[action_number]
        if is_hint(action_number, action_space) and player != 0:
            player_offset = dict_action["target_offset"]
            if (player_offset + player) % nplayers == 0:
                # Player which was targeted was me
                if color == dict_action.get("color", None) or rank == dict_action.get("rank", None):
                    memory[3][0] += reward
        elif dict_action["action_type"] in ("PLAY", "DISCARD") and player == 0: 
            offset = dict_action["card_index"]
            # That means we drew the played card at this moment
            # So no further reward back prop can occur
            if offset == card_index:
                break

            if offset < card_index:
                card_index -= 1
        # Move to next player
        player -= 1
        player %= nplayers


def run_training(
    config, game_parameters, num_episodes=50
):  # !! config, game_parameters necessary ?
    """Play a game, selecting random actions."""
    observation_size = 658*4 # 4 previous observations are viewed to act
    agent1 = DQNAgent(config, encoded_observation_size=observation_size) 
    agent2 = DQNAgent(config, encoded_observation_size=observation_size)

    agents = [agent1, agent2]
    env = rl_env.make()

    total_rewards = 0

    for i_episode in range(num_episodes):
        # Initialize the environment and state
        observation_all = env.reset()
        state = env.state

        episode_memory = []

        episode_hints = [0 for x in agents]

        agent_buffer = [torch.zeros(observation_size) for agent in agents]

        for i in count():
            agent = agents[i % 2]
            observation = observation_all["player_observations"][i % 2]


            # Select and perform an action
            buffer = agent_buffer[i % 2]
            copy = buffer.clone()
            buffer[658:] = copy[:658 * 3]
            buffer[:658] = torch.from_numpy(np.asarray(observation["vectorized"]))
            # if len(episode_memory) > 3 :
            #     effective_observation = np.concatenate((episode_memory[len(episode_memory)-4:][0], observation)) 
            # elif len(episode_memory) == 0:
            #     effective_observation = np.tile(observation,4)
            # else:
            #     effective_observation = np.concatenate((episode_memory[:][0], np.tile(observation,(4-len(episode_memory)))))

            action, action_number = agent.select_action(observation, buffer)

            new_obs_all, reward, done, _ = env.step(action)
            reward = torch.tensor([reward], device=device)
            new_obs = new_obs_all["player_observations"][i % 2]
            backprop_reward_if_card_is_played(episode_memory, action, reward, agent.action_space, len(agents), observation_all["player_observations"][(i + 1) % 2])

            # Prepare next buffer
            next_buffer = buffer.clone()
            next_buffer[658:] = buffer[:658 * 3]
            next_buffer[:658] = torch.from_numpy(np.asarray(new_obs["vectorized"]))

            if is_hint(action_number, agent.action_space):
                episode_hints[i % 2] += 1
           
            # Store the transition in memory
            if done:
                episode_memory.append([
                    buffer.clone(),
                    torch.LongTensor([action_number]),
                    None,
                    reward,
                ])
                total_rewards += reward.item()
                break
            else:
                episode_memory.append([
                    buffer.clone(),
                    torch.LongTensor([action_number]),
                    new_buffer,
                    reward,
                ])

            # Move to the next state
            observation_all = new_obs_all

        # Add episode memory to all agents
        for memory in episode_memory:
            for agent in agents:
                agent.memory.push(memory)


        # Add metric
        for i, agent in enumerate(agents):
            writer.add_scalar(f"hints agent {i + 1}", episode_hints[i], i_episode)

        for agent in agents:
            # Perform one step of the optimization (on the target network)
            optimize_model(agent, i_episode)

            # Update the target network, copying all weights and biases in DQN
            if i_episode % agent.TARGET_UPDATE == 0:
                agent.target_net.load_state_dict(agent.policy_net.state_dict())

    print("Complete, saving...")
    for i, agent in enumerate(agents):
        torch.save(agent.policy_net.state_dict(), f"policy_net_{i}.pt")
    print("")
    print("Game done. Terminal state:")
    print("")
    print(state)
    print("")
    print("score: {}".format(state.score()))

    print("Mean reward:", total_rewards / num_episodes)


if __name__ == "__main__":
    # Check that the cdef and library were loaded from the standard paths.
    assert pyhanabi.cdef_loaded(), "cdef failed to load"
    assert pyhanabi.lib_loaded(), "lib failed to load"
    flags = {"players": 2, "num_episodes": 1, "agent_class": "DQNAgent"}
    run_training(
        {"players": flags["players"], "colors": 5, "ranks": 5, "hand_size": 5},
        {"players": 2, "random_start_player": True},
        num_episodes=200,
    )
