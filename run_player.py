from __future__ import print_function
from mcts_ucb_agent import MCTSUCBAgent

import numpy as np
from hanabi_learning_environment import pyhanabi


def run_game(game_parameters):
  """Play a game, selecting random actions."""

  def print_state(state):
    """Print some basic information about the state."""
    print("")
    print("Current player: {}".format(state.cur_player()))
    print(state)

    # Example of more queries to provide more about this state. For
    # example, bots could use these methods to to get information
    # about the state in order to act accordingly.
    print("### Information about the state retrieved separately ###")
    print("### Information tokens: {}".format(state.information_tokens()))
    print("### Life tokens: {}".format(state.life_tokens()))
    print("### Fireworks: {}".format(state.fireworks()))
    print("### Deck size: {}".format(state.deck_size()))
    print("### Discard pile: {}".format(str(state.discard_pile())))
    print("### Player hands: {}".format(str(state.player_hands())))
    print("")

  def print_observation(observation):
    """Print some basic information about an agent observation."""
    print("--- Observation ---")
    print(observation)

    print("### Information about the observation retrieved separately ###")
    print("### Current player, relative to self: {}".format(
        observation.cur_player_offset()))
    print("### Observed hands: {}".format(observation.observed_hands()))
    print("### Card knowledge: {}".format(observation.card_knowledge()))
    print("### Discard pile: {}".format(observation.discard_pile()))
    print("### Fireworks: {}".format(observation.fireworks()))
    print("### Deck size: {}".format(observation.deck_size()))
    move_string = "### Last moves:"
    for move_tuple in observation.last_moves():
      move_string += " {}".format(move_tuple)
    print(move_string)
    print("### Information tokens: {}".format(observation.information_tokens()))
    print("### Life tokens: {}".format(observation.life_tokens()))
    print("### Legal moves: {}".format(observation.legal_moves()))
    print("--- EndObservation ---")

  def print_encoded_observations(encoder, state, num_players):
    print("--- EncodedObservations ---")
    print("Observation encoding shape: {}".format(encoder.shape()))
    print("Current actual player: {}".format(state.cur_player()))
    for i in range(num_players):
      print("Encoded observation for player {}: {}".format(
          i, encoder.encode(state.observation(i))))
    print("--- EndEncodedObservations ---")

  game = pyhanabi.HanabiGame(game_parameters)
  print(game.parameter_string(), end="")
  obs_encoder = pyhanabi.ObservationEncoder(
      game, enc_type=pyhanabi.ObservationEncoderType.CANONICAL)

  state = game.new_initial_state()
  player1 = MCTSUCBAgent(None)
  player2 = MCTSUCBAgent(None)

  while not state.is_terminal():
    observation = state.observation(state.cur_player())
    if state.cur_player() == 0:
        move = player1.act(observation, state)
    elif state.cur_player() == 1:
        move = player2.act(observation, state)

    print_state(state)
    print_observation(observation)
    print_encoded_observations(obs_encoder, state, game.num_players())

    state.apply_move(move)

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
  run_game({"players": 2, "random_start_player": True})