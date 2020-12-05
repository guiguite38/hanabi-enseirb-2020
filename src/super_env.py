from hanabi_learning_environment import pyhanabi
from hanabi_learning_environment.pyhanabi import color_char_to_idx

from typing import List, Dict


MOVE_TYPES = [_.name for _ in pyhanabi.HanabiMoveType]


class SuperEnv():
    def __init__(self, config, state=None):
        self.config = config
        self.game = pyhanabi.HanabiGame(config)
        if state is not None:
            self.state = state
            # self.game.free()
            self.game._game = self.state._game
        self.observation_encoder = pyhanabi.ObservationEncoder(
            self.game, pyhanabi.ObservationEncoderType.CANONICAL)
        self.players = self.game.num_players()

    def reset(self):
        self.state = self.game.new_initial_state()

        while self.state.cur_player() == pyhanabi.CHANCE_PLAYER_ID:
            self.state.deal_random_card()

        obs = self._make_observation_all_players()
        obs["current_player"] = self.state.cur_player()
        return obs

    def observations(self):
        obs = self._make_observation_all_players()
        obs["current_player"] = self.state.cur_player()
        return obs

    def vectorized_observation_shape(self):
        """Returns the shape of the vectorized observation.

        Returns:
         A list of integer dimensions describing the observation shape.
        """
        return self.observation_encoder.shape()

    def num_moves(self):
        return self.game.max_moves()

    def step(self, action):
        if isinstance(action, dict):
            # Convert dict action HanabiMove
            action = self._build_move(action)
        elif isinstance(action, int):
            # Convert int action into a Hanabi move.
            action = self.game.get_move(action)
        else:
            raise ValueError("Expected action as dict or int, got: {}".format(action))

        last_score = self.state.score()
        # Apply the action to the state.
        self.state.apply_move(action)

        while self.state.cur_player() == pyhanabi.CHANCE_PLAYER_ID:
            self.state.deal_random_card()

        observation = self._make_observation_all_players()
        done = self.state.is_terminal()
        # Reward is score differential. May be large and negative at game end.
        reward = self.state.score() - last_score
        info = {}

        return (observation, reward, done, info)

    def hand_size(self, player_offset: int):
        return len(self.state.player_hands()[player_offset])

    def return_hand(self, player_offset: int):
        for card_index in range(self.hand_size(player_offset)):
            # Return the card in furthest left position (cards always shift downwards)
            return_move = pyhanabi.HanabiMove.get_return_move(card_index=0, player=player_offset)
            self.state.apply_move(return_move)

    def cloned_with_swapped_hand(self, player_offset: int, new_hand: List[Dict]):
        # copy = SuperEnv(self.config, state=self.state.copy())
        self.return_hand(player_offset)
        for card_index, card in enumerate(new_hand):
            color, rank = color_char_to_idx(card["color"]), card["rank"]
            deal_specific_move = pyhanabi.HanabiMove.get_deal_specific_move(card_index, player_offset, color, rank)
            self.state.apply_move(deal_specific_move)
        return self

    def _make_observation_all_players(self):
        """Make observation for all players.

        Returns:
         dict, containing observations for all players.
         """
        obs = {}
        player_observations = [self._extract_dict_from_backend(player_id, self.state.observation(player_id))
                               for player_id in range(self.players)]  # pylint: disable=bad-continuation
        obs["player_observations"] = player_observations
        obs["current_player"] = self.state.cur_player()
        return obs

    def _extract_dict_from_backend(self, player_id, observation):
        """Extract a dict of features from an observation from the backend.

        Args:
         player_id: Int, player from whose perspective we generate the observation.
         observation: A `pyhanabi.HanabiObservation` object.

        Returns:
         obs_dict: dict, mapping from HanabiObservation to a dict.
        """
        obs_dict = {}
        obs_dict["current_player"] = self.state.cur_player()
        obs_dict["current_player_offset"] = observation.cur_player_offset()
        obs_dict["life_tokens"] = observation.life_tokens()
        obs_dict["information_tokens"] = observation.information_tokens()
        obs_dict["num_players"] = observation.num_players()
        obs_dict["deck_size"] = observation.deck_size()

        obs_dict["fireworks"] = {}
        fireworks = self.state.fireworks()
        for color, firework in zip(pyhanabi.COLOR_CHAR, fireworks):
            obs_dict["fireworks"][color] = firework

        obs_dict["legal_moves"] = []
        obs_dict["legal_moves_as_int"] = []
        for move in observation.legal_moves():
            obs_dict["legal_moves"].append(move.to_dict())
            obs_dict["legal_moves_as_int"].append(self.game.get_move_uid(move))

        obs_dict["observed_hands"] = []
        for player_hand in observation.observed_hands():
            cards = [card.to_dict() for card in player_hand]
            obs_dict["observed_hands"].append(cards)

        obs_dict["discard_pile"] = [
           card.to_dict() for card in observation.discard_pile()
        ]

        # Return hints received.
        obs_dict["card_knowledge"] = []
        for player_hints in observation.card_knowledge():
            player_hints_as_dicts = []
            for hint in player_hints:
                hint_d = {}
                if hint.color() is not None:
                    hint_d["color"] = pyhanabi.color_idx_to_char(hint.color())
                else:
                    hint_d["color"] = None
                hint_d["rank"] = hint.rank()
                player_hints_as_dicts.append(hint_d)
            obs_dict["card_knowledge"].append(player_hints_as_dicts)

        # ipdb.set_trace()
        obs_dict["vectorized"] = self.observation_encoder.encode(observation)
        obs_dict["pyhanabi"] = observation

        return obs_dict

    def _build_move(self, action):
        """Build a move from an action dict.

        Args:
         action: dict, mapping to a legal action taken by an agent. The following
           actions are supported:
             - { 'action_type': 'PLAY', 'card_index': int }
             - { 'action_type': 'DISCARD', 'card_index': int }
             - {
                 'action_type': 'REVEAL_COLOR',
                 'color': str,
                 'target_offset': int >=0
               }
             - {
                 'action_type': 'REVEAL_RANK',
                 'rank': str,
                 'target_offset': int >=0
               }

        Returns:
         move: A `HanabiMove` object constructed from action.

        Raises:
         ValueError: Unknown action type.
        """
        assert isinstance(action, dict), "Expected dict, got: {}".format(action)
        assert "action_type" in action, ("Action should contain `action_type`. action: {}").format(action)
        action_type = action["action_type"]
        assert (action_type in MOVE_TYPES), (
           "action_type: {} should be one of: {}".format(action_type, MOVE_TYPES))

        if action_type == "PLAY":
            card_index = action["card_index"]
            move = pyhanabi.HanabiMove.get_play_move(card_index=card_index)
        elif action_type == "DISCARD":
            card_index = action["card_index"]
            move = pyhanabi.HanabiMove.get_discard_move(card_index=card_index)
        elif action_type == "REVEAL_RANK":
            target_offset = action["target_offset"]
            rank = action["rank"]
            move = pyhanabi.HanabiMove.get_reveal_rank_move(
             target_offset=target_offset, rank=rank)
        elif action_type == "REVEAL_COLOR":
            target_offset = action["target_offset"]
            assert isinstance(action["color"], str)
            color = color_char_to_idx(action["color"])
            move = pyhanabi.HanabiMove.get_reveal_color_move(
             target_offset=target_offset, color=color)
        else:
            raise ValueError("Unknown action_type: {}".format(action_type))

        legal_moves = self.state.legal_moves()
        assert (str(move) in map(
           str,
           legal_moves)), "Illegal action: {}. Move should be one of : {}".format(
               move, legal_moves)

        return move


def make(num_players=2, pyhanabi_path=None):
    """Make an environment.

     Args:
       environment_name: str, Name of the environment to instantiate.
       num_players: int, Number of players in this game.
       pyhanabi_path: str, absolute path to header files for c code linkage.

     Returns:
       env: An `Environment` object.

     Raises:
       ValueError: Unknown environment name.
    """

    if pyhanabi_path is not None:
        prefixes = (pyhanabi_path,)
        assert pyhanabi.try_cdef(prefixes=prefixes), "cdef failed to load"
        assert pyhanabi.try_load(prefixes=prefixes), "library failed to load"

    return SuperEnv(
       config={
           "colors":
               5,
           "ranks":
               5,
           "players":
               num_players,
           "max_information_tokens":
               8,
           "max_life_tokens":
               3,
           "observation_type":
               pyhanabi.AgentObservationType.CARD_KNOWLEDGE.value
       })
