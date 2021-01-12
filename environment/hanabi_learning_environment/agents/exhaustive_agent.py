"""Extensive search of all possible moves, """
# On peut, outre le score de partie basique, donner plus de poids à d'autres choses (donc créer une heuristique) -> indices valent des points par exemple,
#ou score non-linéaire (poser le 5 vaut plus que poser le 2)
""" memo:                       {'current_player': 0,
                                  'current_player_offset': 0,
                                  'deck_size': 40,
                                  'discard_pile': [],
                                  'fireworks': {'B': 0,
                                                'G': 0,
                                                'R': 0,
                                                'W': 0,
                                                'Y': 0},
                                  'information_tokens': 8,
                                  'legal_moves': [{'action_type': 'PLAY',
                                                   'card_index': 0},
                                                  {'action_type': 'PLAY',
                                                   'card_index': 1},
                                                  {'action_type': 'PLAY',
                                                   'card_index': 2},
                                                  {'action_type': 'PLAY',
                                                   'card_index': 3},
                                                  {'action_type': 'PLAY',
                                                   'card_index': 4},
                                                  {'action_type':
                                                  'REVEAL_COLOR',
                                                   'color': 'R',
                                                   'target_offset': 1},
                                                  {'action_type':
                                                  'REVEAL_COLOR',
                                                   'color': 'G',
                                                   'target_offset': 1},
                                                  {'action_type':
                                                  'REVEAL_COLOR',
                                                   'color': 'B',
                                                   'target_offset': 1},
                                                  {'action_type': 'REVEAL_RANK',
                                                   'rank': 0,
                                                   'target_offset': 1},
                                                  {'action_type': 'REVEAL_RANK',
                                                   'rank': 1,
                                                   'target_offset': 1},
                                                  {'action_type': 'REVEAL_RANK',
                                                   'rank': 2,
                                                   'target_offset': 1}],
                                  'life_tokens': 3,
                                  'observed_hands': [[{'color': None, 'rank':
                                  -1},
                                                      {'color': None, 'rank':
                                                      -1},
                                                      {'color': None, 'rank':
                                                      -1},
                                                      {'color': None, 'rank':
                                                      -1},
                                                      {'color': None, 'rank':
                                                      -1}],
                                                     [{'color': 'G', 'rank': 2},
                                                      {'color': 'R', 'rank': 0},
                                                      {'color': 'R', 'rank': 1},
                                                      {'color': 'B', 'rank': 0},
                                                      {'color': 'R', 'rank':
                                                      1}]],
                                  'num_players': 2,
                                  'vectorized': [ 0, 0, 1, ... ]}
"""
#bool GetDealSpecificMove(int card_index, int player, int color, int rank, pyhanabi_move_t* move)

# TODO: use " HanabiState : get_deal_specific_move(card_index, player, color, rank) " in conjunction with " HanabiState : apply_move(self, move)" from " pyhanabi.py "
# and maybe " HanabiState : copy(self) " (to save before a move, but it's pretty heavy to do it before every move)

import sys
import os
import numpy as np
import math  
#DOSSIER_COURANT = os.path.dirname(os.path.abspath(__file__))
#DOSSIER_PARENT = os.path.dirname(DOSSIER_COURANT)
#print(DOSSIER_PARENT)
#sys.path.append(DOSSIER_PARENT)


#from hanabi_learning_environment import pyhanabi
from hanabi_learning_environment.rl_env import Agent
from hanabi_learning_environment.pyhanabi import color_char_to_idx, color_idx_to_char
from hanabi_learning_environment.pyhanabi import HanabiGame, HanabiState, HanabiMoveType, HanabiMove
from hanabi_learning_environment.pyhanabi import CHANCE_PLAYER_ID
import hanabi_learning_environment.partial_belief as pb # Remember to use update function
#from hanabi_learning_environment


def recDichoSearchCard(card, cards, i, j):
    """ The cards must be in order, first colors, then ranks (like : (0 (c),0 (r)) (0,1) (1,1)  (1,2) ...) """
    if (j - i) > 0:
        center = (i + j) // 2
        if (
            cards[center]["color"] == card["color"]
            and cards[center]["rank"] == card["rank"]
        ):
            return center
        elif cards[center]["color"] == card["color"]:
            if cards[center]["rank"] > card["rank"]:
                return recDichoSearchCard(card, cards, i, center - 1)
            else:
                return recDichoSearchCard(card, cards, center + 1, j)
        else:
            if cards[center]["color"] > card["color"]:
                return recDichoSearchCard(card, cards, i, center - 1)
            else:
                return recDichoSearchCard(card, cards, center + 1, j)
    elif (
        j - i
    ) == 0:  # Useless, when (j - i) == -1, it will return -1 (all elements already checked)
        if cards[i]["color"] == card["color"] and cards[i]["rank"] == card["rank"]:
            return i
    return -1


def dichoSearchCard(card, cards):
    if card["color"] != None and card["rank"] != -1 and cards != []:
        card["color"] = color_char_to_idx(str(card["color"]))
        i = 0
        j = len(cards) - 1
        return recDichoSearchCard(card, cards, i, j)
    return -1


class ExtensiveAgent(Agent):
    """Agent that applies an exhaustive search to find the best possible move."""

    def __init__(self, config, *args, **kwargs):
        """Initialize the agent."""
        """ Args:
      config: dict, With parameters for the game. Config takes the following
        keys and values.
          - colors: int, Number of colors \in [2,5].
          - ranks: int, Number of ranks \in [2,5].
          - players: int, Number of players \in [2,5].
          - hand_size: int, Hand size \in [4,5].
          - max_information_tokens: int, Number of information tokens (>=0)
          - max_life_tokens: int, Number of life tokens (>=0)
          - seed: int, Random seed.
          - random_start_player: bool, Random start player.
          - max_iteration: int, Maximum Depth of the search. """

        self.config = config
        print(config)
        # Extract max info tokens or set default to 8.
        self.max_information_tokens = config.get("information_tokens", 8)
        self.max_iteration = config.get("max_iteration", 1)
        config["random_start_player"] = False # To start at 0
        self.global_game = HanabiGame(config)
        self.global_game_state = self.global_game.new_initial_state()

        # !!! PUT IT EVERYWHERE IT IS NEEDED
        self.previous_observation = None
        self.local_player_id = None
        self.offset_real_local = None # The offset to go from the real player id to the local player id (because the starting player can't be chosen precisely, only 0 or random)
        #self.players_partial_belief = [ pb.PartialBelief(self.config["players"], 0, 0) for i in range(self.config["players"])] # The partial beliefs (only used to have
                                                                                                                               #easily the probabilities). Every player                                                                                                      #must use its local id (for coherence)
    @staticmethod
    def count_real_moves(l):

        count = 0
        
        for current_history_item in l:
            current_move = current_history_item.move()
            current_move_type = current_move.type()
            
            if (current_move_type == HanabiMoveType.PLAY
                or current_move_type == HanabiMoveType.DISCARD
                or current_move_type == HanabiMoveType.REVEAL_RANK
                or current_move_type == HanabiMoveType.REVEAL_COLOR
                or current_move_type == HanabiMoveType.INVALID):
                count += 1
                
        return count
    
    def select_card(self, condition, respect_condition, type_of_condition, l): # TODO: Check the current hand 
        """ If respect_condition == True, give a good card, if not, give a bad card (relative to the condition). """

        for i in range(len(l)):
            # The first-level condition is used to filter the color-typed condition
            if ((type_of_condition == "color" and respect_condition == (condition == i))
                or (type_of_condition == "rank")):
                for j in range(len(l[i])):
                    if ((type_of_condition == "rank" and respect_condition == (condition == i))
                        or (type_of_condition == "color")):
                        if l[i][j] > 0:
                            l[i][j] -= 1
                            return {"color": color_idx_to_char(i), "rank": j}
        return None
                

    
    def draw_good_card(self, local_id, observation):
        print("Starting draw_good_card:", local_id)
        self.global_game_state.deal_random_card()

        relative = self.relative_id(local_id)

        # If not, it is the agent, and we don't know which card to give 
        if relative != 0:
            # The same bug explanation for this loop
            for i in range(self.config["players"]):
                self.global_game_state.set_individual_card(local_id, self.config["hand_size"] - 1, observation["observed_hands"][relative][-1])

        return 0

    def prepare_hand(self, local_id, observation, type_of_reveal, value):
        print("Starting prepare_hand:", local_id)
        relative = self.relative_id(local_id)

        # If not, it is theorically useless to change the hand (unless somme mistakes have been made previously, creating a wrong state)
        if relative == 0:
            hand_to_be_set = []
            possible_cards = ExtensiveAgent.unseen_cards(observation)
            player_knowledge = observation["pyhanabi"].card_knowledge()[relative]
            for card_knowledge in player_knowledge:
                card = None
                if type_of_reveal == "color":
                    if card_knowledge.color() == value:
                        card = self.select_card(value, True, type_of_reveal, possible_cards)
                    else:
                        card = self.select_card(value, False, type_of_reveal, possible_cards)
                elif type_of_reveal == "rank":
                    if card_knowledge.rank() == value:
                        card = self.select_card(value, True, type_of_reveal, possible_cards)
                    else:
                        card = self.select_card(value, False, type_of_reveal, possible_cards)
                else:
                    print("Bad reveal_type in prepare_hand:", local_id, observation, type_of_reveal, value, flush=True)
                    return -1
                
                if card is not None:
                    hand_to_be_set.append(card)
                else:
                    print("Pas assez de carte pour compléter la main correctement:", hand_to_be_set, possible_cards, flush=True)
                    return -1

            for i in range(self.config["players"]):
                self.global_game_state.set_hand(local_id, hand_to_be_set)

        
        return 0
                

    def print_state_info(self):
        print("Nothing yet to print", flush = True)
        pass

    def local_id(self, i):
        """ The local id of the player in global_game_state. """
        return (i + self.offset_real_local) % self.config["players"]

    def relative_id(self, i):
        """ The relative place of the player (his offset compared to the agent) """
        return (i - self.offset_real_local) % self.config["players"]
                
    def prepare_global_game_state(self, observation):
        """ This function build the state thanks to the informations given in the observation. """
        obs_pyhanabi = observation["pyhanabi"]
        
        ################## FIRST TURN PART ###################
        if self.offset_real_local is None:
            while self.global_game_state.cur_player() == CHANCE_PLAYER_ID:
              self.global_game_state.deal_random_card()

            # The number of players who have played before the first call
            self.offset_real_local = ExtensiveAgent.count_real_moves(obs_pyhanabi.last_moves()) # Mayb another way ?

            current_player = self.global_game_state.cur_player()

            print("Avant l'initialisation", flush = True)
            print(self.global_game_state.player_hands())
            
            # We have to setup the hands in a correct way
            for i, hand in enumerate(observation["observed_hands"]):

                # Our hand is invalid, so we start from the 2nd player 
                if i != 0:
                    self.global_game_state.set_hand(self.local_id(i), hand)

            #There's a bug in set_hand, so we have to correct the current_player by setting another time the last hand
            if self.global_game_state.cur_player() != current_player:
                self.global_game_state.set_hand(self.local_id(self.config["players"] - 1), observation["observed_hands"][self.config["players"] - 1])

            print("Après l'initialisation", flush = True)
            print(self.global_game_state.player_hands())

                
        print("Starting the generalpart of building", flush = True)
        ################## GENERAL PART ###################
        for current_history_item in obs_pyhanabi.last_moves():
            print("Current history item:", current_history_item, flush = True)

            current_move = current_history_item.move()
            current_move_type = current_move.type()
            
            # The ExtensiveAgent is always player 0 in moves, so we have to correct this by adding the local index of the agent.
            #
            # EXAMPLE: the player given is 1, and we play at the second place in a game with 2 players.
            # The current_local_player is (1 + 1) % nb_player = 0, so it's the first player (which is true).
            current_local_player = self.local_id(current_history_item.player()) 
            

            if current_move_type == HanabiMoveType.PLAY: # TODO: put the good cards in hand
                print("Entering the PLAY condition", flush = True)
                if self.global_game_state.move_is_legal(current_move):
                    print("PLAY condition verified", flush = True)
                    self.global_game_state.apply_move(current_move)
                    print("PLAY move applied successfully", flush = True)
                    self.draw_good_card(current_local_player, observation)
                    
                else:
                    print("The PLAY move given isn't legal:", current_move, flush = True)
                    return -1

            elif current_move_type == HanabiMoveType.DISCARD: # TODO: put the good cards in hand
                if self.global_game_state.move_is_legal(current_move):
                    self.global_game_state.apply_move(current_move)
                    self.draw_good_card(current_local_player, observation)
                    
                else:
                    print("The DISCARD move given isn't legal:", current_move, flush = True)
                    return -1

            elif current_move_type == HanabiMoveType.REVEAL_RANK:
                if self.global_game_state.move_is_legal(current_move):
                    self.prepare_hand(current_local_player + current_move.target_offset(), observation, "rank", current_move.rank())
                    self.global_game_state.apply_move(current_move)
                    
                else:
                    print("The REVEAL_RANK move given isn't legal:", current_move, flush = True)
                    return -1

            elif current_move_type == HanabiMoveType.REVEAL_COLOR:
                if self.global_game_state.move_is_legal(current_move):
                    self.prepare_hand(current_local_player + current_move.target_offset(), observation, "color", current_move.color())
                    self.global_game_state.apply_move(current_move)
                    
                else:
                    print("The REVEAL_COLOR move given isn't legal:", current_move, flush = True)
                    return -1

            elif current_move_type == HanabiMoveType.DEAL:
                pass

        self.print_state_info()
        return 0
        

    @staticmethod
    def transform_dict_to_move(dic):
        """ Transform a dict-shaped move into an HanabiMove """
        if dic["action_type"] == "PLAY":
            return HanabiMove.get_play_move(dic["card_index"])
        elif dic["action_type"] == "DISCARD":
            return HanabiMove.get_discard_move(dic["card_index"])
        elif dic["action_type"] == "REVEAL_COLOR":
            return HanabiMove.get_reveal_color_move(dic["target_offset"], dic["color"])
        elif dic["action_type"] == "REVEAL_RANK":
            return HanabiMove.get_reveal_color_move(dic["target_offset"], dic["rank"])
        else:
            print("Ce move n'existe pas, il n'est donc pas possible de le transformer en HanabiMove.")
            return None


    @staticmethod
    def unseen_cards(observation):
        currently_unseen_cards = np.array([
            [3, 2, 2, 2, 1],
            [3, 2, 2, 2, 1],
            [3, 2, 2, 2, 1],
            [3, 2, 2, 2, 1],
            [3, 2, 2, 2, 1]
        ], dtype=np.int) # Thx Theo for the numbers <3
        
        for card in ovservation["discard_pile"]:
            currently_unseen_cards[color_char_to_idx(card["color"])][card["rank"]] -= 1
        for i in range(len(observation["observed_hands"]) - 1):
            for card in observation["observed_hands"][i + 1]:
                currently_unseen_cards[color_char_to_idx(card["color"])][card["rank"]] -= 1
        return currently_unseen_cards
                
    @staticmethod
    def score_game(fireworks):
        """returns the game score displayed by fireworks played up to now in the game.
         for now no heuristic is used to determine which hand is the most promising for a given score"""
        score = 0
        for coloured_firework in fireworks:
            score += fireworks[coloured_firework]
        return score

    def enumerate_hands(self, current_indices_hand, possible_cards_in_each_position): #, current_tested_hand
        """ Use the current tested hand indices, the observation and the possible cards in each hand position to return the next hand to be tested

        current_indices_hand is used to remember indices of the possible_cards_in_each_position 2D list, so we don't have to loose time searching
        for current tested cards at each call."""

        current_position = 0
        while 1:
            current_indices_hand[current_position] += 1
            if len(possible_cards_in_each_position[current_position]) <= current_indices_hand[current_position]: # all the possible cards in this position has been tried
                current_indices_hand[current_position] = 0
                current_position += 1
                if current_position >= self.config["hand_size"]:
                    return None # All the hands have been tested
            else:
                return current_indices_hand, [possible_cards_in_each_position[i][current_indices_hand[i]] for i in range(self.config["hand_size"])]
                

    def _extract_dict_from_backend(self, player_id, observation): # Copied from rl_env !
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
        

    def hand_probability(self, current_indices_hand, possible_cards_in_each_position):
        pass

    def calculate_expected_value(self, observation, iteration_level, state, local_player_offset):

        if iteration_level >= self.max_iteration:
            return state.score() # We will do the wheighted mean score of all children, so we have to return it
        if iteration_level != 0:
            total = 0
            # For each possible hand: # More optimized to iterate on hands, because it's harder to compute
            for action in observation.legal_moves():
                tempo_state = state.copy() # Could be optimized a lot if a function "pop" was created (instead of cloning the entire state each time !)
                tempo_state.apply_move(action)
                if tempo_state.is_terminal() == True:
                    total += tempo_state.score()
                    continue
                
                total += self.calculate_expected_value(tempo_state.observation((self.local_player_id + local_player_offset + 1) % self.config["players"]),
                                                       iteration_level + 1, tempo_state, local_player_offset + 1) * self.hand_probability()
                
            nb_moves = len(observation.legal_moves())
            if nb_moves == 0: # Is it really possible ?
                total = state.score()
                
            return total
        else:
            print("Calculate_expected_value return not yet implemented")
            list_scores = []
            
            
    
    def act(self, observation):
        """Act based on an observation."""
        # The agent only plays on its turn

        print(observation)
        if observation["current_player_offset"] != 0:
            return None
        self.prepare_global_game_state(observation)
        return observation["legal_moves"][0]
        expected_value = self.calculate_expected_value(observation, 0, self.global_game_state, observation.cur_player_offset())
        return observation.np.argmax(expected_value)


