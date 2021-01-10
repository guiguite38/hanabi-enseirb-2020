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
        self.global_game_state = HanabiState(self.global_game)

        # !!! PUT IT EVERYWHERE IT IS NEEDED
        self.last_personal_move = None
        self.previous_observation = None
        self.local_player_id = None
        self.offset_real_local = None # The offset to go from the real player id to the local player id (because the starting player can't be chosen precisely, only 0 or random)
        #self.players_partial_belief = [ pb.PartialBelief(self.config["players"], 0, 0) for i in range(self.config["players"])] # The partial beliefs (only used to have
                                                                                                                               #easily the probabilities). Every player
                                                                                                                               #must use its local id (for coherence)
        

    @staticmethod
    def search_card_index(card, cards):
        """ Useful only when the cards aren't sorted, in the other case, use dichoSearchCard, which is faster. """
        for i in range(len(cards)):
            if card["color"] == cards[i]["color"] and card["rank"] == cards[i]["rank"]:
                return i
        return -1

    def prepare_global_game_state(self, observation): #TODO: rajouter à certains endroits l'offset du joueur actuel
        """La fonction est appelée à chaque fois qu'on appelle la fonction "act", même si ce n'est pas à nous de jouer.
        Cela permet de mettre à jour l'état du jeu action par action."""
                                                        
        
        if self.previous_observation is None: # C'est le premier tour, donc pas d'observation précédente
            # On doit ici mettre toutes les mains des différents joueurs (sauf nous-même) en accord avec la vraie partie
            self.offset_real_local = observation["current_player"]
            self.local_player_id = (self.config["players"] - observation["current_player_offset"]) % self.config["players"]
            for i in range(len(observation["observed_hands"]) - 1):
                self.global_game_state.set_hand( (self.local_player_id + i + 1) % self.config["players"], observation["observed_hands"][i + 1])
                
        else: # Ce n'est pas le premier tour, on peut comparer avec la précedente observation pour obtenir l'action qui a été jouée
            current_player_local = self.global_game_state.cur_player() # Le joueur de l'observation précédente, de qui on va chercher et appliquer le mouvement
            
            if self.previous_observation["current_player_offset"] == 0: # Si le dernier tour, c'était à moi de jouer (j'ai donc pu stocker mon move)
                self.global_game_state.apply_move(self.last_personal_move)
                self.global_game_state.deal_random_card() # On ne sait de toute manière pas quelle carte j'ai pioché, aucun intérêt d'en choisir une
                    
            """else: # Sinon, il faut chercher ce qui a changé sur le terrain
                if len(self.previous_observation["discard_pile"]) < len(observation["discard_pile"]): # Une carte y a donc été rajoutée (volontairement ou non)
                    index = ExtensiveAgent.search_card_index(observation["discard_pile"][-1],
                                              self.previous_observation["observed_hands"][self.previous_observation["current_player_offset"]])
                    if index == -1:
                        print("!!! index == -1 !!! It's not supposed to be possible ...")
                        return -1 # Problème de state
                    if self.previous_observation["life_tokens"] > observation["life_tokens"]: # Une erreur a été commise
                        
                        self.global_game_state.apply_move(HanabiMove.get_play_move(index))
                        self.global_game_state.deal_specific_card(current_player_local,
                                                                    observation["observed_hands"][self.previous_observation["current_player_offset"]][-1]["color"],
                                                                    observation["observed_hands"][self.previous_observation["current_player_offset"]][-1]["rank"],
                                                                    self.config["hand_size"] -1) # Je sais déjà par quoi a été remplacé la carte jouée
                    else: # La défausse était volontaire
                        self.global_game_state.apply_move(HanabiMove.get_discard_move(index))
                        self.global_game_state.deal_specific_card(current_player_local,
                                                                    observation["observed_hands"][self.previous_observation["current_player_offset"]][-1]["color"],
                                                                    observation["observed_hands"][self.previous_observation["current_player_offset"]][-1]["rank"],
                                                                    self.config["hand_size"] -1) # Je sais déjà par quoi a été remplacé la carte jouée
                    return 0 # OK
                
                for key in observation["fireworks"]:
                    if self.previous_observation["fireworks"][key] < observation["fireworks"][key]:
                        index = ExtensiveAgent.search_card_index({"color": key, "rank": observation["fireworks"][key]},
                                                  self.previous_observation["observed_hands"][self.previous_observation["current_player_offset"]])
                        self.global_game_state.apply_move(HanabiMove.get_play_move(index))
                        self.global_game_state.deal_specific_card(current_player_local,
                                                                    observation["observed_hands"][self.previous_observation["current_player_offset"]][-1]["color"],
                                                                    observation["observed_hands"][self.previous_observation["current_player_offset"]][-1]["rank"],
                                                                    self.config["hand_size"] -1) # Je sais déjà par quoi a été remplacé la carte jouée
                        return 0 # OK
                        
                for i in range(len(observation["card_knowledge"])): # On va maintenant chercher les potentiels indices délivrés, vu qu'aucune carte n'a été jouée
                                                                    #(même sur nous, ce qui est plus complexe)
                    for j in range(len(observation["card_knowledge"][i])):
                        if (self.previous_observation["card_knowledge"][i][j]["color"] != observation["card_knowledge"][i][j]["color"]
                            or self.previous_observation["card_knowledge"][i][j]["rank"] != observation["card_knowledge"][i][j]["rank"]): # Un indice a été donné, et on en
                                                                                                                                          #a la teneur sans avoir besoin de
                                                                                                                                          #regarder les autres cartes
                            if self.previous_observation["card_knowledge"][i][j]["color"] != observation["card_knowledge"][i][j]["color"]:
                                indice_type = "color"
                                indice_value = observation["card_knowledge"][i][j]["color"]
                            else:
                                indice_type = "rank"
                                indice_value = observation["card_knowledge"][i][j]["rank"]
                                
                            if i == 0: # Le cas spécial où on va devoir distribuer une main adaptée aux révélations (ni trop, ni trop peu), car on ne connait pas sa main
                                       #exacte, la main dans le self.global_game_state n'a donc aucune valeur
                                available_cards = ExtensiveAgent.unseen_cards(self.previous_observation)
                                hand_to_be_set = []
                                for k in range(len(observation["card_knowledge"][0])): # On pourrait très légèrement optimiser en traitant à part le cas k < j
                                                                                       #(on gagnerait une comparaison et quelques accès)
                                    if (indice_type == "color"):
                                        color_index = color_char_to_idx(indice_value)
                                        
                                        if (self.previous_observation["card_knowledge"][i][j]["color"] != observation["card_knowledge"][i][j]["color"]): # Carte respectant
                                                                                                                                                         #le critère fourni
                                                                                                                                                         #en indice
                                            for l in range(len(available_cards[color_index])):
                                                if available_cards[color_index][l] > 0:
                                                    available_cards[color_index][l] -= 1
                                                    hand_to_be_set.append({"color": indice_value, "rank": l + 1})
                                                    break
                                        else: # Choisir n'importe quelle carte qui ne répond pas au critère
                                            found = False
                                            for l in range(len(available_cards)):
                                                if l != color_index:
                                                    for m in range(len(available_cards[l])):
                                                        if available_cards[l][m] > 0:
                                                            available_cards[l][m] -= 1
                                                            hand_to_be_set.append({"color": l, "rank": m})
                                                            found = True
                                                            break
                                                    if found:
                                                        break
                                        
                                    elif (indice_type == "rank"):
                                        if (self.previous_observation["card_knowledge"][i][j]["rank"] != observation["card_knowledge"][i][j]["rank"]): # Carte respectant
                                                                                                                                                       #le critère fourni
                                                                                                                                                       #en indice
                                            for l in range(len(available_cards)):
                                                if available_cards[l][indice_value] > 0:
                                                    available_cards[l][indice_value] -= 1
                                                    hand_to_be_set.append({"color": color_idx_to_char(l), "rank": indice_value})
                                                    break
                                        else: # Choisir n'importe quelle carte qui ne répond pas au critère
                                            found = False
                                            for l in range(len(available_cards)):
                                                for m in range(len(available_cards[l])):
                                                    if indice_value != m:
                                                        if available_cards[l][m] > 0:
                                                            available_cards[l][m] -= 1
                                                            hand_to_be_set.append({"color": l, "rank": m})
                                                            found = True
                                                            break
                                                    if found:
                                                        break
                                self.global_game_state.set_hand(self.local_player_id, hand_to_be_set)#On va enfin set la main que l'on vient de créer pour répondre au
                                                                                                     #critère voulu
                                self.global_game_state.apply_move(HanabiMove.get_reveal_color_move((self.local_player_id - current_player_local) % self.config["players"], indice_value))
                                       

                            else: # Le cas le plus simple, la main est déjà censée être dans l'état qui convient
                                if indice_type == "color":
                                    self.global_game_state.apply_move(HanabiMove.get_reveal_color_move((self.local_player_id + i - current_player_local) % self.config["players"], indice_value))
                                else:
                                    self.global_game_state.apply_move(HanabiMove.get_reveal_rank_move((self.local_player_id + i - current_player_local) % self.config["players"], indice_value))
                                """
                            
                                
                            
                                   
        self.previous_observation = observation

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
        # !!! SE RAPPELER DE REMPLIR self.last_personal_move, SINON LA CONSTRUCTION DU STATE NE SERA PAS POSSIBLE !!!
        # The agent only plays on its turn

        print(observation)
        self.prepare_global_game_state(observation)
        if observation["current_player_offset"] != 0:
            return None
        self.last_personal_move = ExtensiveAgent.transform_dict_to_move(observation["legal_moves"][0])
        return observation["legal_moves"][0]
        expected_value = self.calculate_expected_value(observation, 0, self.global_game_state, observation.cur_player_offset())
        return observation.np.argmax(expected_value)


