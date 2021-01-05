"""Extensive search of all possible moves, """
#for move_tuple in observation.last_moves(): => donc on a accès aux derniers moves, utile pour construire un state
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

#TODO: Court-circuiter l'appel à new_initial_state, ou modifier ce dernier pour qu'il prenne également une observation et construise le bon state.
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
 
import partial_belief as pb # Remember to use update function


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
        self.max_iteration = config.get("max_iteration", 3)
        self.global_game = HanabiGame(config)
        self.global_game_state = HanabiState(self.global_game)
        self.current_action_registered = 0 # Don't restart from scratch each time

    def prepare_global_game_state(self, observation): # TODO: a changer pour d'abord construire la discard pile et le board state (en utilisant celui deja present)
        #puis en settant les bonnes main une fois que tout est fait
        """ Switch the hands of the other players in the global_game_state variable according to the observations,
        and change the board state and the discard pile. """

        # QUESTION: Is it really useful to gather all the cards needed instead of just replacing hand for each card ? Only nb_players moves 
        """move_length = len(observation.last_moves()[self.current_action_registered:])
        wanted_hands = np.full((self.config["players"], math.ceil(move_length / float(self.config["players"]))), {})
        for i, move_tuple in enumerate(observation.last_moves()[self.current_action_registered:]): # All the cards that will be needed by each player
            current_move = move_tuple.move() 
            
            if current_move.type() == HanabiMoveType.PLAY:
                wanted_hands[ i % self.config["players"]][math.floor(i / self.config["players"])]


        current_pointer = [0] * self.config["players"] # The current card in use, when it's above self.config["hand_size"] * count, set the new hand
        current_count = [1] * self.config["players"]
        for i, move_tuple in enumerate(observation.last_moves()[self.current_action_registered:]):"""

        available_cards = self.unseen_cards(observation) # To use with dichoSearchCard(card, res)
        
        playing_player_id = observation["current_player"] #TODO: IT'S WRONG AT THE FIRST TURN !!! PUT HERE A FORMULA TO TAKE INTO ACCOUNT IT
        for i, move_tuple in enumerate(observation.last_moves()[self.current_action_registered:]):
            current_move = move_tuple.move()

            #self, player_id, card_index, card
            if current_move.type() == HanabiMoveType.PLAY: # TODO: Verify if the card is already here, if yes, useless to change it (be careful with the current_player case)
                self.global_game_state.set_individual_card(playing_player_id , current_move.card_index(), {"color" : color_idx_to_char(move_tuple.color()), "rank" :  move_tuple.rank()})
                apply_move(HanabiMove.get_play_move(current_move.card_index()))
                playing_player_id = ( playing_player_id + 1 ) %  self.config["players"]
                
            elif current_move.type() == HanabiMoveType.DISCARD:
                self.global_game_state.set_individual_card(playing_player_id , current_move.card_index(), {"color" : color_idx_to_char(move_tuple.color()), "rank" :  move_tuple.rank()})
                apply_move(HanabiMove.get_discard_move(current_move.card_index()))
                playing_player_id = ( playing_player_id + 1 ) %  self.config["players"]
                
            elif current_move.type() == HanabiMoveType.DEAL: 
                deal_specific_card(move_tuple.deal_to_player(), current_move.color(), current_move.rank(), card_index)

            elif current_move.type() == HanabiMoveType.REVEAL_COLOR: 
                pass

            elif current_move.type() == HanabiMoveType.REVEAL_RANK:
                pass
        
        
        for i in range(self.config["players"] - 1): # All the players, except the current one
            tempo_id = (observation["current_player_offset"] + i + 1 ) % self.config["players"] # The place of the ith player's hand
            self.global_game_state.set_hand(observation["current_player"] + i + 1, observation["observed_hands"][tempo_id])

        return

    @staticmethod
    def playable_card(card, fireworks):
        """A card is playable if it can be placed on the fireworks pile."""
        #TODO: Replace by state.card_playable_on_fireworks(self, color, rank)
        return card["rank"] == fireworks[card["color"]]

    @staticmethod
    def score_game(fireworks):
        """returns the game score displayed by fireworks played up to now in the game.
         for now no heuristic is used to determine which hand is the most promising for a given score"""
        score = 0
        for coloured_firework in fireworks:
            score += fireworks[coloured_firework]
        return score

    def unseen_cards(self, observation):
        """
    Returns a list of all unseen cards (deck + hidden player hand)
    """
        res = [
            {"color": x, "rank": y} #TODO: put number of cards here
            for x in range(self.config["colors"])
            for y in range(self.config["ranks"])
        ]

        for card in observation["discard_pile"]:
            pos = dichoSearchCard(card, res)
            if pos != -1:
                del res[pos]
        for i in range(self.config["players"]):
            for card in observation["observed_hands"][
                i
            ]:  # It must work even for invalid cards !
                # print(card, "\n")
                pos = dichoSearchCard(card, res)
                if pos != -1:
                    del res[pos]
        return res

        # Take all possible cards
        # remove seen cards from observations (and discard pile !!!)
        # return remaining cards

    def get_card_probabilities(
        self, cards, observation
    ):  # ptet sommer les probas d'une meme carte dans la main, donc differencier uniquement "dans le tas" et "dans la main"
        # We will do approched probabilities for a first approach (full probabilities might be a little costly)
        # color_plausible(c) and rank_plausible(r)
        knowledge = observation["pyhanabi"].card_knowledge()[
            0
        ]  # According to the definition, relative to the player
        nb_hand = self.config["hand_size"]
        res = (
            []
        )  # The probability that an unseen card is at a particular place in the hand
        possible_cards = []  # The possible cards for each card in hand
        having_card = [[]] * len(cards)  # For each unseen card, the possible locations
        dynamic_sum = [-1] * len(cards)
        for i in range(
            nb_hand
        ):  # Harvest all the possible cards for each position in hand
            tempo_possible = []
            for j in range(len(cards)):
                if knowledge[i].color_plausible(cards[j]["color"]) and knowledge[
                    i
                ].rank_plausible(cards[j]["rank"]):
                    tempo_possible.append(j)
                    having_card[j].append(i)
            possible_cards.append(tempo_possible)

        for i in range(nb_hand):  # Current position
            individual_answer = (
                []
            )  # The probabilities for each possible card in that place
            print(possible_cards[i])

            for card in possible_cards[i]:  #  Index of current card in cards

                if dynamic_sum[card] == -1:
                    dynamic_sum[card] = 0
                    for k in having_card[
                        card
                    ]:  # All the possible locations (in hand) of current card
                        tempo_sum = 1.0 / len(possible_cards[k])
                        for l in having_card[card]:
                            if l != k:
                                tempo_sum *= (len(possible_cards[l]) - 1.0) / float(
                                    len(possible_cards[l])
                                )
                        dynamic_sum[
                            card
                        ] += tempo_sum  # To normalize on the entire hand

                tempo_sum = 1.0 / float(len(possible_cards[i]))
                for k in having_card[card]:
                    tempo_sum *= (len(possible_cards[l]) - 1.0) / float(
                        len(possible_cards[l])
                    )
                individual_answer.append([card, tempo_sum / dynamic_sum[card]])
            res += [
                [[i[0], i[1]] for i in individual_answer]
            ]  # Normalize on the position [ [[]]      ]
            for i in range(len(res)):
                tempo_sum = 0
                for j in range(len(res[0])):
                    # print("\n s ",res[i],"\n s ",res[i][j],"\n s ",res[i][j][1]," e\n")
                    tempo_sum += res[i][j][1]
                for j in range(len(res[0])):
                    res[i][j][1] = res[i][j][1] / float(tempo_sum)
        return res

        # From number of remaining cards in deck, and hints, guess how likely it is to get each given card
        # (((There's a class named "HanabiCardKnowledge" and a function that create those, named "card_knowledge" in HanabiObservation class)))
        # We need

    def get_card_scores(self, cards, observation, iteration = 0):
        """
        returns a card list, and the score each would get if it was played
        """
        if iteration >= self.max_iteration:
            # Question: Juste pour le premier ? Vraiment ? Ou appliquer ca pour tous ?
            # TODO: créer tous les states possibles liés à notre observation du terrain (donc toutes nos mains possibles)
            #for possible_move in observation["



            #######################################################################################################
            # First version, for testing purposes : the state is not modified, score is calculated by hand solely
            scores = []
            num_to_letters = ["B", "G", "R", "W", "Y"]

            for card in cards:
                tempo_card = {"color": num_to_letters[card["color"]], "rank": card["rank"]}
                if self.playable_card(tempo_card, observation["fireworks"]):
                    scores.append(
                        self.score_game(observation["fireworks"]) + 1
                    )  # adding a card adds a single point to score wherever it is
                else:
                    scores.append(
                        -1
                    )  # playing the wrong card causes the use of a red marker and is therefore punished
            return scores

    def chose_action(self, expected_value):
        playable_threshold = 0.75 # A small margin
        discardable_threshold = 0 # Must be sure at 100% to discard it
        best_card = expected_value.index(max(expected_value))
        worst_card = expected_value.index(min(expected_value))
        # we only play if this will give us points with a high probability
        if(expected_value[best_card] >= playable_threshold):
            return {'action_type': 'PLAY', 'card_index': best_card}
        # otherwise we can discard a card if any card is likely to bring a bad score
        # !! this will need to be the last option once we know how to deal with hints
        elif(worst_card < discardable_threshold) :
            return {'action_type': 'DISCARD', 'card_index': worst_card}
        # otherwise we can give a hint
        else :
            # !! The scoring system does not yet give scores for hints. Therefore we chose a colour randomly
            return {
                'action_type': 'REVEAL_COLOR',
                'color': random.choice(["B", "G", "R", "W", "Y"]),
                'target_offset': 1 # !! in a two player game, the offset can only be one (this will have to be changed with more players)
            }


    def act(self, observation):
        """Act based on an observation."""
        # The agent only plays on its turn
        if observation["current_player_offset"] != 0:
            return None

        # It computes the score of all possible moves

        # To compute the score of a PLAY / DISCARD action, it tries playing it with all card it could possibly have in its hand
        unseen_cards = self.unseen_cards(observation)
        print(unseen_cards)

        likeliness = self.get_card_probabilities(unseen_cards, observation)
        print(likeliness)
        scores = self.get_card_scores(unseen_cards, observation)
        print(scores)
        # Chose move with highest expected value (probability of occurence * score)
        expected_value = []
        for likely_card in likeliness:
            expected_value.append(
                sum([likely_card[i][1] * scores[i] for i in range(len(likely_card))])
            )  # !! should we numpy array all this ?

        return chose_action(expected_value)


