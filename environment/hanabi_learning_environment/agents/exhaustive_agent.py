"""Extensive search of all possible moves, """

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

from hanabi_learning_environment.rl_env import Agent
from hanabi_learning_environment.pyhanabi import color_char_to_idx
from hanabi_learning_environment.pyhanabi import color_idx_to_char
from hanabi_learning_environment import pyhanabi


def recDichoSearchCard(card, cards, i, j):
    """ The cards must be in order, first colors, then ranks (like : (0 (c),0 (r)) (0,1) (1,1)  (1,2) ...) """
    if (j - i) > 0:
        center = (i+j)//2
        if (cards[center]["color"] == card["color"] and cards[center]["rank"] == card["rank"]):
            return center
        elif (cards[center]["color"] == card["color"]):
            if (cards[center]["rank"] > card["rank"]):
                return recDichoSearchCard(card, cards, i, center - 1)
            else:
                return recDichoSearchCard(card, cards, center + 1, j)
        else:
            if cards[center]["color"] > card["color"]:
                return recDichoSearchCard(card, cards, i, center - 1)
            else:
                return recDichoSearchCard(card, cards, center + 1, j)
    elif (j - i) == 0: # Useless, when (j - i) == -1, it will return -1 (all elements already checked)
        if cards[i]["color"] == card["color"] and cards[i]["rank"] == card["rank"]:
            return i
    return -1

def dichoSearchCard(card, cards):
    if  card["color"] != None and card["rank"] != -1 and cards != []:
        card["color"] = color_char_to_idx(str(card["color"]))
        i = 0
        j = len(cards)-1
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
          - random_start_player: bool, Random start player."""
        self.config = config
        print(config)
        # Extract max info tokens or set default to 8.
        self.max_information_tokens = config.get("information_tokens", 8)

    @staticmethod
    def playable_card(card, fireworks):
        """A card is playable if it can be placed on the fireworks pile."""
        return card["rank"] == fireworks[card["color"]]

    @staticmethod
    def score_game(fireworks):
        """returns the game score displayed by fireworks played up to now in the game.
         for now no heuristic is used to determine which hand is the most promising for a given score"""
        score = 0
        for coloured_firework in fireworks:
            score += coloured_firework
        return score

    def unseen_cards(self, observation):
        """
    Returns a list of all unseen cards (deck + hidden player hand)
    """
        res = [{"color":x,"rank":y} for x in range(self.config["colors"]) for y in range(self.config["ranks"])]
        
        for card in observation["discard_pile"]:
            pos = dichoSearchCard(card, res)
            if pos != -1:
                del res[pos]
        for i in range(self.config["players"]):
            for card in observation["observed_hands"][i]: # It must work even for invalid cards !
                #print(card, "\n")
                pos = dichoSearchCard(card, res)
                if pos != -1:
                    del res[pos]
        return res

        # Take all possible cards
        # remove seen cards from observations (and discard pile !!!)
        # return remaining cards

    def get_card_probabilities(self, cards, observation): #ptet sommer les probas d'une meme carte dans la main, donc differencier uniquement "dans le tas" et "dans la main"
        # We will do approched probabilities for a first approach (full probabilities might be a little costly)
        #color_plausible(c) and rank_plausible(r)
        knowledge = observation["pyhanabi"].card_knowledge()[0] # According to the definition, relative to the player
        nb_hand = self.config["hand_size"]
        res = [] # The probability that an unseen card is at a particular place in the hand
        possible_cards = [] # The possible cards for each card 
        having_card = [[]] * len(cards) # For each unseen card, the possible locations
        dynamic_sum = [-1] * len(cards)
        for i in range(nb_hand): # Harvest all the possible cards for each position in hand
            tempo_possible = []
            for j in range(len(cards)):
                if knowledge[i].color_plausible(cards[j]["color"]) and knowledge[i].rank_plausible(cards[j]["rank"]):
                    tempo_possible.append(j)
                    having_card[j].append(i)
            possible_cards.append(tempo_possible)

        for i in range(nb_hand): # Current position
            individual_answer = [] # The probabilities for each possible card in that place
            print(possible_cards[i])

            for card in possible_cards[i]: #  Index of current card in cards

                if dynamic_sum[card] == -1:
                    dynamic_sum[card] = 0
                    for k in having_card[card]: # All the possible locations (in hand) of current card
                        tempo_sum = 1. /  len(possible_cards[k])
                        for l in having_card[card]:
                            if l != k:
                                tempo_sum *= (len(possible_cards[l]) - 1.) / float(len(possible_cards[l]))
                        dynamic_sum[card] += tempo_sum  # To normalize on the entire hand

                tempo_sum = 1. / float(len(possible_cards[i]))
                for k in having_card[card]:
                    tempo_sum *=  (len(possible_cards[l]) - 1.) / float(len(possible_cards[l]))
                individual_answer.append([card, tempo_sum / dynamic_sum[card]])
            res += [[[i[0], i[1]]  for i in individual_answer]] # Normalize on the position [ [[]]      ]
            for i in range(len(res)):
                tempo_sum = 0
                for j in range(len(res[0])):
                    #print("\n s ",res[i],"\n s ",res[i][j],"\n s ",res[i][j][1]," e\n")
                    tempo_sum += res[i][j][1]
                for j in range(len(res[0])):
                    res[i][j][1] = res[i][j][1] / float(tempo_sum)
        return res 
                

        # From number of remaining cards in deck, and hints, guess how likely it is to get each given card
        # (((There's a class named "HanabiCardKnowledge" and a function that create those, named "card_knowledge" in HanabiObservation class)))
        # We need 

    def get_card_scores(self, cards):
        """
    returns a card list, and the score each would get if it was played
    """
        raise NotImplementedError("Not implemented get_card_scores function")
        for card in cards:
            # simulate playing card
            raise NotImplementedError("Not implemented for loop in get_card_scores")
        # from that state we either :
        # reach the iteration limit, compute a score (heuristically or end-game score)
        # haven't reached the iteration limit, simulate the other agent's move (begin by using a random_agent,  then use this agent as a model of the oponent, then idk yet...)

    def act(self, observation):
        """Act based on an observation."""
        # The agent only plays on its turn
        if observation["current_player_offset"] != 0:
            return None

        # It computes the score of all possible moves
        for legal_move in observation["legal_moves"]:
            

            # To compute the score of a PLAY / DISCARD action, it tries playing it with all card it could possibly have in its hand
            unseen_cards = self.unseen_cards(observation)
            print(unseen_cards)
            # Trick : this will basically be the same method for all 5 cards in hand, with modifications on probability but not on score.
            # scores and simulations for each card should therefore only be computed once, and probabilities each time
            
            # ponderate scores by probability of occurence (if a red hint has been given, occurence of red card is more likely) to get expected value (fr:Esperance)
            # Chose move with highest expected value
            
             # !! Add where those hints are in the observation space, I don't recall
            likeliness = self.get_card_probabilities(unseen_cards, observation)
            print(likeliness)
            raise NotImplementedError("Not implemented for loop in act function")
            scores = self.get_card_scores(unseen_cards)

        # !! Example action formats, kept as an example, to be removed once done
        # return {'action_type': 'PLAY', 'card_index': card_index}
        # return {
        #     'action_type': 'REVEAL_COLOR',
        #     'color': card['color'],
        #     'target_offset': player_offset
        # }
        # return {'action_type': 'DISCARD', 'card_index': 0}
        # return {'action_type': 'PLAY', 'card_index': 0}

