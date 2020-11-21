"""Extensive search of all possible moves, """

from hanabi_learning_environment.rl_env import Agent


class ExtensiveAgent(Agent):
    """Agent that applies an exhaustive search to find the best possible move."""

    def __init__(self, config, *args, **kwargs):
        """Initialize the agent."""
        self.config = config
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
        raise NotImplementedError("Not implemented possible_cards function")

        # Take all possible cards
        # remove seen cards from observations
        # return remaining cards

    def get_card_probabilities(self, cards, observation):
        raise NotImplementedError("Not implemented get_card_probabilities function")

        # From number of remaining cards in deck, and hints, guess how likely it is to get each given card

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
            raise NotImplementedError("Not implemented for loop in act function")

            # To compute the score of a PLAY / DISCARD action, it tries playing it with all card it could possibly have in its hand
            unseen_cards = self.unseen_cards(observation)
            # Trick : this will basically be the same method for all 5 cards in hand, with modifications on probability but not on score.
            # scores and simulations for each card should therefore only be computed once, and probabilities each time
            scores = self.get_card_scores(unseen_cards)
            # ponderate scores by probability of occurence (if a red hint has been given, occurence of red card is more likely) to get expected value (fr:Esperance)
            # Chose move with highest expected value
            hints = observation[
                ""
            ]  # !! Add where those hints are in the observation space, I don't recall
            likeliness = self.get_card_probabilities(unseen_cards, observation)

        # !! Example action formats, kept as an example, to be removed once done
        # return {'action_type': 'PLAY', 'card_index': card_index}
        # return {
        #     'action_type': 'REVEAL_COLOR',
        #     'color': card['color'],
        #     'target_offset': player_offset
        # }
        # return {'action_type': 'DISCARD', 'card_index': 0}
        # return {'action_type': 'PLAY', 'card_index': 0}

