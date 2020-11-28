from typing import Dict

import numpy as np


_FULL_DECK_ = np.array([
    [3, 2, 2, 2, 1],
    [3, 2, 2, 2, 1],
    [3, 2, 2, 2, 1],
    [3, 2, 2, 2, 1],
    [3, 2, 2, 2, 1]
], dtype=np.int)

_COLOR_TO_INDEX = {
    "W": 0,
    "R": 1,
    "Y": 2,
    "G": 3,
    "B": 4,
    None: None
}


class PartialBelief:
    """
    Computes partial belief from the perspective of player ```perspective_of```.
    assuming the observations it receive are from the perspective of ```observations_from```.

    Parameters
    -----------
    - **players**: the number of players
    - **perspective_of**: the player offset whose perspective the beliefs will be build upon
    - **observation_from**: the player offset whose observations will be used to compute the beliefs
    """

    def __init__(self, players: int, perspective_of: int, observations_from: int):
        self.observation_from = observations_from
        self.perspective_of = perspective_of
        self.players = players
        self.hand_size = 5 if players <= 3 else 4

    def update(self, observation: Dict):
        """
        Update this belief according to the specified observations.

        Parameters
        -----------
        - **observation**: the observation dictionnary from the perspective of ```observations_from```.
        """
        deck = _FULL_DECK_.copy()

        # --------------------------------------
        # CK PART
        # --------------------------------------
        # Remove cards from the discard pile
        for card in observation["discard_pile"]:
            color = _COLOR_TO_INDEX[card["color"]]
            rank = card["rank"]
            deck[color, rank] -= 1

        # Remove cards played
        for color, n in observation["fireworks"].items():
            cindex = _COLOR_TO_INDEX[color]
            for rank in range(n):
                deck[cindex, rank] -= 1

        # Note: hints are CK but are treated later

        # --------------------------------------
        # Partial knowledge part
        # --------------------------------------

        # Update based on observed hands
        hands = observation["observed_hands"]
        offset = self.observation_from
        for hand in hands:
            # The player cannot see its own hand
            if offset == self.perspective_of or offset == self.observation_from:
                offset += 1
                continue
            for card in hand:
                color = _COLOR_TO_INDEX[card["color"]]
                rank = card["rank"]
                deck[color, rank] -= 1
            offset += 1
            offset %= self.players

        # --------------------------------------
        # Update based on hints
        # --------------------------------------
        knowledge = observation["card_knowledge"]

        self.virtual_colors = np.zeros(5)
        self.virtual_ranks = np.zeros(5)

        if self.observation_from == self.perspective_of:
            # Observator hand knowledge
            obs_hand_kl = knowledge[0]
            self.hand = obs_hand_kl
        else:
            # Perspective of hand knowledge
            persp_hand_kl = knowledge[(self.perspective_of - self.observation_from) % self.players]
            self.hand = persp_hand_kl

            # Observator hand knowledge
            obs_hand_kl = knowledge[0]
            for card in obs_hand_kl:
                color = _COLOR_TO_INDEX[card["color"]]
                rank = card["rank"]
                if color is None and rank == -1:
                    continue
                if color is None:
                    self.virtual_ranks[rank] += 1
                elif rank == -1:
                    self.virtual_colors[color] += 1
                else:
                    deck[color, rank] -= 1
        self.deck = deck

    def _prob_color(self, card_info: Dict, color: int) -> float:
        if color is None:
            return None

        c = _COLOR_TO_INDEX[card_info["color"]]
        if c == color:
            return 1
        elif c is not None:
            return 0
        total_cards = np.sum(self.deck) - np.sum(self.virtual_colors) + self.virtual_colors[color]
        total_colors = np.sum(self.deck[color, :]) - self.virtual_colors[color]
        return total_colors / total_cards

    def _prob_rank(self, card_info: Dict, rank: int) -> float:
        r = card_info["rank"]
        if r == rank:
            return 1
        elif r is not None and r > -1:
            return 0
        total_cards = np.sum(self.deck) - np.sum(self.virtual_ranks) + self.virtual_ranks[rank]
        total_rank = np.sum(self.deck[:, rank]) - self.virtual_ranks[rank]
        return total_rank / total_cards

    def _prob_card_knowing_color(self, rank: int, color: int) -> float:
        # p1 = P(C=r & R=r)
        # p1 does not take into account hints
        # p2 = P(C=c)
        total_cards = np.sum(self.deck)
        total_in_deck = self.deck[color, rank]
        p1 = total_in_deck / total_cards
        p2 = self._prob_color({"color": None}, color)
        return p1 / p2

    def _prob_card_knowing_rank(self, rank: int, color: int) -> float:
        # p1 = P(C=r & R=r)
        # p1 does not take into account hints
        # p2 = P(R=R)
        total_cards = np.sum(self.deck)
        total_in_deck = self.deck[color, rank]
        p1 = total_in_deck / total_cards
        p2 = self._prob_rank({"rank": None}, rank)
        return p1 / p2

    def probability(self, offset: int, rank: int = -1, color: str = None) -> float:
        """
        Computes the asked probability.
        If only a ```color``` is specified, computes the probability of the card being of the specified ```color```.
        If only a ```rank``` is specified, computes the probability of the card being of the specified ```rank```.
        If no ```color``` nor ```rank``` is specified return ```1```.

        Parameters
        -----------
        - **offset**: the offset of the card in you hand that you want information about
        - **rank**: (*optional*) the rank you want information about
        - **color**: (*optional*) the color you want information about

        Return
        -----------
        A float in ```[0; 1]``` that specifies the probability of the specified card matching the parameters given.
        """
        if rank is None:
            rank = -1
        card_info = self.hand[offset]
        if rank == -1:
            if color is None:
                return 1
            return self._prob_color(card_info, _COLOR_TO_INDEX[color])
        elif color is None:
            return self._prob_rank(card_info, rank)
        else:
            r = card_info["rank"]
            c = card_info["color"]
            cindex = _COLOR_TO_INDEX[color]
            if r == rank and c == color:
                return 1
            elif c is None and rank == r:
                return self._prob_card_knowing_rank(rank, cindex)
            elif c == color and r is None:
                return self._prob_card_knowing_color(rank, cindex)
            elif c is None and r is None:
                total_cards = np.sum(self.deck)
                total_in_deck = self.deck[cindex, rank]
                return total_in_deck / total_cards
            else:
                return 0
