from hanabi_learning_environment.rl_env import Agent, HanabiEnv

import numpy as np


class Node():

    def __init__(self, action, prior=0, parent=None, closed=False):
        self.visits = 0
        self.prior = prior
        self.action_value = 0
        self.total_action_value = 0
        self.action = action
        self.children = []
        self.parent = parent
        self.closed = closed

    def max_value(self):
        '''
        Return the max potential of a node.
        '''
        return self.action_value + (self.prior / (1 + self.visits))

    def select(self, actions=None):
        '''
        Returns self if leaf, else best child
        '''
        if len(self.children) > 0:
            # find best child
            index = np.argmax([child.max_value() for child in self.children])
            child = self.children[index]
            actions = actions or []
            actions.append(child.action)
            return child.select(), actions
        else:
            # node is leaf
            return self, actions            
            
    def expand(self, actions, priors, closed):
        '''
        actions[i] a le prior priors[i]
        '''
        for action, prior in zip(actions, priors, closed):
            self.children.append(Node(action=action, prior=prior, parent=self, closed=closed))

    def backup(self, value):
        self.total_action_value += value
        self.visits += 1
        self.action_value = self.total_action_value / self.visits
        if self.parent is not None:
            self.parent.backup(value)

    def select_move(self):
        index = np.argmax([child.max_value() for child in self.children])
        child = self.children[index]
        return child, child.action
        
    

class MCTSUCBAgent(Agent):
    """Agent that applies a MCTS search."""

    def __init__(self, config, *args, **kwargs):
        """Initialize the agent."""
        self.config = config
        # Extract max info tokens or set default to 8.
        # self.max_information_tokens = config.get('information_tokens', 8)
        

    def act(self, observation, state):
        """Act based on an observation."""
        if observation['current_player_offset'] != 0:
            return None
        self.state = state.copy()
        self.root = Node(None)
        simulations_to_run = 100
        for _ in range(simulations_to_run):
            node, action_path = self.root.select()
            if node.closed:
                node.backup(node.prior)
            else:
                actions, priors, closed = compute_priors(action_path) 
                node.expand(actions, priors, closed)

        self.root, move = self.root.select_move()
        return move


    def heuristic(self, observation, score, done):
        '''
        returns sum of *info token* + *score* + *potential final score* + *lives* (+*known cards*)
        - observation contains all the hands of the players
        '''
        lives = observation['life_tokens']
        known_cards = sum([1 for hint in observation['card_knowledge'][0] if hint['color'] is not None or hint['rank'] is not None])
        potential_score= get_max_possible_score(self.state.discard_pile())
        
        if done:
            return score
        else:
            return 0 if lives < 1 else (score + known_cards + lives - 1) * potential_score / 25


    def get_max_possible_score(self, discard_pile):
        quantities = {
            5: 1,
            4: 2,
            3: 2,
            2: 2,
            1: 3
        }
        max_score = [5] * 5
        for color in range(5):
            for rank in range(0, 4):
                qty = 0
                for card in discard_pile:
                    if card.color() == color and card.rank() == rank:
                        qty += 1
                if qty >= quantities[rank + 1]:
                    max_score[color] = rank
                    break
        return sum(max_score)
        

    def compute_priors(self, action_path):
        tmp = self.environment.copy()

        # Apply all actions in action path
        for action in action_path:
            self.environment.apply_move(action)
        actions = self.list_possible_actions()
        priors = []
        closed = []
        for action in actions:
            save_pt = self.state.copy()
            self.state.apply_move(action)
            observation = self.state.observation(state.cur_player())
            done = self.state.is_terminal()
            score = self.state.score()
            prior = heuristic(observation, score, done)
            priors.append(prior)
            closed.append(done)
            self.state = save_pt
        self.state = tmp
        return actions, priors, closed
        
    def list_possible_actions(self):
        return self.state.legal_moves()
