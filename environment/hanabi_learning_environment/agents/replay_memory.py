from collections import namedtuple
import random


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.rounds_remaining = [0] * self.capacity
        self.position = 0
        self.Transition = namedtuple(
            "Transition", ("state", "action", "next_state", "reward")
        )
        self.rounds_for_important_memories = 2

    def push(self, li):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.rounds_remaining[self.position] = self.rounds_for_important_memories if li[-1] else 0
        self.memory[self.position] = self.Transition(*li[:-1])
        self.position = (self.position + 1) % self.capacity
        while self.position < len(self.memory) and self.rounds_remaining[self.position] > 0:
            self.rounds_remaining[self.position] -= 1
            self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
