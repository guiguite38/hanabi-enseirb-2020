from collections import namedtuple
import random


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.Transition = namedtuple(
            "Transition", ("state", "action", "next_state", "reward", "rounds_remaining")
        )
        self.rounds_for_important_memories = 3

    def push(self, li):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        if li[-1]:
            li[-1] = self.rounds_for_important_memories
        else:
            li[-1] = 0
        self.memory[self.position] = self.Transition(*li)
        self.position = (self.position + 1) % self.capacity
        while self.position < len(self.memory) and self.memory[self.position].rounds_remaining > 0:
            self.memory[self.position].rounds_remaining -= 1
            self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
