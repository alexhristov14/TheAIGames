import numpy as np
import random
import os
import time

class SudokuEnv:
    def __init__(self):
        pass

    def create_random_board(self, dims):
        pass

    def pattern(self, dims, r, c):
        return (dims*(r%dims)+r//dims+c)%(dims**2)

    def shuffle(self, s):
        return random.sample(s,len(s)) 

    def render(self):
        pass

    def reset(self):
        pass

    def backtrack(self):
        pass


env = SudokuEnv()

env.render()