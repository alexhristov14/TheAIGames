import numpy as np

class ConnectFourEnv:
    def __init__(self):
        self.board = np.zeros((6,7))
        self.actions = np.arange(7)

        self.beta
        self.alpha

    def step(self, action):
        pass

    def render(self):
        for row in self.board:
            print("\t")
            print(" | ".join(["X" if cell == 1 else "Y" if cell == 2 else " " for cell in row]))
            print("-"*25)
        print("\n")

    def check_if_won(self):
        pass

    def train_ai(self):
        pass


env = ConnectFourEnv()
env.render()