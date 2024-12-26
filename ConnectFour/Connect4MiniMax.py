import numpy as np

class ConnectFourEnv:
    def __init__(self):
        self.board = np.zeros((6,7))
        self.actions = np.arange(7)
        self.depth = 10

        # self.beta
        # self.alpha

    def step(self, action):
        if action<0 or action>self.board.shape[1]:
            return -10

        for r in range(self.board.shape[0] - 1, -1, -1):
            if self.board[r, action] == 0:
                self.board[r, action] = 1

        self.minimax()

    def render(self):
        for row in self.board:
            print("\t")
            print(" | ".join(["X" if cell == 1 else "Y" if cell == 2 else " " for cell in row]))
            print("-"*25)
        print("\n")

    def minimax(self):
        if self.check_if_won(1):
            return 1000000

        if self.check_if_won(2):
            return -1000000
        
        valid_moves = self.get_valid_moves()
        
    def check_if_won(self, player):
        rows, cols = self.board.shape

        for r in range(rows):
            for c in range(cols-3):
                if self.board[r, c] == self.board[r, c + 1] == self.board[r, c + 2] == self.board[r, c + 3] == player:
                    return True

        for r in range(rows-3):
            for c in range(cols):
                if self.board[r, c] == self.board[r + 1, c] == self.board[r + 2, c] == self.board[r + 3, c] == player:
                    return True

        for r in range(rows-3):
            for c in range(cols-3):
                if self.board[r,c] == self.board[r + 1, c + 1] == self.board[r + 2, c + 2] == self.board[r + 3, c + 3] == player:
                    return True
        
        return False

    def get_valid_moves(self):
        for c in range(self.board.shape[1]):
            if self.board[0, c] != 0:
                valid = np.delete(self.actions, c)

        return valid

    def is_draw(self):
        pass

    def train_ai(self):
        pass


env = ConnectFourEnv()
env.step(1)
env.step(1)
env.step(1)
env.step(1)
env.step(1)
env.step(1)
env.render()