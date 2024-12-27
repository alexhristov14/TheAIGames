import numpy as np

class ConnectFourEnv:
    def __init__(self):
        self.board = np.zeros((6,7))
        self.actions = np.arange(7)
        self.depth = 4
        self.episodes = 10

    def step(self, action):
        if action < 0 or action > self.board.shape[1]:
            return -10

        for r in range(self.board.shape[0] - 1, -1, -1):
            if self.board[r, action] == 0:
                self.board[r, action] = 1

    def reset(self):
        self.board = np.zeros((6,7))

    def render(self):
        for row in self.board:
            print("\t")
            print(" | ".join(["X" if cell == 1 else "Y" if cell == 2 else " " for cell in row]))
            print("-"*25)
        print("\n")

    def minimax(self, depth, maximazingPlayer, alpha, beta):
        if self.check_if_won(1):
            return 1000000

        elif self.check_if_won(2):
            return -1000000
        
        elif depth == 0 or self.is_draw(): # or draw
            return None, self.evaluate_board()
        
        valid_moves = self.get_valid_moves()
        best_move = None

        if maximazingPlayer:
            max_eval = float('-inf')
            for move in valid_moves:
                copy_board = simulate(move, 1)
                depth = depth - 1
                _, evaluation = self.minimax(depth, False, alpha, beta)
                if evaluation > max_eval:
                    max_eval = evaluation
                    best_move = move
                alpha = max(alpha, evaluation)
                if beta <= alpha:
                    break
            return best_move, max_eval


        else:
            min_eval = float('inf')
            for move in valid_moves:
                copy_board = simulate(move, 2)
                depth = depth - 1
                _, evaluation = self.minimax(depth, True, alpha, beta)
                if evalutaion < min_eval:
                    min_eval = evaluation
                    best_move = move
                beta = min(beta, evaluation)
                if beta <= alpha:
                    break
            return best_move, min_eval


    def simulate(self, action, player):
        copy_board = self.boarboard.copy()
        for row in range(copy_board.shape[0]-1, -1, -1):
            if copy_board[row, action] == 0:
                copy_board[row, action] = player
                break

        return copy_board

    def evaluate_board(self, player):
        # if self.check_if_won(1)
        pass
        
        
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
        return all(self.board[0, c] != 0 for c in range(len(self.board[0])))

    def get_state(self):
        return tuple(self.board.flatten())

    def play_this_move(self, action, player):
        for r in range(self.board.shape[0] - 1, -1, -1):
                if self.board[r, action] == 0:
                    self.board[r, action] = player

    def train_ai(self):
        for episode in range(self.episodes):
            env.reset()
            state = self.get_state()
            done = False

            while not done:
                if np.random.random() < self.epsilon:
                    action = np.random.choice(self.actions)
                
                else:
                    action, _ = self.minimax(state)

                self.step(action)

    def play_the_ai(self):
        env.reset()

        while True:    
            if self.is_draw():
                print("Its a draw!")
                break
            env.render()
            col = input("Choose a column: ")
            col -= 1
            if col < 1 or col > 7:
                print("Please choose again!")
                continue

            self.play_this_move(col)

            if self.check_if_won(1):
                print("Player 1 won!")
                break
        
            move, _ self.minimax(4, True, float('-inf'), float('inf'))

            self.play_this_move(move)

            if self.check_if_won(2):
                print("Player 1 won!")
                break            


env = ConnectFourEnv()
env.play_the_ai()