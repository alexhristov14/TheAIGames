import numpy as np

class ConnectFourEnv:
    def __init__(self):
        self.board = np.zeros((6,7))
        self.actions = np.arange(7)
        self.depth = 4
        self.episodes = 10

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
            return None, 1000000
        elif self.check_if_won(2):
            return None, -1000000
        elif depth == 0 or self.is_draw():
            return None, self.evaluate_board(1 if maximazingPlayer else 2)
        
        valid_moves = self.get_valid_moves()
        if len(valid_moves) == 0:
            return None, 0

        best_move = valid_moves[0]

        if maximazingPlayer:
            max_eval = float('-inf')
            for move in valid_moves:
                board_backup = self.board.copy()
                self.play_this_move(move, 1)
                _, evaluation = self.minimax(depth-1, False, alpha, beta)
                self.board = board_backup
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
                board_backup = self.board.copy()
                self.play_this_move(move, 2)
                _, evaluation = self.minimax(depth-1, True, alpha, beta)
                self.board = board_backup
                if evaluation < min_eval:
                    min_eval = evaluation
                    best_move = move
                beta = min(beta, evaluation)
                if beta <= alpha:
                    break
            return best_move, min_eval

    def simulate(self, action, player):
        copy_board = self.board.copy()
        for row in range(copy_board.shape[0]-1, -1, -1):
            if copy_board[row, action] == 0:
                copy_board[row, action] = player
                break

        return copy_board

    def evaluate_board(self, player):
        score = 0
        rows, cols = self.board.shape

        center_column = [self.board[row][len(self.board[0])//2] for row in range(len(self.board))]
        score += np.sum(np.array(center_column) == player) * 3

        for r in range(rows):
            for c in range(cols):
                if self.board[r][c] == player:
                    score += self.evaluate_position(player, r, c)
    
        return score

    def evaluate_position(self, player, row, col):
        score = 0
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]

        for dr, dc in directions:
            count = 0
            for i in range(4):
                r, c = row + dr * i, col + dc * i
                if 0 <= r < self.board.shape[0] and 0 <= c < self.board.shape[1] and self.board[r, c] == player:
                    count += 1
                else:
                    break

            if count == 4:
                score += 1000
            elif count == 3:
                score += 50
            elif count == 2:
                score += 10

        return score
        
        
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
        valid = self.actions.copy()
        for c in range(self.board.shape[1]):
            if self.board[0, c] != 0:
                valid = valid[valid != c]
        return valid

    def is_draw(self):
        return all(self.board[0, c] != 0 for c in range(len(self.board[0])))

    def play_this_move(self, action, player):
        for r in range(self.board.shape[0] - 1, -1, -1):
                if self.board[r, action] == 0:
                    self.board[r, action] = player
                    break

    def play_the_ai(self):
        self.reset()
        game_over = False

        while not game_over:
            self.render()

            while True:
                try:
                    col = int(input("Choose a column (1-7): ")) - 1
                    if 0 <= col < 7 and self.board[0, col] == 0:
                        break
                    print("Invalid move! Please choose again.")
                except ValueError:
                    print("Please enter a number between 1 and 7.")
            
            self.play_this_move(col, 1)
            
            if self.check_if_won(1):
                self.render()
                print("You won! Congratulations!")
                game_over = True
                break
                
            if self.is_draw():
                self.render()
                print("It's a draw!")
                game_over = True
                break

            print("AI is thinking...")
            move, _ = self.minimax(self.depth, True, float('-inf'), float('inf'))
            self.play_this_move(move, 2)
            
            if self.check_if_won(2):
                self.render()
                print("AI won!")
                game_over = True
                break
                
            if self.is_draw():
                self.render()
                print("It's a draw!")
                game_over = True
                break          


env = ConnectFourEnv()
env.play_the_ai()