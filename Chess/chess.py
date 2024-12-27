import numpy as np

class ChessEnv:
    def __init__(self):
        self.board = np.zeros((8,8), dtype=int)
        
        self.piece_map = {
            1: "♟", 2: "♜", 3: "♞", 4: "♝", 5: "♛", 6: "♚",
            -1: "♙", -2: "♖", -3: "♘", -4: "♗", -5: "♕", -6: "♔"
        }

        self.notation_to_piece = {
            "N": 4, "K": 6, "Q": 5, "R": 3, "B": 4
        }

    
        self.chess_map = self.create_chess_map()

        self.reset()

    def step(self, action):
        piece, takes, next_state = self.process_notation(action)
        print(next_state)
        if not self.notation_to_piece[piece]:
            return -10

        start_row, start_col = self.get_piece_starting_location(piece, next_state)
        
        valid_moves = self.valid_moves(piece, start_row, start_col)
        print(valid_moves)

    def render(self):
        symbols = np.vectorize(lambda x: self.piece_map.get(x, "·"))(self.board)
        for row in symbols:
            print(" ".join(row))

    def reset(self):
        for c in range(8):
            self.board[6, c] = 1
            self.board[1, c] = -1
        
        back_row = np.array([2, 3, 4, 5, 6, 4, 3, 2])
        self.board[0, :] = -back_row
        self.board[7, :] = back_row

    def valid_moves(self, piece, start_row, start_col):
        valid_moves = []

        if piece=="N":
            knight_moves = [(2, 1), (2, -1), (-2, 1), (-2, -1), 
                            (1, 2), (1, -2), (-1, 2), (-1, -2)]

            for dr, dc in knight_moves:
                next_row, next_col = start_row+dr, start_col+dc
                if 0 <= next_row < 8 and 0 <= next_col < 8:
                    valid_moves.append((next_row, next_col))

        elif piece=="B":
            bishop_moves = [(1,1), (-1,-1), (-1,1), (1,-1)]

            for dr, dc in bishop_moves:
                for i in range(1, 8):
                    next_row, next_col = start_row+(dr*i), start_col+(dc*i)
                    if 0 <= next_row < 8 and 0 <= next_col < 8:
                        valid_moves.append((next_row, next_col))
                        if self.board[next_row, next_col] != 0:
                            break
                    else:
                        break

        elif piece=="R":
            rook_moves = [(1,0), (-1,0), (0,1), (0,-1)]

            for dr, dc in rook_moves:
                for i in range(1, 8):
                    next_row, next_col = start_row+(dr*i), start_col+(dc*i)
                    if 0 <= next_row < 8 and 0 <= next_col < 8:
                        valid_moves.append((next_row, next_col))
                        if self.board[next_row, next_col] != 0:
                            break
                    else:
                        break

        elif piece=="K":
            king_moves = [(1,0), (-1,0), (0,1), (0,-1),
                            (1,1), (-1,-1), (-1,1), (1,-1)]

            for dr, dc in king_moves:
                next_row, next_col = start_row+dr, start_col+dc
                if 0 <= next_row < 8 and 0 <= next_col < 8:
                    valid_moves.append((next_row, next_col))

        elif piece=="Q":
            valid_moves = self.valid_moves("R", start_row, start_col) + self.valid_moves("B", start_row, start_col)

        return valid_moves

    def get_piece_starting_location(self, piece, next_state):
        pass


    def process_notation(self, action):
        takes = False
        notation_data = list(action)
        piece = notation_data[0]
        if notation_data[1] == "x":
            takes = True
            next_state = notation_data[2:]
        else:
            next_state = notation_data[1:]

        next_state = "".join(next_state)

        return piece, takes, next_state

    def create_chess_map(self):
        chess_map = {}

        for row in range(8):
            for col in range(8):
                square = f"{chr(col + 97)}{8 - row}"
                chess_map[square] = (row, col)

        return chess_map

    def train_ai(self):
        pass

    def play_the_ai(self):
        pass


env = ChessEnv()
env.step("Kc4")
