import numpy as np

class ChessEnv:
    def __init__(self):
        self.board = np.zeros((8,8), dtype=int)
        
        self.piece_map = {
            1: "♟", 2: "♜", 3: "♞", 4: "♝", 5: "♛", 6: "♚",
            -1: "♙", -2: "♖", -3: "♘", -4: "♗", -5: "♕", -6: "♔"
        }

        self.notation_to_piece = {
            "N": 3, "K": 6, "Q": 5, "R": 2, "B": 4, "P": 1
        }

        self.how_pieces_move = {
            "N": [(2, 1), (2, -1), (-2, 1), (-2, -1), (1, 2), (1, -2), (-1, 2), (-1, -2)],
            "R": [(1,0), (-1,0), (0,1), (0,-1)],
            "B": [(1,1), (-1,-1), (-1,1), (1,-1)],
            "K": [(1,0), (-1,0), (0,1), (0,-1), (1,1), (-1,-1), (-1,1), (1,-1)],
            "P": [(-1,0), (-2, 0), (-1,-1), (-1,1)]
        }

    
        self.chess_map = self.create_chess_map()

        self.current_player = "w"

        self.reset()

    def step(self, action):
        piece, takes, next_state = self.process_notation(action)
        next_row, next_col = self.chess_map[next_state]
        if not self.notation_to_piece[piece]:
            return -10

        start_row, start_col = self.get_piece_starting_location(piece, next_state, self.current_player)
        valid_moves = self.valid_moves(piece, start_row, start_col, takes)

        if len(valid_moves) == 0:
            return -10

        self.board[next_row, next_col] = self.notation_to_piece[piece] if self.current_player == "w" else -self.notation_to_piece[piece]
        self.board[start_row, start_col] = 0

        self.current_player = "b" if self.current_player == "w" else "w"
        

    def render(self):
        symbols = np.vectorize(lambda x: self.piece_map.get(x, "·"))(self.board)
        for row in symbols:
            print(" ".join(row))
        
        print('\n')

    def reset(self):
        for c in range(8):
            self.board[6, c] = 1
            self.board[1, c] = -1
        
        back_row = np.array([2, 3, 4, 5, 6, 4, 3, 2])
        self.board[0, :] = -back_row
        self.board[7, :] = back_row

    def valid_moves(self, piece, start_row, start_col, takes=False):
        valid_moves = []

        if piece == "P":
            return self.get_pawn_moves(piece, start_row, start_col, takes)

        elif piece=="N":
            knight_moves = self.how_pieces_move[piece]

            for dr, dc in knight_moves:
                next_row, next_col = start_row+dr, start_col+dc
                if 0 <= next_row < 8 and 0 <= next_col < 8:
                    valid_moves.append((next_row, next_col))

        elif piece=="B":
            bishop_moves = self.how_pieces_move[piece]

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
            king_moves = self.how_pieces_move[piece]

            for dr, dc in king_moves:
                next_row, next_col = start_row+dr, start_col+dc
                if 0 <= next_row < 8 and 0 <= next_col < 8:
                    valid_moves.append((next_row, next_col))

        elif piece=="Q":
            valid_moves = self.valid_moves("R", start_row, start_col) + self.valid_moves("B", start_row, start_col)

        return valid_moves

    def get_pawn_moves(self, piece, start_row, start_col, takes):
        pawn_moves = self.how_pieces_move[piece]

        direction = -1 if self.current_player == "w" else 1
        valid_moves = []

        if not takes:
            dr, dc = pawn_moves[0]
            next_row, next_col = start_row + (dr * direction), start_col + dc
            if 0 <= next_row < 8 and 0 <= next_col < 8:
                valid_moves.append((next_row, next_col))

                if (start_row == 6 and self.current_player == "w") or (start_row == 1 and self.current_player == "b"):
                    dr, dc = pawn_moves[1]
                    next_row, next_col = start_row + (dr * direction), start_col + dc
                    if 0 <= next_row < 8:
                        valid_moves.append((next_row, next_col))

        else:
            for dr, dc in pawn_moves[2:]:
                next_row, next_col = start_row + dr * direction, start_col + dc
                if (
                    0 <= next_row < 8 and
                    0 <= next_col < 8 and
                    self.board[next_row][next_col] != "" and
                    self.board[next_row][next_col].islower() if self.current_player == "w" else self.board[next_row][next_col].isupper()
                ):
                    valid_moves.append((next_row, next_col))

        return valid_moves


    def get_piece_starting_location(self, piece, next_state, player):
        target_row, target_col = self.chess_map[next_state]
        directions = self.how_pieces_move[piece]

        if piece == "P":
            if self.current_player == "b":
                directions = [(x * -1, y * -1) for x, y in directions]

            directions = directions[:2] if (target_row == 3 and self.current_player == "b") or (target_row == 4 and self.current_player == "w") else [directions[0]]

            dr, dc = directions[0]
            r, c = target_row-dr, target_col+dc
            if 0 <= r < 8 and 0 <= c < 8:
                if self.board[r, c] == 1 and self.current_player == "w":
                        return r, c

                elif self.board[r, c] == -1 and self.current_player == "b":
                    return r, c
            
            if len(directions) == 2:
                dr, dc = directions[1]
                r, c = target_row-dr, target_col+dc
                if 0 <= r < 8 and 0 <= c < 8:
                    if self.board[r, c] == 1 and self.current_player == "w":
                            return r, c

                    elif self.board[r, c] == -1 and self.current_player == "b":
                        return r, c

            elif (r == 7 and self.current_player == "b") or (r==0 and self.current_player == "w"):
                self.process_pawn_promotion(c)

        elif piece == "N":
            for dr, dc in directions:
                r, c = target_row+dr, target_col+dc
                if 0 <= r < 8 and 0 <= c < 8:
                    if self.board[r, c] == 3 and self.current_player == "w":
                        return r, c

                    elif self.board[r, c] == -3 and self.current_player == "b":
                        return r, c

        elif piece == "B":
            for dr, dc in directions:
                for i in range(1, 8):
                    r, c = target_row+(dr*i), target_col+(dc*i)
                    if 0 <= r < 8 and 0 <= c < 8:
                        if self.board[r, c] == 4 and player=="w":
                            return r, c
                        elif self.board[r, c] == -4 and player=="b":
                            return r, c
                    else:
                        break

        elif piece == "R":
            for dr, dc in directions:
                for i in range(1, 8):
                    r, c = target_row+(dr*i), target_col+(dc*i)
                    if 0 <= r < 8 and 0 <= c < 8:
                        if self.board[r, c] == 2 and player=="w":
                            return r, c
                        elif self.board[r, c] == -2 and player=="b":
                            return r, c

        return None, None


    def process_notation(self, action):
        notation_data = list(action)
        if len(notation_data) == 2 or notation_data[0] in "abcdefgh":
            return self.process_pawn(action)
        else:
            takes = False
            piece = notation_data[0]
            if notation_data[1] == "x":
                takes = True
                next_state = notation_data[2:]
            else:
                next_state = notation_data[1:]

            next_state = "".join(next_state)

            return piece, takes, next_state

    def process_pawn(self, action):
        notation_data = list(action)
        piece = "P"
        takes = False

        if notation_data[1] == "x":
            takes = True
            next_state = notation_data[2:]
        else:
            next_state = notation_data[:]

        next_state = "".join(next_state)

        return piece, takes, next_state

    def process_pawn_promotion(self, col):
        pass

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
env.render()

while True:
    move = str(input("Enter a chess move in algebreic notation: "))
    env.step(move)
    env.render()
