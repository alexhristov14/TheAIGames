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

        self.piece_to_notation = {
            1: "P", 2: "R", 3: "N", 4: "B", 5: "Q", 6: "Q"
        }

        self.how_pieces_move = {
            "N": [(2, 1), (2, -1), (-2, 1), (-2, -1), (1, 2), (1, -2), (-1, 2), (-1, -2)],
            "R": [(1,0), (-1,0), (0,1), (0,-1)],
            "B": [(1,1), (-1,-1), (-1,1), (1,-1)],
            "K": [(1,0), (-1,0), (0,1), (0,-1), (1,1), (-1,-1), (-1,1), (1,-1)],
            "P": [(-1,0), (-2, 0), (-1,-1), (-1,1)]
        }

        self.piece_material = {
            "P": 1,
            "R": 5,
            "B": 3,
            "N": 3,
            "Q": 9,
            "K": 100
        }

        self.piece_index_to_material = {
            1: 1,
            2: 5,
            3: 3,
            4: 3,
            5: 9,
            6: 100
        }
    
        self.chess_map = self.create_chess_map()

        self.current_player = "w"

        self.reset()

    def step(self, action):
        reward = 0

        piece, takes, next_state, castling = self.process_notation(action)
        if castling:
            return

        next_row, next_col = self.chess_map[next_state]
        if not self.notation_to_piece[piece]:
            return -10

        start_row, start_col = self.get_piece_starting_location(piece, next_state, takes)
        valid_moves = self.valid_moves(piece, start_row, start_col, takes)
                
        if len(valid_moves) == 0:
            return -10

        if takes:
            material = self.takes_material(piece, start_row, start_col, next_row, next_col)
            reward += material

        self.board[next_row, next_col] = self.notation_to_piece[piece] if self.current_player == "w" else -self.notation_to_piece[piece]
        self.board[start_row, start_col] = 0

        self.change_current_player()
        

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

    def get_all_valid_moves(self):
        all_pieces = [1, 2, 3, 4, 5, 6]
        all_valid_moves = []

        if self.current_player == "b":
            all_pieces = [-1,-2,-3,-4,-5,-6]

        for piece in all_pieces:
            indexes = np.where(self.board == piece)
            for i in range(len(indexes[0])):
                row, col = indexes[0][i], indexes[1][i]
                p = self.piece_to_notation[np.absolute(piece)]
                print(p, ": ", self.valid_moves(p, row, col))

    def valid_moves(self, piece, start_row, start_col, takes=False):
        valid_moves = []

        if piece == "P":
            return self.get_pawn_moves(piece, start_row, start_col, takes)

        elif piece=="N":
            knight_moves = self.how_pieces_move[piece]

            for dr, dc in knight_moves:
                next_row, next_col = start_row+dr, start_col+dc
                if 0 <= next_row < 8 and 0 <= next_col < 8 and self.board[next_row, next_col]==0:
                    valid_moves.append((next_row, next_col))

        elif piece=="B":
            bishop_moves = self.how_pieces_move[piece]

            for dr, dc in bishop_moves:
                for i in range(1, 8):
                    next_row, next_col = start_row+(dr*i), start_col+(dc*i)
                    if 0 <= next_row < 8 and 0 <= next_col < 8 and self.board[next_row, next_col]==0:
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
                    if 0 <= next_row < 8 and 0 <= next_col < 8 and self.board[next_row, next_col]==0:
                        valid_moves.append((next_row, next_col))
                        if self.board[next_row, next_col] != 0:
                            break
                    else:
                        break

        elif piece=="K":
            king_moves = self.how_pieces_move[piece]

            for dr, dc in king_moves:
                next_row, next_col = start_row+dr, start_col+dc
                if 0 <= next_row < 8 and 0 <= next_col < 8 and self.board[next_row, next_col]==0:
                    valid_moves.append((next_row, next_col))

        elif piece=="Q":
            valid_moves = self.valid_moves("R", start_row, start_col) + self.valid_moves("B", start_row, start_col)


        return valid_moves

    def get_pawn_moves(self, piece, start_row, start_col, takes):
        pawn_moves = self.how_pieces_move[piece]

        direction = 1 if self.current_player == "w" else -1
        valid_moves = []
    
        if not takes:
            dr, dc = pawn_moves[0]
            next_row, next_col = start_row + (dr * direction), start_col + dc
            if 0 <= next_row < 8 and 0 <= next_col < 8 and self.board[next_row, next_col] == 0:
                valid_moves.append((next_row, next_col))

                if (start_row == 6 and self.current_player == "w") or (start_row == 1 and self.current_player == "b"):
                    dr, dc = pawn_moves[1]
                    next_row, next_col = start_row + (dr * direction), start_col + dc
                    if 0 <= next_row < 8:
                        valid_moves.append((next_row, next_col))

        else:
            for dr, dc in pawn_moves[2:]:
                next_row, next_col = start_row - (dr * direction), start_col + dc
                if self.board[next_row, next_col] != 0:
                    if 0 <= next_row < 8 and 0 <= next_col < 8:
                        valid_moves.append((next_row, next_col))

        return valid_moves


    def get_piece_starting_location(self, piece, next_state, takes=False):
        target_row, target_col = self.chess_map[next_state]
        directions = self.how_pieces_move[piece] if piece != "Q" else None

        if piece == "P":
            if self.current_player == "b":
                directions = [(x * -1, y * -1) for x, y in directions]

            if takes:
                for dr, dc in directions[2:]:
                    r, c = target_row-dr, target_col+dc
                    if 0 <= r < 8 and 0 <= c < 8:
                        if np.absolute(self.board[r, c]) == 1:
                                return r, c

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

            # elif (r == 7 and self.current_player == "b") or (r==0 and self.current_player == "w"):
            #     self.process_pawn_promotion(c)

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
                        if self.board[r, c] == 4 and self.current_player=="w":
                            return r, c
                        elif self.board[r, c] == -4 and self.current_player=="b":
                            return r, c
                    else:
                        break

        elif piece == "R":
            for dr, dc in directions:
                for i in range(1, 8):
                    r, c = target_row+(dr*i), target_col+(dc*i)
                    if 0 <= r < 8 and 0 <= c < 8:
                        if self.board[r, c] == 2 and self.current_player=="w":
                            return r, c
                        elif self.board[r, c] == -2 and self.current_player=="b":
                            return r, c

        elif piece == "Q":
            all_directions = self.how_pieces_move["R"] + self.how_pieces_move["B"]

            for dr, dc in all_directions:
                for i in range(1, 8):
                    r, c = target_row + (dr * i), target_col + (dc * i)

                    if 0 <= r < 8 and 0 <= c < 8:
                        if self.board[r, c] == 5 and self.current_player == "w":
                            return r, c
                        elif self.board[r, c] == -5 and self.current_player == "b":
                            return r, c
                        elif self.board[r, c] == 5 and self.current_player == "w":
                            return r, c
                        elif self.board[r, c] == -5 and self.current_player == "b":
                            return r, c
                    else:
                        break

        return None, None


    def process_notation(self, action):
        if action == "O-O" or action == "O-O-O":
            return self.process_castling(action)

        notation_data = list(action)

        if len(notation_data) > 2 and notation_data[2] == "=":
            print("promoting")
            return self.process_pawn_promotion(action)

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

            return piece, takes, next_state, False

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

        return piece, takes, next_state, False

    def process_castling(self, action):
        if action == "O-O": # Short Castle
            if not self.is_in_check():
                if self.current_player == "w":
                    self.board[7, 6] = 6
                    self.board[7, 5] = 2
                    self.board[7, 7] = 0
                    self.board[7, 4] = 0
                else:
                    self.board[0, 6] = -6
                    self.board[0, 5] = -2
                    self.board[0, 7] = 0
                    self.board[0, 4] = 0

        elif action == "O-O-O": # Long Castle
            if not self.is_in_check():
                if self.current_player == "w":
                    self.board[7, 2] = 6
                    self.board[7, 3] = 2
                    self.board[7, 7] = 0
                    self.board[7, 0] = 0
                else:
                    self.board[0, 2] = -6
                    self.board[0, 3] = -2
                    self.board[0, 0] = 0
                    self.board[0, 4] = 0

        return None, False, None, True

    def process_pawn_promotion(self, action):
        info = list(action)
        next_state = info[:2]
        promotion_piece = info[-1]

        assert promotion_piece in self.notation_to_piece.items()

        return promotion_piece, takes, next_state, False



    def takes_material(self, piece, start_row, start_col, end_row, end_col):
        piece_taken = self.board[end_row, end_col]
        material = self.piece_index_to_material[np.absolute(piece_taken)]

        return material

    def create_chess_map(self):
        chess_map = {}

        for row in range(8):
            for col in range(8):
                square = f"{chr(col + 97)}{8 - row}"
                chess_map[square] = (row, col)

        return chess_map

    def is_in_check(self):
        kings_location = self.get_kings_location()
        # print("Current kings location: ", kings_location)
        
        self.change_current_player()
        for piece, _ in self.notation_to_piece.items():
            if self.get_piece_starting_location(piece, kings_location) != (None, None):
                # print(self.get_piece_starting_location(piece, kings_location))
                self.change_current_player()
                return True
        self.change_current_player()

        return False

    def change_current_player(self):
        self.current_player = "b" if self.current_player == "w" else "w"

    def get_kings_location(self):
        if self.current_player == "w":
            kings_row, kings_col = np.where(self.board == 6)
        else:
            kings_row, kings_col = np.where(self.board == -6)

        kings_location = self.index_to_chess_notation(kings_row[0], kings_col[0])
        return kings_location

    def index_to_chess_notation(self, row, col):
        column = chr(ord('a') + col)
        row_number = 8 - row
        return f"{column}{row_number}"

    def is_checkmate(self):
        if not self.is_in_check():
            return False

        kings_location = self.get_kings_location()
        king_row, king_col = self.chess_map[kings_location]
        directions = self.how_pieces_move["K"]
        
        for dr, dc in directions:
            next_row, next_col = king_row + dr, king_col + dc
            if (0 <= next_row < 8 and 0 <= next_col < 8):
                if (self.current_player == "w" and self.board[next_row, next_col] <= 0) or \
                (self.current_player == "b" and self.board[next_row, next_col] >= 0):
                    original_board = self.board.copy()
                    king_value = 6 if self.current_player == "w" else -6
                    self.board[next_row, next_col] = king_value
                    self.board[king_row, king_col] = 0
                    
                    if not self.is_in_check():
                        self.board = original_board
                        return False
                        
                    self.board = original_board
        
        # TODO: Make it so other pieces might protect it by taking the checking pieces or by blocking
        
        return True

    def is_stalemate(self):
        if self.is_in_check():
            return False

        directions = self.how_pieces_move["K"]
        kings_location = self.get_kings_location()
        r, c = self.chess_map[kings_location]
    
        for dr, dc in directions:
            next_row, next_col = r+dr, c+dc
            if 0 <= next_row < 8 and 0 <= next_col < 8:
                if not self.is_in_check:
                    return False

        return True

    def minimax(self, depth, maximazingPlayer, alpha=float('-inf'), beta=float('inf')):
        if self.is_checkmate("w"):
            return None, 1000
    
        elif self.is_checkmate("b"):
            return None, -1000

        elif depth == 0:
            None, self.evaluate_board("w" if maximazingPlayer else "b")
        
        valid_moves = self.get_all_valid_moves()

        if maximazingPlayer:
            max_eval = float("-inf")


        else:
            min_eval = float("inf")


    def evaluate_board(self, player):
        pass

    def train_ai(self):
        pass

    def play_the_ai(self):
        pass


env = ChessEnv()

# env.step("e4")
# env.step("e5")
# env.step("Nf3")
# env.step("Nf6")
# env.step("Bd3")
# env.step("Bd7")
# env.step("O-O")

env.render()

# env.step("e4")
# env.step("e5")
# env.step("Qh5")
# env.step("Nc6")
# env.step("Bc4")
# env.step("Nf6")
# env.step("Qxf7")
# env.step("Ke7")

# env.is_in_check()
# print(env.is_checkmate())


while True:
    print(env.get_all_valid_moves())
    move = str(input("Enter a chess move in algebreic notation: "))
    env.step(move)
    env.render()