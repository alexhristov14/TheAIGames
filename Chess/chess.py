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
        try:
            piece, takes, next_state, castling = self.process_notation(action)
            
            if castling:
                return 10
                
            if next_state not in self.chess_map:
                return -10
                
            next_row, next_col = self.chess_map[next_state]
            start_row, start_col = self.get_piece_starting_location(piece, next_state, takes)

            if start_row is None or start_col is None:
                return -10
                
            valid_moves = self.valid_moves(piece, start_row, start_col, takes)

            if (next_row, next_col) not in valid_moves:
                return -10
                
            reward = 0
            if takes:
                reward = self.takes_material(piece, start_row, start_col, next_row, next_col)
                
            self.board[next_row, next_col] = self.notation_to_piece[piece] if self.current_player == "w" else -self.notation_to_piece[piece]
            self.board[start_row, start_col] = 0
            
            if self.is_checkmate():
                return 1000
                
            self.change_current_player()
            return reward
            
        except Exception:
            return -10

    def render(self):
        symbols = np.vectorize(lambda x: self.piece_map.get(x, "·"))(self.board)
        print("  a b c d e f g h")
        print("  ---------------")
        for i, row in enumerate(symbols):
            print(f"{8-i}|{' '.join(row)}|{8-i}")
        print("  ---------------")
        print("  a b c d e f g h")
        print(f"\nCurrent player: {'White' if self.current_player == 'w' else 'Black'}")

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
            all_pieces = [-1,-2,-3,-4,-5, -6]

        for piece in all_pieces:
            indexes = np.where(self.board == piece)
            for i in range(len(indexes[0])):
                row, col = indexes[0][i], indexes[1][i]
                p = self.piece_to_notation[np.absolute(piece)]
                for move in self.valid_moves(p, row, col):
                    m = "".join((p, self.index_to_chess_notation(move[0], move[1]))) if p != "P" else self.index_to_chess_notation(move[0], move[1])
                    all_valid_moves.append(m)
        
        return all_valid_moves

    def get_all_valid_opponent_moves(self):
        self.change_current_player()
        valid_moves = self.get_all_valid_moves()
        self.change_current_player()
        return valid_moves

    def get_best_valid_moves(self, valid_moves, player):
        # taking a piece -> check -> quiet move
        prioritized_moves = []

        for move in valid_moves:
            if 'x' in move:
                prioritized_moves.append(move)
            else:
                board_copy = self.board.copy()
                self.step(move)
                if self.is_in_check():  # Check if the move results in a check
                    prioritized_moves.append(move)
                self.board = board_copy

        while len(prioritized_moves) < 10:
            prioritized_moves.append(np.random.choice(valid_moves))

        prioritized_moves = list(set(prioritized_moves))[:10]

        return prioritized_moves

    def valid_moves(self, piece, start_row, start_col, takes=False):
        valid_moves = []

        if piece == "P":
            return self.get_pawn_moves(piece, start_row, start_col, takes)

        elif piece=="N":
            knight_moves = self.how_pieces_move[piece]

            for dr, dc in knight_moves:
                next_row, next_col = start_row+dr, start_col+dc
                if 0 <= next_row < 8 and 0 <= next_col < 8 and ((self.board[next_row, next_col]<=0 and self.current_player=="w") or (self.board[next_row, next_col]>=0 and self.current_player=="b")):
                    valid_moves.append((next_row, next_col))

        elif piece=="B":
            bishop_moves = self.how_pieces_move[piece]

            for dr, dc in bishop_moves:
                for i in range(1, 8):
                    next_row, next_col = start_row+(dr*i), start_col+(dc*i)
                    if 0 <= next_row < 8 and 0 <= next_col < 8 and ((self.board[next_row, next_col]<=0 and self.current_player=="w") or (self.board[next_row, next_col]>=0 and self.current_player=="b")):
                        valid_moves.append((next_row, next_col))
                        if self.board[next_row, next_col] != 0:
                            break
                    else:
                        break

        elif piece=="R":
            rook_moves = self.how_pieces_move[piece]

            for dr, dc in rook_moves:
                for i in range(1, 8):
                    next_row, next_col = start_row+(dr*i), start_col+(dc*i)
                    if 0 <= next_row < 8 and 0 <= next_col < 8 and ((self.board[next_row, next_col]<=0 and self.current_player=="w") or (self.board[next_row, next_col]>=0 and self.current_player=="b")):
                        valid_moves.append((next_row, next_col))
                        if self.board[next_row, next_col] != 0:
                            break
                    else:
                        break

        elif piece=="K":
            king_moves = self.how_pieces_move[piece]

            for dr, dc in king_moves:
                next_row, next_col = start_row+dr, start_col+dc
                if 0 <= next_row < 8 and 0 <= next_col < 8 and ((self.board[next_row, next_col]<=0 and self.current_player=="w") or (self.board[next_row, next_col]>=0 and self.current_player=="b")):
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
                    if 0 <= next_row < 8 and self.board[next_row, next_col] == 0:
                        valid_moves.append((next_row, next_col))

        else:
            for dr, dc in pawn_moves[2:]:
                next_row, next_col = start_row + (dr * direction), start_col + dc
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

                    if np.absolute(self.board[r, c]) not in [0, 4]:
                        break

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

                    if np.absolute(self.board[r, c]) not in [0, 2]:
                        break

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

        elif piece == "K":
            all_directions = self.how_pieces_move["K"]

            for dr, dc in all_directions:
                r, c = target_row + dr, target_col + dc
                if 0 <= r < 8 and 0 <= c < 8:
                    if self.board[r, c] == 6 and self.current_player == "w":
                        return r, c
                    elif self.board[r, c] == -6 and self.current_player == "b":
                        return r, c
                    elif self.board[r, c] == 6 and self.current_player == "w":
                        return r, c
                    elif self.board[r, c] == -6 and self.current_player == "b":
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
        king = 6 if self.current_player == "w" else -6
        king_pos = np.where(self.board == king)
        if king_pos[0].size == 0:  # King not found
            return False
        
        king_row, king_col = king_pos[0][0], king_pos[1][0]
        opponent_moves = self.get_all_valid_opponent_moves()

        for move in opponent_moves:
            target_row, target_col = self.chess_map[move[-2:]]
            if target_row == king_row and target_col == king_col:
                return True
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

        all_moves = self.get_all_valid_moves()
        
        for move in all_moves:
            original_board = self.board.copy()
            original_player = self.current_player
            
            try:
                piece, takes, next_state, castling = self.process_notation(move)
                if castling:
                    continue
                    
                next_row, next_col = self.chess_map[next_state]
                start_row, start_col = self.get_piece_starting_location(piece, next_state, takes)
                
                if start_row is None or start_col is None:
                    continue
                
                self.board[next_row, next_col] = self.board[start_row, start_col]
                self.board[start_row, start_col] = 0
                
                if not self.is_in_check():
                    self.board = original_board
                    self.current_player = original_player
                    return False
                    
                self.board = original_board
                self.current_player = original_player
                
            except Exception as e:
                self.board = original_board
                self.current_player = original_player
                continue

        return True


    def is_stalemate(self):
        if self.is_in_check():
            return False
        return len(self.get_all_valid_moves()) == 0 and not self.is_in_check() 

    def minimax(self, depth, maximizing_player, alpha=float('-inf'), beta=float('inf')):
        if depth == 0 or self.is_checkmate():
            return None, self.evaluate_board("w" if maximizing_player else "b")

        current_player = "w" if maximizing_player else "b"
        valid_moves = self.get_all_valid_moves()
        
        if not valid_moves:
            return None, self.evaluate_board(current_player)

        best_move = valid_moves[0]
        best_value = float("-inf") if maximizing_player else float("inf")
        
        for move in valid_moves:
            board_copy = self.board.copy()
            player_copy = self.current_player
            
            reward = self.step(move)
            _, value = self.minimax(depth - 1, not maximizing_player, alpha, beta)
            
            self.board = board_copy
            self.current_player = player_copy
            
            if maximizing_player and value > best_value:
                best_value = value
                best_move = move
                alpha = max(alpha, value)
            elif not maximizing_player and value < best_value:
                best_value = value
                best_move = move
                beta = min(beta, value)
                
            if beta <= alpha:
                break
                
        return best_move, best_value

    def evaluate_board(self, player):
        piece_values = {
            'P': 100,
            'N': 320,
            'B': 330,
            'R': 500,
            'Q': 900,
            'K': 20000
        }
        
        # Position bonuses for pieces
        pawn_position = [
            [0,  0,  0,  0,  0,  0,  0,  0],
            [50, 50, 50, 50, 50, 50, 50, 50],
            [10, 10, 20, 30, 30, 20, 10, 10],
            [5,  5, 10, 25, 25, 10,  5,  5],
            [0,  0,  0, 20, 20,  0,  0,  0],
            [5, -5,-10,  0,  0,-10, -5,  5],
            [5, 10, 10,-20,-20, 10, 10,  5],
            [0,  0,  0,  0,  0,  0,  0,  0]
        ]
        
        knight_position = [
            [-50,-40,-30,-30,-30,-30,-40,-50],
            [-40,-20,  0,  0,  0,  0,-20,-40],
            [-30,  0, 10, 15, 15, 10,  0,-30],
            [-30,  5, 15, 20, 20, 15,  5,-30],
            [-30,  0, 15, 20, 20, 15,  0,-30],
            [-30,  5, 10, 15, 15, 10,  5,-30],
            [-40,-20,  0,  5,  5,  0,-20,-40],
            [-50,-40,-30,-30,-30,-30,-40,-50]
        ]
        
        bishop_position = [
            [-20,-10,-10,-10,-10,-10,-10,-20],
            [-10,  0,  0,  0,  0,  0,  0,-10],
            [-10,  0,  5, 10, 10,  5,  0,-10],
            [-10,  5,  5, 10, 10,  5,  5,-10],
            [-10,  0, 10, 10, 10, 10,  0,-10],
            [-10, 10, 10, 10, 10, 10, 10,-10],
            [-10,  5,  0,  0,  0,  0,  5,-10],
            [-20,-10,-10,-10,-10,-10,-10,-20]
        ]
        
        rook_position = [
            [0,  0,  0,  0,  0,  0,  0,  0],
            [5, 10, 10, 10, 10, 10, 10,  5],
            [-5,  0,  0,  0,  0,  0,  0, -5],
            [-5,  0,  0,  0,  0,  0,  0, -5],
            [-5,  0,  0,  0,  0,  0,  0, -5],
            [-5,  0,  0,  0,  0,  0,  0, -5],
            [-5,  0,  0,  0,  0,  0,  0, -5],
            [0,  0,  0,  5,  5,  0,  0,  0]
        ]
        
        queen_position = [
            [-20,-10,-10, -5, -5,-10,-10,-20],
            [-10,  0,  0,  0,  0,  0,  0,-10],
            [-10,  0,  5,  5,  5,  5,  0,-10],
            [-5,  0,  5,  5,  5,  5,  0, -5],
            [0,  0,  5,  5,  5,  5,  0, -5],
            [-10,  5,  5,  5,  5,  5,  0,-10],
            [-10,  0,  5,  0,  0,  0,  0,-10],
            [-20,-10,-10, -5, -5,-10,-10,-20]
        ]
        
        king_position = [
            [-30,-40,-40,-50,-50,-40,-40,-30],
            [-30,-40,-40,-50,-50,-40,-40,-30],
            [-30,-40,-40,-50,-50,-40,-40,-30],
            [-30,-40,-40,-50,-50,-40,-40,-30],
            [-20,-30,-30,-40,-40,-30,-30,-20],
            [-10,-20,-20,-20,-20,-20,-20,-10],
            [20, 20,  0,  0,  0,  0, 20, 20],
            [20, 30, 10,  0,  0, 10, 30, 20]
        ]
        
        position_tables = {
            'P': pawn_position,
            'N': knight_position,
            'B': bishop_position,
            'R': rook_position,
            'Q': queen_position,
            'K': king_position
        }
        
        score = 0
        multiplier = 1 if player == "w" else -1
        
        if self.is_checkmate():
            return -10000 * multiplier
        if self.is_in_check():
            score -= 50 * multiplier
            
        for row in range(8):
            for col in range(8):
                piece = self.board[row][col]
                if piece != 0:
                    is_white = piece > 0
                    piece_type = self.piece_to_notation[abs(piece)]
                    position_score = position_tables[piece_type][row][col]
                    
                    # Flip position tables for black pieces
                    if not is_white:
                        position_score = position_tables[piece_type][7-row][col]
                    
                    piece_score = piece_values[piece_type] + position_score
                    score += piece_score if is_white else -piece_score
        
        # Evaluate development in early game (first 10 moves)
        if sum(1 for row in self.board for piece in row if piece != 0) > 28:
            # Penalize undeveloped pieces
            back_rank = 7 if player == "w" else 0
            score -= 10 * len([1 for piece in self.board[back_rank] if abs(piece) in [2,3,4]])  # Undeveloped R,N,B
            
        # Evaluate pawn structure
        for col in range(8):
            white_pawns = len([1 for row in range(8) if self.board[row][col] == 1])
            black_pawns = len([1 for row in range(8) if self.board[row][col] == -1])
            
            # Penalize doubled pawns
            if white_pawns > 1:
                score -= 20
            if black_pawns > 1:
                score += 20
            
            # Penalize isolated pawns
            if col > 0 and col < 7:
                if white_pawns > 0 and sum(1 for row in range(8) for c in [col-1, col+1] if self.board[row][c] == 1) == 0:
                    score -= 10
                if black_pawns > 0 and sum(1 for row in range(8) for c in [col-1, col+1] if self.board[row][c] == -1) == 0:
                    score += 10
        
        return score * multiplier

    def train_ai(self):
        pass

    def play_the_ai(self):
        pass



def play_chess():
    env = ChessEnv()
    env.render()

    ai_or_not = str(input("Do you want to play against the AI? (y/n): "))
    if ai_or_not.lower() == "n":
            while True:
                move = str(input("Enter a chess move in algebreic notation: "))
                reward = env.step(move)

                if reward == -10:
                    print("Invalid move, please try again")
                    continue

                elif env.is_checkmate():
                    print("Checkmate, game over!")
                    exit()

                elif env.is_stalemate():
                    print("Stalemate, game over!")
                    exit()
                
                elif env.is_in_check():
                    print("You're in check")

                env.render()

    elif ai_or_not.lower() == "y":

        choice = str(input("Do you want to play as white or black? (w/b): "))

        if choice.lower() == "w":
            while True:
                move = str(input("Enter a chess move in algebreic notation: "))
                reward = env.step(move)

                if reward == -10:
                    print("Invalid move, please try again")
                    continue
                    
                elif env.is_checkmate():
                    print("You won!")
                    exit()

                action, reward = env.minimax(3, True)
                reward = env.step(action)

                if env.is_checkmate():
                    exit()
                
                elif env.is_in_check():
                    print("You're in check")
                
                elif env.is_stalemate():
                    print("Stalemate, game over!")
                    exit()

                env.render()
        elif choice.lower() == "b":   
            while True:
                action, reward = env.minimax(3, True)
                reward = env.step(action)

                if env.is_checkmate():
                    exit()
                
                elif env.is_in_check():
                    print("You're in check")

                env.render()

                move = str(input("Enter a chess move in algebreic notation: "))
                reward = env.step(move)

                if reward == -10:
                    print("Invalid move, please try again")
                    continue
                    
                elif env.is_checkmate():
                    print("You won!")
                    exit()

                elif env.is_stalemate():
                    print("Stalemate, game over!")
                    exit()

                env.render()

    else:   
        print("Invalid choice, please try again")
        play_chess()

if __name__ == "__main__":
    play_chess()