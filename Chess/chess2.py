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
            1: "P", 2: "R", 3: "N", 4: "B", 5: "Q", 6: "K"  # Fixed: Changed last Q to K
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
            print("start_row, start_col: ", start_row, start_col)

            if start_row is None or start_col is None:
                return -10
                
            valid_moves = self.valid_moves(piece, start_row, start_col, takes)
            print(valid_moves)

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
        self.board = np.zeros((8,8), dtype=int)
        for c in range(8):
            self.board[6, c] = 1
            self.board[1, c] = -1
        
        back_row = np.array([2, 3, 4, 5, 6, 4, 3, 2])
        self.board[0, :] = -back_row
        self.board[7, :] = back_row
        self.current_player = "w"
        return self.board.copy()

    def get_all_valid_moves(self):
        all_pieces = [1, 2, 3, 4, 5, 6] if self.current_player == "w" else [-1,-2,-3,-4,-5,-6]
        all_valid_moves = []

        for piece in all_pieces:
            indexes = np.where(self.board == piece)
            for i in range(len(indexes[0])):
                row, col = indexes[0][i], indexes[1][i]
                p = self.piece_to_notation[abs(piece)]
                for move in self.valid_moves(p, row, col):
                    move_notation = self.index_to_chess_notation(move[0], move[1])
                    m = f"{p}{move_notation}" if p != "P" else move_notation
                    all_valid_moves.append(m)
        
        return all_valid_moves

    def get_all_valid_opponent_moves(self):
        self.change_current_player()
        valid_moves = self.get_all_valid_moves()
        self.change_current_player()
        return valid_moves

    def valid_moves(self, piece, start_row, start_col, takes=False):
        valid_moves = []

        if piece == "P":
            return self.get_pawn_moves(piece, start_row, start_col, takes)

        piece_directions = self.how_pieces_move[piece]
        is_sliding_piece = piece in "RRBQ"

        if is_sliding_piece:
            for dr, dc in piece_directions:
                for i in range(1, 8):
                    next_row, next_col = start_row + (dr * i), start_col + (dc * i)
                    if not self.is_valid_position(next_row, next_col):
                        break
                    target_piece = self.board[next_row, next_col]
                    if self.is_valid_target(target_piece):
                        valid_moves.append((next_row, next_col))
                        if target_piece != 0:  # Stop if we hit a piece
                            break
                    else:
                        break
        else:  # Knight and King
            for dr, dc in piece_directions:
                next_row, next_col = start_row + dr, start_col + dc
                if self.is_valid_position(next_row, next_col) and self.is_valid_target(self.board[next_row, next_col]):
                    valid_moves.append((next_row, next_col))

        return valid_moves

    def is_valid_position(self, row, col):
        return 0 <= row < 8 and 0 <= col < 8

    def is_valid_target(self, piece):
        return (piece <= 0 and self.current_player == "w") or (piece >= 0 and self.current_player == "b")

    def get_pawn_moves(self, piece, start_row, start_col, takes):
        pawn_moves = self.how_pieces_move[piece]
        direction = -1 if self.current_player == "w" else 1
        valid_moves = []
    
        if not takes:
            dr, dc = pawn_moves[0]
            next_row, next_col = start_row + (dr * direction), start_col + dc
            if self.is_valid_position(next_row, next_col) and self.board[next_row, next_col] == 0:
                valid_moves.append((next_row, next_col))

                # Initial two-square move
                if ((start_row == 6 and self.current_player == "w") or 
                    (start_row == 1 and self.current_player == "b")):
                    dr, dc = pawn_moves[1]
                    next_row, next_col = start_row + (dr * direction), start_col + dc
                    if self.is_valid_position(next_row, next_col) and self.board[next_row, next_col] == 0:
                        valid_moves.append((next_row, next_col))

        else:  # Capturing moves
            for dr, dc in pawn_moves[2:]:
                next_row, next_col = start_row + (dr * direction), start_col + dc
                if (self.is_valid_position(next_row, next_col) and 
                    self.board[next_row, next_col] != 0 and 
                    self.is_valid_target(self.board[next_row, next_col])):
                    valid_moves.append((next_row, next_col))

        return valid_moves

    def get_piece_starting_location(self, piece, next_state, takes=False):
        target_row, target_col = self.chess_map[next_state]
        
        if piece == "P":
            return self.get_pawn_starting_location(target_row, target_col, takes)
            
        piece_num = self.notation_to_piece[piece]
        if self.current_player == "b":
            piece_num = -piece_num
            
        piece_locs = np.where(self.board == piece_num)
        for i in range(len(piece_locs[0])):
            row, col = piece_locs[0][i], piece_locs[1][i]
            valid_moves = self.valid_moves(piece, row, col, takes)
            if (target_row, target_col) in valid_moves:
                return row, col
                
        return None, None

    def get_pawn_starting_location(self, target_row, target_col, takes):
        direction = 1 if self.current_player == "w" else -1
        piece_num = 1 if self.current_player == "w" else -1
        
        if takes:
            for dr, dc in [(1, 1), (1, -1)]:
                r = target_row + (dr * direction)
                c = target_col + dc
                if (self.is_valid_position(r, c) and 
                    self.board[r, c] == piece_num):
                    return r, c
        else:
            r = target_row + direction
            # print(self.board[r, target_col])
            # print(piece_num)

            if (self.is_valid_position(r, target_col) and 
                self.board[r, target_col] == 0):
                return r, target_col
            
            if ((target_row == 3 and self.current_player == "w") or 
                (target_row == 4 and self.current_player == "b")):
                r = target_row + (2 * direction)
                if (self.is_valid_position(r, target_col) and 
                    self.board[r, target_col] == piece_num):
                    return r, target_col
                    
        return None, None

    def process_notation(self, action):
        if action in ["O-O", "O-O-O"]:
            return self.process_castling(action)

        notation_data = list(action)

        if len(notation_data) > 2 and notation_data[2] == "=":
            return self.process_pawn_promotion(action)

        if len(notation_data) == 2 or notation_data[0] in "abcdefgh":
            return self.process_pawn(action)
        
        piece = notation_data[0]
        takes = notation_data[1] == "x"
        next_state = "".join(notation_data[2:] if takes else notation_data[1:])

        return piece, takes, next_state, False

    def process_pawn(self, action):
        notation_data = list(action)
        takes = len(notation_data) > 2 and notation_data[1] == "x"
        next_state = "".join(notation_data[2:] if takes else notation_data)

        return "P", takes, next_state, False

    def process_castling(self, action):
        if not self.is_in_check():
            row = 7 if self.current_player == "w" else 0
            king_col = 4
            
            if action == "O-O":  # Short Castle
                rook_from, rook_to = 7, 5
                king_to = 6
            else:  # Long Castle
                rook_from, rook_to = 0, 3
                king_to = 2

            self.board[row, king_to] = 6 if self.current_player == "w" else -6
            self.board[row, rook_to] = 2 if self.current_player == "w" else -2
            self.board[row, king_col] = 0
            self.board[row, rook_from] = 0

        return None, False, None, True

    def process_pawn_promotion(self, action):
        info = list(action)
        next_state = info[:2]
        promotion_piece = info[-1]

        if promotion_piece not in self.notation_to_piece:
            raise ValueError("Invalid promotion piece")

        return promotion_piece, False, next_state, False

    def takes_material(self, piece, start_row, start_col, end_row, end_col):
        piece_taken = self.board[end_row, end_col]
        return self.piece_index_to_material[abs(piece_taken)]

    def create_chess_map(self):
        return {f"{chr(col + 97)}{8 - row}": (row, col) 
                for row in range(8) 
                for col in range(8)}

    def is_in_check(self):
        king = 6 if self.current_player == "w" else -6
        king_pos = np.where(self.board == king)
        
        if king_pos[0].size == 0:
            return False
            
        king_row, king_col = king_pos[0][0], king_pos[1][0]
        
        self.change_current_player()
        opponent_moves = self.get_all_valid_moves()
        self.change_current_player()
        
        king_square = self.index_to_chess_notation(king_row, king_col)
        return any(move.endswith(king_square) for move in opponent_moves)

    def change_current_player(self):
        self.current_player = "b" if self.current_player == "w" else "w"

    def index_to_chess_notation(self, row, col):
        return f"{chr(col + 97)}{8 - row}"

    def is_checkmate(self):
        if not self.is_in_check():
            return False

        original_board = self.board.copy()
        original_player = self.current_player
        
        for move in self.get_all_valid_moves():
            try:
                reward = self.step(move)
                if reward != -10 and not self.is_in_check():
                    self.board = original_board
                    self.current_player = original_player
                    return False
                self.board = original_board
                self.current_player = original_player
            except Exception:
                self.board = original_board
                self.current_player = original_player
                continue
        
        return True

    def is_stalemate(self):
        if self.is_in_check():
            return False
        return len(self.get_all_valid_moves()) == 0

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
            'P': 100, 'N': 320, 'B': 330,
            'R': 500, 'Q': 900, 'K': 20000
        }
        
        position_tables = {
            'P': [
                [0,  0,  0,  0,  0,  0,  0,  0],
                [50, 50, 50, 50, 50, 50, 50, 50],
                [10, 10, 20, 30, 30, 20, 10, 10],
                [5,  5, 10, 25, 25, 10,  5,  5],
                [0,  0,  0, 20, 20,  0,  0,  0],
                [5, -5,-10,  0,  0,-10, -5,  5],
                [5, 10, 10,-20,-20, 10, 10,  5],
                [0,  0,  0,  0,  0,  0,  0,  0]
            ],
            'N': [
                [-50,-40,-30,-30,-30,-30,-40,-50],
                [-40,-20,  0,  0,  0,  0,-20,-40],
                [-30,  0, 10, 15, 15, 10,  0,-30],
                [-30,  5, 15, 20, 20, 15,  5,-30],
                [-30,  0, 15, 20, 20, 15,  0,-30],
                [-30,  5, 10, 15, 15, 10,  5,-30],
                [-40,-20,  0,  5,  5,  0,-20,-40],
                [-50,-40,-30,-30,-30,-30,-40,-50]
            ],
            'B': [
                [-20,-10,-10,-10,-10,-10,-10,-20],
                [-10,  0,  0,  0,  0,  0,  0,-10],
                [-10,  0,  5, 10, 10,  5,  0,-10],
                [-10,  5,  5, 10, 10,  5,  5,-10],
                [-10,  0, 10, 10, 10, 10,  0,-10],
                [-10, 10, 10, 10, 10, 10, 10,-10],
                [-10,  5,  0,  0,  0,  0,  5,-10],
                [-20,-10,-10,-10,-10,-10,-10,-20]
            ],
            'R': [
                [0,  0,  0,  0,  0,  0,  0,  0],
                [5, 10, 10, 10, 10, 10, 10,  5],
                [-5,  0,  0,  0,  0,  0,  0, -5],
                [-5,  0,  0,  0,  0,  0,  0, -5],
                [-5,  0,  0,  0,  0,  0,  0, -5],
                [-5,  0,  0,  0,  0,  0,  0, -5],
                [-5,  0,  0,  0,  0,  0,  0, -5],
                [0,  0,  0,  5,  5,  0,  0,  0]
            ],
            'Q': [
                [-20,-10,-10, -5, -5,-10,-10,-20],
                [-10,  0,  0,  0,  0,  0,  0,-10],
                [-10,  0,  5,  5,  5,  5,  0,-10],
                [-5,  0,  5,  5,  5,  5,  0, -5],
                [0,  0,  5,  5,  5,  5,  0, -5],
                [-10,  5,  5,  5,  5,  5,  0,-10],
                [-10,  0,  5,  0,  0,  0,  0,-10],
                [-20,-10,-10, -5, -5,-10,-10,-20]
            ],
            'K': [
                [-30,-40,-40,-50,-50,-40,-40,-30],
                [-30,-40,-40,-50,-50,-40,-40,-30],
                [-30,-40,-40,-50,-50,-40,-40,-30],
                [-30,-40,-40,-50,-50,-40,-40,-30],
                [-20,-30,-30,-40,-40,-30,-30,-20],
                [-10,-20,-20,-20,-20,-20,-20,-10],
                [20, 20,  0,  0,  0,  0, 20, 20],
                [20, 30, 10,  0,  0, 10, 30, 20]
            ]
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
                    position_score = position_tables[piece_type][row if is_white else 7-row][col]
                    piece_score = piece_values[piece_type] + position_score
                    score += piece_score if is_white else -piece_score
        
        return score * multiplier

def play_chess():
    env = ChessEnv()
    print("Welcome to Chess! Enter moves in algebraic notation (e.g., 'e4', 'Nf3')")
    print("Type 'quit' to exit\n")
    
    env.render()
    
    while True:
        move = input("\nYour move: ").strip()
        if move.lower() == 'quit':
            break
            
        reward = env.step(move)
        if reward == -10:
            print("Invalid move! Try again.")
            continue
            
        print("\nAI is thinking...")
        ai_move, _ = env.minimax(3, False)
        print(f"AI plays: {ai_move}")
        
        env.step(ai_move)
        env.render()
        
        if env.is_checkmate():
            print("Checkmate!")
            break
        elif env.is_stalemate():
            print("Stalemate!")
            break

if __name__ == "__main__":
    play_chess()