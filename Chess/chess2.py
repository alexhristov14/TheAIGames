import numpy as np

class ChessEnv:
    def __init__(self):
        self.board = np.zeros((8, 8), dtype=int)
        self.piece_map = {
            1: "♟", 2: "♜", 3: "♞", 4: "♝", 5: "♛", 6: "♚",
            -1: "♙", -2: "♖", -3: "♘", -4: "♗", -5: "♕", -6: "♔"
        }
        self.notation_to_piece = {"N": 3, "K": 6, "Q": 5, "R": 2, "B": 4, "P": 1}
        self.piece_to_notation = {1: "P", 2: "R", 3: "N", 4: "B", 5: "Q", 6: "K"}
        self.how_pieces_move = {
            "N": [(2, 1), (2, -1), (-2, 1), (-2, -1), (1, 2), (1, -2), (-1, 2), (-1, -2)],
            "R": [(1, 0), (-1, 0), (0, 1), (0, -1)],
            "B": [(1, 1), (-1, -1), (-1, 1), (1, -1)],
            "K": [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (-1, 1), (1, -1)],
            "P": [(-1, 0), (-2, 0), (-1, -1), (-1, 1)]
        }
        self.chess_map = self.create_chess_map()
        self.current_player = "w"
        self.reset()

    def reset(self):
        """Sets up the initial board position."""
        self.board.fill(0)
        self.board[1, :] = -1  # Black pawns
        self.board[6, :] = 1   # White pawns
        self.board[0, :] = [-2, -3, -4, -5, -6, -4, -3, -2]  # Black pieces
        self.board[7, :] = [2, 3, 4, 5, 6, 4, 3, 2]          # White pieces

    def create_chess_map(self):
        """Creates a mapping between chess notation and board indices."""
        chess_map = {}
        for row in range(8):
            for col in range(8):
                square = f"{chr(col + 97)}{8 - row}"
                chess_map[square] = (row, col)
        return chess_map

    def step(self, action):
        """Processes a move in algebraic notation."""
        try:
            piece, takes, next_state, castling = self.process_notation(action)
            if castling:
                return
            start_row, start_col = self.get_piece_starting_location(piece, next_state, takes)
            next_row, next_col = self.chess_map[next_state]
            valid_moves = self.valid_moves(piece, start_row, start_col, takes)
            if (next_row, next_col) not in valid_moves:
                raise ValueError("Invalid move.")
            # Move the piece
            self.board[next_row, next_col] = self.board[start_row, start_col]
            self.board[start_row, start_col] = 0
            self.change_current_player()
        except Exception as e:
            print("Error:", e)

    def process_notation(self, action):
        """Parses algebraic notation to extract move details."""
        if action == "O-O" or action == "O-O-O":
            return self.process_castling(action)

        notation_data = list(action)
        takes = "x" in action
        if notation_data[0] in "abcdefgh":
            piece = "P"
        else:
            piece = notation_data[0]
        next_state = action[-2:]  # Destination square
        return piece, takes, next_state, False

    def process_castling(self, action):
        """Handles castling moves."""
        if action == "O-O":  # Short Castle
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
        elif action == "O-O-O":  # Long Castle
            if self.current_player == "w":
                self.board[7, 2] = 6
                self.board[7, 3] = 2
                self.board[7, 0] = 0
                self.board[7, 4] = 0
            else:
                self.board[0, 2] = -6
                self.board[0, 3] = -2
                self.board[0, 0] = 0
                self.board[0, 4] = 0
        return None, False, None, True

    def valid_moves(self, piece, start_row, start_col, takes=False):
        """Generates valid moves for a piece."""
        moves = []
        directions = self.how_pieces_move.get(piece)
        if not directions:
            return moves

        for dr, dc in directions:
            for i in range(1, 8):
                next_row, next_col = start_row + dr * i, start_col + dc * i
                if 0 <= next_row < 8 and 0 <= next_col < 8:
                    if takes and self.board[next_row, next_col] == 0:
                        continue
                    if not takes and self.board[next_row, next_col] != 0:
                        break
                    moves.append((next_row, next_col))
                    if self.board[next_row, next_col] != 0:
                        break
                else:
                    break
        return moves

    def get_piece_starting_location(self, piece, next_state, takes=False):
        """Finds the starting location of a piece."""
        target_row, target_col = self.chess_map[next_state]
        for dr, dc in self.how_pieces_move.get(piece, []):
            start_row, start_col = target_row - dr, target_col - dc
            if 0 <= start_row < 8 and 0 <= start_col < 8:
                if self.board[start_row, start_col] == self.notation_to_piece[piece]:
                    return start_row, start_col
        return None, None

    def render(self):
        """Prints the current board state."""
        symbols = np.vectorize(lambda x: self.piece_map.get(x, "·"))(self.board)
        for row in symbols:
            print(" ".join(row))
        print('\n')

    def change_current_player(self):
        """Switches the current player."""
        self.current_player = "b" if self.current_player == "w" else "w"


# Example Usage
env = ChessEnv()
env.render()

# Example move
env.step("e2e4")
env.render()
