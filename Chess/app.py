import pygame
import sys

# Initialize pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 800, 800
ROWS, COLS = 8, 8
SQUARE_SIZE = WIDTH // COLS

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
LIGHT_BROWN = (240, 217, 181)
DARK_BROWN = (181, 136, 99)

# Chess pieces (Unicode representation)
PIECES = {
    "white": {
        "K": "\u2654",  # King
        "Q": "\u2655",  # Queen
        "R": "\u2656",  # Rook
        "B": "\u2657",  # Bishop
        "N": "\u2658",  # Knight
        "P": "\u2659",  # Pawn
    },
    "black": {
        "K": "\u265A",  # King
        "Q": "\u265B",  # Queen
        "R": "\u265C",  # Rook
        "B": "\u265D",  # Bishop
        "N": "\u265E",  # Knight
        "P": "\u265F",  # Pawn
    },
}

# Initialize screen
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Chessboard")

# Font for chess pieces
font = pygame.font.SysFont("arial", SQUARE_SIZE - 10)

def draw_board():
    """Draws the chessboard."""
    for row in range(ROWS):
        for col in range(COLS):
            color = LIGHT_BROWN if (row + col) % 2 == 0 else DARK_BROWN
            pygame.draw.rect(screen, color, (col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))

def draw_pieces():
    """Places the chess pieces on the board."""
    # Initial positions for pieces
    initial_positions = [
        ["R", "N", "B", "Q", "K", "B", "N", "R"],
        ["P", "P", "P", "P", "P", "P", "P", "P"],
        ["", "", "", "", "", "", "", ""],
        ["", "", "", "", "", "", "", ""],
        ["", "", "", "", "", "", "", ""],
        ["", "", "", "", "", "", "", ""],
        ["P", "P", "P", "P", "P", "P", "P", "P"],
        ["R", "N", "B", "Q", "K", "B", "N", "R"],
    ]
    colors = ["black", "black", "", "", "", "", "white", "white"]

    for row in range(ROWS):
        for col in range(COLS):
            piece = initial_positions[row][col]
            if piece:
                color = colors[row]
                piece_text = PIECES[color][piece]
                text_surface = font.render(piece_text, True, BLACK if color == "white" else WHITE)
                screen.blit(text_surface, (col * SQUARE_SIZE + SQUARE_SIZE // 4, row * SQUARE_SIZE + SQUARE_SIZE // 8))

def main():
    """Main loop for the game."""
    clock = pygame.time.Clock()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # Draw the board and pieces
        draw_board()
        draw_pieces()

        # Update the display
        pygame.display.flip()
        clock.tick(60)

if __name__ == "__main__":
    main()