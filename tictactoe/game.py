"""Core Tic-Tac-Toe game mechanics."""

import numpy as np
from typing import Tuple, List, Optional

def create_board() -> List[int]:
    """Create an empty 3x3 Tic-Tac-Toe board."""
    return [0] * 9

def state_of_board(board: Tuple[int, ...]) -> int:
    """Check the current state of the board."""
    winning_lines = [
        (0, 1, 2), (3, 4, 5), (6, 7, 8),  # rows
        (0, 3, 6), (1, 4, 7), (2, 5, 8),  # columns
        (0, 4, 8), (2, 4, 6)              # diagonals
    ]
    
    for a, b, c in winning_lines:
        if board[a] == board[b] == board[c] != 0:
            return board[a]
    
    return -1 if 0 in board else 0

def get_empty_cells(board: Tuple[int, ...]) -> List[int]:
    """Return list of empty cell indices."""
    return [i for i in range(9) if board[i] == 0]

def make_move(board: Tuple[int, ...], location: int, player: int) -> Tuple[int, ...]:
    """Apply a move to the board."""
    board_list = list(board)
    board_list[location] = player
    return tuple(board_list)

def display_board(board: Tuple[int, ...]) -> str:
    """Return a string representation of the board."""
    symbols = ['.', 'O', 'X']
    lines = []
    for i in range(0, 9, 3):
        line = ' | '.join(symbols[board[i + j]] for j in range(3))
        lines.append(line)
    return '\n--+---+--\n'.join(lines)