"""Policy implementations for Tic-Tac-Toe."""

import numpy as np
from typing import Dict, Tuple
from .game import get_empty_cells

Policy = Dict[Tuple[int, ...], np.ndarray]

def create_random_policy(perfect_policy: Policy) -> Policy:
    """Create a random policy based on perfect policy structure."""
    random_policy = {}
    for board, probs in perfect_policy.items():
        n_moves = len(probs)
        random_policy[board] = np.ones(n_moves) / n_moves
    return random_policy

def convert_to_fixed_length(policy: Policy) -> Policy:
    """Convert variable-length policy to fixed-length."""
    fixed_policy = {}
    for board, probs in policy.items():
        fixed_probs = np.zeros(9)
        empty_cells = get_empty_cells(board)
        for i, cell in enumerate(empty_cells):
            fixed_probs[cell] = probs[i]
        fixed_policy[board] = fixed_probs
    return fixed_policy

def get_valid_move_probs(board: Tuple[int, ...], policy: Policy) -> np.ndarray:
    """Get valid move probabilities for current board state."""
    empty_cells = get_empty_cells(board)
    
    if board not in policy:
        return np.ones(len(empty_cells)) / len(empty_cells)
    
    probs = policy[board][empty_cells]
    if np.sum(probs) > 0:
        return probs / np.sum(probs)
    return np.ones(len(empty_cells)) / len(empty_cells)