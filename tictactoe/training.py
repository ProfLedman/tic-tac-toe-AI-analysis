"""Reinforcement learning training functions."""

import numpy as np
from collections import defaultdict
from typing import Dict, Tuple
from .game import state_of_board, get_empty_cells, make_move
from .policies import Policy
from .policies import Policy, get_valid_move_probs  

def train_q_learning(
    policy: Policy,
    episodes: int = 10000,
    alpha: float = 0.1,
    gamma: float = 0.9
) -> Policy:
    """Train policy using Q-learning."""
    Q = defaultdict(lambda: np.zeros(9))
    
    for episode in range(episodes):
        board = (0, 0, 0, 0, 0, 0, 0, 0, 0)
        player = 1
        history = []

        # Play episode
        while state_of_board(board) == -1:
            empty_cells = get_empty_cells(board)
            probs = get_valid_move_probs(board, policy)
            action = np.random.choice(empty_cells, p=probs)
            
            history.append((board, action, player))
            board = make_move(board, action, player)
            player = 3 - player

        # Update Q-values
        outcome = state_of_board(board)
        for state, action, p in history:
            reward = 1 if outcome == p else (-1 if outcome == 3 - p else 0)
            Q[state][action] += alpha * (reward - Q[state][action])
            
            # Update policy
            empty_cells = get_empty_cells(state)
            if empty_cells:
                logits = Q[state][empty_cells]
                exp_logits = np.exp(np.clip(logits, -10, 10))
                policy[state] = np.zeros(9)
                policy[state][empty_cells] = exp_logits / np.sum(exp_logits)
    
    return policy