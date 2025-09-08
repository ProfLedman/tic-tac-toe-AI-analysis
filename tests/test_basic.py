#!/usr/bin/env python3
"""
Basic tests for Tic-Tac-Toe game functionality.
Efficient, lightweight, and comprehensive without over-engineering.
"""

import numpy as np
from tictactoe.game import state_of_board, get_empty_cells, make_move

def test_empty_board():
    """Test empty board state and empty cell detection."""
    board = (0, 0, 0, 0, 0, 0, 0, 0, 0)
    assert state_of_board(board) == -1, "Empty board should return -1 (game ongoing)"
    assert get_empty_cells(board) == list(range(9)), "All cells should be empty"

def test_row_win():
    """Test winning by row."""
    # Player 1 wins with top row
    board = (1, 1, 1, 0, 0, 0, 0, 0, 0)
    assert state_of_board(board) == 1, "Player 1 should win with top row"
    
    # Player 2 wins with middle row  
    board = (0, 0, 0, 2, 2, 2, 0, 0, 0)
    assert state_of_board(board) == 2, "Player 2 should win with middle row"

def test_column_win():
    """Test winning by column."""
    # Player 1 wins with first column
    board = (1, 0, 0, 1, 0, 0, 1, 0, 0)
    assert state_of_board(board) == 1, "Player 1 should win with first column"

def test_diagonal_win():
    """Test winning by diagonal."""
    # Player 2 wins with main diagonal
    board = (2, 0, 0, 0, 2, 0, 0, 0, 2)
    assert state_of_board(board) == 2, "Player 2 should win with main diagonal"
    
    # Player 1 wins with anti-diagonal
    board = (0, 0, 1, 0, 1, 0, 1, 0, 0)
    assert state_of_board(board) == 1, "Player 1 should win with anti-diagonal"

def test_draw_condition():
    """Test draw game state."""
    board = (1, 2, 1, 1, 2, 1, 2, 1, 2)  # Full board, no winner
    assert state_of_board(board) == 0, "Full board with no winner should be a draw"

def test_make_move():
    """Test making moves and immutability."""
    original_board = (0, 0, 0, 0, 0, 0, 0, 0, 0)
    
    # Test move placement
    new_board = make_move(original_board, 4, 1)
    assert new_board[4] == 1, "Move should place player 1 at index 4"
    
    # Test immutability - original board unchanged
    assert original_board == (0, 0, 0, 0, 0, 0, 0, 0, 0), "Original board should remain unchanged"
    
    # Test multiple moves
    newer_board = make_move(new_board, 0, 2)
    assert newer_board[0] == 2, "Second move should place player 2 at index 0"
    assert newer_board[4] == 1, "First move should still be present"

def test_get_empty_cells():
    """Test empty cell detection in various board states."""
    # Partially filled board
    board = (1, 0, 2, 0, 1, 0, 0, 0, 2)
    empty_cells = get_empty_cells(board)
    expected_empty = [1, 3, 5, 6, 7]  # Indices of empty cells
    assert empty_cells == expected_empty, f"Expected {expected_empty}, got {empty_cells}"
    
    # Almost full board
    board = (1, 2, 1, 2, 1, 2, 0, 2, 1)
    assert get_empty_cells(board) == [6], "Only cell 6 should be empty"

def test_ongoing_game():
    """Test ongoing game states (no winner yet)."""
    # Game in progress
    board = (1, 0, 0, 0, 2, 0, 0, 0, 1)
    assert state_of_board(board) == -1, "Game should still be ongoing"
    
    # Another ongoing game
    board = (1, 2, 0, 0, 1, 0, 0, 0, 0)
    assert state_of_board(board) == -1, "Game should still be ongoing"

def run_all_tests():
    """Run all test functions with clear progress indication."""
    tests = [
        test_empty_board,
        test_row_win,
        test_column_win, 
        test_diagonal_win,
        test_draw_condition,
        test_make_move,
        test_get_empty_cells,
        test_ongoing_game
    ]
    
    print("Running Tic-Tac-Toe functionality tests...")
    
    for test in tests:
        test_name = test.__name__
        try:
            test()
            print(f":) {test_name}")
        except AssertionError as e:
            print(f"XOX {test_name} FAILED: {e}")
            raise
    
    print("All tests passed! Basic game functionality is working correctly.")

if __name__ == "__main__":
    run_all_tests()