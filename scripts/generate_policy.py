import pickle
import numpy as np

def stateOfBoard(board):
    """Check the current state of the board"""
    winning_lines = [
        (0,1,2), (3,4,5), (6,7,8),  # rows
        (0,3,6), (1,4,7), (2,5,8),  # columns
        (0,4,8), (2,4,6)            # diagonals
    ]

    for a, b, c in winning_lines:
        if board[a] == board[b] == board[c] != 0:
            return board[a]  # 1 if O won, 2 if X won

    if 0 in board:
        return -1  # Game not finished

    return 0  # Draw

def getEmpty(board):
    """Get list of empty cells"""
    return [i for i in range(9) if board[i] == 0]

def makeMove(board, location, player):
    """Make a move on the board"""
    board = list(board)
    board[location] = player
    return tuple(board)

def get_optimal_moves(board, player):
    """Determine optimal moves for a given board state"""
    empty_cells = getEmpty(board)
    if not empty_cells:
        return []
        
    # Check for immediate win
    for move in empty_cells:
        new_board = makeMove(board, move, player)
        if stateOfBoard(new_board) == player:
            return [move]
    
    # Check for blocking opponent's win
    opponent = 3 - player
    for move in empty_cells:
        new_board = makeMove(board, move, opponent)
        if stateOfBoard(new_board) == opponent:
            return [move]
    
    # Strategic moves based on board position
    center = 4
    corners = [0, 2, 6, 8]
    edges = [1, 3, 5, 7]
    
    # Prefer center if available
    if center in empty_cells:
        return [center]
    
    # If opponent is in center, play corners
    if board[center] == opponent:
        available_corners = [c for c in corners if c in empty_cells]
        if available_corners:
            return available_corners
    
    # Otherwise, prefer corners over edges
    available_corners = [c for c in corners if c in empty_cells]
    if available_corners:
        return available_corners
    
    # Finally, play any available edge
    return empty_cells

def generate_perfect_policy():
    """Generate a perfect policy using strategic rules"""
    perfect_policy = {}
    
    # Generate policy for all possible board states
    def generate_all_states(board, player):
        state = stateOfBoard(board)
        if state != -1:  # Terminal state
            return
        
        if board in perfect_policy:
            return
        
        # Get optimal moves
        optimal_moves = get_optimal_moves(board, player)
        empty_cells = getEmpty(board)
        
        # Create probability distribution
        if not optimal_moves:
            # Fallback: if no optimal moves found, use uniform distribution
            prob_dist = np.ones(len(empty_cells)) / len(empty_cells)
        else:
            prob_dist = np.zeros(len(empty_cells))
            for i, move in enumerate(empty_cells):
                if move in optimal_moves:
                    prob_dist[i] = 1.0 / len(optimal_moves)
        
        # Normalize to ensure probabilities sum to 1
        prob_dist /= np.sum(prob_dist)
        
        perfect_policy[board] = prob_dist
        
        # Recursively generate states for all possible moves
        for move in empty_cells:
            new_board = makeMove(board, move, player)
            generate_all_states(new_board, 3 - player)  # Switch player
    
    # Start with empty board and player 1 (O)
    start_board = (0,0,0,0,0,0,0,0,0)
    generate_all_states(start_board, 1)
    
    return perfect_policy

# Generate and save the policy
print("Generating perfect policy...")
perfectPolicy = generate_perfect_policy()
pickle.dump(perfectPolicy, open("perfectPolicy.p", "wb"))

print(f"Perfect policy generated and saved as perfectPolicy.p")
print(f"Number of states in policy: {len(perfectPolicy)}")

# Create a simple random policy for comparison
randomPolicy = {}
for board in perfectPolicy:
    empty_cells = getEmpty(board)
    randomPolicy[board] = np.ones(len(empty_cells)) / len(empty_cells)

pickle.dump(randomPolicy, open("randomPolicy.p", "wb"))
print("Random policy generated and saved as randomPolicy.p")

# Validate that all probability distributions sum to 1
print("Validating policy...")
for board, probs in perfectPolicy.items():
    if abs(np.sum(probs) - 1.0) > 1e-10:
        print(f"Warning: Probabilities for board {board} sum to {np.sum(probs)}")
print("Validation complete.")