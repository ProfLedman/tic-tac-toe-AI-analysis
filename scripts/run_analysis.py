#!/usr/bin/env python3
"""Complete analysis script with visualization."""

import pickle
import numpy as np
from pathlib import Path
from tictactoe.game import state_of_board, get_empty_cells, make_move
from tictactoe.policies import create_random_policy, convert_to_fixed_length
from tictactoe.training import train_q_learning
import matplotlib.pyplot as plt

def play_game(policyA, policyB):
    """Simulate a game between two policies."""
    board = (0, 0, 0, 0, 0, 0, 0, 0, 0)
    next_player = [0, 2, 1]
    player = 1
    
    while state_of_board(board) == -1:
        locations = get_empty_cells(board)
        current_policy = policyA if player == 1 else policyB
        
        # Get move probabilities
        if board in current_policy:
            probs = current_policy[board][locations]
            if np.sum(probs) > 0:
                probs = probs / np.sum(probs)
            else:
                probs = np.ones(len(locations)) / len(locations)
        else:
            probs = np.ones(len(locations)) / len(locations)
        
        chosen_location = np.random.choice(locations, p=probs)
        board = make_move(board, chosen_location, player)
        player = next_player[player]
    
    return state_of_board(board)

def run_simulation(policyA, policyB, num_games=500, description=""):
    """Run a simulation between two policies."""
    print(f"Running {description}...")
    results = [0, 0, 0]  # [Draws, O wins, X wins]
    
    for i in range(num_games):
        outcome = play_game(policyA, policyB)
        results[outcome] += 1
        
        if (i + 1) % 100 == 0:
            print(f"  Completed {i + 1}/{num_games} games")
    
    return results

def create_clear_performance_plot(results_dict, title):
    """Create a clear performance plot showing which policy wins each matchup."""
    strategies = list(results_dict.keys())
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Prepare data with policy-specific interpretation
    draws = [results_dict[s][0] for s in strategies]
    first_policy_wins = []  # Wins for the first policy (plays as O)
    second_policy_wins = []  # Wins for the second policy (plays as X)
    
    for strategy in strategies:
        draws_count, o_wins, x_wins = results_dict[strategy]
        first_policy_wins.append(o_wins)
        second_policy_wins.append(x_wins)
    
    # Create grouped bars
    x_pos = np.arange(len(strategies))
    width = 0.25
    
    # Create bars with policy-specific colors and labels
    bars1 = ax.bar(x_pos - width, first_policy_wins, width, 
                  label='First Policy Wins (plays as O)', color='#3498db', alpha=0.8)
    bars2 = ax.bar(x_pos, second_policy_wins, width, 
                  label='Second Policy Wins (plays as X)', color='#e74c3c', alpha=0.8)
    bars3 = ax.bar(x_pos + width, draws, width, 
                  label='Draws', color='#f39c12', alpha=0.8)
    
    # Customize plot
    ax.set_xlabel('Policy Matchups', fontweight='bold', fontsize=12)
    ax.set_ylabel('Number of Games (out of 500)', fontweight='bold', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    
    # Create clear matchup labels showing which policy is which
    matchup_labels = []
    for strategy in strategies:
        parts = strategy.split(' vs ')
        matchup_labels.append(f"{parts[0]} (O)\nvs\n{parts[1]} (X)")
    
    ax.set_xticklabels(matchup_labels, fontsize=10, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add value labels with explicit policy information
    for i, strategy in enumerate(strategies):
        draws_count, o_wins, x_wins = results_dict[strategy]
        parts = strategy.split(' vs ')
        
        # Label for first policy (O wins)
        ax.text(i - width, o_wins + 10, 
               f'{parts[0]}\n{o_wins} wins', 
               ha='center', va='bottom', fontsize=9, fontweight='bold',
               bbox=dict(facecolor='lightblue', alpha=0.7, boxstyle='round,pad=0.3'))
        
        # Label for second policy (X wins)
        ax.text(i, x_wins + 10, 
               f'{parts[1]}\n{x_wins} wins', 
               ha='center', va='bottom', fontsize=9, fontweight='bold',
               bbox=dict(facecolor='lightcoral', alpha=0.7, boxstyle='round,pad=0.3'))
        
        # Label for draws
        ax.text(i + width, draws_count + 10, 
               f'Draws\n{draws_count}', 
               ha='center', va='bottom', fontsize=9, fontweight='bold',
               bbox=dict(facecolor='lightyellow', alpha=0.7, boxstyle='round,pad=0.3'))
    
    # Add overall winner indicators
    for i, strategy in enumerate(strategies):
        draws_count, o_wins, x_wins = results_dict[strategy]
        parts = strategy.split(' vs ')
        
        if o_wins > x_wins:
            winner_text = f"Winner: {parts[0]}"
            ax.text(i - width, max(o_wins, x_wins) + 40, winner_text, 
                   ha='center', va='bottom', fontweight='bold', color='green', fontsize=11,
                   bbox=dict(facecolor='yellow', alpha=0.8, boxstyle='round,pad=0.5'))
        elif x_wins > o_wins:
            winner_text = f"Winner: {parts[1]}"
            ax.text(i, max(o_wins, x_wins) + 40, winner_text, 
                   ha='center', va='bottom', fontweight='bold', color='green', fontsize=11,
                   bbox=dict(facecolor='yellow', alpha=0.8, boxstyle='round,pad=0.5'))
        else:
            draw_text = "Draw"
            ax.text(i, max(o_wins, x_wins) + 40, draw_text, 
                   ha='center', va='bottom', fontweight='bold', color='orange', fontsize=11,
                   bbox=dict(facecolor='lightyellow', alpha=0.8, boxstyle='round,pad=0.5'))
    # Dynamically extend y-axis to accommodate winner labels
    max_height = max(first_policy_wins + second_policy_wins + draws)
    ax.set_ylim(0, max_height + 60)


    plt.tight_layout()
    return fig

def main():
    """Run the complete analysis pipeline."""
    print("Starting Tic-Tac-Toe AI Analysis")
    print("="*50)
    
    # Create results directory
    Path("results/plots").mkdir(parents=True, exist_ok=True)
    
    # Load policies
    print("1. Loading policies...")
    try:
        with open("data/perfectPolicy.p", "rb") as f:
            perfect_policy = pickle.load(f)
        print("Perfect policy loaded")
    except FileNotFoundError:
        print("perfectPolicy.p not found in data/ directory")
        return
    
    random_policy = create_random_policy(perfect_policy)
    fixed_perfect = convert_to_fixed_length(perfect_policy)
    fixed_random = convert_to_fixed_length(random_policy)
    
    # Train RL policy
    print("2. Training RL policy...")
    trained_policy = train_q_learning(fixed_random.copy(), episodes=1000)
    print("Training completed")
    
    # Run simulations
    print("3. Running simulations...")
    
    results_dict = {}
    
    # Run simulations
    results_dict["Random vs Perfect"] = run_simulation(
        fixed_random, fixed_perfect, 500, "Random vs Perfect"
    )
    
    results_dict["Trained vs Perfect"] = run_simulation(
        trained_policy, fixed_perfect, 500, "Trained vs Perfect"
    )
    
    results_dict["Trained vs Random"] = run_simulation(
        trained_policy, fixed_random, 500, "Trained vs Random"
    )
    
    # Create and save visualization
    print("4. Creating visualizations...")
    fig = create_clear_performance_plot(results_dict, "Tic-Tac-Toe AI Performance (500 games each)")
    plt.savefig("results/plots/performance_comparison.png", dpi=300, bbox_inches='tight')
    print("Plot saved to results/plots/performance_comparison.png")
    
    # Print results
    print("\n" + "="*50)
    print("RESULTS SUMMARY")
    print("="*50)
    
    for name, result in results_dict.items():
        draws, o_wins, x_wins = result
        total = sum(result)
        parts = name.split(' vs ')
        
        print(f"\n{name}:")
        print(f"  {parts[0]} wins (as O): {o_wins} ({o_wins/total*100:.1f}%)")
        print(f"  {parts[1]} wins (as X): {x_wins} ({x_wins/total*100:.1f}%)")
        print(f"  Draws: {draws} ({draws/total*100:.1f}%)")
        
        if o_wins > x_wins:
            print(f" WINNER: {parts[0]}")
        elif x_wins > o_wins:
            print(f" WINNER: {parts[1]}")
        else:
            print(f" DRAW")
    
    print("\n Analysis complete! Check results/plots/ folder for visualizations.")

if __name__ == "__main__":
    main()