import sys
import os
import time
import random
import numpy as np

# Add tetris directory to path
sys.path.append(os.path.join(os.getcwd(), 'tetris'))

import config
import pieces
import shots
import Handler
from ai_weighted import WeightedAI

class RandomAgent:
    def find_best_move(self, shot, piece):
        # Try a few random moves and pick the first valid one
        # Or just return a random valid move
        valid_moves = []
        
        # Simple approach: Try 10 random attempts
        for _ in range(20):
            rot = random.randint(0, 3)
            col = random.randint(-2, config.columns)
            
            # Check validity
            # We need to clone piece to check
            test_piece = pieces.Piece(col, 0, piece.shape)
            test_piece.rotation = rot
            
            if Handler.isValidPosition(shot, test_piece):
                # Check if it can drop at least one block or is just valid at spawn
                # Actually, just returning a valid target (x, rot) is enough
                # The game loop handles the drop
                return (col, rot)
        
        # If no valid move found in random attempts, try to find ANY valid move
        # to be fair (don't die just because RNG was bad on valid moves)
        for rot in range(4):
            for x in range(config.columns):
                test_piece = pieces.Piece(x, 0, piece.shape)
                test_piece.rotation = rot
                if Handler.isValidPosition(shot, test_piece):
                    return (x, rot)
                    
        return None

def run_agent(agent_name, agent, num_games=10, max_lines=200):
    scores = []
    print(f"Testing {agent_name} over {num_games} games...")
    
    for i in range(num_games):
        shot = shots.Shot()
        # Ensure status is init
        if not hasattr(shot, 'status'):
             shot.status = [[0 for _ in range(config.columns)] for _ in range(config.rows)]
        
        game_over = False
        bag = []
        
        while not game_over:
            if not bag:
                bag = list(config.shapes.keys())
                random.shuffle(bag)
            
            shape = bag.pop()
            piece = pieces.Piece(5, 0, shape)
            
            if not Handler.isValidPosition(shot, piece):
                game_over = True
                break
            
            move = agent.find_best_move(shot, piece)
            
            if move:
                tx, trot = move
                piece.x = tx
                piece.rotation = trot
                
                # Drop
                Handler.instantDrop(shot, piece)
                
                # Clear lines & Update Score
                lines, _ = Handler.eliminateFilledRows(shot, piece)
                
                if shot.line_count >= max_lines:
                    game_over = True
            else:
                game_over = True
                
        scores.append(shot.score)
        # print(f"  Game {i+1}: {shot.score}")
        
    avg_score = np.mean(scores)
    std_score = np.std(scores)
    
    return avg_score, std_score

def main():
    # 1. Weighted AI
    weighted_ai = WeightedAI()
    w_avg, w_std = run_agent("Weighted Heuristic AI", weighted_ai, num_games=2, max_lines=10)
    
    # 2. Random Agent
    random_agent = RandomAgent()
    r_avg, r_std = run_agent("Random Agent", random_agent, num_games=5, max_lines=10)
    
    print("\n" + "="*50)
    print("FINAL RESULTS FOR REPORT")
    print("="*50)
    print(f"Weighted Heuristic AI: Avg={w_avg:.2f}, Std={w_std:.2f}")
    print(f"Random Agent:          Avg={r_avg:.2f}, Std={r_std:.2f}")
    print("="*50)

if __name__ == "__main__":
    main()
