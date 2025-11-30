import torch
import torch.nn as nn
import numpy as np
import copy
import config
import Handler

class TetrisNetwork(nn.Module):
    def __init__(self):
        super(TetrisNetwork, self).__init__()
        # 輸入 8 個特徵 -> 輸出 1 個評分 (Value)
        self.fc = nn.Sequential(
            nn.Linear(8, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.fc(x)

class AIPlayerNN:
    def __init__(self, model_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TetrisNetwork().to(self.device)
        
        if model_path:
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                self.model.eval() # 評估模式
                print(f"DQN: Loaded model from {model_path}")
            except Exception as e:
                print(f"DQN: Load failed {e}, starting fresh.")
        
    def evaluate_board(self, features):
        """輸入特徵，回傳模型預測的分數"""
        with torch.no_grad():
            x = torch.tensor(features, dtype=torch.float32).to(self.device)
            return self.model(x).item()

    def find_best_move(self, shot, piece, heuristic_ai_helper):
        """
        在遊戲中使用 (Inference Mode)。
        需要傳入 heuristic_ai_helper 是因為我們需要用它的 get_tetris_features_v8 函式來提取特徵。
        """
        best_score = -float('inf')
        best_move = None
        
        num_rotations = len(config.shapes[piece.shape])
        if piece.shape == 'O': num_rotations = 1
        elif piece.shape in ['S', 'Z', 'I']: num_rotations = 2
        
        for rot in range(num_rotations):
            test_piece = copy.deepcopy(piece)
            test_piece.rotation = rot
            for x in range(-2, config.columns + 1):
                test_piece.x = x
                if not Handler.isValidPosition(shot, test_piece): continue
                
                # 模擬下落
                sim_status = [row[:] for row in shot.status]
                sim_y = test_piece.y
                while True:
                    test_piece.y = sim_y
                    if Handler._can_move(shot, test_piece, 0, 1): sim_y += 1
                    else: break
                test_piece.y = sim_y
                
                # 模擬盤面與消行
                landing_height = config.rows - test_piece.y
                for cy, cx in Handler.getCellsAbsolutePosition(test_piece):
                    if 0 <= cy < config.rows and 0 <= cx < config.columns:
                        sim_status[cy][cx] = 2
                
                lines = 0
                new_board = []
                for r in range(config.rows):
                    if all(sim_status[r][c] == 2 for c in range(config.columns)): lines += 1
                    else: new_board.append(sim_status[r])
                for _ in range(lines): new_board.insert(0, [0]*config.columns)
                
                # --- 關鍵：讓學生看特徵 ---
                feats = heuristic_ai_helper.get_tetris_features_v8(new_board, landing_height, lines)
                
                # 學生打分數
                score = self.evaluate_board(feats)
                
                if score > best_score:
                    best_score = score
                    best_move = (x, rot)
                
                test_piece.y = piece.y # Reset
                
        return best_move