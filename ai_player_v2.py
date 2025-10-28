# ai_player_v2.py (修正後的版本)
import json
import copy
import random
import config
import Handler

DEFAULT_WEIGHTS = {
    'lines_cleared': 0.760666,
    'aggregate_height': -0.510066,
    'holes': -0.35663,
    'bumpiness': -0.184483
}

class AIPlayer:
    """
    一個擁有獨立權重大腦的 AI 玩家類別。
    """
    def __init__(self, weights=None):
        """
        初始化 AI 玩家。
        :param weights: 一個包含權重值的字典。如果為 None，則使用預設權重。
        """
        # --- 核心修正：確保每個實例都有獨立的大腦 ---
        if weights is None:
            # 使用預設權重的「複本」，而不是參考
            self.weights = DEFAULT_WEIGHTS.copy()
        else:
            # 使用傳入權重的「複本」，防止外部修改影響內部狀態
            self.weights = weights.copy()
        
        # 確保所有必要的權重都存在，以防傳入的權重不完整
        for key in DEFAULT_WEIGHTS:
            if key not in self.weights:
                self.weights[key] = DEFAULT_WEIGHTS[key]

    @classmethod
    def from_file(cls, weights_path):
        """
        一個工廠方法，從 .json 檔案創建 AIPlayer 實例。
        """
        try:
            with open(weights_path, 'r') as f:
                loaded_weights = json.load(f)
            print(f"成功從 {weights_path} 載入權重。")
            return cls(weights=loaded_weights)
        except FileNotFoundError:
            print(f"警告：在 {weights_path} 找不到權重檔案。將創建一個使用預設權重的 AI。")
            return cls() # 返回一個使用預設權重的實例

    # ... evaluate_board 和 find_best_move 函式保持不變 ...
    def evaluate_board(self, board_status, lines_cleared):
        heights = [0] * config.columns
        for x in range(config.columns):
            for y in range(config.rows):
                if board_status[y][x] == 2:
                    heights[x] = config.rows - y
                    break
        aggregate_height = sum(heights)

        holes = 0
        for x in range(config.columns):
            if heights[x] > 0:
                for y in range(config.rows - heights[x] + 1, config.rows):
                    if board_status[y][x] == 0:
                        holes += 1
        
        bumpiness = 0
        for x in range(config.columns - 1):
            bumpiness += abs(heights[x] - heights[x+1])

        score = (self.weights['lines_cleared'] * lines_cleared +
                 self.weights['aggregate_height'] * aggregate_height +
                 self.weights['holes'] * holes +
                 self.weights['bumpiness'] * bumpiness)
        
        return score

    def find_best_move(self, shot, piece):
        best_score = -float('inf')
        best_move = None

        for rotation in range(len(config.shapes[piece.shape])):
            sim_piece_template = copy.deepcopy(piece)
            sim_piece_template.rotation = rotation

            for x in range(-2, config.columns + 1):
                sim_shot = copy.deepcopy(shot)
                sim_piece = copy.deepcopy(sim_piece_template)
                sim_piece.x = x

                if not Handler.isValidPosition(sim_shot, sim_piece):
                    continue

                Handler.instantDrop(sim_shot, sim_piece)

                lines_cleared, _ = Handler.eliminateFilledRows(sim_shot, sim_piece)

                if Handler.isDefeat(sim_shot, sim_piece):
                    score = -float('inf')
                else:
                    score = self.evaluate_board(sim_shot.status, lines_cleared)

                if score > best_score:
                    best_score = score
                    best_move = (x, rotation)
        
        return best_move
