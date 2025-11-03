import random
import copy
import numpy as np

# 導入你現有的遊戲邏輯檔案
import config
import Handler
import pieces
import shots

class TetrisEnv:
    def __init__(self):
        self.shot = None
        self.piece = None
        self.next_piece = None
        self.reset()

    def _get_state_properties(self, board_status):
        """從盤面狀態提取特徵向量"""
        # 1. 計算總高度 (Aggregate Height) 和每一行的方塊數
        heights = [0] * config.columns
        for x in range(config.columns):
            for y in range(config.rows):
                if board_status[y][x] == 2:
                    heights[x] = config.rows - y
                    break
        aggregate_height = sum(heights)

        # 2. 計算空洞數 (Holes)
        holes = 0
        for x in range(config.columns):
            if heights[x] > 0:
                for y in range(config.rows - heights[x] + 1, config.rows):
                    if board_status[y][x] == 0:
                        holes += 1

        # 3. 計算粗糙度 (Bumpiness)
        bumpiness = 0
        for x in range(config.columns - 1):
            bumpiness += abs(heights[x] - heights[x+1])
        
        # 4. 回傳特徵
        return np.array([aggregate_height, holes, bumpiness, sum(heights)])

    def get_possible_states(self):
        """
        獲取當前方塊所有可能的最終落點，以及對應的盤面狀態特徵
        """
        possible_states = {}
        # 遍歷所有旋轉
        for rotation in range(len(config.shapes[self.piece.shape])):
            # 遍歷所有可能的 x 座標
            for x in range(-2, config.columns + 1):
                # 模擬這一步
                sim_piece = copy.deepcopy(self.piece)
                sim_piece.rotation = rotation
                sim_piece.x = x

                # 檢查初始位置是否有效
                if not Handler.isValidPosition(self.shot, sim_piece):
                    continue

                # 模擬瞬降
                sim_shot = copy.deepcopy(self.shot)
                Handler.instantDrop(sim_shot, sim_piece)
                
                # 獲取這個模擬盤面的特徵
                state_properties = self._get_state_properties(sim_shot.status)
                possible_states[(x, rotation)] = state_properties

        return possible_states

    def step(self, action):
        """
        執行一個動作並回傳 (next_state, reward, done)
        action: 一個元組 (x, rotation)
        """
        x, rotation = action
        self.piece.x = x
        self.piece.rotation = rotation

        # 檢查移動是否合法 (理論上應該都是合法的)
        if not Handler.isValidPosition(self.shot, self.piece):
            return self._get_state_properties(self.shot.status), -100, True # 給予巨大懲罰

        # 執行瞬降
        Handler.instantDrop(self.shot, self.piece)
        
        # 結算與計算獎勵
        lines_cleared, _ = Handler.eliminateFilledRows(self.shot, self.piece)
        
        # 設計獎勵函數
        reward = 0
        if lines_cleared == 1:
            reward = 1
        elif lines_cleared == 2:
            reward = 4
        elif lines_cleared == 3:
            reward = 9
        elif lines_cleared >= 4:
            reward = 20 # Tetris!
        else:
            reward = -0.1 # 每放一個方塊給予微小懲罰，鼓勵消行

        # 產生新方塊
        self.piece = self.next_piece
        self.next_piece = pieces.Piece(*config.init_start, random.choice(list(config.shapes.keys())))

        # 檢查是否遊戲結束
        done = Handler.isDefeat(self.shot, self.piece)
        if done:
            reward = -50 # 遊戲結束給予巨大懲罰

        return self._get_state_properties(self.shot.status), reward, done

    def reset(self):
        """重置遊戲環境"""
        self.shot = shots.Shot()
        self.piece = pieces.Piece(*config.init_start, random.choice(list(config.shapes.keys())))
        self.next_piece = pieces.Piece(*config.init_start, random.choice(list(config.shapes.keys())))
        return self._get_state_properties(self.shot.status)
