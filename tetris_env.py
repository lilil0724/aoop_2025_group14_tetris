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
                # 模擬瞬降
                sim_shot = copy.deepcopy(self.shot)
                # 檢查初始位置是否有效
                if not Handler.isValidPosition(sim_shot, sim_piece): # <--- 改成呼叫我們剛剛在 Handler 中新增的函式
                    continue

                
                Handler.instantDrop(sim_shot, sim_piece)
                
                # 獲取這個模擬盤面的特徵
                state_properties = self._get_state_properties(sim_shot.status)
                possible_states[(x, rotation)] = state_properties

        return possible_states

    def step(self, action):
        """
        執行一個動作並回傳 (next_state, reward, done)
        (修改後的版本，包含更豐富的獎勵)
        """
        # 取得執行動作前的盤面狀態
        old_properties = self._get_state_properties(self.shot.status)
        old_height = old_properties[0]
        old_holes = old_properties[1]
        old_bumpiness = old_properties[2]

        # ... (執行瞬降等操作) ...
        Handler.instantDrop(self.shot, self.piece)
        lines_cleared, _ = Handler.eliminateFilledRows(self.shot, self.piece)

        # 取得執行動作後的盤面狀態
        new_properties = self._get_state_properties(self.shot.status)
        new_height = new_properties[0]
        new_holes = new_properties[1]
        new_bumpiness = new_properties[2]


        # --- 設計新的獎勵函數 (Reward Shaping) ---
        reward = 0
        
        # 主要獎勵：消行
        if lines_cleared == 1:
            reward += 10
        elif lines_cleared == 2:
            reward += 30
        elif lines_cleared == 3:
            reward += 60
        elif lines_cleared >= 4:
            reward += 120  # 大力獎勵 Tetris

        height_increase = new_height - old_height
        # 只有在沒有消行的情況下，高度增加才應該被視為一個嚴重的負面行為
        if lines_cleared == 0 and height_increase > 0:
            reward -= height_increase * 0.5  # 施加一個與高度增量相關的懲罰
        # 輔助獎勵：鼓勵好的盤面狀態
        
        # 1. 懲罰製造新的空洞
        if new_holes > old_holes:
            reward -= (new_holes - old_holes) * 2
        
        # 2. 獎勵填補空洞
        if new_holes < old_holes:
            reward += (old_holes - new_holes) * 4
            
        # 3. 懲罰增加的粗糙度
        if new_bumpiness > old_bumpiness:
            reward -= (new_bumpiness - old_bumpiness) * 0.1

        
        # 產生新方塊
        self.piece = self.next_piece
        self.next_piece = pieces.Piece(*config.init_start, random.choice(list(config.shapes.keys())))

        done = Handler.isDefeat(self.shot, self.piece)
        if done:
            reward = -200 # 遊戲結束是最大的失敗

        return self._get_state_properties(self.shot.status), reward, done

    def reset(self):
        """重置遊戲環境"""
        self.shot = shots.Shot()
        self.piece = pieces.Piece(*config.init_start, random.choice(list(config.shapes.keys())))
        self.next_piece = pieces.Piece(*config.init_start, random.choice(list(config.shapes.keys())))
        return self._get_state_properties(self.shot.status)
