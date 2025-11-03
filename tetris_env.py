# tetris_env.py

import random
import copy
import numpy as np

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

    def _get_state_tensor(self):
        """
        生成代表完整遊戲狀態的 2D 圖像張量。
        使用兩個通道 (Channel):
        1. 已固定的方塊。
        2. 正在下落的方塊。
        """
        # 通道 1: 盤面固定的方塊 (0=空, 1=有方塊)
        board_channel = np.zeros((config.rows, config.columns), dtype=np.float32)
        board_channel[self.shot.status == 2] = 1.0
        
        # 通道 2: 當前下落的方塊
        piece_channel = np.zeros((config.rows, config.columns), dtype=np.float32)
        if self.piece:
            for y, x in Handler.getCellsAbsolutePosition(self.piece):
                if 0 <= y < config.rows and 0 <= x < config.columns:
                    piece_channel[y, x] = 1.0
                
        # 將兩個通道堆疊起來，形成 (Channels, Height, Width) 的形狀
        state_tensor = np.stack([board_channel, piece_channel], axis=0)
        return state_tensor

    def get_possible_next_states(self):
        """
        遍歷所有可能的動作，並回傳每個動作對應的「下一個狀態」圖像張量。
        """
        possible_states = {}
        if not self.piece:
            return possible_states

        for rotation in range(len(config.shapes[self.piece.shape])):
            for x in range(-2, config.columns + 1):
                sim_piece = copy.deepcopy(self.piece)
                sim_piece.rotation = rotation
                sim_piece.x = x

                if not Handler.isValidPosition(self.shot, sim_piece):
                    continue

                sim_shot = copy.deepcopy(self.shot)
                Handler.instantDrop(sim_shot, sim_piece)
                
                # 為這個模擬的盤面生成狀態張量
                board_channel = np.zeros((config.rows, config.columns), dtype=np.float32)
                board_channel[sim_shot.status == 2] = 1.0
                
                # 在這個模擬的下一步中，還沒有新的 "current_piece"，所以第二個通道是全零
                piece_channel = np.zeros((config.rows, config.columns), dtype=np.float32)
                
                next_state_tensor = np.stack([board_channel, piece_channel], axis=0)
                possible_states[(x, rotation)] = next_state_tensor

        return possible_states

    def step(self, action):
        """執行一個動作並回傳 (next_state_tensor, reward, done)"""
        if self.piece is None:
            return self._get_state_tensor(), -200, True

        x, rotation = action
        self.piece.x = x
        self.piece.rotation = rotation

        Handler.instantDrop(self.shot, self.piece)
        lines_cleared, _ = Handler.eliminateFilledRows(self.shot, self.piece)

        # 設計一個簡單但有效的獎勵
        reward = lines_cleared ** 2 * 10 # 獎勵消行的平方，大力鼓勵一次消多行
        
        # 產生新方塊
        self.piece = self.next_piece
        self.next_piece = pieces.Piece(*config.init_start, random.choice(list(config.shapes.keys())))

        done = Handler.isDefeat(self.shot, self.piece)
        if done:
            reward = -50 # 遊戲結束給予懲罰

        return self._get_state_tensor(), reward, done

    def reset(self):
        """重置遊戲環境"""
        self.shot = shots.Shot()
        self.piece = pieces.Piece(*config.init_start, random.choice(list(config.shapes.keys())))
        self.next_piece = pieces.Piece(*config.init_start, random.choice(list(config.shapes.keys())))
        return self._get_state_tensor()
