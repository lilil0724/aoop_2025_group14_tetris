import random
import numpy as np
import torch
import copy
import config
from pieces import Piece

# 定義簡單的 Reward
REWARD_CLEAR_LINES = [0, 100, 300, 600, 1000] # 0, 1, 2, 3, 4 lines (大幅增加!)
REWARD_GAME_OVER = -200      # 稍微降低死亡懲罰，鼓勵冒險
REWARD_SURVIVE = 1           # 活著只給一點點糖吃
REWARD_HOLE_PENALTY = -5     # 空洞懲罰加重，讓它學會鋪平
REWARD_HEIGHT_PENALTY = -2   # 高度懲罰
REWARD_BUMPINESS_PENALTY = -1 # (新增) 表面不平整的懲罰

class TetrisEnv:
    def __init__(self):
        self.reset()

    def reset(self):
        # 0=Empty, 1=Moving, 2=Fixed
        self.board = np.zeros((config.rows, config.columns), dtype=int)
        self.current_piece = self._get_random_piece()
        self.score = 0
        self.game_over = False
        self.steps = 0
        return self._get_state_tensor(self.board, self.current_piece)

    def _get_random_piece(self):
        shape = random.choice(list(config.shapes.keys()))
        # 預設從上方中間生成
        return Piece(5, 0, shape)

    def step(self, action):
        """
        Action 是一個 tuple: (target_x, target_rotation)
        這是一個簡化的環境，我們假設 AI 可以直接把方塊放到目標位置 (Instant Drop)
        """
        if self.game_over:
            return 0, True

        target_x, target_rot = action
        
        # 複製一個暫時的 piece 來操作
        piece = copy.deepcopy(self.current_piece)
        piece.rotation = target_rot
        piece.x = target_x
        
        # 1. 模擬落地 (Hard Drop)
        # 這裡簡化計算，直接找到該 x 下最低的合法 y
        final_y = piece.y
        while self._is_valid_position(self.board, piece, adj_x=0, adj_y=1):
            piece.y += 1
            final_y = piece.y
            
        # 2. 固定方塊
        self._lock_piece(self.board, piece)
        
        # 3. 檢查消行
        cleared_lines = self._clear_lines(self.board)
        
        reward = REWARD_SURVIVE
        reward += REWARD_CLEAR_LINES[cleared_lines]
        
        # 額外獎勵: 連擊 (Combo) - 鼓勵連續消行
        # 假設 self.combo 在 reset 時初始化為 0，每次消行 +1，沒消行歸零
        if cleared_lines > 0:
            self.combo += 1
            reward += (self.combo * 50) # 連擊獎勵
        else:
            self.combo = 0

        # 狀態懲罰計算
        holes = self._count_holes(self.board)
        height = self._get_aggregate_height(self.board)
        bumpiness = self._get_bumpiness(self.board) # 需實作此 helper function
        
        reward += (holes * REWARD_HOLE_PENALTY)
        reward += (height * REWARD_HEIGHT_PENALTY)
        reward += (bumpiness * REWARD_BUMPINESS_PENALTY)

        # 5. 檢查是否 Game Over (如果生出的新方塊一出來就撞到)
        self.current_piece = self._get_random_piece()
        if not self._is_valid_position(self.board, self.current_piece):
            self.game_over = True
            reward += REWARD_GAME_OVER
            
        return reward, self.game_over

    def get_possible_next_states(self):
        """
        回傳所有可能的下一步狀態圖像
        Returns: 
            states: dictionary {(x, rot): state_tensor_numpy}
        """
        states = {}
        piece = self.current_piece
        
        # 遍歷所有旋轉
        num_rotations = len(config.shapes[piece.shape])
        for rot in range(num_rotations):
            # 遍歷所有可能的 x
            # 為了效能，我們做簡單邊界檢查 (-2 到 columns)
            for x in range(-2, config.columns + 1):
                
                # 模擬這個位置
                sim_piece = copy.deepcopy(piece)
                sim_piece.rotation = rot
                sim_piece.x = x
                sim_piece.y = 0 # 從頂部開始
                
                # 如果一開始這個 (x, rot) 就不合法 (例如卡在牆壁裡)，跳過
                if not self._is_valid_position(self.board, sim_piece):
                    continue
                
                # 模擬下落到底
                while self._is_valid_position(self.board, sim_piece, adj_x=0, adj_y=1):
                    sim_piece.y += 1
                
                # 產生對應的 State Tensor
                # 這裡我們產生 "假設落地後" 的盤面狀態給 AI 評估
                temp_board = self.board.copy()
                self._lock_piece(temp_board, sim_piece)
                
                # 轉換成 Tensor 格式 (Channels, H, W)
                tensor_np = self._board_to_tensor_numpy(temp_board, sim_piece)
                
                states[(x, rot)] = tensor_np
                
        return states

    def _get_state_tensor(self, board, piece):
        return torch.FloatTensor(self._board_to_tensor_numpy(board, piece))

    def _board_to_tensor_numpy(self, board, piece):
        # Channel 0: Fixed Board (0 or 1)
        board_layer = (board == 2).astype(np.float32)
        
        # Channel 1: Current Piece (0 or 1)
        piece_layer = np.zeros_like(board_layer)
        # 這裡不畫出當前 piece，因為 get_possible_next_states 
        # 比較的是「落地後的結果」，所以 piece 已經融合進 board_layer 了
        # 但為了維持輸入格式一致，我們保留這個 channel，或者你可以畫上「預測落點」
        
        return np.stack([board_layer, piece_layer], axis=0)

    def _is_valid_position(self, board, piece, adj_x=0, adj_y=0):
        """檢查方塊是否在合法位置"""
        for y, x in self._get_piece_coords(piece):
            nx, ny = x + adj_x, y + adj_y
            # 邊界檢查
            if nx < 0 or nx >= config.columns or ny >= config.rows:
                return False
            # 碰撞檢查 (忽略 y < 0 的上方區域)
            if ny >= 0 and board[ny][nx] == 2:
                return False
        return True

    def _lock_piece(self, board, piece):
        for y, x in self._get_piece_coords(piece):
            if 0 <= y < config.rows and 0 <= x < config.columns:
                board[y][x] = 2

    def _get_piece_coords(self, piece):
        """取得方塊絕對座標"""
        # 假設 config.shapes 格式為 [ [(y,x), (y,x)...], [旋轉2]... ]
        # 或者如果你的 config 是二維陣列格式，這裡需要對應調整
        # 這裡假設你的 pieces.py 裡的 getCells 邏輯
        shape_template = config.shapes[piece.shape][piece.rotation % len(config.shapes[piece.shape])]
        return [(y + piece.y, x + piece.x) for y, x in shape_template]

    def _clear_lines(self, board):
        lines_to_clear = [i for i, row in enumerate(board) if all(cell == 2 for cell in row)]
        for i in lines_to_clear:
            del_row = np.zeros(config.columns, dtype=int)
            board[:] = np.vstack([del_row, np.delete(board, i, axis=0)])
        return len(lines_to_clear)
        
    def _count_holes(self, board):
        holes = 0
        for col in range(config.columns):
            is_blocked = False
            for row in range(config.rows):
                if board[row][col] == 2:
                    is_blocked = True
                elif is_blocked and board[row][col] == 0:
                    holes += 1
        return holes

    def _get_aggregate_height(self, board):
        total_height = 0
        for col in range(config.columns):
            for row in range(config.rows):
                if board[row][col] == 2:
                    total_height += (config.rows - row)
                    break
        return total_height

    def _get_bumpiness(self, board):
        total_bumpiness = 0
        max_heights = []
        for col in range(config.columns):
            h = 0
            for row in range(config.rows):
                if board[row][col] == 2:
                    h = config.rows - row
                    break
            max_heights.append(h)
        
        for i in range(len(max_heights) - 1):
            total_bumpiness += abs(max_heights[i] - max_heights[i+1])
        return total_bumpiness
