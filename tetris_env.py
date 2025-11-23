import numpy as np
import config
import copy
import random
import torch
from pieces import Piece

# --- 獎勵參數設定 (瘋狂加碼版) ---
REWARD_CLEAR_LINES = [0, 50, 200, 800, 10000] 
REWARD_HOLE_PENALTY = -1.0      
REWARD_HEIGHT_PENALTY = -0.05   
REWARD_BUMPINESS_PENALTY = -0.05 
REWARD_SURVIVE = 1              
REWARD_GAME_OVER = -500

class TetrisEnv:
    def __init__(self):
        self.reset()

    def reset(self):
        self.board = np.zeros((config.rows, config.columns), dtype=int)
        self.current_piece = self._get_random_piece()
        self.score = 0
        self.line_count = 0  
        self.combo = 0       
        self.game_over = False
        self.steps = 0
        self.last_cleared_lines = 0 # 新增
        return self._get_features(self.board)

    def _get_random_piece(self):
        shape = random.choice(list(config.shapes.keys()))
        return Piece(5, 0, shape)

    def step(self, action):
        if self.game_over:
            return 0, True

        target_x, target_rot = action
        
        piece = copy.deepcopy(self.current_piece)
        piece.rotation = target_rot
        piece.x = target_x
        
        # 1. 模擬落地
        while self._is_valid_position(self.board, piece, adj_x=0, adj_y=1):
            piece.y += 1
            
        # 2. 固定方塊
        self._lock_piece(self.board, piece)
        
        # 3. 檢查消行
        cleared_lines = self._clear_lines(self.board)
        self.last_cleared_lines = cleared_lines # 記錄下來
        self.line_count += cleared_lines
        
        # 4. 計算獎勵
        reward = REWARD_SURVIVE
        reward += REWARD_CLEAR_LINES[cleared_lines]
        
        if cleared_lines > 0:
            self.combo += 1
            reward += (self.combo * 50)
        else:
            self.combo = 0

        # 狀態懲罰 (雖然 CMA-ES 不用這個 reward，但保留給 PPO 相容)
        # 這裡簡單算一下，不影響 CMA-ES 的 fitness (它看的是消行數)
        features = self._get_features(self.board)
        reward += (features[3] * REWARD_HOLE_PENALTY)      
        reward += (features[0] * REWARD_HEIGHT_PENALTY)    

        # 5. 檢查 Game Over
        self.current_piece = self._get_random_piece()
        if not self._is_valid_position(self.board, self.current_piece):
            self.game_over = True
            reward += REWARD_GAME_OVER
            
        return reward, self.game_over

    def get_possible_next_states(self):
        """
        回傳所有可能的下一步狀態特徵向量
        """
        states = {}
        piece = self.current_piece
        
        num_rotations = len(config.shapes[piece.shape])
        for rot in range(num_rotations):
            for x in range(-2, config.columns + 1):
                
                sim_piece = copy.deepcopy(piece)
                sim_piece.rotation = rot
                sim_piece.x = x
                sim_piece.y = 0 
                
                if not self._is_valid_position(self.board, sim_piece):
                    continue
                
                while self._is_valid_position(self.board, sim_piece, adj_x=0, adj_y=1):
                    sim_piece.y += 1
                
                temp_board = self.board.copy()
                self._lock_piece(temp_board, sim_piece)
                
                # 計算 5 維特徵
                features = self._get_features(temp_board)
                
                # 標準化 (讓數值不要太大)
                norm_features = features.copy()
                norm_features[0] /= 100.0 # Landing Height
                norm_features[1] /= 20.0  # Row Trans
                norm_features[2] /= 20.0  # Col Trans
                norm_features[3] /= 20.0  # Holes
                norm_features[4] /= 20.0  # Wells
                
                states[(x, rot)] = norm_features
                
        return states

    def _is_valid_position(self, board, piece, adj_x=0, adj_y=0):
        for y, x in self._get_piece_coords(piece):
            nx, ny = x + adj_x, y + adj_y
            if nx < 0 or nx >= config.columns or ny >= config.rows:
                return False
            if ny >= 0 and board[ny][nx] == 2:
                return False
        return True

    def _lock_piece(self, board, piece):
        for y, x in self._get_piece_coords(piece):
            if 0 <= y < config.rows and 0 <= x < config.columns:
                board[y][x] = 2

    def _get_piece_coords(self, piece):
        shape_template = config.shapes[piece.shape][piece.rotation % len(config.shapes[piece.shape])]
        return [(y + piece.y, x + piece.x) for y, x in shape_template]

    def _clear_lines(self, board):
        lines_to_clear = [i for i, row in enumerate(board) if all(cell == 2 for cell in row)]
        count = len(lines_to_clear)
        if count > 0:
            mask = np.ones(config.rows, dtype=bool)
            mask[lines_to_clear] = False
            new_board = np.zeros_like(board)
            new_board[count:] = board[mask]
            board[:] = new_board
        return count

        # --- 8 參數特徵計算 (Tetris 專精版) ---
    def _get_features(self, board):
        # board: (20, 10) array, 0 or 2
        grid = (board == 2).astype(int)
        rows, cols = grid.shape

        # 1. Landing Height (平均高度)
        row_indices = np.arange(rows, 0, -1).reshape(-1, 1)
        height_grid = grid * row_indices
        col_heights = np.max(height_grid, axis=0)
        landing_height = np.mean(col_heights)
        
        # 3. Row Transitions
        row_trans = 0
        for r in range(rows):
            line = np.insert(grid[r], [0, cols], 1)
            row_trans += np.sum(np.abs(np.diff(line)))

        # 4. Column Transitions
        col_trans = 0
        for c in range(cols):
            col = np.insert(grid[:, c], [0, rows], [0, 1])
            col_trans += np.sum(np.abs(np.diff(col)))

        # 5. Number of Holes
        cumsum = np.cumsum(grid, axis=0)
        holes = np.sum((cumsum > 0) & (grid == 0))

        # 6. Well Analysis (詳細分析井)
        # 我們要計算每一列的 "井深"
        well_depths = []
        for c in range(cols):
            if c == 0: left_wall = np.ones(rows)
            else: left_wall = grid[:, c-1]
            
            if c == cols-1: right_wall = np.ones(rows)
            else: right_wall = grid[:, c+1]
            
            mid = grid[:, c]
            
            # 找出所有是井的格子 (左右實，中空)
            is_well = (left_wall == 1) & (right_wall == 1) & (mid == 0)
            
            # 計算深度：從上往下連鎖
            depth = 0
            for r in range(rows):
                if is_well[r]:
                    depth += 1
                else:
                    if depth > 0:
                        well_depths.append(depth)
                    depth = 0
            if depth > 0: well_depths.append(depth)
            
        # Well Sums: 所有井的深度和
        well_sums = sum(well_depths)
        
        # 7. Deep Wells (鼓勵留深坑給 I 方塊)
        # 深度 >= 3 的井，視為 "好井" (準備 Tetris)
        # 但這也是雙面刃，所以權重讓 CMA-ES 去抓
        deep_wells = sum([d for d in well_depths if d >= 3])
        
        # 8. Cumulative Wells (Dellacherie 原版定義: sum(1..d))
        # 深度越深，權重越重。例如深 2 的井 = 1+2=3，深 3 = 1+2+3=6
        cum_wells = sum([d*(d+1)/2 for d in well_depths])

        # 注意：Eroded Piece Cells 需要前後盤面比較，這裡暫時用靜態特徵替代
        # 或者我們把 Max Height 加回來當第 8 個特徵
        max_height = np.max(col_heights)

        # 回傳 8 個特徵
        # [Landing Height, Row Trans, Col Trans, Holes, Well Sums, Deep Wells, Cum Wells, Max Height]
        return np.array([landing_height, row_trans, col_trans, holes, well_sums, deep_wells, cum_wells, max_height], dtype=np.float32)

