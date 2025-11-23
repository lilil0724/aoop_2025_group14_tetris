import numpy as np
import config
import copy
import random
import torch
from pieces import Piece

# --- 獎勵參數設定 (CMA-ES 訓練時其實不太需要，但為了相容性保留) ---
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
        self.last_cleared_lines = 0 # 為了給 CMA-ES 統計 Tetris 數
        self.game_over = False
        self.steps = 0
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
        
        while self._is_valid_position(self.board, piece, adj_x=0, adj_y=1):
            piece.y += 1
            
        self._lock_piece(self.board, piece)
        
        cleared_lines = self._clear_lines(self.board)
        self.last_cleared_lines = cleared_lines
        self.line_count += cleared_lines
        
        reward = REWARD_SURVIVE + REWARD_CLEAR_LINES[cleared_lines]
        
        if cleared_lines > 0:
            self.combo += 1
            reward += (self.combo * 50)
        else:
            self.combo = 0

        self.current_piece = self._get_random_piece()
        if not self._is_valid_position(self.board, self.current_piece):
            self.game_over = True
            reward += REWARD_GAME_OVER
            
        return reward, self.game_over

    def get_possible_next_states(self):
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
                
                # 計算 8 維特徵
                features = self._get_features(temp_board)
                
                # 對 8 個特徵進行標準化
                norm_features = features.copy()
                norm_features[0] /= 10.0   # Landing Height
                norm_features[1] /= 100.0  # Row Trans
                norm_features[2] /= 100.0  # Col Trans
                norm_features[3] /= 40.0   # Holes
                norm_features[4] /= 40.0   # Well Sums
                norm_features[5] /= 40.0   # Deep Wells
                norm_features[6] /= 100.0  # Cumulative Wells
                norm_features[7] /= 20.0   # Max Height
                
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
        grid = (board == 2).astype(int)
        rows, cols = grid.shape

        # 1. Landing Height (平均高度)
        row_indices = np.arange(rows, 0, -1).reshape(-1, 1)
        height_grid = grid * row_indices
        col_heights = np.max(height_grid, axis=0)
        landing_height = np.mean(col_heights)
        
        # 2. Row Transitions
        row_trans = 0
        for r in range(rows):
            line = np.insert(grid[r], [0, cols], 1)
            row_trans += np.sum(np.abs(np.diff(line)))

        # 3. Column Transitions
        col_trans = 0
        for c in range(cols):
            col = np.insert(grid[:, c], [0, rows], [0, 1])
            col_trans += np.sum(np.abs(np.diff(col)))

        # 4. Number of Holes
        cumsum = np.cumsum(grid, axis=0)
        holes = np.sum((cumsum > 0) & (grid == 0))

        # 5. Well Analysis (井)
        well_depths = []
        for c in range(cols):
            if c == 0: left_wall = np.ones(rows)
            else: left_wall = grid[:, c-1]
            
            if c == cols-1: right_wall = np.ones(rows)
            else: right_wall = grid[:, c+1]
            
            mid = grid[:, c]
            is_well = (left_wall == 1) & (right_wall == 1) & (mid == 0)
            
            depth = 0
            for r in range(rows):
                if is_well[r]: depth += 1
                else:
                    if depth > 0: well_depths.append(depth)
                    depth = 0
            if depth > 0: well_depths.append(depth)
            
        # 6. Well Sums (所有井深加總)
        well_sums = sum(well_depths)
        
        # 7. Deep Wells (深度>=3的井)
        deep_wells = sum([d for d in well_depths if d >= 3])
        
        # 8. Cumulative Wells (井深累加)
        cum_wells = sum([d*(d+1)/2 for d in well_depths])
        
        max_height = np.max(col_heights) if len(col_heights) > 0 else 0

        # 返回 8 個特徵
        return np.array([landing_height, row_trans, col_trans, holes, well_sums, deep_wells, cum_wells, max_height], dtype=np.float32)

