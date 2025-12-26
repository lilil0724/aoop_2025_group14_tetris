import numpy as np
import config
import copy
import random
from pieces import Piece

# --- 獎勵參數 ---
REWARD_CLEAR_LINES = [0, 0.1, 0.4, 1.6, 5.0]
REWARD_HOLE_PENALTY = -0.05 
REWARD_HEIGHT_PENALTY = -0.05   
REWARD_BUMPINESS_PENALTY = -0.05 
REWARD_SURVIVE = 0.01                 
REWARD_GAME_OVER = -1.0

class TetrisEnv:
    def __init__(self):
        self.reset()

    def reset(self):
        self.board = np.zeros((config.rows, config.columns), dtype=int)
        self.current_piece = self._get_random_piece()
        self.score = 0
        self.line_count = 0  
        self.combo = 0       
        self.last_cleared_lines = 0
        self.game_over = False
        self.steps = 0
        return self._get_features(self.board)

    def _get_random_piece(self):
        shape = random.choice(list(config.shapes.keys()))
        return Piece(5, 0, shape)

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

    def _get_features(self, board):
        grid = (board == 2).astype(int)
        rows, cols = grid.shape
        row_indices = np.arange(rows, 0, -1).reshape(-1, 1)
        height_grid = grid * row_indices
        col_heights = np.max(height_grid, axis=0)
        landing_height = np.mean(col_heights)
        row_trans = 0
        for r in range(rows):
            line = np.insert(grid[r], [0, cols], 1)
            row_trans += np.sum(np.abs(np.diff(line)))
        col_trans = 0
        for c in range(cols):
            col = np.insert(grid[:, c], [0, rows], [0, 1])
            col_trans += np.sum(np.abs(np.diff(col)))
        cumsum = np.cumsum(grid, axis=0)
        holes = np.sum((cumsum > 0) & (grid == 0))
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
        well_sums = sum(well_depths)
        deep_wells = sum([d for d in well_depths if d >= 3])
        cum_wells = sum([d*(d+1)/2 for d in well_depths])
        max_height = np.max(col_heights) if len(col_heights) > 0 else 0
        return np.array([landing_height, row_trans, col_trans, holes, well_sums, deep_wells, cum_wells, max_height], dtype=np.float32)