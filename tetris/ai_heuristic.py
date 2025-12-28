import numpy as np
import copy
import config
try:
    import tetris_env
except ImportError:
    print("Warning: tetris_env.py not found. Heuristic AI might fail.")

# 特徵權重 (這就是 AI 的性格參數)
BEST_WEIGHTS = np.array([-1.41130507, -2.23926392, -0.78272467, -4.00369693, -0.67902086, -0.449347,
                         -0.1623215, -0.91940282])

def get_tetris_features_v8(board):
    """ 計算盤面特徵 (AI 的眼睛) """
    grid = (np.array(board) == 2).astype(int)
    rows, cols = grid.shape
    
    # 1. Landing Height
    row_indices = np.arange(rows, 0, -1).reshape(-1, 1)
    height_grid = grid * row_indices
    col_heights = np.max(height_grid, axis=0)
    landing_height = np.mean(col_heights)
    
    # 2. Row Transitions .
    row_trans = 0
    for r in range(rows):
        line = np.insert(grid[r], [0, cols], 1)
        row_trans += np.sum(np.abs(np.diff(line)))
        
    # 3. Column Transitions
    col_trans = 0
    for c in range(cols):
        col = np.insert(grid[:, c], [0, rows], [0, 1])
        col_trans += np.sum(np.abs(np.diff(col)))
        
    # 4. Holes
    cumsum = np.cumsum(grid, axis=0)
    holes = np.sum((cumsum > 0) & (grid == 0))
    
    # 5. Well Analysis
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
    
    features = np.array([landing_height, row_trans, col_trans, holes, well_sums, deep_wells, cum_wells, max_height], dtype=np.float32)
    
    # 正規化
    features[0] /= 10.0
    features[1] /= 100.0
    features[2] /= 100.0
    features[3] /= 40.0
    features[4] /= 40.0
    features[5] /= 40.0
    features[6] /= 100.0
    features[7] /= 20.0
    return features

def get_ai_move_heuristic(shot, piece):
    """ 思考最佳移動路徑 (AI 的大腦) """
    # 建立模擬環境
    env = tetris_env.TetrisEnv()
    env.board = np.array(shot.status, dtype=int)
    env.current_piece = copy.deepcopy(piece)
    
    possible_moves = {}
    piece_ref = env.current_piece
    num_rotations = len(config.shapes[piece_ref.shape])
    
    # 窮舉所有可能的落點
    for rot in range(num_rotations):
        for x in range(-2, config.columns + 1):
            sim_piece = copy.deepcopy(piece_ref)
            sim_piece.rotation = rot
            sim_piece.x = x
            sim_piece.y = 0
            
            if not env._is_valid_position(env.board, sim_piece):
                continue
                
            while env._is_valid_position(env.board, sim_piece, adj_x=0, adj_y=1):
                sim_piece.y += 1
            
            temp_board = env.board.copy()
            env._lock_piece(temp_board, sim_piece)
            possible_moves[(x, rot)] = temp_board
            
    if not possible_moves:
        return None
        
    best_score = -float('inf')
    best_move = None
    
    # 評估每個落點的分數
    for move, board_state in possible_moves.items():
        features = get_tetris_features_v8(board_state)
        score = np.dot(BEST_WEIGHTS, features)
        if score > best_score:
            best_score = score
            best_move = move
            
    return best_move
