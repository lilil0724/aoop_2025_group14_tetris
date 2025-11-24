# dataset.py
# 功能：讓最強的 "8-Feature Dellacherie AI" 自動玩遊戲，並收集數據給 Transformer 訓練

import os
import numpy as np
import torch
from torch.utils.data import Dataset

import config
import Handler
import pieces
import shots
import tetris_env
import copy
import random

# ----------------------------------------------------
# 一、Teacher AI (老師) 的大腦
# ----------------------------------------------------

# 最強權重 (8-Feature, Tetris Expert)
BEST_WEIGHTS = np.array([-1.0, -1.0, -1.0, -4.0, -1.0, 0.5, -1.0, -1.0])

def get_tetris_features_v8(board):
    """
    8 參數特徵計算函式 (老師用來觀察盤面的眼睛)
    """
    # board: 20x10 list or array
    grid = (np.array(board) == 2).astype(int)
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
        
    # 5. Well Sums
    well_sums = sum(well_depths)
    
    # 6. Deep Wells (深度 >= 3)
    deep_wells = sum([d for d in well_depths if d >= 3])
    
    # 7. Cumulative Wells
    cum_wells = sum([d*(d+1)/2 for d in well_depths])
    
    # 8. Max Height
    max_height = np.max(col_heights) if len(col_heights) > 0 else 0

    # 回傳 8 個特徵
    features = np.array([landing_height, row_trans, col_trans, holes, well_sums, deep_wells, cum_wells, max_height], dtype=np.float32)
    
    # 標準化
    features[0] /= 10.0   # Landing
    features[1] /= 100.0  # Row Trans
    features[2] /= 100.0  # Col Trans
    features[3] /= 40.0   # Holes
    features[4] /= 40.0   # Well Sums
    features[5] /= 40.0   # Deep Wells
    features[6] /= 100.0  # Cum Wells
    features[7] /= 20.0   # Max Height
    
    return features

def teacher_policy_best_move(shot, piece):
    """
    老師 AI 的決策核心：遍歷所有動作，找分數最高的
    """
    env = tetris_env.TetrisEnv()
    env.board = np.array(shot.status, dtype=int)
    env.current_piece = copy.deepcopy(piece)

    possible_moves = {}
    cur_piece = env.current_piece
    num_rotations = len(config.shapes[cur_piece.shape])

    for rot in range(num_rotations):
        for x in range(-2, config.columns + 3):
            sim_piece = copy.deepcopy(cur_piece)
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

    # 評分
    best_score = -float('inf')
    best_move = None
    
    for move, board_state in possible_moves.items():
        features = get_tetris_features_v8(board_state)
        score = np.dot(BEST_WEIGHTS, features)
        
        if score > best_score:
            best_score = score
            best_move = move
            
    return best_move

# ----------------------------------------------------
# 二、Dataset 類別：給 train_transformer.py 使用
# ----------------------------------------------------

class TetrisDataset(Dataset):
    def __init__(self, npz_path: str):
        data = np.load(npz_path,allow_pickle=True)
        self.boards = data["boards"]      # (N, rows, cols)
        self.piece_ids = data["piece_ids"]  # (N,)
        self.action_ids = data["action_ids"]  # (N,)

        assert len(self.boards) == len(self.piece_ids) == len(self.action_ids)

    def __len__(self):
        return len(self.boards)

    def __getitem__(self, idx):
        board = self.boards[idx]              # (rows, cols)
        piece_id = int(self.piece_ids[idx])   # scalar
        action_id = int(self.action_ids[idx]) # scalar

        return {
            "board": board.astype(np.float32),
            "piece_id": piece_id,
            "action_id": action_id,
        }

# ----------------------------------------------------
# 三、資料收集邏輯
# ----------------------------------------------------

def encode_action(x: int, rot: int, max_rot: int = 4, min_x: int = -2, max_x: int = None) -> int:
    """將 (x, rot) 編碼成 action_id"""
    if max_x is None:
        max_x = config.columns + 3 
    num_x = max_x - min_x + 1
    x_idx = x - min_x
    return rot * num_x + x_idx

def decode_action(action_id: int, max_rot: int = 4, min_x: int = -2, max_x: int = None):
    """將 action_id 解碼回 (x, rot)"""
    if max_x is None:
        max_x = config.columns + 3
    num_x = max_x - min_x + 1
    rot = action_id // num_x
    x_idx = action_id % num_x
    x = x_idx + min_x
    return x, rot

def collect_dataset(
    out_path: str = "tetris_demo_data.npz",
    num_games: int = 100,  # 玩 100 場
    max_steps_per_game: int = 2000, # 每場最多 2000 步
):
    all_boards = []
    all_piece_ids = []
    all_action_ids = []

    shape_list = list(config.shapes.keys())
    shape_to_idx = {s: i for i, s in enumerate(shape_list)}

    print(f"🚀 開始收集資料 (Target: {num_games} Games)...")

    for game_idx in range(num_games):
        shot = shots.Shot()
        current_shape = random.choice(shape_list)
        next_shape = random.choice(shape_list)
        piece = pieces.Piece(5, 0, current_shape)
        next_piece = pieces.Piece(5, 0, next_shape)

        steps = 0
        game_over = False

        while not game_over and steps < max_steps_per_game:
            steps += 1

            # 1. 老師思考
            move = teacher_policy_best_move(shot, piece)
            if move is None:
                break

            # 2. 記錄數據
            # 盤面: 我們要把 2 (固定方塊) 轉成 1，0 還是 0，這樣對神經網路比較好學
            board_np = np.array(shot.status, dtype=np.int32)
            board_np = (board_np == 2).astype(np.int32) # 轉成 0/1 mask
            
            piece_id = shape_to_idx[piece.shape]
            x, rot = move
            action_id = encode_action(x, rot)

            all_boards.append(board_np)
            all_piece_ids.append(piece_id)
            all_action_ids.append(action_id)

            # 3. 執行動作
            piece.x, piece.rotation = x, rot
            Handler.instantDrop(shot, piece)
            Handler.eliminateFilledRows(shot, piece)

            # 4. 下一步
            current_shape = next_shape
            next_shape = random.choice(shape_list)
            piece, next_piece = next_piece, pieces.Piece(5, 0, next_shape)

            if Handler.isDefeat(shot, piece):
                game_over = True
                break
        
        if (game_idx + 1) % 10 == 0:
            print(f"  - 進度: {game_idx + 1}/{num_games} 局完成. 當前樣本數: {len(all_boards)}")

    # 存檔
    boards_arr = np.stack(all_boards, axis=0)
    piece_ids_arr = np.array(all_piece_ids, dtype=np.int64)
    action_ids_arr = np.array(all_action_ids, dtype=np.int64)

    np.savez_compressed(
        out_path,
        boards=boards_arr,
        piece_ids=piece_ids_arr,
        action_ids=action_ids_arr,
    )
    print(f"✅ 資料收集完成！已儲存至 {out_path}")
    print(f"📊 總樣本數: {len(boards_arr)}")

if __name__ == "__main__":
    # 執行此檔案會自動開始收集資料
    collect_dataset()
