import numpy as np
import copy
import config
import Handler

class WeightedAI:
    def __init__(self):
        # Dellacherie's Algorithm Weights (你指定的 8 個權重)
        # 順序對應下方的 get_features 回傳順序
        self.weights = np.array([
            -4.5001,  # 1. Landing Height
            -2.0001,  # 2. Max Height
            -3.2001,  # 3. Row Transitions
            -9.3001,  # 4. Col Transitions
            -7.9001,  # 5. Holes
            -3.4001,  # 6. Well Sums
            -1.0001,  # 7. Deep Wells
            -1.0001   # 8. Cumulative Wells
        ], dtype=np.float32)

    def get_features(self, board, landing_height, lines_cleared):
        """
        提取對應上述權重的 8 個特徵
        """
        rows, cols = config.rows, config.columns
        # 將盤面轉換為 0/1 矩陣 (1代表有方塊)
        grid = (np.array(board) == 2).astype(int)
        
        # --- 1. Landing Height (外部傳入) ---
        feat_landing_height = landing_height

        # 計算每列高度 (Column Heights)
        col_heights = np.zeros(cols, dtype=int)
        for c in range(cols):
            # 找該列第一個非 0 的位置
            r_idx = np.argmax(grid[:, c])
            if grid[r_idx, c] == 0: # 該列全空
                col_heights[c] = 0
            else:
                col_heights[c] = rows - r_idx

        # --- 2. Max Height ---
        feat_max_height = np.max(col_heights)

        # --- 3. Row Transitions ---
        # 計算每一列中，從空到有或從有到空的變換次數 (邊界算實體)
        feat_row_trans = 0
        for r in range(rows):
            # 在左右邊界各加一個 '1' (牆壁)
            line = np.insert(grid[r], [0, cols], 1)
            feat_row_trans += np.sum(np.abs(np.diff(line)))

        # --- 4. Column Transitions ---
        # 計算每一行中，變換次數 (上下邊界算實體)
        feat_col_trans = 0
        for c in range(cols):
            col = np.insert(grid[:, c], [0, rows], 1)
            feat_col_trans += np.sum(np.abs(np.diff(col)))

        # --- 5. Holes ---
        # 定義：上方有實體方塊的空格
        feat_holes = 0
        for c in range(cols):
            if col_heights[c] > 0:
                top_r = rows - col_heights[c]
                # 從最高點往下數，所有的 0 都是洞
                feat_holes += np.sum(grid[top_r+1:, c] == 0)

        # --- 6, 7, 8. Wells (井) ---
        well_depths = []
        for c in range(cols):
            # 定義牆壁：邊界或旁邊有方塊
            left = grid[:, c-1] if c > 0 else np.ones(rows)
            right = grid[:, c+1] if c < cols - 1 else np.ones(rows)
            mid = grid[:, c]
            
            depth = 0
            for r in range(rows):
                if left[r] == 1 and right[r] == 1 and mid[r] == 0:
                    depth += 1
                else:
                    if depth > 0:
                        well_depths.append(depth)
                        depth = 0
            if depth > 0: well_depths.append(depth) # 到底部的井
            
        feat_well_sums = sum(well_depths)
        feat_deep_wells = sum(d for d in well_depths if d >= 2)
        feat_cum_wells = sum((d * (d + 1)) // 2 for d in well_depths)

        # 回傳順序必須嚴格對應 self.weights
        return np.array([
            feat_landing_height,
            feat_max_height,
            feat_row_trans,
            feat_col_trans,
            feat_holes,
            feat_well_sums,
            feat_deep_wells,
            feat_cum_wells
        ], dtype=np.float32)

    def evaluate(self, features):
        return np.dot(features, self.weights)

    def find_best_move(self, shot, piece):
        """
        遍歷所有動作，回傳最佳 (x, rotation)
        """
        best_score = -float('inf')
        best_move = None 
        
        num_rotations = len(config.shapes[piece.shape])
        if piece.shape == 'O': num_rotations = 1
        elif piece.shape in ['S', 'Z', 'I']: num_rotations = 2
        
        for rot in range(num_rotations):
            test_piece = copy.deepcopy(piece)
            test_piece.rotation = rot
            
            for x in range(-2, config.columns + 1):
                test_piece.x = x
                if not Handler.isValidPosition(shot, test_piece):
                    continue
                
                # 模擬下落位置
                sim_y = test_piece.y
                while True:
                    test_piece.y = sim_y
                    if Handler._can_move(shot, test_piece, 0, 1):
                        sim_y += 1
                    else:
                        break
                test_piece.y = sim_y
                
                # 計算 Landing Height (特徵 1)
                landing_height = config.rows - test_piece.y
                
                # 模擬盤面
                sim_board = [row[:] for row in shot.status]
                for cy, cx in Handler.getCellsAbsolutePosition(test_piece):
                    if 0 <= cy < config.rows and 0 <= cx < config.columns:
                        sim_board[cy][cx] = 2
                
                # 模擬消行
                lines_cleared = 0
                final_board = []
                for r in range(config.rows):
                    if all(cell == 2 for cell in sim_board[r]):
                        lines_cleared += 1
                    else:
                        final_board.append(sim_board[r])
                # 補回空行
                for _ in range(lines_cleared):
                    final_board.insert(0, [0]*config.columns)
                
                # 計算分數
                feats = self.get_features(final_board, landing_height, lines_cleared)
                score = self.evaluate(feats)
                
                if score > best_score:
                    best_score = score
                    best_move = (x, rot)
                
                test_piece.y = piece.y # 重置 y
                
        return best_move