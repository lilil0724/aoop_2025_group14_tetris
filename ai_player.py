import config
import Handler
import copy

# --- AI 評估權重 ---
# 這些是 AI 的「大腦」。我們懲罰 (負分) 不好的狀態，獎勵 (正分) 好的狀態。
# 這些權重是 GA (遺傳演算法) 未來可以去「訓練」和「演化」的目標！
WEIGHTS = {
    'aggregate_height': -0.510066, # 總高度 (越低越好)
    'lines_cleared':   0.760666,    # 消行數 (越多越好)
    'holes':           -0.35663,    # 空洞數 (越少越好)
    'bumpiness':       -0.184483    # 粗糙度 (越平坦越好)
}


def find_best_move(shot, piece):
    """
    遍歷所有可能的移動 (x 座標和旋轉)，找到最佳的放置位置。
    """
    best_score = -float('inf')
    best_move = None # 將儲存 (best_x, best_rotation)

    # 1. 遍歷所有 4 種旋轉
    for rotation in range(len(config.shapes[piece.shape])):
        
        # 2. 遍歷所有可能的 x 座標
        # (從 -2 到 10，以涵蓋方塊在邊緣的旋轉)
        for x in range(-2, config.columns + 1):
            
            # --- 模擬這一步 ---
            # (A) 建立遊戲的「深度複製」，這樣我們才不會動到真正的遊戲
            sim_shot = copy.deepcopy(shot)
            sim_piece = copy.deepcopy(piece)
            
            # (B) 應用旋轉和 X 座標
            sim_piece.rotation = rotation
            sim_piece.x = x

            # (C) 檢查這個初始位置是否有效 (例如旋轉後是否撞牆或已固定的方塊)
            # 這是必要的，因為 instantDrop 假設初始位置是 OK 的
            if not _is_valid_position(sim_shot, sim_piece):
                continue
                
            # (D) 模擬瞬降
            Handler.instantDrop(sim_shot, sim_piece)
            
            # (E) 模擬消行
            (lines_cleared, all_clear) = Handler.eliminateFilledRows(sim_shot, sim_piece)
            
            # (F) 評估這個「最終盤面」
            score = evaluate_board(sim_shot.status, lines_cleared)
            if all_clear:
                score += 500 # 給 All Clear 巨量獎勵
            
            # (G) 檢查這是否是目前最好的分數
            if score > best_score:
                best_score = score
                best_move = (x, rotation)
    if best_move:
        print(f"[AI P2] Decision: x={best_move[0]}, rot={best_move[1]}, score={best_score:.2f}")
    else:
        print("[AI P2] ERROR: No valid move found!")
    return best_move


def _is_valid_position(shot, piece):
    """
    輔助函式：檢查方塊在 (piece.x, piece.y, piece.rotation)
    是否與邊界或已固定方塊重疊。
    """
    for y_offset, x_offset in piece.getCells():
        y, x = piece.y + y_offset, piece.x + x_offset
        
        # 1. 檢查邊界 (只檢查左右和底部，頂部 (y<0) 是允許的)
        if x < 0 or x >= config.columns or y >= config.rows:
            return False
            
        # 2. 檢查碰撞 (只在盤面內檢查)
        if 0 <= y < config.rows and shot.status[y][x] == 2:
            return False
            
    return True


# --- 盤面評估函式 (AI 的大腦) ---

def evaluate_board(board_status, lines_cleared):
    """
    根據 board_status 計算啟發式分數。
    """
    # 1. 計算盤面高度 (Height)
    heights = [0] * config.columns
    for x in range(config.columns):
        for y in range(config.rows):
            if board_status[y][x] == 2:
                heights[x] = config.rows - y
                break # 找到該行的最高點，換下一行
                
    aggregate_height = sum(heights)

    # 2. 計算空洞數 (Holes)
    holes = 0
    for x in range(config.columns):
        # 只有在方塊下方才可能產生空洞
        if heights[x] > 0:
            for y in range(config.rows - heights[x] + 1, config.rows):
                if board_status[y][x] == 0:
                    holes += 1

    # 3. 計算粗糙度 (Bumpiness)
    bumpiness = 0
    for x in range(config.columns - 1):
        bumpiness += abs(heights[x] - heights[x+1])

    # 4. 計算總分
    score = (
        WEIGHTS['aggregate_height'] * aggregate_height +
        WEIGHTS['lines_cleared'] * lines_cleared +
        WEIGHTS['holes'] * holes +
        WEIGHTS['bumpiness'] * bumpiness
    )
    
    return score