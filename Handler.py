import config
import random # <-- 新增

def getCellsAbsolutePosition(piece):
    '''取得方塊當前所有方格的座標'''
    return [(y + piece.y, x + piece.x) for y, x in piece.getCells()]


def fixPiece(shot, piece):
    '''固定已落地的方塊，並且在main中自動切到下一個方塊'''
    piece.is_fixed = True
    for y, x in getCellsAbsolutePosition(piece):
        # (修改) 增加邊界檢查
        if 0 <= y < config.rows and 0 <= x < config.columns:
            shot.status[y][x] = 2
            shot.color[y][x] = piece.color

# 小工具：檢查是否能以 (dx, dy) 位移
def _can_move(shot, piece, dx, dy):
    for y, x in getCellsAbsolutePosition(piece):
        ny, nx = y + dy, x + dx
        # 邊界檢查
        if nx < 0 or nx >= config.columns or ny >= config.rows:
            return False
        # 只在可見區域內做碰撞檢查；ny<0 表示仍在天花板上方，先允許移動
        if ny >= 0 and shot.status[ny][nx] == 2:
            return False
    return True

# 向左移動
def moveLeft(shot, piece):
    if _can_move(shot, piece, dx=-1, dy=0):
        piece.x -= 1

# 向右移動
def moveRight(shot, piece):
    if _can_move(shot, piece, dx=+1, dy=0):
        piece.x += 1

# 使方塊下落一格；若無法再下落就固定
def drop(shot, piece):
    if _can_move(shot, piece, dx=0, dy=+1):
        piece.y += 1
    else:
        fixPiece(shot, piece)

# 瞬間掉落：一路落到底再固定
def instantDrop(shot, piece):
    while _can_move(shot, piece, dx=0, dy=+1):
        piece.y += 1
    fixPiece(shot, piece)

# 旋轉方塊（若旋轉後越界或撞到，則還原） //
def rotate(shot, piece):
    old_rot = piece.rotation
    piece.rotation += 1
    for y, x in getCellsAbsolutePosition(piece):
        # 檢查水平越界與落到最底
        if x < 0 or x >= config.columns or y >= config.rows:
            piece.rotation = old_rot
            return
        # 在畫面內才檢查撞固定塊；y<0 還在上方可暫時忽略
        if y >= 0 and shot.status[y][x] == 2:
            piece.rotation = old_rot
            return

# 判斷是否死掉（出局）
def isDefeat(shot, piece):
    # 1) 新方塊生成時就與已固定方塊重疊
    overlap = any(
        (0 <= y < config.rows and 0 <= x < config.columns and shot.status[y][x] == 2)
        for y, x in getCellsAbsolutePosition(piece)
    )
    if overlap:
        return True
    # 2) 方塊固定後，其中任一格仍在天花板上方（y<0）
    if getattr(piece, "is_fixed", False) and any(y < 0 for y, _ in getCellsAbsolutePosition(piece)):
        return True
    return False

# 消去列（自底向上檢查；清除一列後把上面整體下移）
def eliminateFilledRows(shot, piece): # <-- (修改)
    lines = 0
    rows = config.rows
    cols = config.columns

    y = rows - 1
    while y >= 0:
        # 該列是否滿
        if all(shot.status[y][x] == 2 for x in range(cols)):
            lines += 1
            # 這一列以上往下搬一列
            for yy in range(y, 0, -1):
                for x in range(cols):
                    shot.status[yy][x] = shot.status[yy - 1][x]
                    if hasattr(shot, "color"):
                        shot.color[yy][x] = shot.color[yy - 1][x]
            # 最頂列清空
            for x in range(cols):
                shot.status[0][x] = 0
                if hasattr(shot, "color"):
                    shot.color[0][x] = config.background_color # (修改) 用背景色
            # y 留在原位
        else:
            y -= 1

    # 計分
    score_table = getattr(config, "score_count", {1: 40, 2: 100, 3: 300, 4: 1200})
    shot.line_count += lines
    shot.score += score_table.get(lines, 0)
    
    # (新增) 檢查 All Clear
    all_clear = all(shot.status[y][x] == 0 for y in range(rows) for x in range(cols))
    
    return (lines, all_clear) # <-- (修改) 回傳消行數和 PC 狀態


# --- (以下為新增函式) ---

def calculateAttack(clears, combo_count, is_b2b, all_clear):
    """
    根據消行、Combo、B2B、PC 狀態計算攻擊行數
    """
    if clears == 0:
        return 0
    
    # 1. 基礎攻擊
    base_atk = config.ATTACK_BASE.get(clears, 0)
    
    # 2. 判定是否為「威力技」 (目前只有 Tetris)
    is_power_move = (clears == 4)
    
    # 3. B2B 加成 (B2B 狀態 *且* 這次也是威力技)
    b2b_atk = 0
    if is_b2b and is_power_move:
        b2b_atk = config.ATTACK_B2B_BONUS
        
    # 4. Combo 加成
    combo_atk = config.get_combo_bonus(combo_count)
    
    # 5. All Clear 加成
    pc_atk = 0
    if all_clear:
        pc_atk = config.ATTACK_PERFECT_CLEAR
        
    # 總攻擊 = 基礎 + B2B + Combo + PC
    total_atk = base_atk + b2b_atk + combo_atk + pc_atk
    
    return total_atk

def insertGarbage(shot, lines_to_add):
    """
    從底部插入指定行數的垃圾行，並將盤面往上推
    """
    rows = config.rows
    cols = config.columns
    
    # 1. 將現有盤面(status=2) 往上推 `lines_to_add` 行
    # 從上往下(y=0)開始搬
    for y in range(0, rows - lines_to_add):
        for x in range(cols):
            shot.status[y][x] = shot.status[y + lines_to_add][x]
            shot.color[y][x] = shot.color[y + lines_to_add][x]
            
    # 2. 在底部 (y = rows - lines_to_add 到 rows - 1) 生成新的垃圾行
    for y in range(rows - lines_to_add, rows):
        # 決定洞位 (Hole RNG)
        # 根據 config 機率決定是否沿用上一個洞位
        if random.random() > config.GARBAGE_HOLE_REPEAT_PROB:
            # 換一個新洞位，但避免跟上一個完全一樣
            new_pos = random.randint(0, cols - 1)
            if new_pos == shot.garbage_hole_pos:
                new_pos = (new_pos + 1) % cols # 簡單換到隔壁
            shot.garbage_hole_pos = new_pos
        
        hole_x = shot.garbage_hole_pos
        
        # 填入垃圾行
        for x in range(cols):
            if x == hole_x:
                shot.status[y][x] = 0 # 洞
                shot.color[y][x] = config.background_color
            else:
                shot.status[y][x] = 2 # 垃圾 (固定方塊)
                shot.color[y][x] = config.GARBAGE_COLOR