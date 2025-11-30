import config
import random 

def getCellsAbsolutePosition(piece):
    '''取得方塊當前所有方格的座標'''
    return [(y + piece.y, x + piece.x) for y, x in piece.getCells()]


def fixPiece(shot, piece):
    '''固定已落地的方塊，並且在main中自動切到下一個方塊'''
    piece.is_fixed = True
    for y, x in getCellsAbsolutePosition(piece):
        if 0 <= y < config.rows and 0 <= x < config.columns:
            shot.status[y][x] = 2
            shot.color[y][x] = piece.color

def _can_move(shot, piece, dx, dy):
    for y, x in getCellsAbsolutePosition(piece):
        ny, nx = y + dy, x + dx
        if nx < 0 or nx >= config.columns or ny >= config.rows:
            return False
        if ny >= 0 and shot.status[ny][nx] == 2:
            return False
    return True

def moveLeft(shot, piece):
    if _can_move(shot, piece, dx=-1, dy=0):
        piece.x -= 1

def moveRight(shot, piece):
    if _can_move(shot, piece, dx=+1, dy=0):
        piece.x += 1

def drop(shot, piece):
    if _can_move(shot, piece, dx=0, dy=+1):
        piece.y += 1
    else:
        fixPiece(shot, piece)

def instantDrop(shot, piece):
    while _can_move(shot, piece, dx=0, dy=+1):
        piece.y += 1
    fixPiece(shot, piece)

# 順時針旋轉 (Clockwise)
def rotate(shot, piece):
    old_rot = piece.rotation
    piece.rotation += 1
    for y, x in getCellsAbsolutePosition(piece):
        if x < 0 or x >= config.columns or y >= config.rows:
            piece.rotation = old_rot
            return
        if y >= 0 and shot.status[y][x] == 2:
            piece.rotation = old_rot
            return

# [新增] 逆時針旋轉 (Counter-Clockwise)
def rotateCCW(shot, piece):
    old_rot = piece.rotation
    piece.rotation -= 1 # Python 的負數取餘數機制會自動處理 (-1 % 4 = 3)
    for y, x in getCellsAbsolutePosition(piece):
        # 檢查水平越界與落到最底
        if x < 0 or x >= config.columns or y >= config.rows:
            piece.rotation = old_rot
            return
        # 在畫面內才檢查撞固定塊
        if y >= 0 and shot.status[y][x] == 2:
            piece.rotation = old_rot
            return

def isDefeat(shot, piece):
    if getattr(piece, 'is_fixed', False):
        return False
    for y_cell, x_cell in getCellsAbsolutePosition(piece):
        if 0 <= y_cell < config.rows and 0 <= x_cell < config.columns:
            if shot.status[y_cell][x_cell] == 2:
                return True
    return False

def eliminateFilledRows(shot, piece):
    lines = 0
    rows = config.rows
    cols = config.columns

    y = rows - 1
    while y >= 0:
        if all(shot.status[y][x] == 2 for x in range(cols)):
            lines += 1
            for yy in range(y, 0, -1):
                for x in range(cols):
                    shot.status[yy][x] = shot.status[yy - 1][x]
                    if hasattr(shot, "color"):
                        shot.color[yy][x] = shot.color[yy - 1][x]
            for x in range(cols):
                shot.status[0][x] = 0
                if hasattr(shot, "color"):
                    shot.color[0][x] = config.background_color
        else:
            y -= 1

    score_table = getattr(config, "score_count", {1: 40, 2: 100, 3: 300, 4: 1200})
    shot.line_count += lines
    shot.score += score_table.get(lines, 0)
    
    all_clear = all(shot.status[y][x] == 0 for y in range(rows) for x in range(cols))
    
    return (lines, all_clear)

def calculateAttack(clears, combo_count, is_b2b, all_clear):
    if clears == 0:
        return 0
    base_atk = config.ATTACK_BASE.get(clears, 0)
    is_power_move = (clears == 4)
    b2b_atk = 0
    if is_b2b and is_power_move:
        b2b_atk = config.ATTACK_B2B_BONUS
    combo_atk = config.get_combo_bonus(combo_count)
    pc_atk = 0
    if all_clear:
        pc_atk = config.ATTACK_PERFECT_CLEAR
    total_atk = base_atk + b2b_atk + combo_atk + pc_atk
    return total_atk

def insertGarbage(shot, lines_to_add):
    rows = config.rows
    cols = config.columns
    for y in range(0, rows - lines_to_add):
        for x in range(cols):
            shot.status[y][x] = shot.status[y + lines_to_add][x]
            shot.color[y][x] = shot.color[y + lines_to_add][x]
    for y in range(rows - lines_to_add, rows):
        if random.random() > config.GARBAGE_HOLE_REPEAT_PROB:
            new_pos = random.randint(0, cols - 1)
            if new_pos == shot.garbage_hole_pos:
                new_pos = (new_pos + 1) % cols 
            shot.garbage_hole_pos = new_pos
        hole_x = shot.garbage_hole_pos
        for x in range(cols):
            if x == hole_x:
                shot.status[y][x] = 0 
                shot.color[y][x] = config.background_color
            else:
                shot.status[y][x] = 2 
                shot.color[y][x] = config.GARBAGE_COLOR

def isValidPosition(shot, piece):
    cells = getCellsAbsolutePosition(piece)
    for y, x in cells:
        if not (0 <= x < config.columns and y < config.rows):
            return False
        if y >= 0 and shot.status[y][x] == 2:
            return False
    return True