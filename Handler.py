import config


def getCellsAbsolutePosition(piece):
    '''取得方塊當前所有方格的座標'''
    return [(y + piece.y, x + piece.x) for y, x in piece.getCells()]


def fixPiece(shot, piece):
    '''固定已落地的方塊，並且在main中自動切到下一個方塊'''
    piece.is_fixed = True
    for y, x in getCellsAbsolutePosition(piece):
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

# 旋轉方塊（若旋轉後越界或撞到，則還原）
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
    # 兩種常見定義：
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
def eliminateFilledRows(shot, piece):
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
                    # 若你有 color 陣列，記得同步
                    if hasattr(shot, "color"):
                        shot.color[yy][x] = shot.color[yy - 1][x]
            # 最頂列清空
            for x in range(cols):
                shot.status[0][x] = 0
                if hasattr(shot, "color"):
                    shot.color[0][x] = (0, 0, 0)
            # 清完這列後，y 留在原位，因為新的內容下來了，需要再檢查同一 y
        else:
            y -= 1

    # 計分（若 config 有 score_count 就用它，否則備用）
    score_table = getattr(config, "score_count", {1: 40, 2: 100, 3: 300, 4: 1200})
    shot.line_count += lines
    shot.score += score_table.get(lines, 0)
