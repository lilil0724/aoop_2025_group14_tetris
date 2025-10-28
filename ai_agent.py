# ai_agent.py
import copy
import config
import pieces

# 盤面特徵：總高度/空洞數/起伏度(bumpiness) 與 消行數
def _board_features(status):
    rows, cols = config.rows, config.columns
    heights = [0]*cols
    holes = 0
    for x in range(cols):
        block_seen = False
        for y in range(rows):
            if status[y][x] != 0:
                if not block_seen:
                    heights[x] = rows - y
                    block_seen = True
            elif block_seen:
                holes += 1
    bump = sum(abs(heights[x]-heights[x+1]) for x in range(cols-1))
    agg_h = sum(heights)
    return agg_h, holes, bump

def _lines_cleared_after_place(status):
    rows, cols = config.rows, config.columns
    clears = 0
    for y in range(rows):
        if all(status[y][x] != 0 for x in range(cols)):
            clears += 1
    return clears

def _can_place(status, cells):
    rows, cols = config.rows, config.columns
    for y, x in cells:
        if x < 0 or x >= cols or y >= rows:
            return False
        if y >= 0 and status[y][x] != 0:
            return False
    return True

def _drop_down(status, cells):
    # 一路往下直到撞底或撞固定塊
    while True:
        moved = [(y+1, x) for (y, x) in cells]
        if _can_place(status, moved):
            cells = moved
        else:
            break
    return cells

def _place_and_fix(status, cells):
    new_status = [row[:] for row in status]
    for y, x in cells:
        if 0 <= y < config.rows and 0 <= x < config.columns:
            new_status[y][x] = 2
    # 模擬消行（不必真的把上面拉下來，只要估計清幾行即可；更精確可真的搬移）
    rows, cols = config.rows, config.columns
    cleared_rows = [y for y in range(rows) if all(new_status[y][x] == 2 for x in range(cols))]
    # 真實地把上面下移（與你 Handler.eliminateFilledRows 一致）
    for y in cleared_rows:
        for yy in range(y, 0, -1):
            for x in range(cols):
                new_status[yy][x] = new_status[yy-1][x]
        for x in range(cols):
            new_status[0][x] = 0
    return new_status, len(cleared_rows)

def enumerate_plans(shot, piece):
    plans = []
    shape_rotations = config.shapes[piece.shape]
    rows, cols = config.rows, config.columns
    heights = [0]*cols
    for rot in range(len(shape_rotations)):
        # 在該 rot 下的方塊相對座標
        rel = shape_rotations[rot]
        min_x = min(x for (_, x) in rel)
        max_x = max(x for (_, x) in rel)
        # 嘗試所有 x
        for target_x in range(0, cols):
            # 將方塊「置頂」到 (y, x) 再 drop
            # 這裡用 piece 的 x,y 為 0,0 來模擬，等價於絕對位置
            cells = [(y, x + target_x) for (y, x) in rel]
            # 往上拉，確保開始時不相交
            offset_up = 0
            while any(y < 0 for (y, _) in cells):
                cells = [(y+1, x) for (y, x) in cells]
                offset_up += 1
                if offset_up > rows: break
            if not _can_place(shot.status, cells):
                continue
            dropped = _drop_down(shot.status, cells)
            placed, lines = _place_and_fix(shot.status, dropped)
            agg_h, holes, bump = _board_features(placed)
            # 經典權重（可之後用學習來調）
            well_depth = max(heights) - min(heights)
            score = (0.75*agg_h) + (0.5*holes) + (0.25*bump) + (0.35*well_depth) - (1.0*lines)

            plans.append((score, rot, target_x, lines))
    # 分數越小越好（因為高度/空洞/起伏是成本、消行數是收益）
    plans.sort(key=lambda t: t[0])
    return plans

def plan_for_piece(shot, piece):
    plans = enumerate_plans(shot, piece)
    if not plans:
        # 沒解就直接原地瞬降
        return piece.rotation, piece.x, True
    _, best_rot, best_x, _ = plans[0]
    return best_rot, best_x, False

def next_move_commands(piece, target_rot, target_x):
    cmds = []
    # 先旋轉到目標
    cur = piece.rotation % len(config.shapes[piece.shape])
    while cur != (target_rot % len(config.shapes[piece.shape])):
        cmds.append("ROT")
        cur = (cur + 1) % len(config.shapes[piece.shape])
    # 再左右移動到目標 x
    while piece.x > target_x:
        cmds.append("L")
        piece.x -= 1  # 只是用來生成路徑（別擔心，外層會用真的 moveLeft）
    while piece.x < target_x:
        cmds.append("R")
        piece.x += 1
    # 最後瞬降
    cmds.append("DROP")
    return cmds
