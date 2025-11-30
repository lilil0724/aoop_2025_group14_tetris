import pygame as pg
import math # <-- 新增

# scale unit
unit = 6
grid = 6*unit
background_color = (33, 47, 60)
rows = 20
columns = 10

# --- 1v1 Layout Configuration ---
GARBAGE_BAR_WIDTH = 2 * unit 
BOARD_WIDTH = columns * grid
INFO_PANEL_WIDTH = 50 * unit

# P1 positions (調整 X 座標)
P1_GARBAGE_BAR_POS = (0, 0) # (x, y)
P1_OFFSET_X = P1_GARBAGE_BAR_POS[0] + GARBAGE_BAR_WIDTH
P1_INFO_X = P1_OFFSET_X + BOARD_WIDTH
P1_SCORE_POS = (P1_INFO_X + 10*unit, rows * grid // 2)
P1_LINE_POS = (P1_INFO_X + 10*unit, rows * grid // 2 + int(100 * (unit / 10)**1.5))
P1_NEXT_PIECE_POS = (P1_INFO_X + 22*unit, rows * grid // 2 - 30*unit)

# P2 positions (調整 X 座標)
P2_GARBAGE_BAR_POS = (P1_INFO_X + INFO_PANEL_WIDTH, 0) # (x, y)
P2_OFFSET_X = P2_GARBAGE_BAR_POS[0] + GARBAGE_BAR_WIDTH
P2_INFO_X = P2_OFFSET_X + BOARD_WIDTH
P2_SCORE_POS = (P2_INFO_X + 10*unit, rows * grid // 2)
P2_LINE_POS = (P2_INFO_X + 10*unit, rows * grid // 2 + int(100 * (unit / 10)**1.5))
P2_NEXT_PIECE_POS = (P2_INFO_X + 22*unit, rows * grid // 2 - 30*unit)

# Total screen size
# width = P2_INFO_X + INFO_PANEL_WIDTH
# height = rows * grid
width = 1600 # 增加寬度以容納更多玩家
height = 900
# --- End 1v1 Config ---


# others
fps = 60
difficulty = 30   # 調整降下的速度 (保留您的 30)
score_count = {
    1: 40,
    2: 100,
    3: 300,
    4: 1200
}
font = ('Comic Sans MS', int(100 * (unit / 10)**1.5))

ATTACK_BASE = {
    0: 0, # 0 行
    1: 0, # Single
    2: 1, # Double
    3: 2, # Triple
    4: 4  # Tetris
}

# Combo (REN) 攻擊加成
def get_combo_bonus(combo_count):
    # combo_garbage = max(0, floor((combo-1)/2))
    if combo_count <= 1:
        return 0
    return math.floor((combo_count - 1) / 2)

ATTACK_B2B_BONUS = 1         # B2B 加成 (目前僅 Tetris)
ATTACK_PERFECT_CLEAR = 4   # All Clear 額外加成

GARBAGE_INSERT_DELAY = 30  # 垃圾行插入節拍 (30 幀插入一次)
GARBAGE_LINES_PER_INSERT = 1 # 每次插入 1 行
GARBAGE_HOLE_REPEAT_PROB = 0.7 # 垃圾洞位沿用機率
GARBAGE_COLOR = (100, 100, 100) # 垃圾行顏色



# shapes: S, Z, I, O, J, L, T
shapes = {
    'S': [
        [(0, 0), (0, 1), (1, -1), (1, 0)],
        [(-1, 0), (0, 0), (0, 1), (1, 1)],
    ],
    'Z': [
        [(0, -1), (0, 0), (1, 0), (1, 1)],
        [(-1, 0), (0, -1), (0, 0), (1, -1)],
    ],
    'I': [
        [(-2, 0), (-1, 0), (0, 0), (1, 0)],
        [(-1, -2), (-1, -1), (-1, 0), (-1, 1)],
    ],
    'O': [
        [(0, -1), (0, 0), (1, -1), (1, 0)],
    ],
    'J': [
        [(-1, -1), (0, -1), (0, 0), (0, 1)],
        [(-1, 0), (-1, 1), (0, 0), (1, 0)],
        [(1, 1), (0, -1), (0, 0), (0, 1)],
        [(-1, 0), (1, -1), (0, 0), (1, 0)],
    ],
    'L': [
        [(-1, 1), (0, -1), (0, 0), (0, 1)],
        [(-1, 0), (0, 0), (1, 0), (1, 1)],
        [(1, -1), (0, -1), (0, 0), (0, 1)],
        [(-1, 0), (0, 0), (1, 0), (-1, -1)],
    ],
    'T': [
        [(-1, 0), (0, -1), (0, 0), (0, 1)],
        [(-1, 0), (1, 0), (0, 0), (0, 1)],
        [(1, 0), (0, -1), (0, 0), (0, 1)],
        [(-1, 0), (0, -1), (0, 0), (1, 0)],
    ],
}

SHADOW = (192, 192, 192)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (231, 76, 60)
PURPLE = (108, 52, 131)
BLUE = (40, 116, 166)
GREEN = (17, 122, 101)
YELLOW = (244, 208, 63)
LIGHT_PURPLE = (195, 155, 211)

shape_colors = {
    'S': GREEN,
    'Z': RED,
    'I': YELLOW,
    'O': BLUE,
    'J': PURPLE,
    'L': WHITE,
    'T': LIGHT_PURPLE
}