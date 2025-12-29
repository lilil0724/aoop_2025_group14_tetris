import pygame as pg

# Global settings
AI_MOVE_DELAY = 5 
AI_THINKING_DELAY = 10 
SHOW_GHOST = True  # 預設開啟投影

#  AI 速度等級 (1: Slow, 2: Normal, 3: Fast, 4: God/Instant)
AI_SPEED_LEVEL = 2

# Keybindings
KEYBINDS = {
    'P1': {
        'ROTATE': pg.K_w,
        'SOFT_DROP': pg.K_s,
        'LEFT': pg.K_a,
        'RIGHT': pg.K_d,
        'HARD_DROP': pg.K_SPACE, # Default for Solo/PvE
        'HARD_DROP_PVP': pg.K_LSHIFT, # Default for PVP
        'ROTATE_CCW': pg.K_l
    },
    'P2': {
        'ROTATE': pg.K_UP,
        'SOFT_DROP': pg.K_DOWN,
        'LEFT': pg.K_LEFT,
        'RIGHT': pg.K_RIGHT,
        'HARD_DROP': pg.K_RSHIFT
    }
}
